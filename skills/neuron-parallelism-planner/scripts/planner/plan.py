"""§4 sweep + selection: enumerate the config x context-bucket grid (via planner.build.cell)
and select a recommendation per bucket under the serving objective.

Gate-then-rank (design.md §4.2): SLA ceilings decide which configs are valid, per-node
throughput ranks the valid set. One replica serves BOTH phases, so a config is valid only
if it meets BOTH the prefill TTFT ceiling and the decode ITL ceiling. The grid knobs
(TP/CP/DP choices, N_BUCKETS, batch granularity) are the constants below.

SCOPE: single-node, colocated serving ONLY. Every planned config must fit inside one node
(`TP * DP <= hw.cores_per_node`); the grid enforces this in `build_grid`. Splitting prefill
and decode into separate pools (disaggregated serving) and any topology spanning 2+ nodes
or instances are out of scope and are not modeled -- see `max_cores()` below and
`entitlement_formulas._link_bw`, which raises rather than pricing a cross-node link.
"""

from __future__ import annotations

import math
import os
from typing import List

from planner.build import ModelBundle, ep_default_enabled, cell
from planner.shard import ShardConfig

USABLE_FRACTION = 0.90
# TP degrees are candidate divisors; each is kept per-model in build_grid only if it EXACTLY
# divides every TP-sharded dimension (query heads, vocab, MLP intermediate) -- see _tp_is_valid.
# So the grid is not powers-of-two only: a 96-Q-head model can use TP 6/12/24/48 when those
# also divide its vocab and intermediate, and the latency-optimal config can land on one of
# them (TP=48 meets a 400ms TTFT ceiling where TP=32 misses it).
TP_CHOICES = (4, 6, 8, 12, 16, 24, 32, 48)
CP_CHOICES = (1, 2, 4, 8)
# DP replicates within ONE node. Capped so that TP*DP can never exceed a single node's
# logical-core count (see max_cores); multi-node replication is out of scope.
DP_CHOICES = (1, 2, 4, 8, 16)
BUCKET_STEP = 4096
N_BUCKETS = 6
BS_GRANULARITY = 2
PREFILL_ITPS_FLOOR = 30000.0


def max_cores(hw) -> int:
    """Hard ceiling on cores a single planned config may use = ONE node's logical cores.

    Single-node serving is a scope constraint, not a tunable: a config spanning 2+ nodes is
    never emitted. This replaces the former module-level MAX_CORES=256 (four 64-core nodes),
    which allowed the sweep to return multi-node configs that were then silently priced at
    the cross-node link bandwidth.
    """
    return hw.cores_per_node

# Axes the sweep may use. A DISABLED axis is pinned to degree 1 (EP: pinned OFF), so it never
# appears in any planned config -- for what-if analysis (e.g. TP-only planning). PP is NOT an
# axis here: it is out of scope for the whole engine (not implemented by vllm-neuron;
# ShardConfig has no PP field), so it is excluded rather than default-disabled. Nothing in
# {TP,CP,DP,EP} is disabled by default.
#
# This is a BEHAVIOR knob, and it is CLI-ONLY: `--disable-cp` / `--enable-cp` etc. in __main__,
# or set_disabled_axes() in-process. It used to also read $PARALLELISM_PLANNER_DISABLE_AXES; that
# was removed because an exported variable silently changes what every later run plans -- a
# reader of the output cannot see it, and this family now reads exactly two environment variables
# (NEURON_PROJECT_ROOT, NEURON_AGENTIC_DEVELOPMENT_ROOT), both of which are locations, not
# behavior. Behavior belongs on the command line, where it appears in the transcript.
GRID_AXES = ("TP", "CP", "DP", "EP")

DISABLED_AXES = set()


def set_disabled_axes(axes) -> None:
    """Replace the disabled-axis set (used by the CLI). Each entry must be in GRID_AXES."""
    axes = {a.strip().upper() for a in axes}
    bad = axes - set(GRID_AXES)
    if bad:
        raise ValueError("unknown axes %r; valid axes are %s."
                         % (sorted(bad), ", ".join(GRID_AXES)))
    global DISABLED_AXES
    DISABLED_AXES = axes


def _axis_choices(name, full):
    """Degrees this axis may take: (1,) if disabled, else its full choice tuple."""
    return (1,) if name in DISABLED_AXES else full


def usable_hbm(hw) -> float:
    # Per-core memory budget = capacity x usable fraction. The fraction is a HW field
    # (settable in the hardware JSON, overridable per run-spec / CLI); USABLE_FRACTION is
    # the module default/fallback for a HW built without one. Signature stays hw-only so
    # every call site (build_grid, max_feasible_bs, prefix_cache_headroom, emit._min_footprint)
    # is untouched.
    return hw.hbm_capacity_bytes_per_core * getattr(hw, "usable_fraction", USABLE_FRACTION)


def _activation_bytes(a, S, bs, prefill, TP, CP) -> float:
    live = (S * bs / CP) if prefill else bs
    return 4 * live * a.hidden_size * a.dtype_bytes / TP


def peak_hbm(b: ModelBundle, cfg: ShardConfig, S, bs, prefill) -> float:
    c = cell(b, cfg, S, max(bs, 0) if bs else 0, prefill)
    act = _activation_bytes(b.arch, S, bs, prefill, cfg.TP, cfg.CP)
    return c["per_device_weight_bytes"] + c["per_device_kv_bytes"] + act


def max_feasible_bs(b: ModelBundle, cfg: ShardConfig, S, prefill) -> int:
    budget = usable_hbm(b.hw)
    fixed = peak_hbm(b, cfg, S, 0, prefill)                 # weights (bs-independent)
    per_tok = peak_hbm(b, cfg, S, 1, prefill) - fixed
    if per_tok <= 0:
        return 1 if fixed <= budget else 0
    bs = int((budget - fixed) // per_tok)
    if bs < 1:
        return 0
    return max(bs - bs % BS_GRANULARITY, 1)


def _ep_enabled(a):
    """EP on iff the model has routed experts AND the EP axis is not disabled."""
    return ep_default_enabled(a) and "EP" not in DISABLED_AXES


def _dp_choices(a):
    # DP only widens the grid when EP is live (world = TP*DP); otherwise DP is a free outer
    # multiplier, not a swept axis. Honors a disabled DP axis.
    full = DP_CHOICES if _ep_enabled(a) else (1,)
    return _axis_choices("DP", full)


def max_q_heads(a) -> int:
    """The TP ceiling: TP cannot exceed the query-head count of the narrowest attention group
    (see references/tp-feasibility-and-kv-replication.md)."""
    return min(g.n_q_heads for g in a.attn_groups)


def _tp_is_valid(a, TP) -> bool:
    """TP must EXACTLY divide every dimension it shards, and not exceed the query-head ceiling.

    A non-divisor TP would need padding, which the cost model does not represent and which the
    serving stack may reject outright -- so recommending one here risks a config that is then
    BLOCKED at deploy (e.g. "TP=24 ... blocked by vocab_size % 24 != 0"). The TP-sharded
    dimensions, all checked here:
      - each attention group's query-head count -- Q/O projections shard on it. (K/V replicate
        to `min(TP, n_kv_heads)`, so they never gate TP; see max_q_heads.)
      - the vocab table -- the embedding / LM-head is vocab-sharded.
      - each PRESENT MLP intermediate width -- the always-on dense path and, when the layer has
        routed experts, the per-expert MoE intermediate (gate/up/down shard the intermediate).

    This is deliberately conservative toward stacks that PAD vocab/intermediate up to a TP
    multiple: such a config is technically runnable, but its (small) padding cost is unmodeled,
    so the planner declines to recommend it rather than quote an optimistic floor -- the same
    stance the head check already takes.
    """
    if TP > max_q_heads(a):
        return False
    if a.vocab_size % TP:
        return False
    for g in a.attn_groups:
        if g.n_q_heads % TP:
            return False
        if g.dense_inter(a) and g.dense_inter(a) % TP:
            return False
        if g.num_experts(a) and g.moe_inter(a) % TP:
            return False
    return True


def build_grid(b: ModelBundle) -> List[ShardConfig]:
    a, hw = b.arch, b.hw
    ep = _ep_enabled(a)
    cap = usable_hbm(hw)
    core_cap = max_cores(hw)
    grid = []
    for TP in _axis_choices("TP", TP_CHOICES):
        if not _tp_is_valid(a, TP):
            continue
        for CP in _axis_choices("CP", CP_CHOICES):
            if TP % CP:
                continue
            for DP in _dp_choices(a):
                cores = TP * DP
                # single-node scope: a config may never span more than one node.
                if cores > core_cap:
                    continue
                cfg = ShardConfig(TP=TP, CP=CP, DP=DP, ep_enabled=ep)
                # weight-fit: per-device weights (bs=0) must fit one core.
                if peak_hbm(b, cfg, 8192, 0, False) > cap:
                    continue
                grid.append(cfg)
    return sorted(grid, key=lambda c: (c.world, c.TP))


def _bucket_best_efficiency(b: ModelBundle, grid, S) -> float:
    """Best decode itl-otps tradeoff at context S: max over configs of (per-node throughput /
    ITL) at each config's max-feasible batch. Rewards a context that delivers high aggregate
    throughput at low inter-token latency."""
    best = 0.0
    N = b.hw.cores_per_node
    for cfg in grid:
        bs = max_feasible_bs(b, cfg, S, False)
        if not bs:
            continue
        c = cell(b, cfg, S, bs, False)
        if c["step_ms"] <= 0:
            continue
        otps_per_node = (bs * 1e3 / c["step_ms"]) * N / c["cores"]
        eff = otps_per_node / c["step_ms"]
        best = max(best, eff)
    return best


def select_buckets(b: ModelBundle, n=N_BUCKETS) -> List[int]:
    """n context buckets by best itl-otps tradeoff, from the 4k-multiple grid. Full context is
    pinned last; the rest are log-spread into n-1 geometric bins, each keeping the
    best-efficiency 4k-multiple."""
    mx_ctx = b.arch.max_context
    mx = (mx_ctx // BUCKET_STEP) * BUCKET_STEP
    cands = list(range(2 * BUCKET_STEP, mx + 1, BUCKET_STEP))
    if len(cands) <= n:
        return cands
    lo, hi = float(cands[0]), float(mx)
    edges = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
    grid = build_grid(b)
    picks = []
    for blo, bhi in zip(edges[:-1], edges[1:]):
        binc = [S for S in cands if blo - 1 <= S <= bhi + 1 and S < mx and S not in picks]
        if not binc:
            continue
        picks.append(max(binc, key=lambda S: _bucket_best_efficiency(b, grid, S)))
    return sorted(set(picks))[:n - 1] + [mx]


def min_kv_heads(a) -> int:
    return min(g.n_kv_heads for g in a.attn_groups)


def ttft_ceiling_ms(S, itps_floor=PREFILL_ITPS_FLOOR) -> float:
    """Prefill TTFT SLA: constant input-token-throughput floor -> ceiling linear in context."""
    return S / itps_floor * 1e3


# --- shared building blocks ----------------------------------------------------

def is_pareto(row, others) -> bool:
    """3-axis decode Pareto within a bucket: high throughput (otps), low ITL (step_ms), low
    cores. A row is dominated iff some other row is >= otps AND <= ITL AND <= cores, strictly
    better on at least one. Mirrors the old sweep.py::is_pareto."""
    for o in others:
        if (o["otps"] >= row["otps"] and o["step_ms"] <= row["step_ms"]
                and o["cores"] <= row["cores"]):
            if (o["otps"] > row["otps"] or o["step_ms"] < row["step_ms"]
                    or o["cores"] < row["cores"]):
                return False
    return True


def is_pareto_pn(row, others) -> bool:
    """2-axis DP-normalized decode Pareto within a bucket: high per-node throughput
    (tput_per_node), low ITL (step_ms). Footprint is normalized out via no-overhead DP, so
    cores drops as an axis. Mirrors the old visualize.py::_is_pareto_2d."""
    for o in others:
        if o["tput_per_node"] >= row["tput_per_node"] and o["step_ms"] <= row["step_ms"]:
            if o["tput_per_node"] > row["tput_per_node"] or o["step_ms"] < row["step_ms"]:
                return False
    return True


def decode_grid(b: ModelBundle, grid, S) -> list:
    """Every fitting config's decode cell at max-feasible batch, annotated with per-node
    throughput and BOTH Pareto flags (3-axis `pareto`, DP-normalized 2-axis `pareto_pn`).
    This is the full landscape the reports show (transparency); selection picks a row from it,
    it is not itself filtered."""
    N = b.hw.cores_per_node
    rows = []
    for cfg in grid:
        bs = max_feasible_bs(b, cfg, S, False)
        if not bs:
            continue
        c = cell(b, cfg, S, bs, False)
        otps = bs * 1e3 / c["step_ms"]
        rows.append({**c, "max_bs": bs, "otps": otps, "tput_per_node": otps * N / c["cores"]})
    for r in rows:
        r["pareto"] = is_pareto(r, [o for o in rows if o is not r])
        r["pareto_pn"] = is_pareto_pn(r, [o for o in rows if o is not r])
    return sorted(rows, key=lambda r: (r["cores"], r["TP"]))


def _bs_ladder(max_bs: int) -> list:
    """A batch ladder from 1 up to max_bs for the interactivity sweep: powers of two, plus
    max_bs itself, each snapped to BS_GRANULARITY parity (bs=1 exempt), deduped, sorted.
    O(log max_bs) points -- enough to trace the ITL/throughput curve without a dense scan."""
    if max_bs < 1:
        return []
    xs, b = [1], 2
    while b < max_bs:
        xs.append(b)
        b *= 2
    xs.append(max_bs)
    snapped = {(x if x == 1 else x - x % BS_GRANULARITY) for x in xs if x >= 1}
    return sorted(v for v in snapped if v >= 1)


def decode_bs_sweep(b: ModelBundle, grid, S) -> list:
    """Interactivity landscape: every fitting config swept over a batch ladder (1 -> max
    feasible bs), not just at max batch like decode_grid. Each (config, bs) is a full decode
    cell annotated with otps / tput_per_node / peak_hbm_gb, flagged `pareto_pn` (the existing
    DP-normalized per-node-throughput-vs-ITL dominance). This is the below-max-batch companion
    to decode_grid: it traces the ITL/throughput tradeoff the throughput-capacity landscape
    (all cells at max batch, ITL ~flat) cannot show. The max-bs point of each config's sweep
    coincides with that config's decode_grid row."""
    N = b.hw.cores_per_node
    rows = []
    for cfg in grid:
        mbs = max_feasible_bs(b, cfg, S, False)
        if not mbs:
            continue
        for bs in _bs_ladder(mbs):
            c = cell(b, cfg, S, bs, False)
            if c["step_ms"] <= 0:
                continue
            otps = bs * 1e3 / c["step_ms"]
            rows.append({**c, "max_bs": mbs, "otps": otps,
                         "tput_per_node": otps * N / c["cores"],
                         "peak_hbm_gb": peak_hbm(b, cfg, S, bs, False) / 1e9})
    for r in rows:
        r["pareto_pn"] = is_pareto_pn(r, [o for o in rows if o is not r])
    return sorted(rows, key=lambda r: (r["cores"], r["TP"], r["bs"]))


def ttft_by_config(b: ModelBundle, grid, S) -> dict:
    """Prefill TTFT at bs=1 for every config -> {(TP,CP,DP): ttft_ms}. The SLA-validity gate
    (§4.2): a config is valid at S iff its bs=1 TTFT <= ttft_ceiling(S)."""
    out = {}
    for cfg in grid:
        c = cell(b, cfg, S, 1, True)
        out[(cfg.TP, cfg.CP, cfg.DP)] = c["step_ms"]
    return out


def prefill_grid(b: ModelBundle, grid, S) -> list:
    """Every fitting config's PREFILL cell at max-feasible batch -- the prefill counterpart of
    decode_grid. Mirrors the old sweep's prefill half (prefill evaluated at its own
    max-feasible bs, not bs=1)."""
    rows = []
    for cfg in grid:
        bs = max_feasible_bs(b, cfg, S, True)
        if not bs:
            continue
        c = cell(b, cfg, S, bs, True)
        rows.append({**c, "max_bs": bs})
    return sorted(rows, key=lambda r: (r["cores"], r["TP"]))


def sweep_cells(b: ModelBundle, grid=None, buckets=None):
    """Full sweep: every (config x bucket) as a COMPLETE cell dict, decode and prefill each at
    its max-feasible batch. Mirrors the old sweep.py::sweep_cells -- the raw material behind
    both selection and the saved sweep_results.json. Returns (decode_cells, prefill_cells),
    flat across buckets."""
    grid = build_grid(b) if grid is None else grid
    buckets = select_buckets(b) if buckets is None else buckets
    decode, prefill = [], []
    for S in buckets:
        decode.extend(decode_grid(b, grid, S))
        prefill.extend(prefill_grid(b, grid, S))
    return decode, prefill


def sweep_results(b: ModelBundle) -> dict:
    """The full-sweep dump (same shape as the old sweep_results.json): grid/bucket metadata
    plus the COMPLETE decode + prefill cell dicts. Serialized to disk by the CLI; nothing in
    this dict is slimmed -- it is every cell's full output at max-feasible bs."""
    grid = build_grid(b)
    buckets = select_buckets(b)
    decode, prefill = sweep_cells(b, grid, buckets)
    # interactivity landscape: the below-max-batch bs sweep, flat across buckets.
    decode_bs = [r for S in buckets for r in decode_bs_sweep(b, grid, S)]
    return {"model": b.raw.get("model"), "hardware": b.raw.get("hardware_name", "hw"),
            "disabled_axes": sorted(DISABLED_AXES),
            "grid_size": len(grid), "buckets": buckets, "bs_policy": "max_feasible",
            "decode": decode, "prefill": prefill, "decode_bs_sweep": decode_bs}


def _cfg_key(r) -> tuple:
    return (r["TP"], r["CP"], r.get("DP", 1))


def _annotate(dgrid, ttfts, ttft_ceil, itl_ceil):
    """Attach each decode row's bs=1 TTFT and both SLA verdicts (§4.2 gates)."""
    out = []
    for r in dgrid:
        t = ttfts.get(_cfg_key(r))
        if t is None:
            continue
        out.append({**r, "ttft_ms": t, "ttft_ceiling_ms": ttft_ceil, "itl_ceiling_ms": itl_ceil,
                    "ttft_ok": t <= ttft_ceil, "itl_ok": r["step_ms"] <= itl_ceil})
    return out


def select_colocated(dgrid, ttfts, ttft_ceil, itl_ceil) -> dict:
    """§4.2 colocated: a config is VALID iff it meets BOTH the prefill TTFT ceiling (bs=1) and
    the decode ITL ceiling (max-feasible bs). Among the valid set, pick max per-node decode
    throughput (tie -> fewer cores). Unreachable -> closest to admissible (see _fallback)."""
    rows = _annotate(dgrid, ttfts, ttft_ceil, itl_ceil)
    if not rows:
        return None
    valid = [r for r in rows if r["ttft_ok"] and r["itl_ok"]]
    if valid:
        best = max(valid, key=lambda r: (r["tput_per_node"], -r["cores"]))
        return {**best, "sla_met": True}
    return _fallback(rows, both=True)


def _fallback(rows, both) -> dict:
    """Unreachable-gate fallback for colocated: the config closest to admissible. With both
    gates, prefer the one that misses by the least on the phase that fails -- rank by whether
    each gate is met, then by the tighter miss. Simplest robust choice: lowest TTFT among those
    meeting ITL if any, else lowest TTFT overall."""
    itl_ok = [r for r in rows if r["itl_ok"]]
    pool = itl_ok or rows
    return {**min(pool, key=lambda r: (r["ttft_ms"], r["cores"])), "sla_met": False}


# --- selection modes -----------------------------------------------------------

def plan(b: ModelBundle, serving="colocated", prefill_itps_floor=PREFILL_ITPS_FLOOR,
         decode_itl_ceiling_ms=50.0, prefill_ttft_ceiling_ms=None) -> dict:
    """Per-bucket plan (§4.2). Gate on the SLA ceilings, rank valid configs by per-node
    throughput.

    Serving is COLOCATED and SINGLE-NODE by construction: one replica serves both phases, so
    a config is VALID iff its bs=1 prefill TTFT <= ceiling AND its max-bs decode ITL <=
    ceiling. Valid configs are ranked by per-node throughput. Splitting the phases into
    separate prefill/decode pools is out of scope; `serving` is accepted only as "colocated"
    and any other value raises.

    prefill_ttft_ceiling_ms: flat prefill TTFT ceiling (ms) applied at every context; when set,
    it replaces the context-scaled ttft_ceiling(S) = S / prefill_itps_floor.
    """
    if serving != "colocated":
        raise ValueError(
            f"serving={serving!r} is not supported: this planner models single-node, "
            "colocated serving only. Disaggregated / split prefill-decode pools and any "
            "multi-node or multi-instance topology are out of scope."
        )
    grid = build_grid(b)
    buckets = select_buckets(b)
    out = {"serving": "colocated", "buckets": buckets, "grid_size": len(grid),
           "disabled_axes": sorted(DISABLED_AXES),
           "prefill_itps_floor": prefill_itps_floor,
           "prefill_ttft_ceiling_ms": prefill_ttft_ceiling_ms,
           "decode_itl_ceiling_ms": decode_itl_ceiling_ms,
           "max_cores": max_cores(b.hw),
           "decode_grid": {}, "decode_bs_grid": {}, "selected": {}}

    for S in buckets:
        dgrid = decode_grid(b, grid, S)
        ttfts = ttft_by_config(b, grid, S)
        ttft_ceil = (prefill_ttft_ceiling_ms if prefill_ttft_ceiling_ms is not None
                     else ttft_ceiling_ms(S, prefill_itps_floor))
        itl_ceil = decode_itl_ceiling_ms
        out["decode_grid"][S] = dgrid
        # below-max-batch interactivity landscape (consumed by visualize.pareto_per_node)
        out["decode_bs_grid"][S] = decode_bs_sweep(b, grid, S)
        out["selected"][S] = select_colocated(dgrid, ttfts, ttft_ceil, itl_ceil)
    return out


def kv_replication_factor(a, TP) -> float:
    """Worst-case KV replication across groups at this TP: max over groups of TP/min(TP,n_kv).
    1 = no group replicates (TP <= min_kv_heads); >1 = the cache is duplicated that many ways."""
    return max(TP / min(TP, g.n_kv_heads) for g in a.attn_groups)


def prefix_cache_headroom(b: ModelBundle, cfg: ShardConfig, S: int) -> dict:
    """HBM left for a prefix cache after weights + bs=1 activations, and how many full-context
    sequences it holds. Replication-aware: the per-device KV bytes already divide by the
    min(TP,n_kv) ceiling, so a replicating config self-reports a larger per-seq KV and thus
    fewer cacheable sequences -- no extra penalty needed. A CAPACITY statement, not a benefit
    one (hit rate needs a traffic model we do not have). Applies to the single colocated
    replica; there is no separate prefill pool to size."""
    c = cell(b, cfg, S, 1, True)
    weights = c["per_device_weight_bytes"]
    act = _activation_bytes(b.arch, S, 1, True, cfg.TP, cfg.CP)
    free = usable_hbm(b.hw) - weights - act
    kv_per_seq = c["per_device_kv_bytes"]   # one full-context seq, this device (bs=1)
    seqs = free / kv_per_seq if kv_per_seq > 0 else 0.0
    return {"free_hbm_gb": free / 1e9, "kv_bytes_per_seq_gb": kv_per_seq / 1e9,
            "cacheable_full_ctx_seqs": seqs,
            "global_kv_replication": kv_replication_factor(b.arch, cfg.TP)}
