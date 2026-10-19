"""Per-OPERATOR hardware-entitlement floor for a single parallelism config.

The config-level entitlement (config_entitlement.py) gives the binding bound for a
whole forward step. This script OPENS THAT BOUND UP to the operator level: for one
``{TP, PP, CP, bucket, bs, prefill}`` cell it reports, for every matmul-class
operator the model owns, the minimum-viable per-device

  - compute floor   = op FLOPs / TE peak
  - weight-read floor = op weight bytes / HBM BW
  - KV-read floor     = op KV bytes / HBM BW   (attention only)
  - activation interface volume = input + output bytes

The default model floor defines attention and FFN components as their compute bodies after
the input gather and before the output reduce-scatter. Collectives are not contained within
them. Internal operator interfaces are replaced with one mandatory body-input read and one
body-output write. The sequence-parallel residual/norm bridge is outside the current floor.
An explicit ``per_operator`` run-spec policy instead charges every listed interface. The
MUST-binding is the max of the compute floor and selected mandatory HBM contract. A measured
per-kernel profile is diffed operator-for-operator against these floors to localize the gap
and route each bottleneck.

DATA SOURCES (split by trustworthiness):
  - attention-projection, FFN/router, and lm_head FLOPs and parameter shapes come from the
    shape-exact meta-device ORACLE.
  - SDPA FLOPs, KV/state bytes, activation interfaces, and embedding/lm-head table
    corrections are ANALYTIC planner-owned terms.

SHARDING is per-device and comes from ``planner.shard`` -- the engine's single owner of every
parallelism divisor -- never restated here: Q/O and the MLP shard 1/TP; K/V projections and
the KV cache shard only to ``min(TP, n_kv_heads)`` and then replicate; CP shards the KV
sequence on decode and prefill token-compute only under DCP prefill (standard prefill keeps
the whole prompt at full TP); EP shards the routed-expert weights across TP*EP; the MoE router
(``moe_router``) is REPLICATED, so it is the one row whose per-device weight does not fall as
sharding widens; PP holds ``num_layers/PP`` of each layer type.

A "reconciliation" block sums the per-op component floors and checks them against the
analytic config-level cell (the planner's ``cell_entitlement``), proving the operator
breakdown is a DECOMPOSITION of the same bound, not a parallel re-derivation.

The planner also returns its native component summaries. Cost-equivalent bodies are
compressed by component variant after receiving ``max(TE, HBM)``. This script exposes those
summaries instead of rebuilding the same floor from operator rows. Collectives remain
outside the component-only result and are added by the config-level planner. The certified
binding-engine scope is explicitly ``["te", "hbm"]``. An engine outside that scope is
unmodeled by this floor and excluded from its binding maximum; it may still bind measured
execution and is not zero work.

    python3.12 operator_entitlement.py --tp 8 --bucket 8192 --bs 8            # decode
    python3.12 operator_entitlement.py --tp 32 --cp 4 --bucket 131072 --bs 1 --prefill
    python3.12 operator_entitlement.py --runspec ./agent_artifacts/tmp/my-model-run.json ...
"""

import argparse
import json
import os
import sys

import _planner

_planner.ensure_engine_on_path()

from entitlement_counter import EntitlementCounter  # noqa: E402
# `embed_table_bytes` moved from the retired sweep.py into entitlement_formulas.py (878948e);
# `cell_entitlement` has no modern equivalent and is re-exposed by _planner on the current
# engine (see _planner.cell_entitlement).
from entitlement_formulas import active_attention_pairs, embed_table_bytes  # noqa: E402
from oracle_model import build_oracle, make_inputs  # noqa: E402
# The planner's cost engine owns component grouping/composition policy.
from planner.activation import (  # noqa: E402
    ATTN_OPS,
    COMPONENT_BOUNDARY,
    MLA_ATTN_OPS,
    PER_OPERATOR,
    act_io_elems as _shared_act_io_elems,
    group_of_layer as _ordered_group_of_layer,
    operator_activation_bytes,
)
from planner.cost import CERTIFIED_ENGINE_SCOPE, OperatingPoint  # noqa: E402
# planner.shard is the engine's declared single owner of every parallelism divisor
# (TP/CP/DP/EP). Deriving the per-op divisors FROM it, instead of restating TP/CP arithmetic
# here, is what keeps this breakdown a decomposition of the analytic cell instead of a
# parallel derivation that drifts -- it had drifted twice: K/V projections divided by TP
# rather than min(TP, n_kv), and standard prefill divided token-compute by CP, which the
# engine does not do (CP only shards prefill compute under DCP prefill).
from planner.shard import (  # noqa: E402
    ShardConfig, flop_divisor, kv_cache_divisor, weight_divisor,
)

HERE = os.path.dirname(os.path.abspath(__file__))

MLP_OPS = ("gate_proj", "up_proj", "down_proj")
# Row-name prefix for the ROUTED-expert copy of an FFN op. The oracle runs two parallel FFN
# paths per layer -- an always-on dense MLP (`dense`, which is also where arch_from_hf folds
# any shared experts) and a routed pool (`moe`) -- at two different intermediate widths. The
# dense path keeps the bare op names (in a pure-dense model it is the only FFN), so routed
# rows are prefixed to keep the rollup keys distinct.
_MOE_PREFIX = "moe_"

# Each operator's weight/FLOP class in planner.shard's taxonomy. MLA's whole low-rank set is
# `attn_qo`: the engine folds it there (attn_kv_params returns 0 for MLA), so no MLA op ever
# takes the K/V replication ceiling.
_OP_CLASS = {
    "q_proj": "attn_qo", "o_proj": "attn_qo",
    "k_proj": "attn_kv", "v_proj": "attn_kv",
    "q_a_proj": "attn_qo", "q_b_proj": "attn_qo",
    "kv_a_proj": "attn_qo", "kv_b_proj": "attn_qo",
    "sdpa": "attn_score", "lm_head": "embed",
    # the MoE gate: its own class, because it is REPLICATED while the experts shard on world.
    "moe_router": "router",
}


def class_of(op, is_mla=False):
    """The planner.shard weight/FLOP class an operator row belongs to."""
    if op in _OP_CLASS:                     # explicit map wins over the moe_ prefix rule
        cls = _OP_CLASS[op]
        return "attn_qo" if (is_mla and cls == "attn_kv") else cls
    if op.startswith(_MOE_PREFIX):
        return "routed_experts"
    if op in MLP_OPS:
        return "dense_mlp"
    raise KeyError(op)


def shard_config(arch, cfg):
    """(ShardConfig, regime) for a cell, in the engine's own terms.

    The old expert-parallel DEGREE E maps to DP=E with EP enabled, exactly as
    ``_planner.cell_entitlement`` maps it, so both sides shard routed experts identically.
    The regime distinguishes standard prefill from DCP prefill because they shard attention
    differently (``apply_prefill_dcp`` is a run-spec serving knob, not a config fact).
    """
    EP = int(cfg.get("EP", 1))
    sh = ShardConfig(TP=cfg["TP"], CP=cfg["CP"], DP=max(EP, 1), ep_enabled=EP > 1)
    if not cfg["prefill"]:
        return sh, "decode"
    return sh, "prefill_dcp" if getattr(arch, "apply_prefill_dcp", False) else "prefill"


def group_of_layer(arch, i):
    """(group_index, AttnGroup) owning layer i in exact model execution order."""
    return _ordered_group_of_layer(arch, i)


def group_label(arch, gi, g):
    """Human label for a group: 'full' (uncapped) or 'sliding' (windowed); when two
    groups share a kind, the index disambiguates so rollup keys never collide."""
    kind = "full" if g.kv_seq_cap is None else "sliding"
    kinds = [("full" if x.kv_seq_cap is None else "sliding") for x in arch.attn_groups]
    return kind if kinds.count(kind) == 1 else f"{kind}{gi}"


def _subtree(flop_counts, fqn):
    d = flop_counts.get(fqn)
    return sum(d.values()) if d else 0


def oracle_counts(arch, prefill, bucket):
    """One meta-device forward -> (flop_counts, weight_bytes). bs=1; FLOPs are linear
    in batch so callers scale by the real bs."""
    model = build_oracle(arch)
    q_len = bucket if prefill else 1
    with EntitlementCounter(model, depth=6) as c:
        model(make_inputs(1, q_len), bucket, prefill)
    rep = c.report()
    return rep.flop_counts, rep.weight_bytes


def sdpa_flops_one_layer(g, S, prefill):
    """Analytic QK^T + PV FLOPs for one layer and one sequence.

    Decode has one query over the effective cache length. Empty-prefix prefill
    uses the exact causal active-pair count; multiplying a length-S per-token
    formula by S would include masked future positions.
    """
    positions = (
        active_attention_pairs(S, g.kv_seq_cap)
        if prefill
        else (S if g.kv_seq_cap is None else min(S, g.kv_seq_cap))
    )
    if g.mla is not None:
        score_dim = g.mla.qk_head_dim + g.mla.v_head_dim
        return 2 * g.n_q_heads * score_dim * positions
    return 4 * g.n_q_heads * g.head_dim * positions


def kv_bytes_one_layer(arch, g, S, bs):
    """Analytic KV cache bytes one attention layer reads, whole batch (matches
    ``kv_bytes_total``). Returns (bytes, kv_shards); the caller shards by min(TP, kv_shards)*CP.

    GQA/MHA: factor 2 = K and V; per_tok = 2*n_kv_heads*head_dim; shards to min(TP, n_kv).
    MLA: ONE compressed latent (kv_lora_rank + qk_rope_head_dim) per token per layer -- no
    factor 2, no n_kv_heads. The latent is MQA-like and fully REPLICATED across TP (not
    sharded), so kv_shards=1.

    Element size is ``kv_elem_bytes``, NOT the model dtype: the KV cache is a serving choice
    and is often held at a lower precision than the weights (run-spec ``kv_dtype_bytes``).
    Charging the model dtype here would report a 2x-high KV floor for an fp8-KV run and put
    this breakdown out of step with the analytic cell it is supposed to decompose."""
    eff_ctx = S if g.kv_seq_cap is None else min(S, g.kv_seq_cap)
    if g.mla is not None:
        return g.mla.kv_cache_dim * arch.kv_elem_bytes * eff_ctx * bs, 1
    return 2 * g.n_kv_heads * g.head_dim * arch.kv_elem_bytes * eff_ctx * bs, g.n_kv_heads


def act_io_elems(op, arch, g, cfg, regime="decode", inter=None):
    """Compatibility surface for the planner-owned activation shape authority."""
    sh = cfg if isinstance(cfg, ShardConfig) else ShardConfig(TP=int(cfg))
    return _shared_act_io_elems(op, arch, g, sh, regime, inter)


def routed_flops(flop_counts, moe_fc, op, k):
    """Routed-expert FLOPs for an MLP op: sum the k experts the oracle computed. FLOPs
    route to k experts regardless of EP -- a token's expert work is the same whether the
    expert is local or fetched, so this is divided by TP only (never EP), matching the
    analytic matmul (which gives EP no compute benefit)."""
    return sum(_subtree(flop_counts, f"{moe_fc}.experts.{e}.{op}") for e in range(k))


def routed_weight(weight_bytes, moe_wb, op, num_experts):
    """Resident routed-expert weight bytes for an MLP op: ALL n_routed experts occupy HBM
    (FLOPs route to k, weights stay resident). Under EP these shard num_local = n_routed/EP
    per device, so the caller divides this by TP*EP."""
    return sum(weight_bytes.get(f"{moe_wb}.experts.{e}.{op}", 0) for e in range(num_experts))


def build_op_rows(arch, hw, cfg):
    """Per-instance operator rows with per-device sharded FLOPs/weight/KV + floor times.

    Attention is MLA (q_a/q_b/kv_a/kv_b/o) or GQA (q/k/v/o) per group. The FFN is the SUM of
    two parallel paths, either of which may be absent: an always-on dense MLP (bare op names,
    at ``dense_inter``, incl. the shared experts ``arch_from_hf`` folds in) and a routed pool
    (``moe_``-prefixed rows, at ``moe_inter``).

    Every divisor comes from ``planner.shard`` -- including the routed-expert classes -- so the
    per-op floors shard exactly as the analytic cell does. Note what the engine's EP model
    actually is (design.md 2.2): tokens are NOT dispatched to a core subset. Every rank in the
    ``world = TP*DP`` holds all tokens and runs only its LOCAL experts, and an all-reduce
    combines the partial sums. So a rank's share of a token's k-expert work is
    ``k/(ep_degree*ep_tp) = k/world``, and BOTH the routed weights and the routed FLOPs divide
    by ``world``. (Under a token-dispatch implementation the compute would divide by TP alone;
    that is a different machine from the one this engine models.)
    """
    TP, PP, CP = cfg["TP"], cfg["PP"], cfg["CP"]
    EP = int(cfg.get("EP", 1))
    S, bs, prefill = cfg["bucket"], cfg["bs"], cfg["prefill"]
    flop_counts, weight_bytes = oracle_counts(arch, prefill, S)

    te, bw = hw.te_peak_flops, hw.hbm_bw_bytes_per_s
    sh, regime = shard_config(arch, cfg)
    operating_point = OperatingPoint(S=S, bs=bs, regime=regime)
    activation_policy = getattr(arch, "activation_io_policy", COMPONENT_BOUNDARY)
    # The oracle is built at the MODEL dtype; an FP8 run-spec serves weights at
    # `weight_dtype_bytes`. Scale the oracle's weight bytes to the served precision, or an FP8
    # config reports the BF16 weight floor -- 2x high on the term that binds decode.
    w_scale = arch.w_elem_bytes / arch.dtype_bytes

    def w_div(op, g):
        return weight_divisor(class_of(op, g.mla is not None), sh, regime,
                              n_kv=g.n_kv_heads, is_mla=g.mla is not None)

    def f_div(op, g):
        return flop_divisor(class_of(op, g.mla is not None), sh, regime, cap=g.kv_seq_cap,
                            n_kv=g.n_kv_heads, is_mla=g.mla is not None)

    def add_row(rows, layer, gi, glabel, g, op, flops, wbytes, kvbytes):
        act_in, act_out = operator_activation_bytes(
            op, arch, g, sh, operating_point)
        act_bytes = act_in + act_out
        # In component-boundary mode these interfaces remain decomposition metadata; the
        # planner charges the component graph cut. Per-operator mode explicitly replaces
        # that graph cut with each row's interface traffic.
        mem_bytes = wbytes + kvbytes + (
            act_bytes if activation_policy == PER_OPERATOR else 0.0)
        t = {
            "compute_ms": flops / te * 1e3,
            "weight_read_ms": wbytes / bw * 1e3,
            "kv_read_ms": kvbytes / bw * 1e3,
            "activation_io_ms": act_bytes / bw * 1e3,
            "memory_ms": mem_bytes / bw * 1e3,
        }
        binding = "compute" if t["compute_ms"] >= t["memory_ms"] else "memory"
        rows.append({"layer": layer, "group_index": gi, "group": glabel, "op": op,
                     "flops": flops, "weight_bytes": wbytes, "kv_bytes": kvbytes,
                     "act_input_bytes": act_in, "act_output_bytes": act_out,
                     "act_bytes": act_bytes, "mem_bytes": mem_bytes, **t,
                     "binding": binding,
                     "arith_intensity": (flops / mem_bytes) if mem_bytes else 0.0})

    rows = []
    n_layers = arch.num_layers
    for i in range(n_layers):
        gi, g = group_of_layer(arch, i)
        glabel = group_label(arch, gi, g)
        attn_fc, attn_wb = f"OracleModel.layers.{i}.attn", f"layers.{i}.attn"
        # The oracle's FFN is TWO parallel paths, either of which may be absent: an
        # always-on dense MLP (`dense`, which is where arch_from_hf folds shared experts)
        # and a routed pool (`moe`). There is no single `mlp` node to read.
        dense_fc, dense_wb = f"OracleModel.layers.{i}.dense", f"layers.{i}.dense"
        moe_fc, moe_wb = f"OracleModel.layers.{i}.moe", f"layers.{i}.moe"

        attn_ops = MLA_ATTN_OPS if g.mla is not None else ATTN_OPS
        for op in attn_ops:
            if (op == "v_proj" and arch.attention_k_eq_v
                    and g.kv_seq_cap is None and g.mla is None):
                continue
            f = _subtree(flop_counts, f"{attn_fc}.{op}")
            add_row(rows, i, gi, glabel, g, op, f * bs / f_div(op, g),
                    weight_bytes.get(f"{attn_wb}.{op}", 0) * w_scale / w_div(op, g), 0)

        # always-on dense MLP: computed + resident for every token, replicated across EP.
        if g.dense_inter(arch) > 0:
            for op in MLP_OPS:
                f = _subtree(flop_counts, f"{dense_fc}.{op}")
                wb = weight_bytes.get(f"{dense_wb}.{op}", 0)
                add_row(rows, i, gi, glabel, g, op, f * bs / f_div(op, g),
                        wb * w_scale / w_div(op, g), 0)

        # routed experts: k experts' FLOPs, all n_routed weights (sharded by TP*EP).
        n_routed = g.num_experts(arch)
        if n_routed > 0:
            k = g.experts_per_tok(arch)
            for op in MLP_OPS:
                mop = f"{_MOE_PREFIX}{op}"
                rf = routed_flops(flop_counts, moe_fc, op, k)
                rw = routed_weight(weight_bytes, moe_wb, op, n_routed)
                add_row(rows, i, gi, glabel, g, mop, rf * bs / f_div(mop, g),
                        rw * w_scale / w_div(mop, g), 0)
            # the router: one matmul per token over all experts, weights replicated. Read from
            # the oracle's `moe.gate` subtree, the same source as the expert rows.
            rop = f"{_MOE_PREFIX}router"
            gf = _subtree(flop_counts, f"{moe_fc}.gate")
            gw = weight_bytes.get(f"{moe_wb}.gate", 0)
            add_row(rows, i, gi, glabel, g, rop, gf * bs / f_div(rop, g),
                    gw * w_scale / w_div(rop, g), 0)

        sdpa_flops = sdpa_flops_one_layer(g, S, prefill) * bs / f_div("sdpa", g)
        kv_raw, _kv_shards = kv_bytes_one_layer(arch, g, S, bs)
        kv_dev = kv_raw / kv_cache_divisor(sh, regime, g.kv_seq_cap, g.n_kv_heads,
                                           g.mla is not None)
        add_row(rows, i, gi, glabel, g, "sdpa", sdpa_flops, 0, kv_dev)

    # lm_head: root residual FLOPs above the per-layer subtree sum; streams the
    # vocab x hidden table every step. embed_table_bytes already uses the served table
    # precision, counts the table ONCE when tied and TWICE when untied, and therefore
    # needs no oracle model-dtype scaling here.
    layer_sum = sum(_subtree(flop_counts, f"OracleModel.layers.{i}") for i in range(n_layers))
    lm_flops = max(_subtree(flop_counts, "OracleModel") - layer_sum, 0)
    head_g = arch.attn_groups[-1]
    add_row(rows, None, -1, "head", head_g, "lm_head",
            lm_flops * bs / f_div("lm_head", head_g),
            embed_table_bytes(arch) / w_div("lm_head", head_g), 0)
    return rows


_SCALED = ("flops", "weight_bytes", "kv_bytes",
           "act_input_bytes", "act_output_bytes", "act_bytes", "mem_bytes",
           "compute_ms", "weight_read_ms", "kv_read_ms", "activation_io_ms", "memory_ms")


def rollup_by_type(rows, PP):
    """Aggregate per-instance rows into per (op, group) totals for ONE device. PP holds
    count/PP instances of each per-layer op; lm_head (layer None) -> 1 instance."""
    counts, rep = {}, {}
    for r in rows:
        key = (r["op"], r["group_index"])
        counts[key] = counts.get(key, 0) + 1
        rep.setdefault(key, r)
    out = []
    for key, r in rep.items():
        n_dev = 1 if r["layer"] is None else counts[key] / PP
        agg = {"op": r["op"], "group": r["group"], "group_index": r["group_index"],
               "instances_total": counts[key], "instances_per_device": n_dev}
        for f in _SCALED:
            agg[f] = r[f] * n_dev
        agg["arith_intensity"] = (agg["flops"] / agg["mem_bytes"]) if agg["mem_bytes"] else 0.0
        agg["binding"] = "compute" if agg["compute_ms"] >= agg["memory_ms"] else "memory"
        out.append(agg)
    out.sort(key=lambda x: -max(x["compute_ms"], x["memory_ms"]))
    return out


def reconcile(by_type, analytic_cell, activation_policy):
    """Sum per-op components and check against the analytic cell (cell_entitlement),
    proving the operator breakdown is a decomposition of the same bound."""
    comp = sum(t["compute_ms"] for t in by_type)
    wread = sum(t["weight_read_ms"] for t in by_type)
    kvread = sum(t["kv_read_ms"] for t in by_type)
    act = sum(t["activation_io_ms"] for t in by_type)

    def rel(x, y):
        return abs(x - y) / y if y else (0.0 if x == 0 else float("inf"))

    rel_error = {
        "compute_vs_matmul": rel(comp, analytic_cell["matmul"]),
        "weight_vs_weight_load": rel(wread, analytic_cell["weight_load"]),
        "kv_vs_kv_load": rel(kvread, analytic_cell["kv_load"]),
    }
    if activation_policy == PER_OPERATOR:
        rel_error["activation_vs_activation_load"] = rel(
            act, analytic_cell["activation_load"])

    return {
        "operator_sum": {
            "compute_ms": comp,
            "weight_read_ms": wread,
            "kv_read_ms": kvread,
            "activation_interface_ms": act,
        },
        "analytic_cell": {
            "matmul_ms": analytic_cell["matmul"],
            "weight_load_ms": analytic_cell["weight_load"],
            "kv_load_ms": analytic_cell["kv_load"],
            "activation_load_ms": analytic_cell["activation_load"],
        },
        "activation_reconciliation": (
            "operator_interfaces_match_selected_hbm_policy"
            if activation_policy == PER_OPERATOR
            else "component_graph_cut_is_reconciled_by_component_records"),
        "rel_error": rel_error,
        "binding_constraint": analytic_cell["binding_constraint"],
    }


def analyze(arch, hw, cfg, model_name="model", hardware_name="hardware",
            analytic_cell=None):
    rows = build_op_rows(arch, hw, cfg)
    by_type = rollup_by_type(rows, cfg["PP"])
    if analytic_cell is None:
        analytic_cell = _planner.cell_entitlement(
            arch, hw, cfg["TP"], cfg["PP"], cfg["CP"],
            cfg["bucket"], cfg["bs"], cfg["prefill"], EP=cfg.get("EP", 1))
    activation_policy = getattr(arch, "activation_io_policy", COMPONENT_BOUNDARY)
    rec = reconcile(by_type, analytic_cell, activation_policy)
    if activation_policy == COMPONENT_BOUNDARY:
        activation_policy_report = {
            "mode": COMPONENT_BOUNDARY,
            "hbm_source": "unique_component_graph_cut_inputs_and_outputs",
            "internal_operator_interfaces": "assumed_ideally_fused_or_resident",
        }
    else:
        activation_policy_report = {
            "mode": PER_OPERATOR,
            "hbm_source": "sum_of_operator_input_and_output_interfaces",
            "internal_operator_interfaces": "materialized_by_selected_decomposition",
        }
    return {
        "source": "operator-entitlement (oracle projection/FFN/lm_head FLOPs+weights + "
                  "analytic SDPA/KV/activation), "
                  f"{model_name}, {hardware_name}",
        "model": model_name, "hardware": hardware_name, "config": cfg,
        "engine_scope": list(CERTIFIED_ENGINE_SCOPE),
        "sharding": "per-device, divisors from planner.shard: Q/O + MLP 1/TP; K/V "
                    "projections and KV cache to min(TP,n_kv) then replicated; CP shards "
                    "the KV sequence on decode and prefill compute only under DCP prefill; "
                    "EP shards routed-expert weights across TP*EP; PP holds num_layers/PP "
                    "of each layer type.",
        "activation_policy": activation_policy_report,
        "by_type": by_type,
        "instances": rows,
        "components": analytic_cell["components"],
        "component_composition": analytic_cell["component_composition"],
        "reconciliation": rec,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runspec", default=None, required=True, help="path to a run-spec JSON (the one the agent wrote, typically under agent_artifacts/tmp/)")
    ap.add_argument("--project-root", default=None,
                    help="the PROJECT directory this run belongs to; the floor JSON lands in "
                         "<project-root>/agent_artifacts/tmp/. Defaults to $NEURON_PROJECT_ROOT, "
                         "then to the cwd -- pass it when the cwd is not the project (e.g. the "
                         "run-spec is read in place from elsewhere). Must already exist.")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--cp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--bucket", type=int, default=8192)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--prefill", action="store_true")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    # Resolve the destination BEFORE the oracle forward pass: an unusable project root must
    # surface as a one-line operator instruction up front, not as an error after the work.
    regime = "prefill" if args.prefill else "decode"
    out = args.out or os.path.join(_planner.resolve_outdir(args.project_root),
                                   f"operator_entitlement_{regime}.json")

    runspec = _planner.resolve_runspec(args.runspec)
    arch, hw, _, raw, _ = _planner.load_runspec(runspec)
    model_name = raw.get("model", "model")
    hardware_name = raw.get("hardware_name", "hardware")

    cfg = {"TP": args.tp, "PP": args.pp, "CP": args.cp, "EP": args.ep,
           "bucket": args.bucket, "bs": args.bs, "prefill": args.prefill}
    result = analyze(arch, hw, cfg, model_name, hardware_name)

    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    rec = result["reconciliation"]
    print(f"=== operator entitlement: {regime} TP{cfg['TP']} PP{cfg['PP']} CP{cfg['CP']} "
          f"EP{cfg['EP']} bucket={cfg['bucket']} bs={cfg['bs']} ({model_name}) ===")
    print(f"{'op':<12} {'group':<8} {'inst/dev':>8} {'compute':>9} {'HBM':>9} "
          f"({'wt':>6} {'kv':>6}) {'act-iface':>10} {'AI':>7}  MUST-bound")
    for t in result["by_type"]:
        print(f"{t['op']:<12} {t['group']:<8} {t['instances_per_device']:>8.1f} "
              f"{t['compute_ms']:>8.3f}m {t['memory_ms']:>8.3f}m "
              f"({t['weight_read_ms']:>5.2f}m {t['kv_read_ms']:>5.2f}m) "
              f"{t['act_bytes'] / 1024**2:>8.1f}MiB "
              f"{t['arith_intensity']:>7.1f}  {t['binding']}")
    print(f"\nreconciliation vs analytic cell (binding={rec['binding_constraint']}):")
    for k, v in rec["rel_error"].items():
        print(f"  {k:<24} rel_err={v:.4%}")
    comp = result["component_composition"]
    totals = comp["engine_totals_ms"]
    scope = "/".join(engine.upper() for engine in result["engine_scope"])
    print(f"\ncomponent composition ({scope} certified scope; collectives excluded):")
    print(f"  components={comp['component_count']} in "
          f"{comp['component_record_count']} reported records "
          f"({comp['cost_record_count']} cost records)  TE={totals['te']:.3f} ms  "
          f"HBM={totals['hbm']:.3f} ms")
    print(f"  global-engine diagnostic={comp['global_engine_floor_ms']:.3f} ms  "
          f"serialized component floor={comp['component_floor_ms']:.3f} ms")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
