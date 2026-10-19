"""Diff the analytic ``planner`` engine against the meta-device counter ORACLE, PER SEGMENT.

Runs ``oracle_model`` on the ``meta`` device under ``EntitlementCounter`` (ground truth:
torch's per-aten FLOP formulas + a weight-byte walk) and compares it to the analytic engine
NOT as three whole-model scalars but per ``(component, layer_type)`` SEGMENT -- the same key
the planner actually sweeps on (segments -> shard -> cost). So a mismatch localizes to a cell
(e.g. ``attention/global`` FLOPs, or ``ffn/dense+moe`` weights) instead of smearing into one
model-level percentage, and the lm_head projection gets its OWN row -- verified, not assumed.

Both sides are driven by the run-spec, so this re-derives for any model. Give a direct path:
    python3.12 validate_oracle.py /path/to/run.json

The analytic side is ``planner.cost.per_type_totals`` at TP=CP=DP=1 (unsharded == raw model
totals). The oracle side buckets its per-module ``flop_counts`` / ``weight_bytes`` up to the
SAME ``(component, layer_type)`` key:
  - ``OracleModel.layers.<i>.<attn|dense|moe>`` -> layer i's group -> (component, layer_type),
    via the contiguous per-group index ranges (0..count0, count0..count0+count1, ...) and the
    same ``_attn_type_label`` / ``_ffn_label`` the segments use;
  - the final lm_head ``F.linear`` (not under any ``layers.<i>``) -> ``lm_head/-``.

HIERARCHY is just the key depth: ``--by component`` rolls layer_type up (attention / ffn /
lm_head); ``--by type`` (default) keeps ``attention/sliding``, ``attention/global``, ... .

WHAT TO EXPECT:
  - weight bytes: oracle == analytic per segment to the byte (the analytic corrections -- the
    tied embed/lm_head table and the k_eq_v phantom-v_proj drop -- land in the lm_head and
    attention rows respectively).
  - FLOPs (decode): each segment should match tightly. The attention-score term is one query
    over the effective cache length; ffn and lm_head are exact dense contractions. lm_head is
    a validated row rather than a lumped-in residual.
  - KV bytes: the oracle reports 0 (on meta the SDPA decomposes into bmm+softmax, so the
    counter's fused-attention hook never fires), so analytic KV is the authority -- per row,
    same as before, just not diffed.
  - Prefill softmax-attention FLOPs differ by construction: meta SDPA executes/counts the dense
    SxS score/value rectangle, while the model floor charges only causal active pairs. The exact
    masked-future delta is pinned in tests; it is not a tolerance or approximation.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from entitlement_counter import EntitlementCounter  # noqa: E402
from entitlement_formulas import load_runspec  # noqa: E402
from oracle_model import build_oracle, make_inputs  # noqa: E402
from planner.cost import per_type_totals, OperatingPoint  # noqa: E402
from planner.segments import _attn_type_label, _ffn_label  # noqa: E402
from planner.shard import ShardConfig  # noqa: E402

# No default model: the skill ships none, and no environment variable names one. The run-spec is
# the command-line argument -- a path, or the name of a model shipped with the install.

# Oracle module-FQN component tag -> analytic component name. The oracle names attention
# `attn`, the always-on MLP `dense`, and the routed pool `moe`; the analytic rolls dense+moe
# into the single `ffn` component.
_ORACLE_COMPONENT = {"attn": "attention", "dense": "ffn", "moe": "ffn"}


def _layer_type_ranges(arch):
    """(start, end, attn_label, ffn_label) per attention group, over contiguous layer indices
    -- the oracle builds `g.count` layers per group in order, so index -> group is a range map."""
    ranges, start = [], 0
    for g in arch.attn_groups:
        ranges.append((start, start + g.count, _attn_type_label(g), _ffn_label(arch, g)))
        start += g.count
    return ranges


def _seg_of_layer(i, ranges):
    """(attn_label, ffn_label) for layer index i."""
    for lo, hi, attn_lt, ffn_lt in ranges:
        if lo <= i < hi:
            return attn_lt, ffn_lt
    raise IndexError(f"layer index {i} outside any attention group range {ranges}")


def _oracle_by_segment(model, arch, S, bs, prefill, depth_type=True):
    """Bucket the oracle's per-module FLOPs and weight bytes into (component, layer_type).

    FLOPs: sum each `OracleModel.layers.<i>.<attn|dense|moe>` node's op counts into its segment;
    the lm_head linear is everything in `Global` not under any layer component. Weight bytes:
    the per-module parameter walk, keyed the same way (embed/lm_head table -> lm_head)."""
    import re
    inp = make_inputs(bs, S if prefill else 1)
    with EntitlementCounter(model, depth=6) as counter:
        model(inp, S, prefill)
    rep = counter.report()
    ranges = _layer_type_ranges(arch)

    def key(comp, attn_lt, ffn_lt):
        lt = attn_lt if comp == "attention" else (ffn_lt if comp == "ffn" else "")
        return (comp, lt if depth_type else "")

    flops, weights = {}, {}
    comp_pat = re.compile(r"^OracleModel\.layers\.(\d+)\.(attn|dense|moe)$")
    layer_flops_total = 0
    for fqn, opcounts in rep.flop_counts.items():
        m = comp_pat.match(fqn)
        if not m:
            continue
        i, tag = int(m.group(1)), m.group(2)
        attn_lt, ffn_lt = _seg_of_layer(i, ranges)
        k = key(_ORACLE_COMPONENT[tag], attn_lt, ffn_lt)
        v = sum(opcounts.values())
        flops[k] = flops.get(k, 0) + v
        layer_flops_total += v
    # lm_head FLOPs = everything not attributed to a layer component (the final F.linear).
    lm_flops = sum(rep.flop_counts.get("Global", {}).values()) - layer_flops_total
    if lm_flops:
        flops[("lm_head", "")] = flops.get(("lm_head", ""), 0) + lm_flops

    # weight bytes: the param walk keys LEAF modules (e.g. layers.0.attn.k_proj), so match the
    # `layers.<i>.<attn|dense|moe>` PREFIX (bytes live one level below the component node).
    # Deduped tie-aware; the embed table (owned by `embed`, tied to lm_head) -> lm_head row.
    # Anything matching NOTHING is surfaced as an `unattributed` row rather than dropped:
    # silently skipping unknown owners once hid an entire untied lm_head table (3.2 GB, -1.3%
    # on the total) behind what looked like a rounding-sized diff. Norms are the only expected
    # residue and they are ~0 bytes.
    wpat = re.compile(r"(?:^|\.)layers\.(\d+)\.(attn|dense|moe)\b")
    for fqn, nbytes in rep.weight_bytes.items():
        # Norms FIRST: the QK-norms live *inside* the attention module
        # (`layers.N.attn.q_norm`), so a layer-prefix match would fold their scale params into
        # the attention row -- against an analytic side that models no norm bytes at all. That
        # makes the row inexact in a way that masks real drift of the same size.
        if "norm" in fqn:
            continue
        m = wpat.match(fqn)
        if m:
            i, tag = int(m.group(1)), m.group(2)
            attn_lt, ffn_lt = _seg_of_layer(i, ranges)
            k = key(_ORACLE_COMPONENT[tag], attn_lt, ffn_lt)
        elif "embed" in fqn or "lm_head" in fqn:
            k = ("lm_head", "")
        else:
            k = ("unattributed", fqn or "<root module>")
        weights[k] = weights.get(k, 0) + nbytes
    return flops, weights


def _analytic_by_segment(b, S, bs, prefill, depth_type=True):
    """Analytic (component, layer_type) FLOP + weight-byte totals at TP=CP=DP=1 (unsharded)."""
    cfg = ShardConfig(TP=1, CP=1, DP=1, ep_enabled=b.arch_ep)
    op = OperatingPoint(S, bs, "prefill" if prefill else "decode")
    pt = per_type_totals(b.weight_segs, b.kv_segs, b.flop_segs, cfg, op, b.meta, b.arch.dtype_bytes)
    flops, weights = {}, {}
    for (comp, lt), t in pt.items():
        k = (comp, lt if depth_type else "")
        flops[k] = flops.get(k, 0.0) + t.flops
        weights[k] = weights.get(k, 0.0) + t.weight_bytes
    return flops, weights


def _pct(o, a):
    return "n/a" if not a else f"{(o / a - 1) * 100:+.1f}%"


def _fmt_key(k):
    comp, lt = k
    return f"{comp}/{lt}" if lt else comp


def main():
    ap = argparse.ArgumentParser(description="Segment-level oracle validation.")
    ap.add_argument("--by", choices=("type", "component"), default="type",
                    help="segment granularity: 'type' = (component, layer_type) [default]; "
                         "'component' = rolled up to attention/ffn/lm_head.")
    ap.add_argument("runspec", nargs="?", default=None,
                    help="path to a run-spec JSON. Required.")
    args = ap.parse_args()
    if not args.runspec:
        raise SystemExit(
            "error: a run-spec path is required (samples/granite4-8b-walkthrough.md writes out a "
            "complete example).")
    if not os.path.isfile(args.runspec):
        raise SystemExit(f"error: run-spec not found: {args.runspec}")
    depth_type = args.by == "type"

    # build_bundle gives the analytic side; expose the EP flag it would sweep with.
    from planner.build import build_bundle, ep_default_enabled
    b = build_bundle(args.runspec)
    b.arch_ep = ep_default_enabled(b.arch)
    arch = b.arch
    model = build_oracle(arch)

    print(f"model: {b.raw.get('model', args.runspec)}  layers={arch.num_layers}  "
          f"dtype_bytes={arch.dtype_bytes}  k_eq_v={arch.attention_k_eq_v}  by={args.by}\n")

    cells = [("decode", 4096, 8, False), ("prefill", 2048, 1, True)]
    for name, S, bs, prefill in cells:
        of, ow = _oracle_by_segment(model, arch, S, bs, prefill, depth_type)
        af, aw = _analytic_by_segment(b, S, bs, prefill, depth_type)
        keys = sorted(set(of) | set(aw) | set(af) | set(ow), key=_fmt_key)
        print(f"=== {name} S{S} bs{bs} ===")
        hdr = f"{'segment':22s} {'oracle FLOP':>16s} {'analytic FLOP':>16s} {'d%':>7s}   " \
              f"{'oracle wt(MB)':>14s} {'analytic wt(MB)':>16s} {'d%':>7s}"
        print(hdr)
        print("-" * len(hdr))
        for k in keys:
            ofl, afl = of.get(k, 0), af.get(k, 0)
            owt, awt = ow.get(k, 0), aw.get(k, 0)
            print(f"{_fmt_key(k):22s} {ofl:>16,} {int(afl):>16,} {_pct(ofl, afl):>7s}   "
                  f"{owt / 1e6:>14.1f} {awt / 1e6:>16.1f} {_pct(owt, awt):>7s}")
        print(f"{'TOTAL':22s} {sum(of.values()):>16,} {int(sum(af.values())):>16,} "
              f"{_pct(sum(of.values()), sum(af.values())):>7s}   "
              f"{sum(ow.values()) / 1e6:>14.1f} {sum(aw.values()) / 1e6:>16.1f} "
              f"{_pct(sum(ow.values()), sum(aw.values())):>7s}")
        print()


if __name__ == "__main__":
    main()
