"""Wiring: run-spec -> (Arch, HW, segments, meta), and a one-call per-cell entitlement.

Thin glue so the sweep / selection / oracle layers get everything from one entry point,
built on the §1/§2/§3 modules (segments / shard / cost).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from entitlement_formulas import Arch, HW, load_runspec
from entitlement_formulas import comm_total_ms   # reuse the existing topology-aware comm model

from planner.segments import (
    build_weight_segments, build_kv_segments, build_flop_segments, _attn_type_label,
)
from planner import cost
from planner.activation import (
    COMPONENT_BOUNDARY,
    activation_instances,
    activation_records,
)
from planner.cost import (
    OperatingPoint,
    apply_activation_records,
    per_type_totals,
    roll_up_components,
    step_time,
)
from planner.shard import ShardConfig


@dataclass
class ModelBundle:
    """Everything a planner needs for one model, built once from the run-spec."""
    arch: Arch
    hw: HW
    weight_segs: list
    kv_segs: list
    flop_segs: list
    meta: Dict[str, Tuple[int, bool]]   # attn_type_label -> (n_kv, is_mla)
    raw: dict
    serving: str = "colocated"          # only "colocated" is supported (single-node scope)
    prefill_itps_floor: float = 30000.0  # prefill TTFT SLA (input-tokens/s floor)
    prefill_ttft_ceiling_ms: float = None  # flat prefill TTFT ceiling (ms); overrides the floor
    decode_itl_ceiling_ms: float = 50.0  # decode ITL SLA (max inter-token latency, ms)


# "colocated" = ONE single-node replica serves both prefill and decode, gated by both the
# prefill TTFT SLA and the decode ITL SLA. It is the only supported mode: splitting the
# phases across separate prefill/decode pools, and any topology spanning 2+ nodes or
# instances, are out of scope for this planner and are not modeled.
_SERVING_MODES = ("colocated",)


def require_lnc2(hw) -> None:
    """The per-rank TE, HBM bandwidth/capacity, and rank-count constants are LNC2 values.

    LNC2 fuses two physical NeuronCores into one logical rank. LNC1 would halve the per-rank
    engine/memory values and double the rank count, so the planner refuses any other geometry.
    Link-bandwidth fields are separate collective-axis contracts and are not LNC-scaled.
    """
    if hw.lnc != 2:
        raise ValueError(
            "this sweep only supports LNC2 (2 physical cores per logical core); "
            "got hw.lnc=%r. Set \"lnc\": 2 in the hardware config." % (hw.lnc,))


def build_bundle(runspec_path: str) -> ModelBundle:
    arch, hw, _configs, raw = load_runspec(runspec_path)
    require_lnc2(hw)
    meta: Dict[str, Tuple[int, bool]] = {}
    for g in arch.attn_groups:
        meta[_attn_type_label(g)] = (g.n_kv_heads, g.mla is not None)
    serving = raw.get("serving", "colocated")
    if serving not in _SERVING_MODES:
        raise ValueError(
            f"run-spec 'serving' must be one of {_SERVING_MODES}, got {serving!r}. "
            "Disaggregated / split prefill-decode serving and any multi-node or "
            "multi-instance topology are out of scope for this planner."
        )
    # SLA ceilings (policy knobs, run-spec overridable): prefill TTFT and decode ITL (max
    # inter-token latency, ms). The single colocated replica gates on BOTH (§4.2).
    # The prefill TTFT SLA is stated one of two mutually-exclusive ways: `prefill_itps_floor`,
    # an input-tokens/s floor giving a context-scaled ceiling (ttft_ceiling(S) = S / floor);
    # or `prefill_ttft_ceiling_ms`, a flat ceiling in ms applied at every context.
    if raw.get("prefill_itps_floor") is not None and raw.get("prefill_ttft_ceiling_ms") is not None:
        raise ValueError("set only one of 'prefill_itps_floor' or 'prefill_ttft_ceiling_ms' "
                         "(two ways to state the same prefill TTFT SLA).")
    itps = float(raw.get("prefill_itps_floor", 30000.0))
    ttft_ceil = raw.get("prefill_ttft_ceiling_ms")
    ttft_ceil = float(ttft_ceil) if ttft_ceil is not None else None
    itl = float(raw.get("decode_itl_ceiling_ms", 50.0))
    return ModelBundle(
        arch=arch, hw=hw,
        weight_segs=build_weight_segments(arch),
        kv_segs=build_kv_segments(arch),
        flop_segs=build_flop_segments(arch),
        meta=meta, raw=raw, serving=serving,
        prefill_itps_floor=itps, prefill_ttft_ceiling_ms=ttft_ceil,
        decode_itl_ceiling_ms=itl,
    )


def ep_default_enabled(arch: Arch) -> bool:
    """EP is on by default iff the model has routed experts (matches the old _ep_for)."""
    return any(g.num_experts(arch) > 0 for g in arch.attn_groups)


def cell(bundle: ModelBundle, cfg: ShardConfig, S: int, bs: int, prefill: bool) -> dict:
    """One (config, bucket, bs, regime) entitlement, on the new pipeline.

    regime: decode, or prefill (standard). apply_prefill_dcp is a run-spec serving flag; when
    set, prefill uses the prefill_dcp regime. cores = world = TP*DP (CP carved from TP)."""
    a = bundle.arch
    regime = "decode"
    if prefill:
        regime = "prefill_dcp" if getattr(a, "apply_prefill_dcp", False) else "prefill"
    op = OperatingPoint(S=S, bs=bs, regime=regime)
    activation_policy = a.activation_io_policy
    activation_by_instance = activation_instances(a, cfg, op, activation_policy)
    activation_by_record = activation_records(a, cfg, op, activation_policy)

    # §3.1: boundary-distinct records retain every distinct execution component for latency.
    # After costing, the report clubs roles into one row per (component, variant). The work
    # roll-up remains useful for aggregate memory/FLOP reporting and fit checks.
    base_per_type = per_type_totals(
        bundle.weight_segs, bundle.kv_segs, bundle.flop_segs,
        cfg, op, bundle.meta, a.dtype_bytes, a.w_elem_bytes,
        a.vocab_table_effective_elem_bytes)
    per_type = apply_activation_records(base_per_type, activation_by_record)
    triples = roll_up_components(per_type)

    # comm: reuse the existing topology-aware model. EP span = world when EP on else 1.
    ep_on = cfg.ep_enabled
    ep_span = cfg.world if ep_on else 1
    comm = comm_total_ms(a, bundle.hw, cfg.TP, 1, cfg.CP, ep_span, S, bs, prefill)["total"]

    sc = step_time(
        per_type, bundle.hw, comm,
        activation_io_policy=activation_policy)
    step = sc.as_dict()
    reported_components = sc.component_summaries()
    components = [c.as_dict() for c in reported_components]
    component_composition = sc.component_composition()

    pdw = sum(t.weight_bytes for t in triples.values())
    pdk = sum(t.kv_bytes for t in triples.values())
    pda = sum(t.activation_bytes for t in triples.values())
    pdf = sum(t.flops for t in triples.values())
    return {
        "TP": cfg.TP, "CP": cfg.CP, "DP": cfg.DP, "cores": cfg.world,
        "bucket": S, "bs": bs, "prefill": prefill, "regime": regime,
        "per_device_weight_bytes": pdw,
        "per_device_kv_bytes": pdk,
        "per_device_activation_hbm_bytes": pda,
        "per_device_flops": pdf,
        # total-model AI (§3 Triple.ai over the summed triple): FLOPs / mandatory HBM bytes.
        "ai": pdf / (pdw + pdk + pda) if (pdw + pdk + pda) else 0.0,
        "triples": {k: (v.weight_bytes, v.kv_bytes, v.flops) for k, v in triples.items()},
        "step_ms": sc.step_ms, "comm_ms": comm, "binding": step["binding"],
        "step": step,
        "components": components,
        "activation_boundaries": [
            instance.as_dict() for instance in activation_by_instance],
        "component_composition": component_composition,
        # §3.1 per-layer-type breakdown (component/type -> bytes/flops + load/flop/time ms):
        "by_type": {
            c.name: {
                "layers": c.layers,
                "weight_bytes": c.weight_bytes,
                "kv_bytes": c.kv_bytes,
                "component_input_hbm_bytes": c.activation_input_bytes,
                "component_output_hbm_bytes": c.activation_output_bytes,
                "component_activation_hbm_bytes": c.activation_bytes,
                "mandatory_hbm_bytes": c.hbm_bytes,
                "flops": c.flops,
                "load_ms": c.load_ms,
                "flop_ms": c.flop_ms,
                "time_ms": c.time_ms,
                "per_layer_ms": c.time_ms / c.layers if c.layers else 0.0,
                "ai": c.ai,
                "bound": c.binding,
            }
            for c in reported_components
        },
    }
