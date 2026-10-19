"""Section 3 -- per-device totals + cost model.

per_device_totals(segments, cfg, regime, op, meta) -> {component: Triple}
  Combines §1 segments x §2 divisors x operating point (S, bs). One uniform per-segment form,
  eval factors gated by quantity (§3.1 "what scales what"). Sums into semantic-component
  reporting totals, each a (weight_bytes, kv_bytes, activation_bytes, flops) Triple.

step_time(component_totals, hw, comm_ms) -> StepCost
  Applies one roofline max((weight+kv+component activation)/BW, flop/TE) to every supplied
  cost record, then serializes the records and adds communication.
  ``apply_activation_records`` attaches the ordered component-body activation records and
  recompresses cost-equivalent instances. Reporting then clubs those already-costed records
  by ``(component, variant)`` without applying a second roofline maximum.

meta : {attn_type_label -> (n_kv, is_mla)} so the divisor layer can apply the KV ceiling /
       cp_g per attention-type without cost.py importing arch internals.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from entitlement_formulas import active_attention_pairs
from planner.segments import Segment
from planner import shard
from planner.shard import ShardConfig, Regime


CERTIFIED_ENGINE_SCOPE = ("te", "hbm")


@dataclass(frozen=True)
class ComponentRecordKey:
    """Identity for one cost-equivalent set of serialized component instances."""

    component: str
    variant: str = ""
    qualifier: str = ""


@dataclass
class Triple:
    weight_bytes: float = 0.0
    kv_bytes: float = 0.0
    activation_input_bytes: float = 0.0
    activation_output_bytes: float = 0.0
    flops: float = 0.0
    layers: int = 0                # multiplicity (layer count) for this (component, type)
    instance_ids: Tuple[str, ...] = ()

    def __iadd__(self, o: "Triple") -> "Triple":
        self.weight_bytes += o.weight_bytes
        self.kv_bytes += o.kv_bytes
        self.activation_input_bytes += o.activation_input_bytes
        self.activation_output_bytes += o.activation_output_bytes
        self.flops += o.flops
        self.layers += o.layers
        self.instance_ids += o.instance_ids
        return self

    @property
    def activation_bytes(self) -> float:
        return self.activation_input_bytes + self.activation_output_bytes

    @property
    def hbm_bytes(self) -> float:
        return self.weight_bytes + self.kv_bytes + self.activation_bytes

    @property
    def ai(self) -> float:
        """Arithmetic intensity: FLOPs / mandatory HBM bytes -- the roofline
        x-axis for this triple. A property (derived from the triple's own fields), so it is
        defined once here in §3 and consumed, never recomputed, downstream."""
        return self.flops / self.hbm_bytes if self.hbm_bytes else 0.0

    def times(self, hw) -> dict:
        """load_ms (mandatory HBM / BW), flop_ms (FLOPs / TE peak), and max (§3.2)."""
        load_ms = self.hbm_bytes / hw.hbm_bw_bytes_per_s * 1e3
        flop_ms = self.flops / hw.te_peak_flops * 1e3
        return {"load_ms": load_ms, "flop_ms": flop_ms, "time_ms": max(load_ms, flop_ms),
                "per_layer_ms": max(load_ms, flop_ms) / self.layers if self.layers else 0.0,
                "ai": self.ai,
                "bound": max(
                    (("weight_load", self.weight_bytes),
                     ("kv_load", self.kv_bytes),
                     ("activation_load", self.activation_bytes)),
                    key=lambda item: item[1])[0] if load_ms >= flop_ms else "flop"}


@dataclass
class OperatingPoint:
    S: int
    bs: int
    regime: Regime = "decode"           # decode | prefill | prefill_dcp
    sampled_tokens: int = 0             # lm_head positions; 0 => derive (one per sequence = bs)

    @property
    def is_prefill(self) -> bool:
        return self.regime in ("prefill", "prefill_dcp")

    @property
    def n_tok(self) -> int:
        return self.S * self.bs if self.is_prefill else self.bs

    @property
    def lm_head_tokens(self) -> int:
        if self.sampled_tokens:
            return self.sampled_tokens
        # lm_head projects only the SAMPLED positions (a real stack index_selects the
        # sampling positions before the logit projection -- vllm-neuron gemma4 model.py):
        #   prefill -> the last position per sequence only = bs;
        #   decode  -> every decoded position = n_tok (q_len*bs; q_len=1 today, >1 for MTP).
        return self.bs if self.is_prefill else self.n_tok


def eff_ctx(cap: Optional[int], S: int) -> int:
    """Canonical §3 derived quantity: tokens a segment caches/attends over. cap None => full S."""
    return S if cap is None else min(S, cap)


def _meta_for(seg: Segment, meta: Dict[str, Tuple[int, bool]]) -> Tuple[int, bool]:
    return meta.get(seg.layer_type, (0, False))


def per_type_totals(
    weight_segs: List[Segment],
    kv_segs: List[Segment],
    flop_segs: List[Segment],
    cfg: ShardConfig,
    op: OperatingPoint,
    meta: Dict[str, Tuple[int, bool]],
    dtype_bytes: int,
    weight_dtype_bytes: Optional[float] = None,
    vocab_table_effective_dtype_bytes: Optional[float] = None,
) -> Dict[Tuple[str, str], Triple]:
    """§3.1 step 2: per-device work keyed by (component, layer_type) -- e.g.
    ('attention','sliding'), ('attention','global'), ('ffn','dense'), ('lm_head',''). This is
    the finest aggregation the design defines; per-component (step 3) is a roll-up of it
    (`roll_up_components`). Also records `layers` (multiplicity) per key for per-layer views."""
    out: Dict[Tuple[str, str], Triple] = defaultdict(Triple)
    layers: Dict[Tuple[str, str], int] = defaultdict(int)

    def K(s):
        return (s.component, s.layer_type)

    for s in weight_segs:
        n_kv, is_mla = _meta_for(s, meta)
        d = shard.weight_divisor(s.cls, cfg, op.regime, n_kv=n_kv, is_mla=is_mla)
        common_w_bytes = dtype_bytes if weight_dtype_bytes is None else weight_dtype_bytes
        w_bytes = (vocab_table_effective_dtype_bytes
                   if s.cls == "embed" and vocab_table_effective_dtype_bytes is not None
                   else common_w_bytes)
        out[K(s)].weight_bytes += s.multiplicity * s.unit * w_bytes / d
        layers[K(s)] = max(layers[K(s)], s.multiplicity)

    for s in kv_segs:
        n_kv, is_mla = _meta_for(s, meta)
        d = shard.kv_cache_divisor(cfg, op.regime, s.cap, n_kv, is_mla)
        out[K(s)].kv_bytes += s.multiplicity * s.unit / d * eff_ctx(s.cap, op.S) * op.bs
        layers[K(s)] = max(layers[K(s)], s.multiplicity)

    for s in flop_segs:
        n_kv, is_mla = _meta_for(s, meta)
        d = shard.flop_divisor(
            s.cls, cfg, op.regime, cap=s.cap, n_kv=n_kv, is_mla=is_mla)
        if s.ctx_scaled:
            # Attention score/value contractions scale with ACTIVE causal pairs.
            # A dense ``S * eff_ctx`` rectangle includes masked future positions
            # during prefill and is not mandatory semantic work.
            positions = (
                active_attention_pairs(op.S, s.cap) * op.bs
                if op.is_prefill
                else eff_ctx(s.cap, op.S) * op.bs
            )
            val = s.multiplicity * s.unit / d * positions
        else:
            tokens = op.lm_head_tokens if s.sampled_only else op.n_tok
            val = s.multiplicity * s.unit / d * tokens
        out[K(s)].flops += val
        layers[K(s)] = max(layers[K(s)], s.multiplicity)

    for k in out:
        out[k].layers = layers[k]
    return dict(out)


def apply_activation_records(per_type, activation_records):
    """Attach activation HBM after expanding each base type to cost-equivalent records.

    The segment engine intentionally compresses identical layers. Split the base
    TE/weight/KV work evenly over represented component-body instances, then recompress
    only records with equal activation geometry. The generic split remains useful for
    architectures whose instances of one model variant have different body layouts.
    """

    by_base = defaultdict(list)
    for record in activation_records:
        by_base[(record.component, record.variant)].append(record)

    missing = set(by_base) - set(per_type)
    if missing:
        raise ValueError(
            "activation records have no base TE/HBM component work: %r"
            % sorted(missing))

    out = {}
    for base_key, triple in per_type.items():
        records = by_base.get(base_key)
        if not records:
            out[ComponentRecordKey(*base_key)] = triple
            continue
        if triple.activation_bytes:
            raise ValueError(
                "%r already contains activation bytes before record expansion"
                % (base_key,))

        expected = triple.layers or 1
        represented = sum(record.multiplicity for record in records)
        if represented != expected:
            raise ValueError(
                "%r has %d segment instances but %d activation instances"
                % (base_key, expected, represented))

        role_counts = defaultdict(int)
        for record in records:
            role_counts[record.boundary_role] += 1
        role_seen = defaultdict(int)
        split_required = len(records) > 1
        for record in records:
            role_seen[record.boundary_role] += 1
            qualifier = ""
            if split_required:
                qualifier = record.boundary_role
                if role_counts[record.boundary_role] > 1:
                    qualifier += ".%d" % role_seen[record.boundary_role]
            fraction = record.multiplicity / expected
            key = ComponentRecordKey(
                component=record.component,
                variant=record.variant,
                qualifier=qualifier,
            )
            out[key] = Triple(
                weight_bytes=triple.weight_bytes * fraction,
                kv_bytes=triple.kv_bytes * fraction,
                activation_input_bytes=record.input_bytes,
                activation_output_bytes=record.output_bytes,
                flops=triple.flops * fraction,
                layers=record.multiplicity,
                instance_ids=record.instance_ids,
            )
    return out


def _component_key_parts(key):
    if isinstance(key, ComponentRecordKey):
        return key.component, key.variant, key.qualifier
    if isinstance(key, tuple):
        if len(key) == 2:
            return key[0], key[1], ""
        if len(key) == 3:
            return key
    return key, "", ""


def roll_up_components(per_type: Dict[Tuple[str, str], Triple]) -> Dict[str, Triple]:
    """Sum per-(component, variant) triples for aggregate byte/FLOP reporting.

    This roll-up is deliberately not used for latency: applying a roofline after variants have
    been combined can hide a compute-bound variant under a memory-bound one. ``step_time`` costs
    the unrolled records first. Component names remain opaque here so new components require no
    cost-model change.
    """
    out: Dict[str, Triple] = defaultdict(Triple)
    for key, t in per_type.items():
        comp, _variant, _qualifier = _component_key_parts(key)
        out[comp] += t
    return dict(out)


def per_device_totals(weight_segs, kv_segs, flop_segs, cfg, op, meta, dtype_bytes,
                      weight_dtype_bytes=None, vocab_table_effective_dtype_bytes=None):
    """Per-device aggregate work per component. Not a latency composition."""
    return roll_up_components(
        per_type_totals(weight_segs, kv_segs, flop_segs, cfg, op, meta, dtype_bytes,
                        weight_dtype_bytes, vocab_table_effective_dtype_bytes))


# --- §3.2 cost model -----------------------------------------------------------

@dataclass
class ComponentCost:
    component: str
    variant: str
    load_ms: float
    flop_ms: float
    weight_bytes: float
    kv_bytes: float
    activation_input_bytes: float = 0.0
    activation_output_bytes: float = 0.0
    flops: float = 0.0
    layers: int = 0
    qualifier: str = ""
    instance_ids: Tuple[str, ...] = ()
    source_records: Tuple["ComponentCost", ...] = ()

    @classmethod
    def summarize(cls, records):
        """Roll one component variant up after every source record has been costed."""

        records = tuple(records)
        if not records:
            raise ValueError("a component summary needs at least one cost record")
        identities = {
            (record.component, record.variant)
            for record in records
        }
        if len(identities) != 1:
            raise ValueError(
                "component summaries cannot mix component variants: %r"
                % sorted(identities))

        def total(field):
            return sum(getattr(record, field) for record in records)

        return cls(
            component=records[0].component,
            variant=records[0].variant,
            load_ms=total("load_ms"),
            flop_ms=total("flop_ms"),
            weight_bytes=total("weight_bytes"),
            kv_bytes=total("kv_bytes"),
            activation_input_bytes=total("activation_input_bytes"),
            activation_output_bytes=total("activation_output_bytes"),
            flops=total("flops"),
            layers=total("layers"),
            instance_ids=tuple(
                instance_id
                for record in records
                for instance_id in record.instance_ids
            ),
            source_records=records,
        )

    @property
    def name(self) -> str:
        """Stable record identity. The cost model treats both fields as opaque labels."""
        name = f"{self.component}/{self.variant}" if self.variant else self.component
        return f"{name}@{self.qualifier}" if self.qualifier else name

    @property
    def time_ms(self) -> float:
        if self.source_records:
            return sum(record.time_ms for record in self.source_records)
        return max(self.load_ms, self.flop_ms)

    @property
    def activation_bytes(self) -> float:
        return self.activation_input_bytes + self.activation_output_bytes

    @property
    def hbm_bytes(self) -> float:
        return self.weight_bytes + self.kv_bytes + self.activation_bytes

    @property
    def ai(self) -> float:
        """Arithmetic intensity for this component: FLOPs / mandatory HBM bytes."""
        return self.flops / self.hbm_bytes if self.hbm_bytes else 0.0

    @property
    def binding_engine(self) -> str:
        if self.source_records:
            engines = {
                record.binding_engine for record in self.source_records
            }
            return next(iter(engines)) if len(engines) == 1 else "mixed"
        return "te" if self.flop_ms > self.load_ms else "hbm"

    @property
    def binding(self) -> str:
        if self.source_records:
            bindings = {record.binding for record in self.source_records}
            return next(iter(bindings)) if len(bindings) == 1 else "mixed"
        # Load vs flop; the largest byte source is diagnostic within a binding HBM term.
        if self.flop_ms > self.load_ms:
            return "flop"
        return max(
            (("weight_load", self.weight_bytes),
             ("kv_load", self.kv_bytes),
             ("activation_load", self.activation_bytes)),
            key=lambda item: item[1])[0]

    def as_dict(self) -> dict:
        """Serializable component record shared by the planner and Hardware Entitlement."""
        cost_records = self.source_records or (self,)
        return {
            "component_id": self.name,
            "component": self.component,
            "variant": self.variant or None,
            "boundary_role": self.qualifier or None,
            "multiplicity": self.layers,
            "instance_ids": list(self.instance_ids),
            "cost_record_count": len(cost_records),
            "cost_records": [record.name for record in cost_records],
            "flops": self.flops,
            "weight_bytes": self.weight_bytes,
            "kv_bytes": self.kv_bytes,
            "component_input_hbm_bytes": self.activation_input_bytes,
            "component_output_hbm_bytes": self.activation_output_bytes,
            "component_activation_hbm_bytes": self.activation_bytes,
            "mandatory_hbm_bytes": self.hbm_bytes,
            # Backward-compatible alias used by current report consumers.
            "hbm_bytes": self.hbm_bytes,
            "te_ms": self.flop_ms,
            "hbm_ms": self.load_ms,
            "floor_ms": self.time_ms,
            "floor_policy": (
                "sum_cost_record_floors"
                if self.source_records
                else "single_cost_record_max"),
            "per_instance_ms": self.time_ms / self.layers if self.layers else self.time_ms,
            "arith_intensity": self.ai,
            "binding_engine": self.binding_engine,
            "binding": self.binding,
        }


@dataclass
class StepCost:
    components: List[ComponentCost]
    comm_ms: float
    activation_io_policy: str = None

    @property
    def component_floor_ms(self) -> float:
        return sum(c.time_ms for c in self.components)

    @property
    def step_ms(self) -> float:
        return self.component_floor_ms + self.comm_ms

    def component_summaries(self) -> List[ComponentCost]:
        """Club costed boundary roles into one reported row per component variant."""

        grouped = {}
        order = []
        for component in self.components:
            key = (component.component, component.variant)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(component)
        return [
            ComponentCost.summarize(grouped[key])
            for key in order
        ]

    def grouped_components(self) -> dict:
        """Roll reported variants up by semantic component for step-level display."""
        out = {}
        bindings = defaultdict(set)
        records = defaultdict(list)
        for c in self.component_summaries():
            g = out.setdefault(c.component, {
                "time_ms": 0.0,
                "load_ms": 0.0,
                "flop_ms": 0.0,
                "weight_bytes": 0.0,
                "kv_bytes": 0.0,
                "activation_input_bytes": 0.0,
                "activation_output_bytes": 0.0,
                "flops": 0.0,
            })
            g["time_ms"] += c.time_ms
            g["load_ms"] += c.load_ms
            g["flop_ms"] += c.flop_ms
            g["weight_bytes"] += c.weight_bytes
            g["kv_bytes"] += c.kv_bytes
            g["activation_input_bytes"] += c.activation_input_bytes
            g["activation_output_bytes"] += c.activation_output_bytes
            g["flops"] += c.flops
            bindings[c.component].add(c.binding)
            records[c.component].append(c.name)

        for name, g in out.items():
            g["activation_bytes"] = (
                g["activation_input_bytes"] + g["activation_output_bytes"])
            denom = g["weight_bytes"] + g["kv_bytes"] + g["activation_bytes"]
            g["ai"] = g["flops"] / denom if denom else 0.0
            g["binding"] = (next(iter(bindings[name]))
                            if len(bindings[name]) == 1 else "mixed")
            g["records"] = records[name]
        return out

    def _binding(self, grouped) -> str:
        if not grouped:
            raise ValueError("step binding requires at least one TE/HBM component")
        best_name, best = max(grouped.items(), key=lambda item: item[1]["time_ms"])
        return f"{best_name}:{best['binding']}"

    @property
    def binding(self) -> str:
        """The semantic TE/HBM component contributing the most serialized floor time."""
        return self._binding(self.grouped_components())

    def component_composition(self) -> dict:
        summaries = self.component_summaries()
        te_ms = sum(c.flop_ms for c in self.components)
        hbm_ms = sum(c.load_ms for c in self.components)
        return {
            "engine_scope": list(CERTIFIED_ENGINE_SCOPE),
            "within_component_policy": "ideal_overlap_max",
            "between_component_policy": "serialized_sum",
            "collectives_included": False,
            "activation_io_policy": self.activation_io_policy,
            "component_count": sum(c.layers if c.layers else 1 for c in self.components),
            "component_record_count": len(summaries),
            "cost_record_count": len(self.components),
            "reported_record_policy": "component_variant_after_costing",
            "engine_totals_ms": {"te": te_ms, "hbm": hbm_ms},
            "global_engine_floor_ms": max(te_ms, hbm_ms),
            "component_floor_ms": self.component_floor_ms,
        }

    def as_dict(self) -> dict:
        grouped = self.grouped_components()
        return {
            "step_ms": self.step_ms,
            "comm_ms": self.comm_ms,
            "binding": self._binding(grouped),
            "components": grouped,
        }


def step_time(component_totals, hw, comm_ms: float, activation_io_policy=None) -> StepCost:
    """Apply the roofline to every supplied component record, then serialize.

    Keys may be ``(component, variant)`` records from :func:`per_type_totals`, or bare
    component names for callers that only have one record per component. No component or
    variant name is hard-coded here.
    """
    bw = hw.hbm_bw_bytes_per_s
    te = hw.te_peak_flops
    comps = []
    for key, t in component_totals.items():
        component, variant, qualifier = _component_key_parts(key)
        load_ms = t.hbm_bytes / bw * 1e3
        flop_ms = t.flops / te * 1e3
        comps.append(ComponentCost(
            component=str(component),
            variant=str(variant),
            load_ms=load_ms,
            flop_ms=flop_ms,
            weight_bytes=t.weight_bytes,
            kv_bytes=t.kv_bytes,
            activation_input_bytes=t.activation_input_bytes,
            activation_output_bytes=t.activation_output_bytes,
            flops=t.flops,
            layers=t.layers,
            qualifier=str(qualifier),
            instance_ids=t.instance_ids,
        ))
    return StepCost(
        comps, comm_ms,
        activation_io_policy=activation_io_policy)
