"""Activation-HBM contracts shared by the planner and Hardware Entitlement.

The model floor uses a graph cut around each collective-delimited component. Internal
operator interfaces may stay fused/resident, while each unique external component input
is read once and each external output is written once. ``per_operator`` is an explicit
alternative decomposition: it replaces those graph-cut bytes with every operator
interface and is useful for eager/decomposed execution or Floor(D)-style experiments.

Both policies are built from the same ordered architecture and shape helpers so the
planner's component records and Hardware Entitlement's operator rows cannot drift.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from planner.segments import _attn_type_label, _ffn_label
from planner.shard import (
    ShardConfig,
    activation_row_divisor,
    activation_width_divisor,
    cp_g,
    routed_expert_activation_multiplier,
)


COMPONENT_BOUNDARY = "component_boundary"
PER_OPERATOR = "per_operator"
ACTIVATION_IO_POLICIES = (COMPONENT_BOUNDARY, PER_OPERATOR)

ATTN_OPS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLA_ATTN_OPS = ("q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "o_proj")
MLP_OPS = ("gate_proj", "up_proj", "down_proj")
MOE_PREFIX = "moe_"


@dataclass
class ActivationTraffic:
    """Per-device HBM traffic owned by one compressed component record."""

    input_bytes: float = 0.0
    output_bytes: float = 0.0

    @property
    def total_bytes(self) -> float:
        return self.input_bytes + self.output_bytes

    def add(self, input_bytes: float, output_bytes: float) -> None:
        self.input_bytes += input_bytes
        self.output_bytes += output_bytes


@dataclass(frozen=True)
class BoundaryTensor:
    """One local tensor view crossing a component's registered HBM boundary."""

    tensor_id: str
    elements: float
    storage_bytes: float
    shape_role: str

    @property
    def hbm_bytes(self) -> float:
        return self.elements * self.storage_bytes

    def as_dict(self) -> dict:
        return {
            "tensor_id": self.tensor_id,
            "elements": self.elements,
            "storage_bytes": self.storage_bytes,
            "hbm_bytes": self.hbm_bytes,
            "shape_role": self.shape_role,
        }


@dataclass(frozen=True)
class ActivationInstance:
    """Activation traffic for one ordered component invocation."""

    instance_id: str
    component: str
    variant: str
    boundary_role: str
    input_bytes: float
    output_bytes: float
    inputs: Tuple[BoundaryTensor, ...] = ()
    outputs: Tuple[BoundaryTensor, ...] = ()

    @property
    def total_bytes(self) -> float:
        return self.input_bytes + self.output_bytes

    def as_dict(self) -> dict:
        return {
            "component_id": self.instance_id,
            "component": self.component,
            "variant": self.variant or None,
            "boundary_role": self.boundary_role,
            "component_input_hbm_bytes": self.input_bytes,
            "component_output_hbm_bytes": self.output_bytes,
            "component_activation_hbm_bytes": self.total_bytes,
            "inputs": [tensor.as_dict() for tensor in self.inputs],
            "outputs": [tensor.as_dict() for tensor in self.outputs],
        }


@dataclass(frozen=True)
class ActivationRecord:
    """Cost-equivalent activation instances compressed for the roofline engine."""

    component: str
    variant: str
    boundary_role: str
    instances: Tuple[ActivationInstance, ...]

    @property
    def multiplicity(self) -> int:
        return len(self.instances)

    @property
    def instance_ids(self) -> Tuple[str, ...]:
        return tuple(instance.instance_id for instance in self.instances)

    @property
    def input_bytes(self) -> float:
        return sum(instance.input_bytes for instance in self.instances)

    @property
    def output_bytes(self) -> float:
        return sum(instance.output_bytes for instance in self.instances)


def validate_policy(policy: str) -> str:
    policy = str(policy)
    if policy not in ACTIVATION_IO_POLICIES:
        raise ValueError(
            "activation_io_policy must be one of %r, got %r"
            % (ACTIVATION_IO_POLICIES, policy))
    return policy


def group_of_layer(arch, layer: int):
    """Return the ordered ``(group_index, AttnGroup)`` for one layer."""

    if layer < 0:
        raise IndexError(layer)
    for i, pair in enumerate(arch.ordered_layer_groups()):
        if i == layer:
            return pair
    raise IndexError(layer)


def local_tokens(g, cfg: ShardConfig, op) -> float:
    if not op.is_prefill:
        return float(op.bs)
    return op.n_tok / cp_g(
        cfg, op.regime, g.kv_seq_cap, g.n_kv_heads, g.mla is not None)


def act_io_elems(op_name, arch, g, cfg: ShardConfig, regime: str, inter=None):
    """Per-token input/output elements for one local operator invocation.

    The semantic axis of each width is declared here; ``planner.shard`` is the
    single owner of every TP/CP/EP divisor. Attention heads use TP/CP under DCP
    prefill, K/V stop at their head replication ceiling, and routed experts use
    their local expert-TP degree.
    """

    H = arch.hidden_size
    base = op_name[len(MOE_PREFIX):] if op_name.startswith(MOE_PREFIX) else op_name
    routed = op_name.startswith(MOE_PREFIX) and base != "router"
    if inter is None:
        inter = (g.moe_inter(arch)
                 if routed
                 else g.dense_inter(arch))
    I = inter

    def local(width, axis="replicated", n_heads=0):
        return width / activation_width_divisor(
            axis, cfg, regime, n_heads=n_heads)

    q_heads = local(
        g.n_q_heads * (g.mla.qk_head_dim if g.mla is not None else g.head_dim),
        "heads", g.n_q_heads)
    v_heads = local(
        g.n_q_heads * (g.mla.v_head_dim if g.mla is not None else g.head_dim),
        "heads", g.n_q_heads)
    mla_kv_b_heads = local(
        g.n_q_heads * (
            g.mla.qk_nope_head_dim + g.mla.v_head_dim
            if g.mla is not None else g.head_dim),
        "heads", g.n_q_heads)
    kv_heads = local(
        g.n_kv_heads * g.head_dim, "heads", g.n_kv_heads)
    mlp_axis = "expert_tensor_parallel" if routed else "tensor_parallel"
    local_inter = local(I, mlp_axis)
    local_vocab = local(arch.vocab_size, "vocab")

    if g.mla is not None:
        m = g.mla
        table = {
            "q_a_proj": (H, m.q_lora_rank),
            "q_b_proj": (m.q_lora_rank, q_heads),
            "kv_a_proj": (H, m.kv_cache_dim),
            "kv_b_proj": (m.kv_lora_rank, mla_kv_b_heads),
            "o_proj": (v_heads, H),
            # K/V cache reads are owned by the KV term. The activation interface records
            # the query input and attention output so per-op mode does not charge them twice.
            "sdpa": (q_heads, v_heads),
            "gate_proj": (H, local_inter),
            "up_proj": (H, local_inter),
            "down_proj": (local_inter, H),
            "lm_head": (H, local_vocab),
            "router": (H, g.num_experts(arch)),
        }
        return table[base]

    table = {
        "q_proj": (H, q_heads),
        "k_proj": (H, kv_heads),
        "v_proj": (H, kv_heads),
        "o_proj": (q_heads, H),
        "gate_proj": (H, local_inter),
        "up_proj": (H, local_inter),
        "down_proj": (local_inter, H),
        "sdpa": (q_heads, q_heads),
        "lm_head": (H, local_vocab),
        "router": (H, g.num_experts(arch)),
    }
    return table[base]


def operator_activation_bytes(op_name, arch, g, cfg: ShardConfig, op):
    """Per-device ``(input_bytes, output_bytes)`` for one layer's operator row."""

    base = op_name[len(MOE_PREFIX):] if op_name.startswith(MOE_PREFIX) else op_name
    tokens = float(op.lm_head_tokens) if base == "lm_head" else local_tokens(g, cfg, op)
    multiplier = 1.0

    if op_name.startswith(MOE_PREFIX) and base != "router":
        multiplier = routed_expert_activation_multiplier(
            cfg, g.experts_per_tok(arch))

    a_in, a_out = act_io_elems(op_name, arch, g, cfg, op.regime)
    scale = arch.dtype_bytes * tokens * multiplier
    return a_in * scale, a_out * scale


def _dedupe_tensors(tensors) -> Tuple[BoundaryTensor, ...]:
    by_id = {}
    for tensor in tensors:
        previous = by_id.setdefault(tensor.tensor_id, tensor)
        if previous != tensor:
            raise ValueError(
                "boundary tensor %r has inconsistent local views" % tensor.tensor_id)
    return tuple(by_id.values())


def _instance(instance_id, component, variant, role, inputs, outputs):
    inputs = _dedupe_tensors(inputs)
    outputs = _dedupe_tensors(outputs)
    return ActivationInstance(
        instance_id=instance_id,
        component=component,
        variant=variant,
        boundary_role=role,
        input_bytes=sum(tensor.hbm_bytes for tensor in inputs),
        output_bytes=sum(tensor.hbm_bytes for tensor in outputs),
        inputs=inputs,
        outputs=outputs,
    )


def _activation_tensor(
        tensor_id, rows, width, arch, cfg, regime,
        row_layout, width_axis, shape_role):
    return BoundaryTensor(
        tensor_id=tensor_id,
        elements=(
            rows
            / activation_row_divisor(row_layout, cfg, regime)
            * width
            / activation_width_divisor(width_axis, cfg, regime)
        ),
        storage_bytes=arch.dtype_bytes,
        shape_role=shape_role,
    )


def _hidden_tensor(
        tensor_id, rows, arch, cfg, regime, row_layout,
        shape_role="hidden"):
    return _activation_tensor(
        tensor_id=tensor_id,
        rows=rows,
        width=arch.hidden_size,
        arch=arch,
        cfg=cfg,
        regime=regime,
        row_layout=row_layout,
        width_axis="hidden",
        shape_role=shape_role,
    )


def _component_boundary_instances(arch, cfg: ShardConfig, op):
    """Build the ordered body-only component graph cut before compression.

    Attention and FFN bodies begin after their input gather and end before their
    output reduce-scatter. Each body therefore reads one gathered hidden tensor and
    writes one pre-collective partial. The sequence-parallel residual/norm bridge
    between reduce-scatter and all-gather is outside the current certified floor.
    The lm_head reads one sampled final-hidden tensor and materializes its local
    logit shard.
    """

    instances: List[ActivationInstance] = []
    ordered = list(arch.ordered_layer_groups())
    if not ordered:
        return instances

    rows = float(op.n_tok)

    for layer, (_gi, g) in enumerate(ordered):
        attention_input = _hidden_tensor(
            "layer.%d.attention.input" % layer,
            rows, arch, cfg, op.regime,
            "attention_body", "component_input")
        attention_output = _hidden_tensor(
            "layer.%d.attention.partial" % layer,
            rows, arch, cfg, op.regime,
            "attention_body", "component_output")
        instances.append(_instance(
            "layer.%d.attention" % layer,
            "attention",
            _attn_type_label(g),
            "body",
            [attention_input],
            [attention_output],
        ))

        if g.dense_inter(arch) > 0 or g.num_experts(arch) > 0:
            ffn_input = _hidden_tensor(
                "layer.%d.ffn.input" % layer,
                rows, arch, cfg, op.regime,
                "ffn_body", "component_input")
            ffn_output = _hidden_tensor(
                "layer.%d.ffn.partial" % layer,
                rows, arch, cfg, op.regime,
                "ffn_body", "component_output")
            instances.append(_instance(
                "layer.%d.ffn" % layer,
                "ffn",
                _ffn_label(arch, g),
                "body",
                [ffn_input],
                [ffn_output],
            ))

    sampled_rows = float(op.lm_head_tokens)
    head_inputs = [
        _hidden_tensor(
            "final.hidden.sampled",
            sampled_rows,
            arch,
            cfg,
            op.regime,
            "sampled",
            "sampled_final_hidden")
    ]
    logits = _activation_tensor(
        tensor_id="lm_head.local_logits",
        rows=sampled_rows,
        width=arch.vocab_size,
        arch=arch,
        cfg=cfg,
        regime=op.regime,
        row_layout="sampled",
        width_axis="vocab",
        shape_role="local_logits",
    )
    instances.append(_instance(
        "lm_head",
        "lm_head",
        "",
        "final",
        head_inputs,
        [logits],
    ))
    return instances


def _per_operator_instances(arch, cfg: ShardConfig, op):
    instances: List[ActivationInstance] = []
    ordered = list(arch.ordered_layer_groups())
    for layer, (_gi, g) in enumerate(ordered):
        attn_in = 0.0
        attn_out = 0.0
        attn_ops = MLA_ATTN_OPS if g.mla is not None else ATTN_OPS
        for op_name in attn_ops:
            if (op_name == "v_proj" and arch.attention_k_eq_v
                    and g.kv_seq_cap is None and g.mla is None):
                continue
            a_in, a_out = operator_activation_bytes(op_name, arch, g, cfg, op)
            attn_in += a_in
            attn_out += a_out
        a_in, a_out = operator_activation_bytes("sdpa", arch, g, cfg, op)
        attn_in += a_in
        attn_out += a_out
        instances.append(ActivationInstance(
            instance_id="layer.%d.attention" % layer,
            component="attention",
            variant=_attn_type_label(g),
            boundary_role="operator_interfaces",
            input_bytes=attn_in,
            output_bytes=attn_out,
        ))

        ffn_in = 0.0
        ffn_out = 0.0
        if g.dense_inter(arch) > 0:
            for op_name in MLP_OPS:
                a_in, a_out = operator_activation_bytes(op_name, arch, g, cfg, op)
                ffn_in += a_in
                ffn_out += a_out
        if g.num_experts(arch) > 0:
            for op_name in MLP_OPS:
                a_in, a_out = operator_activation_bytes(
                    MOE_PREFIX + op_name, arch, g, cfg, op)
                ffn_in += a_in
                ffn_out += a_out
            a_in, a_out = operator_activation_bytes(
                MOE_PREFIX + "router", arch, g, cfg, op)
            ffn_in += a_in
            ffn_out += a_out
        if ffn_in or ffn_out:
            instances.append(ActivationInstance(
                instance_id="layer.%d.ffn" % layer,
                component="ffn",
                variant=_ffn_label(arch, g),
                boundary_role="operator_interfaces",
                input_bytes=ffn_in,
                output_bytes=ffn_out,
            ))

    if ordered:
        _gi, last = ordered[-1]
        a_in, a_out = operator_activation_bytes("lm_head", arch, last, cfg, op)
        instances.append(ActivationInstance(
            instance_id="lm_head",
            component="lm_head",
            variant="",
            boundary_role="operator_interfaces",
            input_bytes=a_in,
            output_bytes=a_out,
        ))
    return instances


def activation_instances(arch, cfg: ShardConfig, op, policy=None):
    """Return exact ordered component invocations under the selected HBM policy."""

    policy = validate_policy(policy or getattr(
        arch, "activation_io_policy", COMPONENT_BOUNDARY))
    if policy == COMPONENT_BOUNDARY:
        return _component_boundary_instances(arch, cfg, op)
    return _per_operator_instances(arch, cfg, op)


def activation_records(arch, cfg: ShardConfig, op, policy=None):
    """Compress only instances with equal component, variant, role, and byte geometry."""

    grouped = {}
    order = []
    for instance in activation_instances(arch, cfg, op, policy):
        key = (
            instance.component,
            instance.variant,
            instance.boundary_role,
            instance.input_bytes,
            instance.output_bytes,
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(instance)
    return [
        ActivationRecord(
            component=key[0],
            variant=key[1],
            boundary_role=key[2],
            instances=tuple(grouped[key]),
        )
        for key in order
    ]


def activation_traffic(arch, cfg: ShardConfig, op, policy=None):
    """Return reporting totals keyed by the pre-existing component/variant axis."""

    out: Dict[Tuple[str, str], ActivationTraffic] = defaultdict(ActivationTraffic)
    for instance in activation_instances(arch, cfg, op, policy):
        out[(instance.component, instance.variant)].add(
            instance.input_bytes, instance.output_bytes)
    return dict(out)
