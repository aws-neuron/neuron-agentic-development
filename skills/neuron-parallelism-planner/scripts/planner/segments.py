"""Section 1 -- aggregation: decompose a model into per-class Segments.

A Segment is one piece of the model's per-forward workload, BEFORE any device split. It
carries §1 facts only -- no divisors, no S/bs, no hardware. §2 attaches a divisor by the
segment's (cls, layer_type) key; §3 multiplies in the operating point.

Three quantities, each a list of Segments:
  - weight : resident params, 6 classes (attn_qo, attn_kv, dense_mlp, routed_experts,
             router, embed)
  - kv     : per-token-per-layer KV-cache bytes, 1 class (attn_kv), attention-type segmented
  - flop   : per-token-per-layer FLOPs, keyed to the producing weight class (+ weightless
             attention-score term borrowing attn_qo)

The (cls, layer_type) pair is the KEY §2 shards and §3 groups on. layer_type is the segment's
partition label: an attention-type for attention/KV/score classes, an ffn-type for FFN
classes, and "" for the embed singleton. The two type-axes are independent (§1.1): attention
classes segment on attention-type, FFN classes on ffn-type -- no cross-product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from entitlement_formulas import (
    Arch,
    AttnGroup,
    attn_kv_params,
    attn_qo_params,
    attn_score_flops_per_token,
    dense_mlp_flops_per_token,
    moe_flops_per_token,
    router_params,
    _mlp_block_params,
)


# --- component grouping (§3.1 step 3): which classes roll up into which component ---
COMPONENT_OF = {
    "attn_qo": "attention",
    "attn_kv": "attention",
    "attn_score": "attention",
    "dense_mlp": "ffn",
    "routed_experts": "ffn",
    "router": "ffn",
    "embed": "lm_head",
}


@dataclass(frozen=True)
class Segment:
    """One (class, layer_type) piece of the model workload. §1 facts only.

    multiplicity : layer count of this segment's type -- so ONE segment spans ALL layers of
                   its type (no per-layer loop downstream).
    unit         : per-copy quantity for weights (params); per-token for kv (bytes/token) and
                   flop (flops/token).
    cap          : kv_seq_cap for this type (None => full context). Drives eff_ctx for the kv
                   segment and the ctx_scaled flop (attention scores). Irrelevant to weights.
    ctx_scaled   : flop only -- the attention-score term scales with eff_ctx (§1.3).
    sampled_only : flop only -- lm_head projects the SAMPLED positions, not every token (§1.3).
    """

    quantity: str          # "weight" | "kv" | "flop"
    cls: str               # attn_qo | attn_kv | attn_score | dense_mlp | routed_experts |
                           # router | embed
    layer_type: str        # attention-type / ffn-type label; "" for the embed singleton
    multiplicity: int
    unit: float
    cap: Optional[int] = None
    ctx_scaled: bool = False
    sampled_only: bool = False

    @property
    def component(self) -> str:
        return COMPONENT_OF[self.cls]


# --- §1 intrinsic helpers not already in entitlement_formulas ------------------

def kv_unit_per_token(a: Arch, g: AttnGroup) -> float:
    """Per-token-per-layer KV-cache bytes for one layer of group g (§1.2).

    GQA/MHA: 2 * n_kv * head_dim * kv_elem_bytes. MLA: one latent (kv_cache_dim), no x2/n_kv.
    Uses kv_elem_bytes so fp8-KV with bf16 weights is honored.
    """
    if g.mla is not None:
        return g.mla.kv_cache_dim * a.kv_elem_bytes
    return 2 * g.n_kv_heads * g.head_dim * a.kv_elem_bytes


def _embed_params(a: Arch) -> int:
    """Tied/untied embedding + lm_head table params (vocab*hidden; x2 if untied) (§1.1)."""
    one = a.vocab_size * a.hidden_size
    return one if a.tie_embeddings else 2 * one


def _attn_type_label(g: AttnGroup) -> str:
    """A stable attention-type label for a group's partition key."""
    if g.mla is not None:
        return "mla"
    return "sliding" if g.kv_seq_cap is not None else "global"


def _ffn_label(a: Arch, g: AttnGroup) -> str:
    """A stable ffn-type label: which parallel FFN paths this group runs."""
    dense = g.dense_inter(a) > 0
    moe = g.num_experts(a) > 0
    if dense and moe:
        return "dense+moe"
    if moe:
        return "moe"
    return "dense"


# --- §1.1 weights --------------------------------------------------------------

def build_weight_segments(a: Arch) -> List[Segment]:
    """Resident-weight Segments, 5 classes. Attention classes segment on attention-type;
    FFN classes on ffn-type; embed is a singleton. Corrections (k_eq_v v_proj drop, tied
    embed) are folded into the unit (§1.1) -- attn_kv_params already returns 0 for MLA and the
    real K+V params otherwise; the k_eq_v phantom-v drop is applied here per global group."""
    segs: List[Segment] = []
    for g in a.attn_groups:
        at = _attn_type_label(g)
        qo = attn_qo_params(a, g)
        kv = attn_kv_params(a, g)
        # k_eq_v: V aliases K on global (full-context) real-KV groups -> drop v_proj from unit.
        if a.attention_k_eq_v and g.kv_seq_cap is None and g.mla is None:
            kv -= a.hidden_size * g.n_kv_heads * g.head_dim
        segs.append(Segment("weight", "attn_qo", at, g.count, qo))
        if kv > 0:
            segs.append(Segment("weight", "attn_kv", at, g.count, kv))
    # FFN: dense + routed experts, per ffn-type (folded across attention partition).
    for g in a.attn_groups:
        ft = _ffn_label(a, g)
        if g.dense_inter(a) > 0:
            segs.append(Segment("weight", "dense_mlp", ft, g.count,
                                 _mlp_block_params(a, g.dense_inter(a))))
        if g.num_experts(a) > 0:
            segs.append(Segment("weight", "routed_experts", ft, g.count,
                                 _mlp_block_params(a, g.moe_inter(a)) * g.num_experts(a)))
            segs.append(Segment("weight", "router", ft, g.count, router_params(a, g)))
    # embed / lm_head singleton.
    segs.append(Segment("weight", "embed", "", 1, _embed_params(a)))
    return _merge(segs)


# --- §1.2 KV -------------------------------------------------------------------

def build_kv_segments(a: Arch) -> List[Segment]:
    """KV-cache Segments (§1.2): one attn_kv class, attention-type segmented, per-token unit,
    carrying the type's cap. No S/bs here -- that is §3."""
    segs: List[Segment] = []
    for g in a.attn_groups:
        segs.append(Segment("kv", "attn_kv", _attn_type_label(g), g.count,
                            kv_unit_per_token(a, g), cap=g.kv_seq_cap))
    return _merge(segs)


# --- §1.3 FLOPs ----------------------------------------------------------------

def build_flop_segments(a: Arch) -> List[Segment]:
    """Per-token FLOP Segments keyed to the producing weight class (§1.3). Projection terms
    are 2*(class params); the attention-score term is weightless, keyed to attn_qo, ctx_scaled.
    lm_head is a real matmul (2*vocab*hidden) over sampled positions."""
    segs: List[Segment] = []
    for g in a.attn_groups:
        at = _attn_type_label(g)
        segs.append(Segment("flop", "attn_qo", at, g.count, 2 * attn_qo_params(a, g)))
        kv = attn_kv_params(a, g)
        if a.attention_k_eq_v and g.kv_seq_cap is None and g.mla is None:
            kv -= a.hidden_size * g.n_kv_heads * g.head_dim
        if kv > 0:
            segs.append(Segment("flop", "attn_kv", at, g.count, 2 * kv))
        # attention scores: per-token unit is score-flops / eff_ctx (ctx applied in §3).
        score_full = attn_score_flops_per_token(a, g, seq_len=10**9)  # uncapped -> per-eff_ctx
        # attn_score_flops = coeff * eff_ctx; recover coeff = value / eff_ctx (uncapped => 1e9)
        coeff = score_full / (10**9 if g.kv_seq_cap is None else min(10**9, g.kv_seq_cap))
        segs.append(Segment("flop", "attn_score", at, g.count, coeff,
                            cap=g.kv_seq_cap, ctx_scaled=True))
    for g in a.attn_groups:
        ft = _ffn_label(a, g)
        d = dense_mlp_flops_per_token(a, g)
        if d > 0:
            segs.append(Segment("flop", "dense_mlp", ft, g.count, d))
        m = moe_flops_per_token(a, g)
        if m > 0:
            segs.append(Segment("flop", "routed_experts", ft, g.count, m))
        # router: one matmul per token against every expert (2 = MAC). Replicated, not
        # sharded -- see planner.shard.weight_divisor("router").
        r = router_params(a, g)
        if r > 0:
            segs.append(Segment("flop", "router", ft, g.count, 2 * r))
    # lm_head logit projection: 2 * vocab * hidden, sampled positions only.
    segs.append(Segment("flop", "embed", "", 1, 2 * a.vocab_size * a.hidden_size,
                        sampled_only=True))
    return _merge(segs)


# --- merge segments that share a key (keeps the list minimal) ------------------

def _merge(segs: List[Segment]) -> List[Segment]:
    """Combine segments that share a full key (same quantity/cls/layer_type/unit/flags) by
    summing multiplicity. Uniform-FFN models thus collapse their duplicate per-group FFN
    segments to one -- e.g. gemma's dense MLP over 50 sliding + 10 global layers becomes a
    single (dense, 60) segment instead of two. Different units stay distinct."""
    by_full = {}
    order = []
    for s in segs:
        fkey = (s.quantity, s.cls, s.layer_type, round(s.unit, 6),
                s.cap, s.ctx_scaled, s.sampled_only)
        if fkey not in by_full:
            by_full[fkey] = s
            order.append(fkey)
        else:
            prev = by_full[fkey]
            by_full[fkey] = Segment(prev.quantity, prev.cls, prev.layer_type,
                                    prev.multiplicity + s.multiplicity, prev.unit,
                                    prev.cap, prev.ctx_scaled, prev.sampled_only)
    return [by_full[k] for k in order]
