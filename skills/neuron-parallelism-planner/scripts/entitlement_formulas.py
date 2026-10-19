"""Model + hardware primitives shared by the planner engine and the oracle validator.

This module is the single source of truth for:
  - the parsed model (`Arch`/`AttnGroup`/`MLAParams`) and `arch_from_hf` / `load_runspec`;
  - the hardware constants (`HW`, `load_hardware`);
  - the raw per-model accounting the `planner/` segments build on (`weights_attn`,
    `attn_qo_params`, `attn_kv_params`, `_mlp_block_params`, the per-token FLOP terms);
  - the analytic totals the oracle is diffed against (`weights_total`, `flops_per_token`,
    `kv_bytes_total`) and the oracle-driven weight corrections (`corrected_weight_bytes`,
    `embed_table_bytes`, `phantom_global_vproj_bytes`);
  - the topology-aware communication model (`comm_total_ms` and its per-strategy terms).

Sharding, per-device cost, the roofline step time, and config selection live in the
`planner/` package (segments -> shard -> cost -> plan); they import the primitives above so
there is one parse of each model and no drift between the engine and the oracle.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MLAParams:
    """Multi-head Latent Attention projection ranks.

    MLA replaces the standard Q/K/V/O projections with a low-rank factorization and
    caches a single compressed *latent* per token instead of per-KV-head K/V:

      Q:  hidden -> q_a_proj (rank q_lora_rank) -> q_b_proj (n_heads * qk_head_dim)
      KV: hidden -> kv_a_proj (kv_lora_rank + qk_rope_head_dim) -> kv_b_proj
          (n_heads * (qk_nope_head_dim + v_head_dim))
      O:  (n_heads * v_head_dim) -> hidden

    The KV cache holds ONLY the kv_a output -- the compressed latent plus the shared
    RoPE key -- so cached bytes/token/layer = (kv_lora_rank + qk_rope_head_dim), NOT
    2 * n_kv_heads * head_dim. qk_head_dim = qk_nope_head_dim + qk_rope_head_dim.
    """
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_cache_dim(self) -> int:
        """Latent KV-cache width per token per layer (the kv_a_proj output)."""
        return self.kv_lora_rank + self.qk_rope_head_dim


@dataclass
class AttnGroup:
    """A set of identical attention layers (e.g. all the sliding-window layers).

    kv_seq_cap caps how many tokens each layer keeps in its KV cache: the
    sliding-window size for sliding layers, or None (= full context) for global
    layers. This is what makes long-context KV correct for mixed-attention models.

    `mla` is set for Multi-head Latent Attention layers; when present the standard GQA
    head_dim/n_kv_heads byte+FLOP formulas are bypassed in favor of the low-rank MLA path.

    FFN model: each group's feed-forward is the SUM of two independent, PARALLEL paths,
    EITHER of which may be absent:
      - an ALWAYS-ON dense MLP of size `dense_intermediate` (run for every token), and
      - `mlp_num_experts` routed experts of size `moe_intermediate`, `mlp_experts_per_tok`
        active per token.
    The dense path is NOT replaced by the expert path -- they coexist and are summed.
    Set `dense_intermediate=0` for a layer with no always-on MLP; set `mlp_num_experts=0`
    for a pure-dense layer (Granite/Llama). gemma4-A4B sets BOTH (dense MLP and routed MoE
    in parallel). The two intermediates are kept separate.

    SHARED EXPERTS ARE FOLDED INTO `dense_intermediate`: a shared expert is, in every
    reference implementation, just one always-on dense MLP run on every token and added to
    the routed output (no routing, no separate cost). So `arch_from_hf` adds the shared
    width into `dense_intermediate`; there is no shared-expert concept downstream.
    `*_intermediate=None` falls back to the Arch-level `intermediate_size`.
    """
    count: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    kv_seq_cap: Optional[int]  # None => caches the full sequence
    mla: Optional[MLAParams] = None
    # Always-on dense MLP path (incl. folded shared experts). None => a.intermediate_size;
    # 0 => no always-on MLP.
    dense_intermediate: Optional[int] = None
    # Routed-expert path size. None => a.intermediate_size (used when experts present).
    moe_intermediate: Optional[int] = None
    mlp_num_experts: int = 0        # routed experts (0 => no MoE path)
    mlp_experts_per_tok: int = 0    # active routed experts per token

    def dense_inter(self, a: "Arch") -> int:
        return a.intermediate_size if self.dense_intermediate is None else self.dense_intermediate

    def moe_inter(self, a: "Arch") -> int:
        return a.intermediate_size if self.moe_intermediate is None else self.moe_intermediate

    def num_experts(self, a: "Arch") -> int:
        return self.mlp_num_experts

    def experts_per_tok(self, a: "Arch") -> int:
        return self.mlp_experts_per_tok


@dataclass
class Arch:
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_experts: int
    experts_per_tok: int
    dtype_bytes: int
    attn_groups: List[AttnGroup]
    tie_embeddings: bool = True
    max_context: Optional[int] = None  # max_position_embeddings (text config); None if absent
    # V aliases K on full-context (global) groups: no v_proj in the checkpoint. Drives the
    # phantom-global-v_proj weight correction and the oracle's v_proj omission.
    attention_k_eq_v: bool = False
    # Bytes per cached K/V element. The KV cache is often served at a lower precision than
    # the weights (e.g. fp8 KV with bf16 weights), so it is tracked separately. None falls
    # back to dtype_bytes (KV cached at the model dtype).
    kv_dtype_bytes: Optional[int] = None
    # WEIGHT element bytes, when the served weights are a different precision from the
    # model dtype -- e.g. an FP8 checkpoint served FP8-native is 1 byte/param while
    # activations stay BF16 (2). None => weights at the model dtype. This is a serving
    # choice (run-spec `weight_dtype_bytes`), not a model-config fact, so it never touches
    # the pristine HF config. Decode is weight-HBM-bound, so halving this roughly halves
    # the decode floor; it must be modeled IN the engine, not applied as a factor outside it.
    weight_dtype_bytes: Optional[float] = None
    # Optional served precision for a SEPARATE, untied lm_head. Most models either tie the
    # table or serve every weight at ``w_elem_bytes``. Some quantized checkpoints explicitly
    # exclude an untied lm_head; that exception must be declared rather than inferred from
    # the model dtype. A tied model cannot assign a different precision to the same object.
    lm_head_dtype_bytes: Optional[float] = None
    # Serving knob (NOT a model-config fact): on a disaggregated-prefill server with
    # apply_prefill_dcp=True, PREFILL attention weights shard across the DCP sub-group
    # (TP/CP) and replicate across CP -- so per-device prefill attention bytes are CP x
    # larger than full-TP sharding. Default False = standard serving, where BOTH prefill
    # and decode shard attention across full TP (vendor llama3/model.py:244, gated on this
    # flag). See doc/.../parallelism/dcp.md "DCP Prefill". Only affects prefill cells.
    apply_prefill_dcp: bool = False
    # Ordered layer -> attn_groups index map. ``attn_groups`` intentionally compresses
    # identical layer variants for weight/FLOP accounting, so its list order alone cannot
    # represent an interleaved hybrid schedule such as Gemma's five sliding layers followed
    # by one global layer. Component boundaries and the structural oracle use this map.
    # ``None`` preserves the historical contiguous group-by-group order for directly
    # constructed Arch fixtures.
    layer_group_indices: Optional[List[int]] = None
    # Activation-HBM decomposition contract. Component-boundary traffic is the model-floor
    # default; per_operator is an explicit decomposition/Floor(D)-style alternative that
    # replaces the graph-cut bytes with every operator interface.
    activation_io_policy: str = "component_boundary"

    @property
    def num_layers(self) -> int:
        return sum(g.count for g in self.attn_groups)

    def ordered_layer_groups(self):
        """Yield ``(group_index, group)`` in model execution order.

        The explicit map is validated here so every order-sensitive consumer fails at the
        same boundary. Older hand-built Arch objects omit it and retain contiguous
        group-by-group behavior.
        """
        if self.layer_group_indices is None:
            for gi, g in enumerate(self.attn_groups):
                for _ in range(g.count):
                    yield gi, g
            return
        if len(self.layer_group_indices) != self.num_layers:
            raise ValueError(
                "layer_group_indices has %d entries, expected %d layers"
                % (len(self.layer_group_indices), self.num_layers))
        counts = [0] * len(self.attn_groups)
        for gi in self.layer_group_indices:
            if gi < 0 or gi >= len(self.attn_groups):
                raise ValueError(
                    "layer_group_indices contains invalid group index %r for %d groups"
                    % (gi, len(self.attn_groups)))
            counts[gi] += 1
        expected = [g.count for g in self.attn_groups]
        if counts != expected:
            raise ValueError(
                "layer_group_indices multiplicities %r do not match attn_groups counts %r"
                % (counts, expected))
        for gi in self.layer_group_indices:
            yield gi, self.attn_groups[gi]

    @property
    def kv_elem_bytes(self) -> int:
        """Bytes per cached K/V element: kv_dtype_bytes if set, else the model dtype."""
        return self.kv_dtype_bytes if self.kv_dtype_bytes is not None else self.dtype_bytes

    @property
    def w_elem_bytes(self) -> float:
        """Bytes per WEIGHT element: weight_dtype_bytes if set, else the model dtype."""
        return (self.weight_dtype_bytes if self.weight_dtype_bytes is not None
                else self.dtype_bytes)

    @property
    def lm_head_elem_bytes(self) -> float:
        """Bytes per untied lm_head element: explicit override, else served weight bytes."""
        return (self.lm_head_dtype_bytes
                if self.lm_head_dtype_bytes is not None else self.w_elem_bytes)

    @property
    def vocab_table_effective_elem_bytes(self) -> float:
        """Average bytes/parameter for the planner's aggregate vocab-table segment.

        Tied models own one table at the common weight precision. Untied models own an
        embedding at the common precision plus an lm_head that may declare another precision;
        their aggregate segment has twice the parameters, so its effective size is the mean.
        """
        if self.tie_embeddings:
            return self.w_elem_bytes
        return (self.w_elem_bytes + self.lm_head_elem_bytes) / 2


@dataclass
class HW:
    # Dense BF16 tensor-engine ceiling PER LOGICAL NeuronCore. Under the supported LNC2
    # geometry this is the aggregate of two physical NeuronCore-v3 tensor engines.
    te_peak_flops: float
    # HBM bandwidth available to one LOGICAL NeuronCore/LNC2 rank. This is a direct share of
    # the device HBM subsystem, not a sum of per-physical-core DMA bandwidths.
    hbm_bw_bytes_per_s: float
    # Collective-axis contracts. These are not per-physical-core rates and must not be
    # multiplied by ``lnc``. They remain analytical collective inputs rather than part of the
    # certified TE/HBM component floor.
    link_bw_within_node_bytes_per_s: float
    link_bw_cross_node_bytes_per_s: float
    # Number of LOGICAL NeuronCores/ranks exposed by the instance.
    cores_per_node: int
    # HBM capacity PER LOGICAL NeuronCore, in bytes. Generation-specific:
    # trn1 = 16 GiB, trn2 = 24 GiB (96 GiB/device / 4 LNC2 ranks), trn3 = 144 GiB.
    # This is the per-device memory budget the sweep's fit / max-feasible-batch
    # filters charge against. Defaults to 16 GiB to keep older flat configs valid.
    hbm_capacity_bytes_per_core: float = 16 * 1024**3
    # Logical-NeuronCore configuration: how many PHYSICAL NeuronCores are fused into
    # one logical core. Trainium2 LNC2 = 2 physical cores per logical core, which is
    # the ONLY mode this analysis supports. TE peak is aggregated over those two cores;
    # HBM bandwidth and capacity are direct per-LNC2 values from the device HBM subsystem.
    # cores_per_node already reflects the logical rank count. Link fields are
    # collective-axis contracts and are deliberately not multiplied by LNC. LNC1 would
    # change the per-rank engine/memory values and rank count. The planner assumes
    # lnc == 2 (LNC2); None denotes a non-Trainium fabric such as switched NVL72.
    lnc: Optional[int] = None
    # Interconnect topology. "torus" (TPU pod / Trainium rack): rings map natively to mesh
    # axes, so all-reduce/all-gather/reduce-scatter are bandwidth-optimal along an axis.
    # "switched" (NVL72 NVSwitch / fat-tree Clos): ~full bisection and ~uniform pairwise
    # latency, so a non-ring collective is also cheap.
    # NOTE: every collective the cost model now charges is a RING (TP/EP all-reduce, CP K/V
    # ring, PP p2p), so no term consults this field today -- the EP all-to-all it existed to
    # penalize is gone (comm_ep_ms models the replicated-token AllReduce instead, per
    # design.md 2.2). Kept because it is part of the documented hardware-JSON schema and
    # would be needed again by any non-ring collective.
    topology: str = "torus"
    # Bisection bandwidth available to a collective that cannot ride a single ring. If None we
    # derive it: switched -> within-node BW; torus -> cross-node BW. Also unused by the current
    # comm model, for the same reason as `topology` above.
    bisection_bw_bytes_per_s: Optional[float] = None
    # Latency floor charged once per collective (dispatch/sync), in seconds. The ring terms use
    # a per-HOP latency constant instead (COLLECTIVE_PER_HOP_LATENCY_S); this field is only read
    # by the CP ring and PP p2p terms.
    collective_latency_s: float = 0.0
    # Fraction of hbm_capacity_bytes_per_core the planner may fill with weights + KV +
    # activations; the rest is runtime/fragmentation/allocator headroom. This is the
    # per-core memory BUDGET the fit / max-feasible-batch filters charge against (via
    # plan.usable_hbm). Default 0.90 preserves the original behavior. Lower it to model a
    # tighter budget ("I can't fully fit everything yet") without editing code -- settable
    # in the hardware JSON, overridable per run-spec ("usable_hbm_fraction") or CLI
    # (--usable-fraction).
    usable_fraction: float = 0.90

    @property
    def is_switched(self) -> bool:
        return self.topology.lower() in ("switched", "nvswitch", "fat_tree", "clos")

    def alltoall_bw(self) -> float:
        """Bisection BW a non-ring collective would get. Unused by the current comm model
        (every term it charges is a ring); see the `topology` field note."""
        if self.bisection_bw_bytes_per_s is not None:
            return self.bisection_bw_bytes_per_s
        return self.link_bw_within_node_bytes_per_s if self.is_switched \
            else self.link_bw_cross_node_bytes_per_s


# --- derived weight/FLOP counts (per layer, before sharding) -------------------

def weights_attn(a: Arch, g: AttnGroup) -> int:
    """Params in one attention block of group `g`.

    GQA/MHA: Q + (K,V) + O projections. MLA: the low-rank factorization
    (q_a + q_b + kv_a + kv_b + o), which is the checkpoint's actual weight set.
    """
    if g.mla is not None:
        m = g.mla
        q_a = a.hidden_size * m.q_lora_rank
        q_b = m.q_lora_rank * g.n_q_heads * m.qk_head_dim
        kv_a = a.hidden_size * m.kv_cache_dim
        kv_b = m.kv_lora_rank * g.n_q_heads * (m.qk_nope_head_dim + m.v_head_dim)
        o = g.n_q_heads * m.v_head_dim * a.hidden_size
        return q_a + q_b + kv_a + kv_b + o
    q = a.hidden_size * g.n_q_heads * g.head_dim
    kv = 2 * a.hidden_size * g.n_kv_heads * g.head_dim
    o = g.n_q_heads * g.head_dim * a.hidden_size
    return q + kv + o


def _mlp_block_params(a: Arch, inter: int) -> int:
    """Params in one SwiGLU/GeGLU MLP block of intermediate `inter`: gate + up + down."""
    return 3 * a.hidden_size * inter


def router_params(a: Arch, g: AttnGroup) -> int:
    """Params in the MoE ROUTER (gate) of one layer of group `g`: hidden x num_experts.

    Zero unless the group has routed experts. Tiny in aggregate (~0.04% of a MoE model's
    weights) but it is REPLICATED on every rank rather than TP-sharded, so its per-device share
    grows with TP -- ~1.5% of the routed-FFN weight at TP=32 on gemma-4-26B-A4B, and linear in
    num_experts. The softmax / top-k over the router logits is not a matmul and is not counted
    (the oracle does not count it either).
    """
    return a.hidden_size * g.num_experts(a) if g.num_experts(a) > 0 else 0


def _mlp_resident_params(a: Arch, g: AttnGroup) -> int:
    """All FFN params held resident for one layer of group `g`.

    The dense MLP and the expert pool are PARALLEL paths and both count (either may be
    absent): dense_intermediate (always-on, incl. folded shared experts) + num_experts
    routed experts at moe_intermediate + the router that picks between them.
    """
    dense = _mlp_block_params(a, g.dense_inter(a)) if g.dense_inter(a) > 0 else 0
    n_routed = g.num_experts(a)
    experts = _mlp_block_params(a, g.moe_inter(a)) * n_routed if n_routed > 0 else 0
    return dense + experts + router_params(a, g)


def weights_total(a: Arch) -> int:
    """Total transformer-block params loaded per forward step, summed over groups.

    Per layer: attention + dense MLP + all routed experts (+ shared experts). The dense
    and expert paths coexist (gemma4-A4B) or either is absent (pure-dense / pure-MoE).
    """
    total = 0
    for g in a.attn_groups:
        total += g.count * (weights_attn(a, g) + _mlp_resident_params(a, g))
    return total


# --- oracle-driven weight corrections (Gemma-specific, harmless elsewhere) ------
# weights_total counts transformer blocks only. Two corrections reconcile it with the
# meta-device oracle: (+) the tied embedding / lm_head table (counted once), and (-) the
# phantom global v_proj that attention_k_eq_v removes (V aliases K on global layers). These
# live here (next to weights_total, their only dependency) so the engine, the planner, and
# validate_oracle share one definition. At model precision corrected_weight_bytes matches the
# oracle to the byte; serving knobs then scale each resident class explicitly.

def embed_table_bytes(a: Arch) -> float:
    """Served resident bytes for embedding and lm_head storage.

    A tied model owns one vocab x hidden table at the common served-weight precision. An
    untied model owns an embedding at that precision plus a separate lm_head, which may use
    ``lm_head_dtype_bytes`` when the deployment explicitly excludes it from quantization.
    """
    params = a.vocab_size * a.hidden_size
    if a.tie_embeddings:
        return params * a.w_elem_bytes
    return params * (a.w_elem_bytes + a.lm_head_elem_bytes)


def phantom_global_vproj_bytes(a: Arch) -> float:
    """V-projection params weights_total counts on global layers but attention_k_eq_v removes
    (V aliases K there). Zero unless the config sets attention_k_eq_v."""
    if not a.attention_k_eq_v:
        return 0
    total = 0
    for g in a.attn_groups:
        if g.kv_seq_cap is None:  # global (full-context) layers alias V to K
            total += g.count * (a.hidden_size * g.n_kv_heads * g.head_dim)
    return total * a.w_elem_bytes


def corrected_weight_bytes(a: Arch) -> float:
    """Resident weight bytes reconciled with the oracle: blocks + embed table - phantom vproj."""
    return weights_total(a) * a.w_elem_bytes + embed_table_bytes(a) - phantom_global_vproj_bytes(a)


def flops_per_token(a: Arch, seq_len: int) -> float:
    """Matmul FLOPs to process ONE token, summed over attention groups.

    factor 2 = multiply-add. Attention scores (QK^T) and context (PV) scale with
    the per-layer effective context = min(seq_len, kv_seq_cap) -- so sliding-window
    layers stop growing past their window. MoE routes to `experts_per_tok` experts
    (+ shared experts run every token). The attention-score term uses the per-head
    score width: qk_head_dim for the scores, v_head_dim for the context, for MLA.
    """
    total = 0.0
    for g in a.attn_groups:
        eff_ctx = seq_len if g.kv_seq_cap is None else min(seq_len, g.kv_seq_cap)
        attn_proj = 2 * weights_attn(a, g)
        if g.mla is not None:
            score_dim = g.mla.qk_head_dim + g.mla.v_head_dim
            attn_scores = 2 * g.n_q_heads * score_dim * eff_ctx
        else:
            attn_scores = 4 * g.n_q_heads * g.head_dim * eff_ctx
        # FFN FLOPs: the always-on dense MLP (incl. folded shared experts) PLUS the active
        # routed experts, each path at its own intermediate, PLUS the router that scores every
        # token against every expert. 6 = 2 (MAC) * 3 (gate/up/down); the router is one matmul,
        # so 2 (MAC) * hidden * num_experts.
        dense_mlp = 6 * a.hidden_size * g.dense_inter(a) if g.dense_inter(a) > 0 else 0
        active = g.experts_per_tok(a)
        moe_mlp = 6 * a.hidden_size * g.moe_inter(a) * active if active > 0 else 0
        router = 2 * router_params(a, g)
        total += g.count * (attn_proj + attn_scores + dense_mlp + moe_mlp + router)
    return total


# --- per-component accounting (consumed by the planner/ segments) --------------
# Attention splits into Q/O vs K/V because they shard differently under TP: Q/O shard
# across the query-head count (full TP), but K/V shard only to n_kv_heads and then
# REPLICATE (TP/n_kv copies). Keeping them separate lets the planner's weight AND flop
# segments both charge the KV replication correctly. MLA has no separate per-KV-head K/V
# projection (it caches a shared latent), so its whole attention block is treated as
# Q/O-style (no kv-replication split) -- a documented simplification.

def attn_qo_params(a: Arch, g: AttnGroup) -> int:
    """Attention params that shard across the full query-head count (TP): Q + O (GQA),
    or the entire MLA projection set (q_a/q_b/kv_a/kv_b/o) since MLA has no replicated
    per-KV-head K/V projection."""
    if g.mla is not None:
        return weights_attn(a, g)  # whole MLA block; no kv-replication split
    q = a.hidden_size * g.n_q_heads * g.head_dim
    o = g.n_q_heads * g.head_dim * a.hidden_size
    return q + o


def attn_kv_params(a: Arch, g: AttnGroup) -> int:
    """Attention K+V projection params -- the part that shards to min(TP, n_kv) and then
    REPLICATES. Zero for MLA (folded into attn_qo_params)."""
    if g.mla is not None:
        return 0
    return 2 * a.hidden_size * g.n_kv_heads * g.head_dim


def active_attention_pairs(seq_len: int, cap: Optional[int] = None) -> int:
    """Causal query/key pairs for an empty-prefix prefill.

    Query position ``q`` (1-indexed) can see ``min(q, cap)`` keys for a capped
    attention window and all ``q`` preceding/current keys for full attention.
    Counting the dense ``S * min(S, cap)`` rectangle would include masked future
    positions and can overstate a model floor by almost 2x.
    """
    seq_len = int(seq_len)
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative, got %r" % seq_len)
    if seq_len == 0:
        return 0
    if cap is None:
        return seq_len * (seq_len + 1) // 2
    cap = int(cap)
    if cap <= 0:
        raise ValueError("attention cap must be positive, got %r" % cap)
    active = min(seq_len, cap)
    return active * (active + 1) // 2 + max(seq_len - active, 0) * active


def attn_score_flops_per_token(a: Arch, g: AttnGroup, seq_len: int) -> float:
    """Decode QK^T + PV FLOPs for one token (factor already includes MAC).

    Prefill is not ``this value * seq_len``: a causal prompt has triangular
    active-pair geometry. The segment cost engine uses ``active_attention_pairs``
    for prefill and this helper only for the per-pair coefficient.
    """
    eff_ctx = seq_len if g.kv_seq_cap is None else min(seq_len, g.kv_seq_cap)
    if g.mla is not None:
        score_dim = g.mla.qk_head_dim + g.mla.v_head_dim
        return 2 * g.n_q_heads * score_dim * eff_ctx
    return 4 * g.n_q_heads * g.head_dim * eff_ctx


def dense_mlp_flops_per_token(a: Arch, g: AttnGroup) -> float:
    """Always-on dense MLP FLOPs/token for one layer of group g (0 if no dense path)."""
    return 6 * a.hidden_size * g.dense_inter(a) if g.dense_inter(a) > 0 else 0.0


def moe_flops_per_token(a: Arch, g: AttnGroup) -> float:
    """Active routed-expert FLOPs/token for one layer of group g (0 if dense-only)."""
    active = g.experts_per_tok(a)
    return 6 * a.hidden_size * g.moe_inter(a) * active if active > 0 else 0.0


# --- analytic totals (the oracle-validation reference) + comm model ------------


def kv_bytes_total(a: Arch, seq_len: int, bs: int) -> float:
    """K + V cache bytes for the whole batch, summed over attention groups.

    Each group caches min(seq_len, kv_seq_cap) tokens -- sliding-window layers are
    capped at their window, so they do NOT grow with long context.
    """
    total = 0.0
    for g in a.attn_groups:
        eff_ctx = seq_len if g.kv_seq_cap is None else min(seq_len, g.kv_seq_cap)
        if g.mla is not None:
            # MLA caches ONE compressed latent (kv_lora_rank + qk_rope_head_dim) per
            # token per layer -- no factor 2, no n_kv_heads. This is the whole point
            # of MLA: the cache is ~(2*n_kv_heads*head_dim / kv_cache_dim)x smaller.
            per_tok = g.mla.kv_cache_dim
        else:
            per_tok = 2 * g.n_kv_heads * g.head_dim
        total += g.count * per_tok * a.kv_elem_bytes * eff_ctx * bs
    return total


def tokens_per_step(seq_len: int, bs: int, prefill: bool) -> int:
    """Tokens processed in one timed step.

    Prefill processes the whole prompt at once (S*bs tokens). Decode produces one
    new token per sequence per step, so the batch yields `bs` tokens per step.
    """
    return seq_len * bs if prefill else bs


def _axis_link_bw(hw: HW, degree: int) -> float:
    """Link BW for a ring/axis collective spanning `degree` ranks.

    A collective that fits inside one node rides the within-node link. Single-node
    serving is a hard scope constraint of this model, so a degree exceeding one
    node's logical-core count is a caller bug, not a config to price: it raises
    rather than silently falling back to the cross-node link (which previously let
    the sweep return, and rank, multi-node configs).

    `link_bw_cross_node_bytes_per_s` is retained on HW as the torus bisection proxy behind
    HW.alltoall_bw(), which no current comm term reads (see the HW.topology note).
    """
    if degree > hw.cores_per_node:
        raise ValueError(
            f"collective degree {degree} exceeds cores_per_node {hw.cores_per_node}: "
            "multi-node / multi-instance serving is out of scope for this planner. "
            "Keep TP * DP within a single node."
        )
    return hw.link_bw_within_node_bytes_per_s


def _act_bytes(a: Arch, seq_len: int, bs: int, prefill: bool) -> float:
    """Activation-tensor message size: the per-step tokens x hidden, in bytes."""
    return tokens_per_step(seq_len, bs, prefill) * a.hidden_size * a.dtype_bytes


def gqa_ratio(a: Arch) -> float:
    """Query-per-KV-head ratio g = n_q / n_kv (max over groups). MHA -> 1; GQA -> g>1.

    CP ships the K/V tensors around the ring, so its volume divides by g: GQA caches
    g-fewer KV heads than query heads, so there is g-fewer K/V to exchange.
    """
    return max(g.n_q_heads / g.n_kv_heads for g in a.attn_groups)


# --- intra-node collective cost model (per-hop latency + achieved bandwidth) ----------
# Ring/KangaRing collective cost = latency + transfer (AWS Neuron intra-node collectives):
#   https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/explore/intranode-collective-comm.html
# Per-hop Ring latency is ~1-2 us (HBM-load dependent); we use 1.5 us/hop. The transfer
# term rides an ACHIEVED intra-node bandwidth of ~100 GB/s -- the doc's own throughput
# reference (100 GB/s => 1 MB in 10 us), well below the raw link peak because the small
# activation messages a decode step issues never reach link BW.
# NOTE: this is an INTRA-NODE model -- all ranks of the collective within a single node.
# Cross-node / multi-node collective scaling is out of scope for these terms.
COLLECTIVE_PER_HOP_LATENCY_S = 1.5e-6        # ~1-2 us/hop Ring latency (doc); we pick 1.5 us
COLLECTIVE_INTRANODE_BW_BYTES_PER_S = 100e9  # achieved intra-node BW (doc's 100 GB/s reference)


def ring_allreduce_s(n: int, msg_bytes: float) -> float:
    """Cost (seconds) of ONE ring AllReduce over `n` ranks on a `msg_bytes` tensor.

    ReduceScatter (n-1 hops) + AllGather (n-1 hops) = 2*(n-1) hops, moving 2*(n-1)/n of the
    message per rank; cost = per-hop latency * hops + transfer / achieved intra-node BW. Ring
    collectives map natively onto a torus axis, so there is no topology derate here -- that
    only applies to a collective which cannot ride a single ring.

    Defined once and shared by every AllReduce term (TP, EP) so the terms cannot drift apart.
    """
    if n <= 1:
        return 0.0
    hops = 2 * (n - 1)
    return (hops * COLLECTIVE_PER_HOP_LATENCY_S
            + (2 * (n - 1) / n) * msg_bytes / COLLECTIVE_INTRANODE_BW_BYTES_PER_S)


def comm_tp_ms(a: Arch, hw: HW, TP: int, PP: int, seq_len: int, bs: int, prefill: bool) -> float:
    """Tensor-parallel AllReduce. 2 collectives/layer on the activation tensor.

    Ring AllReduce = ReduceScatter (TP-1 hops) + AllGather (TP-1 hops) = 2*(TP-1) hops,
    moving 2*(TP-1)/TP * msg_bytes per rank. Cost per collective is latency + transfer
    (AWS Neuron intra-node collective doc, linked above):
      latency  = 2*(TP-1) hops * per-hop latency (1.5 us)
      transfer = 2*(TP-1)/TP * msg_bytes / achieved-intra-node-BW (100 GB/s)
    INTRA-NODE model: assumes the TP group is within one node. For the small per-token
    activation messages of decode this is latency-dominated (transfer << latency); the
    transfer term only takes over at large messages (~0.5 MB, i.e. large batch).
    """
    if TP <= 1:
        return 0.0
    msg = _act_bytes(a, seq_len, bs, prefill)
    n_coll = 2 * (a.num_layers / PP)
    return n_coll * ring_allreduce_s(TP, msg) * 1e3


# Alias: the TP AllReduce term is also exposed as comm_ms; the full per-strategy total is
# comm_total_ms below (the one the planner uses).
comm_ms = comm_tp_ms


def comm_cp_ms(a: Arch, hw: HW, CP: int, PP: int, seq_len: int, bs: int, prefill: bool) -> float:
    """Context-parallel ring K/V exchange: 2*bsh*(CP-1)/CP * (1/g) per step.

    CP shards the sequence; each rank rings its K/V shard to the others so every rank
    can attend over the full context. Volume is the K/V activation (not Q), so it
    divides by the GQA ratio g. On a torus the ring is the *native* primitive (ideal,
    and in prefill it hides behind O(s^2) attention compute); on a switched fabric the
    ring underuses full bisection but costs the same bandwidth-wise. Topology-neutral
    in volume, like TP.
    """
    if CP <= 1:
        return 0.0
    link_bw = _axis_link_bw(hw, CP)
    msg = _act_bytes(a, seq_len, bs, prefill) / gqa_ratio(a)
    n_coll = a.num_layers / PP        # one K/V ring per layer
    vol_term = (2 * (CP - 1) / CP) * msg / link_bw
    return n_coll * (vol_term + hw.collective_latency_s) * 1e3


def comm_pp_ms(a: Arch, hw: HW, PP: int, seq_len: int, bs: int, prefill: bool) -> float:
    """Pipeline P2P: (PP-1) inter-stage activation sends of bsh bytes.

    A microbatch's activations hand off stage->stage; total cross-stage volume is
    (PP-1)*bsh. Neighbor links only -- cheap and contention-free on a torus axis,
    trivial on a switch. Does NOT help/hurt single-request latency (already modeled
    via step_ms*PP elsewhere); this is the wire cost of the handoffs themselves.
    """
    if PP <= 1:
        return 0.0
    link_bw = _axis_link_bw(hw, PP)
    msg = _act_bytes(a, seq_len, bs, prefill)
    return (PP - 1) * (msg / link_bw + hw.collective_latency_s) * 1e3


def comm_ep_ms(a: Arch, hw: HW, TP: int, EP: int, PP: int,
               seq_len: int, bs: int, prefill: bool) -> float:
    """Expert-parallel combine: a MoE layer's FFN AllReduce widens from TP to the expert grid.

    This models the EP implementation the divisor model describes (design.md 2.2): tokens are
    NOT dispatched to a core subset. Every rank in the expert grid holds all tokens, runs only
    its LOCAL experts, and ONE AllReduce sums the partial FFN outputs (a token's unrouted
    experts contribute zero). There is therefore no all-to-all dispatch/combine to charge --
    and no bisection-bandwidth term, because a ring AllReduce maps natively onto a torus axis.

    ``EP`` here is the expert-grid span (``world = TP*DP``; ``ep_degree * ep_tp == world``, and
    the reduction has to cross both factors). ``comm_tp_ms`` ALREADY charges an AllReduce over
    TP for every layer's FFN, so this term is the INCREMENT from widening that one collective
    to the expert grid, on the MoE layers only. Charging a whole additional collective here
    would double-count the TP-span reduction that is already in the TP term.

    Consequences worth knowing: at DP=1 the expert grid IS the TP group, so the increment is
    zero -- a pure-TP MoE model pays no extra collective, its expert reduction is the FFN
    AllReduce it already pays. Dense models (num_experts<=1) have no term at all. Only the MoE
    layers are charged: the dense (first_k_dense_replace) layers reduce over TP like any dense
    FFN, so counting a.num_layers would overcharge a hybrid dense-then-MoE model.
    """
    if EP <= 1 or a.num_experts <= 1 or EP <= TP:
        return 0.0
    n_moe_layers = sum(g.count for g in a.attn_groups if g.num_experts(a) > 1)
    if not n_moe_layers:
        return 0.0
    msg = _act_bytes(a, seq_len, bs, prefill)
    increment_s = ring_allreduce_s(EP, msg) - ring_allreduce_s(TP, msg)
    return (n_moe_layers / PP) * max(increment_s, 0.0) * 1e3


def comm_total_ms(a: Arch, hw: HW, TP: int, PP: int, CP: int, EP: int,
                  seq_len: int, bs: int, prefill: bool) -> dict:
    """All parallel-strategy communication terms (ms), plus their sum.

    Returns a dict so callers can see the per-strategy breakdown and pick either the
    total (these collectives are largely sequential within a step) or the dominant
    one. EP is dense-safe (0 when num_experts<=1)."""
    tp = comm_tp_ms(a, hw, TP, PP, seq_len, bs, prefill)
    cp = comm_cp_ms(a, hw, CP, PP, seq_len, bs, prefill)
    pp = comm_pp_ms(a, hw, PP, seq_len, bs, prefill)
    # EP is the expert-grid SPAN (world), and its term is the increment over the TP-span FFN
    # AllReduce already counted in `tp` -- hence TP is passed in.
    ep = comm_ep_ms(a, hw, TP, EP, PP, seq_len, bs, prefill)
    return {"tp": tp, "cp": cp, "pp": pp, "ep": ep, "total": tp + cp + pp + ep}


# --- bound combination + derived axes ------------------------------------------

def roofline_envelope(hw: HW, ai: float) -> float:
    """Roofline ceiling at a given arithmetic intensity, in FLOP/s.

    min(TE peak, AI * HBM BW). A HARDWARE function (the ceiling), distinct from AI itself
    (a workload property, defined on the cost Triple in planner/cost.py). Divide by 1e12
    for TFLOP/s.
    """
    return min(hw.te_peak_flops, ai * hw.hbm_bw_bytes_per_s)


# --- config loading ------------------------------------------------------------

def arch_from_hf(hf: dict) -> Arch:
    """Build an Arch from a HuggingFace config.json.

    Reads `text_config` (vision/audio towers are ignored for text-inference
    entitlement). Handles Gemma-style mixed attention: `layer_types` lists each
    layer as 'sliding_attention' or 'full_attention'; sliding layers cache only
    `sliding_window` tokens and use (num_key_value_heads, head_dim), while full
    layers cache the whole sequence and use (num_global_key_value_heads,
    global_head_dim). Falls back to a single uniform group when those keys are
    absent (standard Llama/Qwen-style configs).
    """
    tc = hf.get("text_config", hf)  # tolerate a flat config too
    h = tc["hidden_size"]
    n_q = tc["num_attention_heads"]

    dtype = (tc.get("dtype") or hf.get("dtype") or "bfloat16").lower()
    dtype_bytes = 4 if "32" in dtype else 2

    dense_intermediate = tc["intermediate_size"]  # standalone dense MLP size

    # --- FFN parse. The FFN of a layer reduces to exactly two quantities for entitlement:
    #   (1) ALWAYS-ON intermediate width  -- run for every token (resident bytes + active
    #       FLOPs). This is the standalone dense MLP PLUS any shared experts. We FOLD
    #       shared experts into the dense width: a shared expert is, in every reference
    #       impl (deepseek DeepseekV3MLP, qwen2/3 Qwen*MLP, granitemoehybrid shared_mlp),
    #       just one always-on dense MLP run on every token and added -- there is no
    #       separate "shared" cost. N shared experts of width W == one dense MLP of width
    #       N*W. So we have NO shared-expert concept downstream; it is dense width.
    #   (2) ROUTED pool -- (num_experts resident, experts_per_tok active) at moe_intermediate.
    # Field names vary by model family; we accept every spelling we have verified in the
    # transformers configs, so the parse is robust to which convention a provider uses.
    moe_block_cfg = bool(tc.get("enable_moe_block"))
    n_routed = tc.get("n_routed_experts")

    # Routed-expert count: n_routed_experts (deepseek) | num_experts (qwen3_moe, mixtral,
    # olmoe, gpt_oss) | num_local_experts (mixtral, phimoe, granitemoe).
    num_experts = (n_routed if n_routed is not None
                   else tc.get("num_experts") if tc.get("num_experts") is not None
                   else tc.get("num_local_experts")) or 0
    # Active per token: num_experts_per_tok (universal) | top_k_experts (gemma4).
    experts_per_tok = tc.get("num_experts_per_tok") or tc.get("top_k_experts") or 0
    has_routed = num_experts > 0
    # Routed-expert intermediate: moe_intermediate_size (qwen*_moe, deepseek, qwen3_next) |
    # expert_intermediate_size, else the dense intermediate (mixtral/phimoe/olmoe/gpt_oss
    # reuse intermediate_size for experts).
    moe_intermediate = (tc.get("moe_intermediate_size") or tc.get("expert_intermediate_size")
                        or dense_intermediate)

    # Shared-expert width, FOLDED INTO the always-on dense path. Two config encodings:
    #   - a SIZE: shared_expert_intermediate_size (qwen2_moe, qwen3_next) |
    #     shared_intermediate_size (granitemoehybrid)
    #   - a COUNT: n_shared_experts (deepseek) * moe_intermediate
    shared_width = (tc.get("shared_expert_intermediate_size")
                    or tc.get("shared_intermediate_size")
                    or (tc.get("n_shared_experts", 0) or 0) * moe_intermediate)

    # --- MLA detection: low-rank q/kv with a compressed latent KV cache. Presence of
    # kv_lora_rank is the signal.
    mla = None
    if tc.get("kv_lora_rank") is not None:
        mla = MLAParams(
            q_lora_rank=tc["q_lora_rank"],
            kv_lora_rank=tc["kv_lora_rank"],
            qk_nope_head_dim=tc["qk_nope_head_dim"],
            qk_rope_head_dim=tc["qk_rope_head_dim"],
            v_head_dim=tc["v_head_dim"])

    n_layers = tc["num_hidden_layers"]
    head_dim = tc.get("head_dim") or (h // n_q)
    n_kv = tc.get("num_key_value_heads", n_q)

    layer_types = tc.get("layer_types")

    # FFN fields for the regimes where every layer has the SAME FFN (uniform): pure dense
    # (no routed experts), or routed-MoE-in-parallel-with-dense (enable_moe_block, gemma4).
    # always-on dense width = standalone dense MLP + folded shared experts.
    def _uniform_ffn():
        d = dict(dense_intermediate=dense_intermediate + shared_width)
        if has_routed:
            d.update(moe_intermediate=moe_intermediate, mlp_num_experts=num_experts,
                     mlp_experts_per_tok=experts_per_tok)
        return d

    if has_routed and n_routed is not None and not moe_block_cfg:
        # MoE REPLACES the standalone dense MLP after first_k_dense_replace layers
        # (deepseek). Dense layers: pure dense. MoE layers: routed experts + the shared
        # expert as the always-on dense width (NO standalone dense -> just shared_width).
        # This convention does not co-occur with layer_types in any known model.
        if layer_types:
            raise NotImplementedError(
                "arch_from_hf: n_routed_experts + layer_types (split-FFN x mixed-attention) "
                "is not modeled; no known model needs it.")
        n_dense = tc.get("first_k_dense_replace", 0)
        n_moe = n_layers - n_dense
        groups = []
        layer_group_indices = []
        if n_dense:
            groups.append(AttnGroup(
                count=n_dense, n_q_heads=n_q, n_kv_heads=n_kv, head_dim=head_dim,
                kv_seq_cap=None, mla=mla, dense_intermediate=dense_intermediate))
            layer_group_indices.extend([len(groups) - 1] * n_dense)
        groups.append(AttnGroup(
            count=n_moe, n_q_heads=n_q, n_kv_heads=n_kv, head_dim=head_dim,
            kv_seq_cap=None, mla=mla,
            dense_intermediate=shared_width,  # always-on = the shared expert(s) only
            moe_intermediate=moe_intermediate, mlp_num_experts=num_experts,
            mlp_experts_per_tok=experts_per_tok))
        layer_group_indices.extend([len(groups) - 1] * n_moe)
    elif layer_types and any("sliding" in t for t in layer_types):
        # GENUINELY mixed sliding/global attention (gemma). An all-`full_attention`
        # layer_types (e.g. qwen2_moe) is not mixed -- it falls through to the uniform
        # branch below. FFN (A or C) is uniform and attaches to both groups.
        ffn = _uniform_ffn()
        n_sliding = sum(1 for t in layer_types if "sliding" in t)
        n_full = sum(1 for t in layer_types if "full" in t)
        groups = []
        kind_to_group = {}
        if n_sliding:
            groups.append(AttnGroup(
                count=n_sliding, n_q_heads=n_q,
                n_kv_heads=n_kv, head_dim=head_dim,
                kv_seq_cap=tc.get("sliding_window"), **ffn))
            kind_to_group["sliding"] = len(groups) - 1
        if n_full:
            groups.append(AttnGroup(
                count=n_full, n_q_heads=n_q,
                n_kv_heads=tc.get("num_global_key_value_heads", n_kv),
                head_dim=tc.get("global_head_dim", head_dim),
                kv_seq_cap=None, **ffn))
            kind_to_group["full"] = len(groups) - 1
        layer_group_indices = [
            kind_to_group["sliding" if "sliding" in layer_type else "full"]
            for layer_type in layer_types
        ]
    else:
        # uniform attention; FFN (A or C) uniform.
        groups = [AttnGroup(
            count=n_layers, n_q_heads=n_q,
            n_kv_heads=n_kv, head_dim=head_dim,
            kv_seq_cap=tc.get("sliding_window"), mla=mla, **_uniform_ffn())]
        layer_group_indices = [0] * n_layers

    return Arch(
        hidden_size=h, intermediate_size=dense_intermediate, vocab_size=tc["vocab_size"],
        num_experts=num_experts, experts_per_tok=experts_per_tok,
        dtype_bytes=dtype_bytes, attn_groups=groups,
        tie_embeddings=bool(tc.get("tie_word_embeddings", hf.get("tie_word_embeddings", False))),
        max_context=tc.get("max_position_embeddings"),
        attention_k_eq_v=bool(tc.get("attention_k_eq_v", False)),
        layer_group_indices=layer_group_indices)


def load_hardware(json_path: str) -> HW:
    """Load the trn2 hardware-constants file (Spike S2 output). Raises on TODOs."""
    with open(json_path) as f:
        raw = json.load(f)
    todos = [k for k, v in raw.items()
             if not k.startswith("_") and isinstance(v, str) and v.startswith("TODO")]
    if todos:
        raise ValueError(f"{json_path} still has unfilled placeholders: {todos}")
    return HW(**{k: v for k, v in raw.items() if not k.startswith("_")})


def apply_runspec_knobs(arch, hw, raw: dict):
    """Apply every run-spec SERVING knob to a freshly parsed ``(arch, hw)`` pair.

    Split out of :func:`load_runspec` so that ANY caller which parses a run-spec itself
    (e.g. the hardware-entitlement skill's ``_planner.load_runspec``, which needs a
    different return shape) applies the identical knobs. Two parsers that each pick up
    their own subset silently produce different floors from the SAME run-spec, which
    defeats the point of cross-checking one against the other.

    Mutates ``arch`` and ``hw`` in place, stamps ``raw["_precision"]`` for reporting, and
    returns ``(arch, hw)`` for convenience. Order matters: the precision block reads
    ``arch.w_elem_bytes`` after ``weight_dtype_bytes`` is set, and must capture the
    tensor-engine peak BEFORE scaling it.
    """
    # KV-cache dtype is a serving choice, not a model-config fact, so it lives in the
    # run-spec (kept off the pristine HF config). Optional; absent => model dtype.
    if raw.get("kv_dtype_bytes") is not None:
        arch.kv_dtype_bytes = raw["kv_dtype_bytes"]
    # Serving knob: apply_prefill_dcp (disaggregated-prefill DCP attention sharding).
    # Run-spec field, default False. Absent => standard full-TP attention both regimes.
    arch.apply_prefill_dcp = bool(raw.get("apply_prefill_dcp", False))
    # HBM decomposition contract. The default charges one graph-cut read/write per
    # collective-delimited component. ``per_operator`` explicitly replaces that traffic
    # with every operator interface for eager/decomposed experiments.
    activation_policy = str(raw.get("activation_io_policy", "component_boundary"))
    valid_activation_policies = ("component_boundary", "per_operator")
    if activation_policy not in valid_activation_policies:
        raise ValueError(
            "run-spec 'activation_io_policy' must be one of %r, got %r"
            % (valid_activation_policies, activation_policy))
    arch.activation_io_policy = activation_policy
    # Serving knob: usable HBM fraction (the per-core memory budget the fit / max-batch
    # filters charge against). Run-spec wins over the hardware JSON's value -- it's the
    # serving-time "how tight is my budget" lever. Absent => leave the HW default (0.90).
    if raw.get("usable_hbm_fraction") is not None:
        hw.usable_fraction = float(raw["usable_hbm_fraction"])
    # --- Serving knob: WEIGHT / COMPUTE PRECISION -------------------------------------
    # `weight_dtype_bytes` sets the served weight element size (1 = FP8, 2 = BF16). It
    # changes the weight-HBM term, which is what binds decode.
    # `compute_precision: "fp8"` additionally scales the tensor-engine peak by
    # `fp8_te_peak_multiplier` (default 2.0), which is what binds PREFILL. Modeling only
    # the byte halving and not the FLOP speedup understates an FP8 prefill by ~2x, so the
    # two are set together here rather than by a caller-side fudge factor.
    if raw.get("weight_dtype_bytes") is not None:
        arch.weight_dtype_bytes = float(raw["weight_dtype_bytes"])
    if raw.get("lm_head_dtype_bytes") is not None:
        lm_head_bytes = float(raw["lm_head_dtype_bytes"])
        if arch.tie_embeddings and lm_head_bytes != arch.w_elem_bytes:
            raise ValueError(
                "run-spec 'lm_head_dtype_bytes' cannot differ from weight_dtype_bytes for "
                "a tied embedding/lm_head table")
        arch.lm_head_dtype_bytes = lm_head_bytes
    prec = str(raw.get("compute_precision", "bf16")).lower()
    if prec not in ("bf16", "fp8"):
        raise ValueError(f"run-spec 'compute_precision' must be 'bf16' or 'fp8', got {prec!r}")
    mult = float(raw.get("fp8_te_peak_multiplier", 2.0)) if prec == "fp8" else 1.0
    raw["_precision"] = {"compute_precision": prec, "te_peak_multiplier": mult,
                         "te_peak_flops_bf16": hw.te_peak_flops,
                         "weight_dtype_bytes": arch.w_elem_bytes,
                         "lm_head_dtype_bytes": arch.lm_head_elem_bytes}
    # Scale the peak ONCE here so every downstream consumer (cost.Triple.times, step_time,
    # roofline_flops, the figures) sees a consistent effective peak with no signature churn.
    hw.te_peak_flops = hw.te_peak_flops * mult
    return arch, hw


def load_runspec(json_path: str):
    """Load a run-spec JSON: an HF config path, a hardware path, and the cell list.

    Returns (Arch, HW, configs, raw). Keeps the HF model config pristine (you drop
    in the real config.json unmodified) and separates the trn2 constants + the
    parallelism cells we want to plot.
    """
    with open(json_path) as f:
        raw = json.load(f)
    base = os.path.dirname(os.path.abspath(json_path))
    # os.path.join so an hf_config given as a bare filename (the co-located agent_artifacts/tmp
    # case) OR as an absolute path both resolve; a hand-rolled f"{base}/{...}" mangles an absolute
    # path into base + "/" + abs. The entitlement skill's own loader already does this -- keeping
    # the two identical is what lets one run-spec drive both.
    with open(os.path.join(base, raw["hf_config"])) as f:
        hf = json.load(f)
    arch = arch_from_hf(hf)
    hw = load_hardware(os.path.join(base, raw["hardware"]))
    # Every serving knob lives in apply_runspec_knobs so the hardware-entitlement skill's
    # own run-spec loader can apply the identical set (see that function's docstring).
    apply_runspec_knobs(arch, hw, raw)
    return arch, hw, raw["configs"], raw
