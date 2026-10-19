"""Shape-exact torch ORACLE for entitlement counting, driven by the run-spec ``Arch``.

This is the ground-truth side of the entitlement analysis. It is a parameter-shape-exact
``nn.Module` that runs on the ``meta`` device (no weights, no compute) so the
``EntitlementCounter`` (``entitlement_counter.py``) can read off the real per-forward
FLOP / weight-byte / KV-byte totals via ``torch.utils.flop_counter``. Those totals are
what ``validate_oracle.py`` diffs the analytic ``entitlement_formulas.py`` engine against
before we trust the engine to sweep configs.

WHY A HAND-WRITTEN TORCH ORACLE (and not the production vLLM/Neuron model)?
  The vendor model measures an *implementation*; the oracle measures *entitlement* (the
  theoretical floor, implementation-independent). Three reasons the vendor model cannot
  serve as the oracle (see references/oracle-construction.md for the long form):
    1. Custom kernels are invisible to ``torch.utils.flop_counter`` -- it intercepts ATEN
       dispatch (``aten.mm``, ``aten._scaled_dot_product_*``...). A fused NKI/CUDA kernel
       is one opaque op the counter sees as ZERO FLOPs. The single most FLOP-heavy parts
       (attention, fused MLP) would not be counted.
    2. The vendor model needs a live distributed process group, paged-KV metadata, and
       bound cache tensors just to run -- there is no cheap meta-device path.
    3. It bakes in implementation choices that are not entitlement (redundant matmuls, TP
       collectives, quant/dequant, sampler), so it would measure THIS backend, not the
       floor.

THE KERNEL-LANGUAGE QUESTION. If the target model has an operator implemented only in a
kernel language (NKI/CUDA/Triton), you still do NOT port the kernel here. The op's cost is
fixed by its arithmetic (matmul shapes, softmax over seq length, bytes moved), not by the
kernel source. So either (a) write a plain-torch stand-in with the same I/O shapes and the
same dominant contraction so the counter sees the same cost (what this oracle does -- it
replaces every attention kernel with one ``F.scaled_dot_product_attention``), or (b) if the
cost is genuinely not torch-expressible (novel sparsity/approximation), add a closed-form
cost TERM to ``entitlement_formulas.py`` and leave a zero-cost placeholder here so shapes
still flow. A behaviourally-accurate kernel replica buys nothing for entitlement and just
adds a second thing that can drift.

DELIBERATELY FAITHFUL TO THE THINGS THAT MOVE BYTES/FLOPS, and ignores the rest:
  - tied embedding / lm_head: ONE table (vocab x hidden), counted once for weight bytes.
  - heterogeneous layers: each ``AttnGroup`` (e.g. sliding vs global) gets its own
    head_dim / KV-head count / sliding-window KV cap, exactly as ``arch_from_hf`` parsed.
  - ``attention_k_eq_v`` on full-context groups: V aliases K, so there is NO v_proj there.
  - sliding-window KV cap: sliding groups cache only ``min(S, window)`` tokens.
  - MoE: holds all ``num_experts`` expert MLPs (weight bytes) but routes only
    ``experts_per_tok`` per token (FLOPs), matching the analytic engine.

INTENTIONALLY OMITTED because they move ~0 of the three counted quantities: the exact
per-layer RMSNorm count (2 vs Gemma's 4), per-layer learned scalars, RoPE, softcap,
biases. Adding them would not change FLOPs/weight-bytes/KV-bytes measurably but would make
the oracle model-specific; the entitlement floor does not depend on them.

THE ONE GOTCHA TO DECIDE PER MODEL -- lm_head over how many tokens? A real serving stack
projects logits only for the SAMPLED positions (~1 token/sequence), not every prefill
token. ``project_all_tokens=False`` (default) matches that and keeps the oracle comparable
to the analytic engine (which omits the lm_head matmul entirely -- the residual is a small,
known additive term). Set ``project_all_tokens=True`` only if your entitlement definition
charges a full-vocab projection for every position.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from entitlement_formulas import Arch, AttnGroup


class _OracleAttention(nn.Module):
    """Standard GQA/MHA attention block. SDPA stands in for any attention kernel."""

    def __init__(self, a: Arch, g: AttnGroup) -> None:
        super().__init__()
        self.hd = g.head_dim
        self.n_q = g.n_q_heads
        self.n_kv = g.n_kv_heads
        # V aliases K on full-context (global) groups when the checkpoint sets k_eq_v.
        self.k_eq_v = a.attention_k_eq_v and g.kv_seq_cap is None
        self.kv_cap = g.kv_seq_cap

        self.q_proj = nn.Linear(a.hidden_size, self.n_q * self.hd, bias=False)
        self.k_proj = nn.Linear(a.hidden_size, self.n_kv * self.hd, bias=False)
        if not self.k_eq_v:
            self.v_proj = nn.Linear(a.hidden_size, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_q * self.hd, a.hidden_size, bias=False)
        # QK-norm: owns params (head_dim each), ~0 matmul FLOPs.
        self.q_norm = nn.RMSNorm(self.hd, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.hd, eps=1e-6)

    def forward(self, hidden: torch.Tensor, S: int, prefill: bool) -> torch.Tensor:
        bs, q_len, _ = hidden.shape
        eff_ctx = S if self.kv_cap is None else min(S, self.kv_cap)

        q = self.q_proj(hidden).view(bs, q_len, self.n_q, self.hd).transpose(1, 2)
        k_new = self.k_proj(hidden).view(bs, q_len, self.n_kv, self.hd).transpose(1, 2)
        q = self.q_norm(q)
        k_new = self.k_norm(k_new)

        # Project the NEW token(s)' V (aliased to K on k_eq_v layers). This ALWAYS runs -- a
        # real decode step projects the incoming token through v_proj too; only the PRIOR
        # cache is not recomputed. (The earlier version faked the whole cache with
        # torch.empty and so skipped v_proj on decode, under-counting non-aliased layers.)
        v_new = k_new if self.k_eq_v else self.v_proj(hidden).view(bs, q_len, self.n_kv, self.hd).transpose(1, 2)

        if prefill:
            k, v = k_new, v_new
        else:
            # decode: the new token(s) attend over the whole KV cache (length eff_ctx). The
            # prior (eff_ctx - q_len) cached positions are not recomputed -- an empty stand-in
            # (their contents don't affect the FLOP/byte count) -- with the projected new
            # token(s) appended, so the score length is eff_ctx and v_proj is charged.
            prior = torch.empty(bs, self.n_kv, eff_ctx - q_len, self.hd, dtype=hidden.dtype, device=hidden.device)
            k = torch.cat([prior, k_new], dim=2)
            v = k if self.k_eq_v else torch.cat([prior, v_new], dim=2)

        attn = F.scaled_dot_product_attention(q, k, v, enable_gqa=(self.n_kv != self.n_q))
        attn = attn.transpose(1, 2).reshape(bs, q_len, self.n_q * self.hd)
        return self.o_proj(attn)


class _OracleMLA(nn.Module):
    """Multi-head Latent Attention block.

    Built in the UNABSORBED form -- q_a/q_b, kv_a/kv_b, o -- so the FLOP counter sees
    the same projection matmuls ``weights_attn`` charges. Whether a kernel absorbs
    kv_b into q is an implementation choice; the entitlement floor counts the algebra.
    Per-head q/k carry qk_head_dim, v carries v_head_dim (SDPA allows a v dim != qk
    dim, output takes v_head_dim), matching ``flops_per_token``'s MLA score term.
    """

    def __init__(self, a: Arch, g: AttnGroup) -> None:
        super().__init__()
        m = g.mla
        self.m = m
        self.n_q = g.n_q_heads
        self.kv_cap = g.kv_seq_cap

        self.q_a_proj = nn.Linear(a.hidden_size, m.q_lora_rank, bias=False)
        self.q_a_norm = nn.RMSNorm(m.q_lora_rank, eps=1e-6)
        self.q_b_proj = nn.Linear(m.q_lora_rank, self.n_q * m.qk_head_dim, bias=False)
        self.kv_a_proj = nn.Linear(a.hidden_size, m.kv_cache_dim, bias=False)
        self.kv_a_norm = nn.RMSNorm(m.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(m.kv_lora_rank, self.n_q * (m.qk_nope_head_dim + m.v_head_dim), bias=False)
        self.o_proj = nn.Linear(self.n_q * m.v_head_dim, a.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, S: int, prefill: bool) -> torch.Tensor:
        bs, q_len, _ = hidden.shape
        m = self.m
        eff_ctx = S if self.kv_cap is None else min(S, self.kv_cap)

        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(hidden)))
        q = q.view(bs, q_len, self.n_q, m.qk_head_dim).transpose(1, 2)

        kv_a = self.kv_a_proj(hidden)
        compressed = self.kv_a_norm(kv_a[..., :m.kv_lora_rank])
        kv_b = self.kv_b_proj(compressed).view(bs, q_len, self.n_q, m.qk_nope_head_dim + m.v_head_dim)
        k_nope = kv_b[..., :m.qk_nope_head_dim]
        v_new = kv_b[..., m.qk_nope_head_dim:].transpose(1, 2)
        # k carries qk_head_dim (nope + rope); the rope part is a per-head broadcast of
        # k_pe -- shape-exact via a zero-cost expand so q/k score dims match.
        k_pe = kv_a[..., m.kv_lora_rank:].unsqueeze(2).expand(bs, q_len, self.n_q, m.qk_rope_head_dim)
        k_new = torch.cat([k_nope, k_pe], dim=-1).transpose(1, 2)

        if prefill:
            k, v = k_new, v_new
        else:
            k = torch.empty(bs, self.n_q, eff_ctx, m.qk_head_dim, dtype=hidden.dtype, device=hidden.device)
            v = torch.empty(bs, self.n_q, eff_ctx, m.v_head_dim, dtype=hidden.dtype, device=hidden.device)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(bs, q_len, self.n_q * m.v_head_dim)
        return self.o_proj(attn)


class _OracleMLP(nn.Module):
    """Dense SwiGLU/GeGLU MLP (gate, up, down). num_experts==1 path."""

    def __init__(self, a: Arch, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(a.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(a.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, a.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class _OracleMoE(nn.Module):
    """Routed-expert pool: holds all E routed experts (weight bytes) but computes only
    experts_per_tok (FLOPs), at moe_intermediate.

    Matches the analytic engine, which counts E routed experts of weight
    (``weights_total``) but routes ``experts_per_tok`` per token (``flops_per_token``).
    We run the first k routed experts on the full token batch -- the COST is identical to
    true top-k routing (k expert MLPs worth of matmul), and the counter cares only about
    the matmul shapes, not which tokens land on which expert. Shared experts are NOT here:
    they are folded into the layer's dense MLP (see _OracleLayer / arch_from_hf).
    """

    def __init__(self, a: Arch, g: AttnGroup) -> None:
        super().__init__()
        inter = g.moe_inter(a)
        n_exp = g.num_experts(a)
        self.k = g.experts_per_tok(a)
        self.gate = nn.Linear(a.hidden_size, n_exp, bias=False)  # router
        self.experts = nn.ModuleList([_OracleMLP(a, inter) for _ in range(n_exp)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.gate(x)  # router matmul (tiny), counted for completeness
        out = None
        for i in range(self.k):
            out = self.experts[i](x) if out is None else out + self.experts[i](x)
        return out


class _OracleLayer(nn.Module):
    """One transformer layer. The FFN is the SUM of two parallel paths, either of which
    may be absent: an always-on dense MLP (dense_inter > 0, incl. folded shared experts)
    and a routed-expert pool (num_experts > 0). gemma4-A4B has both; pure-dense and
    pure-MoE have one."""

    def __init__(self, a: Arch, g: AttnGroup) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(a.hidden_size, eps=1e-6)
        self.post_attn_norm = nn.RMSNorm(a.hidden_size, eps=1e-6)
        self.attn = _OracleMLA(a, g) if g.mla is not None else _OracleAttention(a, g)
        self.dense = _OracleMLP(a, g.dense_inter(a)) if g.dense_inter(a) > 0 else None
        self.moe = _OracleMoE(a, g) if g.num_experts(a) > 0 else None

    def forward(self, h: torch.Tensor, S: int, prefill: bool) -> torch.Tensor:
        h = h + self.attn(self.input_norm(h), S, prefill)
        x = self.post_attn_norm(h)
        ffn = None
        if self.dense is not None:
            ffn = self.dense(x)
        if self.moe is not None:
            ffn = self.moe(x) if ffn is None else ffn + self.moe(x)
        return h + ffn


class OracleModel(nn.Module):
    """Shape-exact transformer for entitlement counting, built straight from an ``Arch``.

    Building from ``Arch`` (the same object ``arch_from_hf`` hands the analytic engine and
    the sweep) is deliberate: a config fact can never silently diverge between the oracle
    and the engine it validates, because there is only one parse of the config.
    """

    def __init__(self, a: Arch, project_all_tokens: bool = False) -> None:
        super().__init__()
        self.a = a
        self.project_all_tokens = project_all_tokens
        self.embed = nn.Embedding(a.vocab_size, a.hidden_size)

        # Preserve the exact hybrid-layer schedule. ``attn_groups`` is a compressed
        # accounting representation; iterating it group-by-group would turn an interleaved
        # five-sliding/one-global model into one sliding block followed by one global block.
        layers: List[_OracleLayer] = [
            _OracleLayer(a, g) for _gi, g in a.ordered_layer_groups()
        ]
        self.layers = nn.ModuleList(layers)

        self.final_norm = nn.RMSNorm(a.hidden_size, eps=1e-6)
        # The logit projection is a NAMED SUBMODULE when untied, and absent when tied (the
        # forward reuses the embedding table). Every weight this model owns must live under a
        # named module: `weight_bytes_by_module` keys bytes by owning-module fqn, so a
        # Parameter assigned directly here would be attributed to the ROOT module -- fqn "" --
        # which no consumer can identify. That is not hypothetical: an earlier version held
        # `lm_head_weight` as a root Parameter, and validate_oracle silently dropped it (its
        # bucketing matches `layers.N.*` / `embed` / `lm_head`), reporting HALF the lm_head
        # weight bytes on every untied model. Tied models hid the bug, because the root alias
        # was dropped and `embed` alone happened to be the right answer.
        self.lm_head = None if a.tie_embeddings else nn.Linear(
            a.hidden_size, a.vocab_size, bias=False)

    @property
    def lm_head_weight(self) -> torch.Tensor:
        """The logit-projection weight: the embedding table when tied, else lm_head's own.

        A property, NOT a registered Parameter -- see the note in ``__init__``. Tied models
        therefore own exactly one vocab x hidden table and untied models own two, which is what
        makes the per-module byte map reconcile with ``embed_table_bytes``.
        """
        return self.embed.weight if self.lm_head is None else self.lm_head.weight

    def forward(self, input_ids: torch.Tensor, S: int, prefill: bool) -> torch.Tensor:
        h = self.embed(input_ids) * math.sqrt(self.a.hidden_size)
        for layer in self.layers:
            h = layer(h, S, prefill)
        h = self.final_norm(h)
        # lm_head projects only the SAMPLED positions (a real serving stack index_selects the
        # sampling positions before the logit projection): in PREFILL only the last position
        # per sequence is sampled (project 1), in DECODE every query position is sampled
        # (project all q_len -- 1 today, >1 for MTP). project_all_tokens forces the full prompt
        # (e.g. scoring) if a caller's entitlement definition wants it.
        if not self.project_all_tokens and prefill:
            h = h[:, -1:, :]
        return F.linear(h, self.lm_head_weight)


def build_oracle(a: Arch, project_all_tokens: bool = False) -> OracleModel:
    """Construct the meta-device oracle from an ``Arch`` (no weights, no compute)."""
    with torch.device("meta"):
        dtype = torch.float32 if a.dtype_bytes == 4 else torch.bfloat16
        return OracleModel(a, project_all_tokens=project_all_tokens).to(dtype)


def make_inputs(bs: int, q_len: int) -> torch.Tensor:
    """Token-id inputs on the meta device. q_len = S for prefill, 1 for a decode step."""
    with torch.device("meta"):
        return torch.zeros(bs, q_len, dtype=torch.long)
