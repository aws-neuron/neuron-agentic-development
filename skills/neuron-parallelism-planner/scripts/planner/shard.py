"""Section 2 -- sharding: the single owner of all parallelism divisors.

`divisor(cls, layer_type, cfg, regime)` answers "how many ranks share one copy of this
(class, type) piece?" -- per-device = aggregate / divisor. This is the ONLY layer that knows
TP/CP/DP/EP. It encodes, in one place (so the three §3 consumers cannot drift):

  - attention Q/O: full TP (decode / standard prefill) or TP/CP (apply_prefill_dcp prefill).
  - attention K/V weight: min(head_tp, n_kv) -- the replication ceiling.
  - dense MLP: full TP (framework-prescribed; EP re-groups only the routed experts -- §2.2).
  - routed experts: world = TP*DP  (= ep_degree * ep_tp), the only EP-sharded class.
  - embed / lm_head: full TP.
  - component activations: body inputs/outputs use the row layout between their input
    all-gather and output reduce-scatter.
  - CP (cp_g): shards the KV cache (decode) / attention sequence-compute (prefill-DCP),
    per attention-type -- capped types (sliding window) -> 1 (our convention, §2.1).

PP is deferred (§2 scope); ShardConfig has no PP field. n_kv per attention-type is supplied
by the caller from the segment's arch group (kept out of this module so it stays arch-free).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShardConfig:
    TP: int
    CP: int = 1
    DP: int = 1
    ep_enabled: bool = False
    ep_degree: Optional[int] = None   # expert-dim split; None => full EP (= world)

    @property
    def world(self) -> int:
        return self.TP * self.DP

    def ep_grid(self) -> tuple[int, int]:
        """(ep_degree, ep_tp) with ep_degree*ep_tp == world. Full EP => (world, 1)."""
        w = self.world
        ep_deg = w if self.ep_degree is None else int(self.ep_degree)
        ep_tp = max(w // max(ep_deg, 1), 1)
        return ep_deg, ep_tp


# regime is "decode" | "prefill" | "prefill_dcp" (the DI apply_prefill_dcp path)
Regime = str


def head_tp(cfg: ShardConfig, regime: Regime) -> int:
    """The head-shard-axis size for attention weights: TP normally, TP/CP under prefill-DCP."""
    return max(cfg.TP // cfg.CP, 1) if regime == "prefill_dcp" else cfg.TP


def head_divisor(cfg: ShardConfig, regime: Regime, n_heads: int) -> int:
    """Ranks sharing a head-partitioned width, capped when heads replicate."""
    return max(min(head_tp(cfg, regime), max(int(n_heads), 1)), 1)


def expert_tp(cfg: ShardConfig) -> int:
    """Tensor-parallel degree inside one local routed expert."""
    return cfg.ep_grid()[1] if cfg.ep_enabled else cfg.TP


def activation_width_divisor(
        axis: str, cfg: ShardConfig, regime: Regime, n_heads: int = 0) -> int:
    """Divisor for one activation width axis.

    Activation ownership is expressed semantically by the caller; all TP/CP/EP
    arithmetic remains here:

      - ``hidden`` / ``replicated``: full hidden/latent width on every rank;
      - ``heads``: a query/key/value head width, capped at ``n_heads``;
      - ``tensor_parallel`` / ``vocab``: the ordinary TP axis;
      - ``expert_tensor_parallel``: the TP axis within a local routed expert.
    """
    if axis in ("hidden", "replicated"):
        return 1
    if axis == "heads":
        if n_heads <= 0:
            raise ValueError("heads activation axis requires n_heads > 0")
        return head_divisor(cfg, regime, n_heads)
    if axis in ("tensor_parallel", "vocab"):
        return cfg.TP
    if axis == "expert_tensor_parallel":
        return expert_tp(cfg)
    raise KeyError("unknown activation sharding axis %r" % axis)


def activation_row_divisor(layout: str, cfg: ShardConfig, regime: Regime) -> int:
    """Divisor for the row axis of a component-boundary activation.

    The component graph declares the tensor's semantic layout; this function owns
    the TP/CP arithmetic:

      - ``attention_body``: the gathered attention input and pre-reduce-scatter
        output. Standard prefill has all rows; prefill-DCP keeps the rank's CP
        query slice; decode has the local batch.
      - ``ffn_body``: the gathered FFN input and pre-reduce-scatter output, with
        all rows on every rank.
      - ``sampled``: selected rows gathered for the final lm_head.

    The sequence-parallel residual/norm bridge between reduce-scatter and
    all-gather is outside the body-only component floor.
    """
    if layout == "attention_body":
        return cfg.CP if regime == "prefill_dcp" else 1
    if layout in ("ffn_body", "sampled"):
        return 1
    raise KeyError("unknown activation row layout %r" % layout)


def routed_expert_activation_multiplier(cfg: ShardConfig, active_experts: int) -> float:
    """Average active expert invocations resident on one rank for one token."""
    if not cfg.ep_enabled:
        return float(active_experts)
    ep_degree, _ep_tp = cfg.ep_grid()
    return float(active_experts) / ep_degree


def cp_g(cfg: ShardConfig, regime: Regime, cap: Optional[int], n_kv: int, is_mla: bool) -> int:
    """Effective CP for an attention-type's sequence (KV cache / prefill token compute).

      - capped (sliding-window) type -> 1 (constant cache; our convention, §2.1).
      - standard prefill -> 1 (whole-prompt KV at full TP, cache not CP-sharded).
      - decode / prefill-DCP -> min(CP, TP // n_kv), the replication CP reclaims (MLA: n_kv=1).
    """
    if cap is not None:
        return 1
    if regime == "prefill":
        return 1
    nkv = 1 if is_mla else max(n_kv, 1)
    return min(cfg.CP, max(cfg.TP // nkv, 1))


def weight_divisor(cls: str, cfg: ShardConfig, regime: Regime,
                   n_kv: int = 0, is_mla: bool = False) -> int:
    """Per-device WEIGHT divisor for a class (§2.1/§2.2)."""
    if cls == "attn_qo":
        return head_tp(cfg, regime)
    if cls == "attn_kv":
        nkv = 1 if is_mla else max(n_kv, 1)
        return head_divisor(cfg, regime, nkv)
    if cls == "dense_mlp":
        return cfg.TP                       # framework-prescribed full TP (§2.2)
    if cls == "routed_experts":
        return cfg.world if cfg.ep_enabled else cfg.TP
    if cls == "router":
        # REPLICATED on every rank, not sharded -- VERIFIED against the Neuron stack, not
        # assumed. NeuronxDistributed modules/moe/routing.py builds the gate as a plain
        #     self.linear_router = torch.nn.Linear(hidden_size, num_experts, ...)
        # (not a Column/RowParallelLinear, though that module imports parallel_layers for other
        # purposes), and get_router_logits() returns all `num_experts` logits per rank with no
        # expert-dim all-gather and no hidden-dim all-reduce -- either of which a sharded gate
        # would require before top-k. The Mantle/Pumice kernel path agrees: router_weights is
        # [H, E_global] while the expert MLPs are sliced to E_local under EP.
        # Consequence: the router does NOT shrink with TP, so its share of per-device FFN
        # weight GROWS with TP (~1.5% of routed-FFN weight at TP=32 on gemma-4-26B-A4B) and is
        # linear in num_experts.
        return 1
    if cls == "embed":
        return cfg.TP
    raise KeyError(f"unknown weight class {cls!r}")


def kv_cache_divisor(cfg: ShardConfig, regime: Regime,
                     cap: Optional[int], n_kv: int, is_mla: bool) -> int:
    """Per-device KV-CACHE divisor: head divisor x sequence (CP) divisor (§2.1).

    head divisor = min(head_tp, n_kv); sequence divisor = cp_g. In prefill-DCP head_tp=TP/CP;
    in decode/standard-prefill head_tp=TP.
    """
    nkv = 1 if is_mla else max(n_kv, 1)
    head_div = head_divisor(cfg, regime, nkv)
    return head_div * cp_g(cfg, regime, cap, n_kv, is_mla)


def flop_divisor(cls: str, cfg: ShardConfig, regime: Regime,
                 cap: Optional[int] = None, n_kv: int = 0, is_mla: bool = False) -> int:
    """Per-device FLOP divisor (§2.3): the class's weight divisor, with the DCP
    query-row split applied to attention work. attn_score borrows attn_qo's divisor.

    Attention runs on the rank's DCP query slice. FFN and router bodies first
    all-gather the full sequence over TP, so their FLOPs use only their weight
    divisors.
    """
    wclass = "attn_qo" if cls == "attn_score" else cls
    base = weight_divisor(wclass, cfg, regime, n_kv=n_kv, is_mla=is_mla)
    if regime in ("prefill", "prefill_dcp") and cls in (
            "attn_qo", "attn_kv", "attn_score"):
        base *= cp_g(cfg, regime, cap, n_kv, is_mla)
    return base
