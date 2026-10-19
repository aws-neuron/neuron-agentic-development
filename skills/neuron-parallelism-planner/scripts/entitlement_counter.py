"""Per-operator / per-module FLOP and HBM-byte counting for entitlement analysis.

Wraps ``torch.utils.flop_counter.FlopCounterMode`` so we get its battle-tested
per-aten-op FLOP formulas and module-hierarchy attribution for free, then layers
on the two HBM-traffic terms the theoretical roofline needs (see
``entitlement_formulas.py``):

  - **weight bytes**  -- static; a parameter walk over the module tree. No dispatch
    hook is needed because weight traffic is just ``sum(numel * element_size)`` over
    the parameters, deduped across tied weights.
  - **KV bytes**      -- read off the key/value tensor shapes at each fused
    scaled-dot-product / flash / efficient attention dispatch. During a forward pass
    those tensors *are* the KV cache for that step, so the attention op is exactly
    where the FLOP formula already intercepts them.

Activation HBM traffic is intentionally NOT modelled here: interfaces internal to a
component are assumed resident or fused unless compiled evidence proves an HBM boundary.
The theoretical weights+KV analysis does not depend on it.

Everything runs on shapes + dtypes, so it works on the ``meta`` device with no
accelerator and no real compute:

    import torch
    from entitlement_counter import profile_module

    with torch.device("meta"):
        model = build_my_module(...)
        inputs = make_example_inputs(...)
    report = profile_module(model, args=(inputs,))
    print(report.summary())
    # report.total_flops, report.total_weight_bytes, report.total_kv_bytes,
    # report.arithmetic_intensity feed straight into the roofline.

The caller supplies the module + example inputs; this module has no dependency on
``transformers`` or any model zoo. KV bytes are captured only for the fused
attention ops listed in ``DEFAULT_ATTENTION_OP_NAMES`` -- a stack whose attention
decomposes into plain ``bmm`` + ``softmax`` (no fused op) will report zero KV bytes
unless you extend ``attn_packets``.

**Speculative decoding (optional).** ``speculative_report`` amortizes a target
verify pass + ``gamma`` draft forwards into a per-accepted-token entitlement,
showing the weight-read-bound decode speedup. It is opt-in and REQUIRES the caller
to supply the draft model's per-forward requirements (weight bytes, KV state, and
optionally FLOPs) via ``SpeculativeDraft`` -- either by profiling the draft module
the same way as the target, or from the draft's HF config. Nothing about the draft
is inferred.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.flop_counter import FlopCounterMode

logger = logging.getLogger(__name__)

# Fused attention ops whose key/value arguments give the per-step KV-cache bytes.
# The ``_for_cpu`` variant is what CPU/meta dispatch usually selects and is missing
# from torch's own flop_registry, so we both detect its KV bytes here and (below)
# wire its FLOPs through torch's sdpa formula.
DEFAULT_ATTENTION_OP_NAMES: Tuple[str, ...] = (
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
    "_flash_attention_forward",
    "_efficient_attention_forward",
)


def _build_attention_packets(names: Sequence[str]) -> set:
    """Resolve aten op names to OpOverloadPackets, skipping ones absent in this torch."""
    aten = torch.ops.aten
    packets = set()
    for name in names:
        op = getattr(aten, name, None)
        if op is not None:
            packets.add(op)
    return packets


def _register_cpu_sdpa_flops(flop_registry: Dict[object, object]) -> None:
    """Teach ``flop_registry`` the CPU sdpa op, which torch omits from its own table.

    Mutates ``flop_registry`` in place. We reuse the *already shape-wrapped* sdpa
    formula torch registered for the flash-attention op rather than wrapping the raw
    one ourselves -- wrapping it twice would pass ``out_shape`` to the formula twice.
    """
    op = getattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu", None)
    if op is None or op in flop_registry:
        return
    from torch.utils.flop_counter import flop_registry as torch_flop_registry

    wrapped = torch_flop_registry.get(torch.ops.aten._scaled_dot_product_flash_attention)
    if wrapped is None:
        logger.debug("flash-attention sdpa formula not in torch registry; CPU sdpa FLOPs will be uncounted")
        return
    flop_registry[op] = wrapped


def kv_bytes_from_args(args: Tuple[object, ...]) -> int:
    """KV-cache bytes read by an attention op: key tensor bytes + value tensor bytes.

    Every fused attention op in DEFAULT_ATTENTION_OP_NAMES takes (query, key, value, ...)
    positionally, so key/value are args[1]/args[2]. Non-tensor or absent args contribute
    nothing (e.g. a fake/meta tensor still reports numel and element_size correctly).
    """
    total = 0
    for idx in (1, 2):
        if idx < len(args):
            t = args[idx]
            if isinstance(t, torch.Tensor):
                total += t.numel() * t.element_size()
    return total


def weight_bytes_by_module(model: torch.nn.Module) -> Dict[str, int]:
    """Per-module weight bytes from parameters owned directly by each module.

    Keyed by module fully-qualified name (root module -> ""). Uses
    ``parameters(recurse=False)`` so each parameter is attributed to its owning module
    exactly once; the per-module values therefore sum to the (non-deduped) parameter
    total. Use ``total_weight_bytes`` for the tie-aware total.
    """
    out: Dict[str, int] = {}
    for name, module in model.named_modules():
        nbytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
        if nbytes:
            out[name] = nbytes
    return out


def total_weight_bytes(model: torch.nn.Module) -> int:
    """Total weight bytes, deduping tied parameters (shared embedding / lm_head)."""
    seen: set = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * p.element_size()
    return total


@dataclass
class EntitlementReport:
    """FLOPs + HBM-byte decomposition for one forward pass.

    ``arithmetic_intensity`` is FLOPs / (weight bytes + KV bytes) -- the x-axis of the
    roofline. ``flop_counts`` and ``kv_byte_counts`` are keyed [module_fqn][op_packet]
    (module names follow torch's ModuleTracker); ``weight_bytes`` is keyed by the
    parameter-walk module fqn (root -> ""). Totals are authoritative regardless of how
    the two keying schemes line up.
    """

    total_flops: int
    total_weight_bytes: int
    total_kv_bytes: int
    flop_counts: Dict[str, Dict[object, int]] = field(default_factory=dict)
    kv_byte_counts: Dict[str, Dict[object, int]] = field(default_factory=dict)
    weight_bytes: Dict[str, int] = field(default_factory=dict)

    @property
    def total_hbm_bytes(self) -> int:
        return self.total_weight_bytes + self.total_kv_bytes

    @property
    def arithmetic_intensity(self) -> float:
        denom = self.total_hbm_bytes
        return self.total_flops / denom if denom else 0.0

    def summary(self) -> str:
        return f"FLOPs={self.total_flops:,}  " f"weight_bytes={self.total_weight_bytes:,}  " f"kv_bytes={self.total_kv_bytes:,}  " f"AI={self.arithmetic_intensity:.2f} FLOP/byte"


class EntitlementCounter(FlopCounterMode):
    """``FlopCounterMode`` that also accumulates per-module KV bytes during dispatch.

    FLOPs and the module hierarchy come straight from the base class. We override
    ``_count_flops`` to additionally tally key/value tensor bytes for attention ops,
    using the same ``mod_tracker.parents`` the base uses so the KV keys line up with
    the FLOP keys. Weight bytes are computed from the model on demand in ``report()``.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        *,
        display: bool = False,
        depth: int = 2,
        attn_packets: Optional[set] = None,
        custom_mapping: Optional[Dict[object, object]] = None,
    ) -> None:
        super().__init__(depth=depth, display=display, custom_mapping=custom_mapping or {})
        _register_cpu_sdpa_flops(self.flop_registry)
        self._model = model
        self._attn_packets = attn_packets if attn_packets is not None else _build_attention_packets(DEFAULT_ATTENTION_OP_NAMES)
        self.kv_byte_counts: Dict[str, Dict[object, int]] = defaultdict(lambda: defaultdict(int))

    def __enter__(self) -> "EntitlementCounter":
        self.kv_byte_counts.clear()
        super().__enter__()
        return self

    def _count_flops(self, func_packet, out, args, kwargs):  # type: ignore[no-untyped-def]
        out = super()._count_flops(func_packet, out, args, kwargs)
        if func_packet in self._attn_packets:
            kv = kv_bytes_from_args(args)
            if kv:
                for parent in set(self.mod_tracker.parents):
                    self.kv_byte_counts[parent][func_packet] += kv
        return out

    def total_kv_bytes(self) -> int:
        # "Global" holds every op once (module parents are subsets of it), so summing
        # only "Global" avoids the double counting that summing all modules would cause.
        return sum(self.kv_byte_counts.get("Global", {}).values())

    def report(self) -> EntitlementReport:
        if self._model is None:
            raise ValueError("EntitlementCounter needs a model to compute weight bytes; pass model=... or use profile_module()")
        return EntitlementReport(
            total_flops=self.get_total_flops(),
            total_weight_bytes=total_weight_bytes(self._model),
            total_kv_bytes=self.total_kv_bytes(),
            flop_counts=self.get_flop_counts(),
            kv_byte_counts={k: dict(v) for k, v in self.kv_byte_counts.items()},
            weight_bytes=weight_bytes_by_module(self._model),
        )


def profile_module(
    model: torch.nn.Module,
    args: Tuple[object, ...] = (),
    kwargs: Optional[Dict[str, object]] = None,
    *,
    depth: int = 2,
    attn_packets: Optional[set] = None,
    custom_mapping: Optional[Dict[object, object]] = None,
) -> EntitlementReport:
    """Run one forward pass under an EntitlementCounter and return the report.

    ``model`` and the example ``args`` / ``kwargs`` should live on the same device;
    the ``meta`` device profiles arbitrarily large models with no real compute.
    """
    kwargs = kwargs or {}
    with EntitlementCounter(model, depth=depth, attn_packets=attn_packets, custom_mapping=custom_mapping) as counter:
        model(*args, **kwargs)
    return counter.report()


# ---------------------------------------------------------------------------
# Speculative decoding (OPTIONAL extension)
# ---------------------------------------------------------------------------
#
# Speculative decoding is NOT a sharding axis, so it does not slot into the
# four-component roofline the way TP/PP/CP do. Its knobs are the draft length
# ``gamma`` (tokens proposed per target verify step) and the acceptance rate
# ``alpha`` (mean per-token accept probability). The payoff is the expected
# number of ACCEPTED tokens emitted per target verify pass:
#
#     N(gamma, alpha) = (1 - alpha**(gamma+1)) / (1 - alpha)   (alpha < 1)
#                     = gamma + 1                               (alpha == 1)
#
# bounded by gamma+1 and approaching 1/(1-alpha) as gamma grows. The entitlement
# win is that the target streams its weights ONCE per verify (over gamma+1 tokens)
# yet emits ~N tokens, so the weight-read-bound decode cost drops ~Nx -- at the
# price of (a) gamma sequential draft forwards and (b) the draft's resident weights
# + its own KV cache competing for the same HBM.
#
# This extension is OPT-IN and makes NO assumptions about the draft: the caller
# MUST supply the draft's per-forward requirements (weight bytes, KV-state bytes,
# and optionally FLOPs) -- either by profiling the draft module the same way as the
# target (SpeculativeDraft.from_report) or by passing the numbers directly (e.g.
# weight_bytes = draft_param_count * element_size from the draft's HF config, via
# entitlement_formulas.weights_total on an arch_from_hf(draft_config)).


def expected_accepted_tokens(gamma: int, acceptance_rate: float) -> float:
    """Mean tokens ACCEPTED per target verify step (truncated-geometric model).

    A draft proposes ``gamma`` tokens; the target verifies them in one pass and emits
    up to the first rejection (plus a bonus token if all accept). With mean per-token
    acceptance probability ``alpha`` the expectation is

        N = (1 - alpha**(gamma+1)) / (1 - alpha)   (alpha < 1),   gamma+1 (alpha == 1).

    High alpha (a good draft) is worth far more than high gamma: N is capped at gamma+1
    AND at 1/(1-alpha), so once the draft quality saturates, extra gamma is wasted work.
    """
    if gamma < 1:
        raise ValueError(f"gamma (draft length) must be >= 1, got {gamma}")
    if not 0.0 <= acceptance_rate <= 1.0:
        raise ValueError(f"acceptance_rate (alpha) must be in [0, 1], got {acceptance_rate}")
    if acceptance_rate == 1.0:
        return float(gamma + 1)
    return (1.0 - acceptance_rate ** (gamma + 1)) / (1.0 - acceptance_rate)


@dataclass
class SpeculativeDraft:
    """Per-forward HBM/FLOP requirements of the DRAFT model -- a REQUIRED input.

    The cost model will not guess the draft's size. Obtain these either by profiling
    the draft module (``SpeculativeDraft.from_report``, the same meta-device forward you
    run for the target -- profile it on a SINGLE decode token, since the draft generates
    its ``gamma`` proposals one token at a time) or by passing the numbers directly when
    you only have the draft's config:

      - ``weight_bytes`` -- bytes streamed for ONE draft forward = draft_param_count *
        element_size (read once per proposed token). From an HF config without
        instantiating: ``entitlement_formulas.weights_total(arch_from_hf(cfg)) *
        dtype_bytes`` (plus the tied embedding table if any).
      - ``kv_bytes``     -- draft KV-cache bytes touched in one draft decode step. The
        draft keeps its OWN cache; this is the extra HBM it reads/occupies and is the
        term that competes with the target model's prefix cache on the same replica.
      - ``flops``        -- matmul FLOPs for one draft forward (optional; 0 if unknown.
        Decode is weight/KV-read bound, so the memory terms dominate the entitlement).
    """

    weight_bytes: int
    kv_bytes: int = 0
    flops: int = 0

    @classmethod
    def from_report(cls, report: "EntitlementReport") -> "SpeculativeDraft":
        """Build from an EntitlementReport of ONE draft forward (single decode token)."""
        return cls(
            weight_bytes=report.total_weight_bytes,
            kv_bytes=report.total_kv_bytes,
            flops=report.total_flops,
        )


@dataclass
class SpeculativeReport:
    """Speculative-decoding entitlement, amortized per ACCEPTED output token.

    Built by ``speculative_report`` from a TARGET verify-pass report (profiled over the
    gamma+1 verify tokens) and the draft's per-forward requirements. Per spec iteration
    the hardware streams the target weights once and runs ``gamma`` draft forwards; the
    iteration emits ``expected_accepted`` (N) tokens, so every byte/FLOP total divides by
    N to give the per-output-token figures that are directly comparable to plain decode.

    ``weight_read_speedup_vs_decode`` is the headline: plain autoregressive decode reads
    the full target weights for EACH token (pure memory-bound waste), so the speedup on
    that dominant term is N / (1 + gamma * draft_weight / target_weight). It shrinks as
    the draft grows heavier relative to the target.
    """

    gamma: int
    expected_accepted: float
    # per accepted output token (amortized) -- comparable to a plain-decode per-token report
    weight_bytes_per_token: float
    kv_bytes_per_token: float
    flops_per_token: float
    # per spec iteration (1 target verify + gamma draft forwards), before amortization
    iter_weight_bytes: int
    iter_kv_bytes: int
    iter_flops: int
    # diagnostics
    target_weight_bytes: int
    draft_weight_bytes: int
    weight_read_speedup_vs_decode: float

    @property
    def total_hbm_bytes_per_token(self) -> float:
        return self.weight_bytes_per_token + self.kv_bytes_per_token

    @property
    def arithmetic_intensity(self) -> float:
        denom = self.total_hbm_bytes_per_token
        return self.flops_per_token / denom if denom else 0.0

    @property
    def draft_weight_fraction(self) -> float:
        """gamma * draft weight bytes / target weight bytes -- the spec overhead ratio."""
        return (self.gamma * self.draft_weight_bytes / self.target_weight_bytes) if self.target_weight_bytes else 0.0

    def summary(self) -> str:
        return (
            f"gamma={self.gamma}  N_accepted={self.expected_accepted:.2f}  "
            f"weight_bytes/tok={self.weight_bytes_per_token:,.0f}  "
            f"kv_bytes/tok={self.kv_bytes_per_token:,.0f}  "
            f"AI={self.arithmetic_intensity:.2f} FLOP/byte  "
            f"weight-read speedup vs decode={self.weight_read_speedup_vs_decode:.2f}x"
        )


def speculative_report(
    target: EntitlementReport,
    draft: "SpeculativeDraft | EntitlementReport",
    *,
    gamma: int,
    acceptance_rate: Optional[float] = None,
    expected_accepted: Optional[float] = None,
) -> SpeculativeReport:
    """Amortize a speculative-decoding step into a per-accepted-token entitlement.

    ``target`` -- EntitlementReport of ONE target verify pass, profiled over the
        ``gamma + 1`` verify tokens (its weight bytes are read once per verify; its
        FLOPs/KV scale with the gamma+1 positions).
    ``draft``  -- the draft's per-forward requirements (REQUIRED, never inferred): a
        ``SpeculativeDraft`` or an EntitlementReport of ONE draft decode token (coerced
        via ``SpeculativeDraft.from_report``). The draft runs ``gamma`` times per
        iteration.
    Provide EXACTLY ONE of ``acceptance_rate`` (alpha -> N computed via the
    truncated-geometric model) or ``expected_accepted`` (N measured directly).
    """
    if isinstance(draft, EntitlementReport):
        draft = SpeculativeDraft.from_report(draft)
    elif not isinstance(draft, SpeculativeDraft):
        raise TypeError(
            "draft must be a SpeculativeDraft or EntitlementReport; the draft model's "
            "requirements must be supplied explicitly, not inferred"
        )
    if gamma < 1:
        raise ValueError(f"gamma (draft length) must be >= 1, got {gamma}")
    if (acceptance_rate is None) == (expected_accepted is None):
        raise ValueError("pass exactly one of acceptance_rate or expected_accepted")

    if expected_accepted is None:
        n = expected_accepted_tokens(gamma, acceptance_rate)  # type: ignore[arg-type]
    else:
        if not 0.0 < expected_accepted <= gamma + 1:
            raise ValueError(f"expected_accepted must be in (0, gamma+1]={gamma + 1}, got {expected_accepted}")
        n = float(expected_accepted)

    iter_weight = target.total_weight_bytes + gamma * draft.weight_bytes
    iter_kv = target.total_kv_bytes + gamma * draft.kv_bytes
    iter_flops = target.total_flops + gamma * draft.flops

    # plain decode reads the full target weights per emitted token; spec reads
    # iter_weight per N tokens -> speedup on the dominant weight-read term.
    speedup = (target.total_weight_bytes / (iter_weight / n)) if iter_weight else 0.0

    return SpeculativeReport(
        gamma=gamma,
        expected_accepted=n,
        weight_bytes_per_token=iter_weight / n,
        kv_bytes_per_token=iter_kv / n,
        flops_per_token=iter_flops / n,
        iter_weight_bytes=iter_weight,
        iter_kv_bytes=iter_kv,
        iter_flops=iter_flops,
        target_weight_bytes=target.total_weight_bytes,
        draft_weight_bytes=draft.weight_bytes,
        weight_read_speedup_vs_decode=speedup,
    )
