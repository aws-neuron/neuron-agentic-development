# Building the entitlement oracle

The analytic engine (`entitlement_formulas.py`) is only trustworthy if something
independent confirms its FLOP/weight/KV counts. That something is the **oracle**: a
shape-exact `torch` module run on the `meta` device under `entitlement_counter.py`, whose
totals `validate_oracle.py` diffs against the engine. Raw prefill SDPA is compared
separately with the analytic causal active-pair count. This doc explains *why the oracle is
hand-written* (not lifted from the production model), how to build one for a new model, and
the gotchas that bite.

Read this before adding a new model whose architecture differs from the shipped
gemma-4-31B example (different attention, MoE, a custom kernel op, etc.).

## Contents
- [What the oracle is for](#what-the-oracle-is-for)
- [Why not just use the production (vLLM/Neuron) model?](#why-not-just-use-the-production-vllmneuron-model)
- [The kernel-language question](#the-kernel-language-question)
- [How to build an oracle for a new model](#how-to-build-an-oracle-for-a-new-model)
- [Gotchas](#gotchas)

## What the oracle is for

Entitlement is a **theoretical floor** — the FLOPs, weight bytes, and KV bytes a forward
step *must* cost, independent of how well any kernel realizes it. Those quantities are
fixed by **shapes and dtypes**, so they can be read off a weightless module on the `meta`
device with no accelerator and no real compute. The oracle exists so the analytic engine's
closed-form formulas have a ground truth to be checked against; once they agree, the engine
is trusted to sweep the whole `{TP, PP, CP}` × bucket grid (which the oracle is too slow /
too manual to do). For prefill SDPA, the comparison accounts for the raw oracle's dense
rectangle versus the floor's causal active pairs.

The division of labor:
- **`oracle_model.py`** — the shape-exact module, built from an `Arch`.
- **`entitlement_counter.py`** — subclasses `torch.utils.flop_counter.FlopCounterMode`;
  counts per-aten FLOPs + a weight/KV byte walk.
- **`validate_oracle.py`** — runs the oracle under the counter and diffs the three totals
  against `entitlement_formulas.py` (+ entitlement_formulas.py's corrections), classifying
  the raw-SDPA prefill difference separately.

## Why not just use the production (vLLM/Neuron) model?

Tempting — it's "the accurate implementation," so why maintain a second module that can
drift? Because the vendor model measures an **implementation**, and entitlement is
**implementation-independent**. Three concrete blockers, each fatal on its own:

1. **Custom kernels are invisible to the FLOP counter.** `FlopCounterMode` intercepts
   *aten dispatch* (`aten.mm`, `aten.bmm`, `aten._scaled_dot_product_*`, …). A fused
   NKI/CUDA/Triton kernel is a single opaque custom op — the counter sees **zero FLOPs**
   for it. In the Neuron Gemma4 model the MLP goes through a fused `NF.mlp(...)` op and
   decode attention through a `gemma4_attention_decode(...)` NKI kernel; the two most
   FLOP-heavy parts of the model would count as nothing.
2. **It can't be instantiated cheaply.** The vendor attention module calls
   `get_tp_group()` in `__init__` — constructing it needs a live distributed process
   group. `forward` needs a fully-populated paged-KV `attn_metadata` dict per layer, bound
   KV-cache tensors, and real positions. There is no "run on meta with zero setup" path.
3. **It bakes in non-entitlement work.** Redundant V matmul on `k_eq_v` global layers
   (the fused QKV duplicates K into the V slot), TP all-gather / reduce-scatter
   collectives, FP8 quant/dequant, the sampler. Measuring it would report *this backend's*
   cost, which changes when the kernels change — the opposite of a stable floor.

By the time you stub out (2), swap every custom kernel for a torch reference to fix (1),
and strip the redundant/quant work for (3), you have **rebuilt the oracle anyway**, just
more fragilely. So we keep a thin torch oracle and manage the drift risk a different way
(below).

**Managing the drift risk:** the thing that actually drifts is not the arithmetic (shapes
fix that) — it's a **config fact** (a head_dim, the sliding/global layer split, `k_eq_v`,
tied embeddings). So the oracle is built from the **same `Arch`** that `arch_from_hf`
hands the analytic engine and the sweep (`build_oracle(arch)`), not from an independent
re-parse of the HF JSON. One parse of the config → it cannot diverge between the two
sides. `validate_oracle.py` is the backstop that catches anything that slips through.

## The kernel-language question

> What if the target model has an operator implemented only in a kernel language, not in
> torch?

You **still do not port the kernel** into the oracle. An op's entitlement cost is set by
its arithmetic — matmul shapes, softmax over sequence length, bytes moved — not by the
kernel source. Options, in order of preference:

1. **Torch stand-in with the same shapes (default).** Write a plain-torch op with the same
   I/O shapes and the same dominant contraction, and let the counter count *that*. The
   oracle uses `F.scaled_dot_product_attention` as the SDPA shape stand-in. Its raw count is
   validation evidence; the planner and Hardware Entitlement source the SDPA floor from the
   analytic active-pair formula.
2. **Analytic cost term (when the cost isn't torch-expressible).** If the kernel exploits
   sparsity, a novel algorithm with different asymptotic cost, or a custom approximation
   that no cheap torch op mirrors, don't fake it with a torch op of the wrong cost. Add a
   closed-form term to `entitlement_formulas.py` and leave a **zero-cost placeholder** in
   the oracle so shapes still flow through the module.
3. **Real kernel on hardware (different question entirely).** If you need *achieved*
   performance rather than entitlement, or numerical correctness, that is the
   profiler/benchmark and verification path — not the oracle. The oracle is a floor, never
   a measurement.

Writing a behaviourally-accurate kernel replica in torch buys **nothing** for entitlement
and adds a second thing that can drift. Don't.

## How to build an oracle for a new model

> **Read first — two silent-drop failures this step is designed to prevent.** The
> `arch_from_hf` parser has silently dropped large fractions of real models twice, each
> time producing a plausible-looking but wrong entitlement:
> - **MoE (missed entirely):** a 197B / 288-expert model whose config used
>   `moe_num_experts` / `moe_top_k` / `moe_layers_enum` — none of which matched the
>   fields the parser looked for (`n_routed_experts`, `num_experts_per_tok`,
>   `first_k_dense_replace`). The parser fell through to the dense branch and reported a
>   `gate_proj` weight read of 0.062ms (dense-intermediate scale), ~10× too small for a
>   288-expert MoE. Not caught until after reporting.
> - **GDN linear attention (75% of layers dropped):** a 64-layer hybrid model
>   (48 `"linear_attention"` + 16 `"full_attention"`). The `layer_types` branch matched
>   only `"sliding"`/`"full"` by substring, so the 48 linear layers mapped to **zero
>   groups** and the parser produced a 16-layer / ~8.6B model out of 27B. Caught (before
>   reporting) *only* because the parser was re-read and `num_layers` was checked against
>   the known 64 first.
>
> **Two guardrails, always, before trusting a parse of a new model:**
> 1. **`layer_types` (and any enum-like config list) is EXHAUSTIVE, not an allow-list.**
>    After building groups assert `sum(g.count) == num_hidden_layers`; if a distinct
>    `layer_types` value maps to zero groups, that must be an **ERROR** naming the
>    unrecognized strings — never a silent skip.
> 2. **MoE/variant fields detected-but-unparsed must warn.** If the config contains
>    `moe_num_experts` / `use_moe` / `enable_moe_block`-like keys but the parsed
>    `n_routed_experts` is `None`, raise or warn rather than silently modeling dense. Add
>    field-name fallbacks: `n_routed_experts = tc.get("n_routed_experts") or tc.get("moe_num_experts")`.
>
> And after any parser change, **grep for duplicated formulas.** The KV-byte formula lived
> in three files (`entitlement_formulas.py::kv_bytes_total`, `planner/cost.py`,
> `operator_entitlement.py::kv_bytes_one_layer`); fixing one left the others wrong,
> surfacing as a 7× KV inflation / 85.7% reconciliation gap. A group-kind fix must land in
> all copies (better: call one shared function).

1. **Make sure `arch_from_hf` captures the model's shape facts.** It currently handles flat
   or `text_config`-nested configs, mixed sliding/global attention (`layer_types`), MoE
   (`enable_moe_block`), tied embeddings, and `attention_k_eq_v`. It does **not** model GDN
   or other recurrent linear-attention layers. Do not run such a config through the current
   engine. If your model has a fact that changes bytes/FLOPs and isn't parsed (a new
   attention variant, per-layer intermediate sizes, …), **add it to `Arch` / `AttnGroup`
   and `arch_from_hf` first** — that keeps the single-source-of-truth property. Add the
   matching correction to `entitlement_formulas.py` if it's a weight-byte adjustment.
   **Extend the engine; never fake the number by scoping down to the layers the parser
   happened to catch.**
2. **Express it in `oracle_model.py` only if the existing groups don't already cover it.**
   `OracleModel` is built generically from `Arch.attn_groups` + the MoE/dense switch, so
   many models need *no oracle code change at all* — a new run-spec is enough. Add a module
   only for a genuinely new structure.
3. **Run `validate_oracle.py`** with your run-spec
   (`python3.12 validate_oracle.py <model-or-path>`) and read the table:
   - **weight bytes** should match `corrected_weight_bytes` to well under 1% (the residual
     is RMSNorm params the engine omits — negligible at real hidden sizes).
   - **decode FLOPs** should match within a few %. A large gap localizes a missing engine
     term — fix the *engine*, not the oracle.
   - **KV bytes**: the oracle reports 0 (meta SDPA decomposes into bmm+softmax, so the
     counter's fused-attention hook never fires). The analytic `kv_bytes_total` is the
     authority for KV; the oracle ground-truths FLOPs and weight bytes.
   - **prefill raw SDPA** differs from the analytic floor by the documented
     dense-versus-causal-active pair count.
4. **Add structural tests** (see `test_oracle_model.py`): assert the config facts that move
   bytes/FLOPs (layer count, `k_eq_v` → no v_proj on global, tied storage, expert count),
   then the two cross-checks (weight bytes == corrected engine; decode FLOPs ≈ engine).

## Gotchas

- **lm_head over how many tokens?** A real serving stack projects logits only for the
  **sampled** positions (~1 token/sequence), not every prefill token. `OracleModel`
  defaults to `project_all_tokens=False` (last token only), which matches that and keeps
  the oracle comparable to the analytic engine (which omits the lm_head matmul entirely —
  a small, known additive residual). Flip to `True` only if your entitlement definition
  charges a full-vocab projection for every position; expect prefill FLOPs to jump by
  ~`S×` the lm_head cost.
- **Raw SDPA is validation evidence, not the floor source.** The planner and Hardware
  Entitlement use the analytic causal/windowed active-pair count for SDPA. The meta
  oracle's raw SDPA count appears only in `validate_oracle.py`; decode matches tightly,
  while prefill must differ by exactly the classified dense-versus-active pairs.
- **Norm params inflate tiny-arch weight bytes.** The oracle counts RMSNorm weights; the
  engine doesn't. At hidden=64 that's ~0.4% of the total (visible in unit tests with toy
  arches); at a real hidden size it's noise. Don't "fix" it by adding norms to the engine —
  compare with a small relative tolerance instead.
- **`attention_k_eq_v` is per-group, not global.** The oracle aliases V to K only on
  full-context (global) groups, and only when the config sets the flag; `entitlement_formulas.py`'s
  `phantom_global_vproj_bytes` subtracts the matching weight. A model with global layers
  but a real v_proj must **not** set the flag — both sides gate on
  `Arch.attention_k_eq_v`.
- **MoE: hold all experts, route k.** The oracle instantiates all `num_experts` expert
  MLPs (weight bytes) but runs only `experts_per_tok` of them (FLOPs), matching the engine.
  It runs the first k experts on the whole batch — the *cost* is identical to true top-k
  routing, and the counter cares only about matmul shapes.
- **GDN / linear attention is unsupported.** The current `Arch` has no representation for
  its recurrent state or mandatory non-TE work. A config containing `"linear_attention"`
  must not be treated as ordinary GQA attention or used for entitlement until the parser,
  work model, oracle, and reconciliation tests support it.
- **A reconciliation gap > ~1% is a STOP, not a footnote.** `validate_oracle.py` (weight
  ~0%, decode FLOPs within a few %) and the operator reconciliation (weight/KV ~0%,
  compute within the known oracle delta) are the **gate**, not a diagnostic printed
  alongside a result. When `config_entitlement`/`operator_entitlement` prints
  `!! operator reconciliation off by X%`, halt and chase it to zero before reporting — that
  message is exactly what exposed the duplicated-KV-formula bug. Never present a cell with
  an open >1% reconciliation gap.
- **Validate the parse against a known fact before trusting any number.** For a model you
  know the size of, check `num_hidden_layers` (does it equal the config's layer count?) and
  eyeball one weight-read magnitude (e.g. a 288-expert MoE `gate_proj` should be ~10× a
  dense intermediate, not equal to it). A single sanity check on the parse catches the
  silent-drop failures before they reach a report.
