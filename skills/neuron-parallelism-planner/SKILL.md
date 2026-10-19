---
name: neuron-parallelism-planner
description: Plans tensor/pipeline/context-parallelism (TP/PP/CP) and context-bucket configurations for transformer LLM inference on AWS Neuron (Trainium trn1/trn2/trn3) using an analytic hardware-entitlement model validated against a meta-device FLOP-counter oracle. Use when deciding how to shard a model for serving, choosing parallelism degrees or batch sizes, estimating decode ITL / prefill TTFT / throughput / per-device HBM before running on hardware, reasoning about KV-cache replication or prefix-cache headroom, comparing throughput-capacity vs interactivity serving objectives, or modeling speculative-decoding speedups. Plans SINGLE-NODE, COLOCATED serving only — disaggregated prefill/decode and any multi-node or multi-instance topology are out of scope and are never recommended. Driven by a run-spec (HF config.json + hardware constants); ships gemma-4-31B / gemma-4-26B-A4B run-specs, with granite-4.1-8B on trn2 as the worked walkthrough (samples/).
---

# Neuron Parallelism Planner

## Overview

Turns a model (HuggingFace `config.json`) + hardware spec into a parallelism and
context-bucket recommendation, using an **analytic hardware-entitlement model** — a
theoretical lower bound that picks where to point the first on-hardware sweep. The engine
(`scripts/planner/`, see [references/design.md](references/design.md)) is a three-layer
pipeline: **aggregate** the model into per-(class, layer-type) segments →  **shard** each by
its TP/CP/DP/EP divisor → attach the ordered component-body activation graph cut → **cost** it
as a per-component roofline step time, then **sweep** the config × context-bucket grid and
**select** under a serving objective.

The aggregation is validated against a meta-device FLOP-counter **oracle** (a shape-exact
`torch` module run on the `meta` device — no accelerator). Projection/FFN/router/lm-head
FLOP and weight counts are ground-truthed, while raw prefill SDPA is compared separately
with the analytic causal active-pair count (`validate_oracle.py`).

## When to use

Use this skill to answer, *before touching hardware*:
- What TP/CP/DP (and EP, for MoE) should I serve this model with, and at what batch size?
- What decode ITL, prefill TTFT, throughput, and per-device HBM should I expect?
- Will TP=N replicate the KV cache? How much prefix-cache headroom is left?
- Which single-node colocated config meets my prefill-TTFT and decode-ITL SLAs at once?
- What speedup can speculative decoding give, given a specific draft model?

## Quick start

Run it from your PROJECT directory (where the harness already is) with the skill's `scripts/`
on `PYTHONPATH` — do NOT `cd` into the skill: generated artifacts are written under the project
root, and the scripts refuse to write inside the skill package rather than pollute it.

```bash
SKILL=$NEURON_AGENTIC_DEVELOPMENT_ROOT/skills/neuron-parallelism-planner/scripts
PYTHONPATH=$SKILL python3.12 -m planner granite4-8b           # a model shipped with the install
PYTHONPATH=$SKILL python3.12 -m planner ./my-model-run.json   # or a run-spec you wrote
```

**The project root defaults to the cwd; name it only if the cwd is not your project.** Artifacts
land in `<project root>/agent_artifacts/tmp/`. Pass `--project-root <dir>` (or set
`$NEURON_PROJECT_ROOT`, honored by every skill in this family so one analysis keeps one evidence
trail) when the run-spec lives somewhere other than the project you want the artifacts in — a
test harness driving a shared, maintained run-spec is the case this exists for. The named root
must already exist. There is no separate "write them exactly here" variable: the project root is
the only knob, so to send a run somewhere specific, point the root there.

**The run-spec is required — the skill ships no model.** There is no `configs/` directory and no
default: a planner that silently analyzes some other model when the argument is missing is worse
than one that stops. To get started from something known-good, copy the complete worked example
written out in [samples/granite4-8b-walkthrough.md](samples/granite4-8b-walkthrough.md) (run-spec +
pristine HF config + trn2 constants) into a directory and point at it.

Point it at any run-spec — a path, or the NAME of a model shipped with the install, which
resolves to `$NEURON_AGENTIC_DEVELOPMENT_ROOT/runspecs/<model>-run.json`. No environment
variable names a run-spec: one that did would let a run silently analyze whatever model was
last exported, which looks exactly like a real result. Override the serving mode / SLAs with
`--serving`, `--prefill-itps-floor`, `--decode-itl-ceiling-ms`, and the per-core memory budget
with `--usable-fraction` (default 0.90; lower it to model a tighter "can't fully fit
everything yet" budget).

**Interpreter:** the planner CLI needs only the standard library. `entitlement_counter.py` /
`oracle_model.py` / `validate_oracle.py` and the tests need `torch` (meta device, CPU) +
`pytest`. Use an interpreter that has the relevant deps (developed with `python3.12`).

## Prerequisites

Settle these before planning a new model.

### Objectives & SLAs

Config selection is **gate-then-rank**: SLA ceilings decide which configs are valid, a
throughput metric ranks the survivors. The objective and SLAs are therefore what shape the
recommendation. The parenthesized defaults are fallbacks for when the user has no specific
target.

**Serving mode is fixed: `"colocated"` on a SINGLE NODE.** One replica serves both phases, so
a config is valid only if it meets **both** the prefill-TTFT and decode-ITL ceilings, and it
must fit within one node (`TP * DP <= cores_per_node`). Splitting prefill and decode into
separate pools, and any topology spanning 2+ nodes or instances, are **out of scope**: the
engine raises on `serving` values other than `"colocated"` and on any collective degree
exceeding `cores_per_node`. Do not propose them, and do not reintroduce them as a way to
satisfy an SLA that a single colocated config misses — report the miss instead.

That leaves one decision:

1. **Optimization objective** — which metric the planner selects a config for. Three options,
   described below.

#### The three objectives

**1. Maximize OTPS under SLA ceilings (default).** Select the config with the highest decode
throughput per node among those that meet the latency SLAs. This objective requires the SLA
ceilings as input:
- **Decode ITL ceiling** (`decode_itl_ceiling_ms`) — the maximum acceptable inter-token
  latency, in ms.
- **Prefill TTFT ceiling** — the maximum acceptable time-to-first-token, set via
  `prefill_itps_floor` (an input-tokens/s floor) or `prefill_ttft_ceiling_ms` (a flat ms
  ceiling).

Gate on BOTH ceilings (colocated serves both phases from one replica). Keep
only the configs that stay under the applicable ceiling(s), then pick the one with the highest
OTPS. Both ceilings are required inputs for this objective.

**2. Minimize prefill TTFT.** For a user who cares only about first-token latency. Select the
config with the lowest prefill TTFT; do not consider decode throughput or footprint. Because
prefill is compute-bound and parallelizes efficiently, this favors configs with more cores.

**3. Maximize decode OTPS.** For a user who cares only about decode throughput. Select the
config with the highest per-node OTPS, without applying the ITL ceiling. This favors
large-batch configs, which achieve high throughput but higher inter-token latency.

#### Why the objective matters

Latency and throughput are controlled by the same two settings — batch size and core count —
and they pull those settings in opposite directions:
- **Batch size.** OTPS is highest when the batch fills the available memory, but a larger batch
  raises ITL, because each decode step processes more sequences. Low ITL therefore requires a
  smaller batch than maximum OTPS.
- **Core count / TP.** Low TTFT and low ITL improve with more cores and higher TP. Per-node
  throughput improves with the opposite: smaller footprint per replica, scaled out with data
  parallelism.

Because of this, no single config is best on all three metrics. The default objective maximizes
throughput within the latency limits you specify; the other two optimize a single metric with
no latency constraint. Choosing an objective determines which point in the latency-throughput
tradeoff the planner selects.

### Model & hardware (the run-spec)

Everything is driven by a **run-spec JSON** (a complete one is written out in
[samples/granite4-8b-walkthrough.md](samples/granite4-8b-walkthrough.md)).
To plan a different model/hardware, write a new run-spec and point the CLI at it:

```bash
python3.12 -m planner /path/to/my-model-run.json
```

A run-spec needs:
- `"model"` — display name.
- `"hardware_name"` — short label (e.g. `"trn2"`).
- `"hf_config"` — path (relative to the run-spec) to the model's **pristine** HF
  `config.json`. Drop it in unmodified; `arch_from_hf` parses it (tolerates flat or
  `text_config`-nested, dense or MoE — both the `enable_moe_block` and the
  `n_routed_experts` conventions, with optional always-on shared experts and a
  dense-then-MoE layer split via `first_k_dense_replace` — standard GQA/MHA or
  Multi-head Latent Attention via `kv_lora_rank`, mixed sliding/global attention via
  `layer_types`, tied or untied embeddings, `max_position_embeddings`).
- `"hardware"` — path to a hardware-constants JSON, resolved relative to the run-spec. This
  skill ships trn2's as [references/trn2_constants.json](references/trn2_constants.json);
  point your run-spec at it, or copy it and calibrate (see
  the hardware-entitlement skill's `references/microbench-calibration.md`). Every number the
  planner prints divides by these, so a stale constant is a silently wrong plan:
- `"serving"` — `"colocated"` only (the default). Any other value raises.
- `"decode_itl_ceiling_ms"` — decode inter-token-latency ceiling (50).
- `"prefill_itps_floor"` — prefill TTFT SLA as an input-tokens/s floor (30000). Higher floor
  ⇒ tighter TTFT ceiling (`ttft_ceiling(S) = S / floor`, scaled with context).
- `"prefill_ttft_ceiling_ms"` — alternative way to state the prefill SLA: a flat TTFT ceiling
  in ms applied at every context. Mutually exclusive with `prefill_itps_floor`.

Physical serving config (run-spec fields, all optional; defaults in parentheses). Unlike the
objective/SLA fields above, these change the *system being modeled*, not what selection
optimizes for:
- `"kv_dtype_bytes"` — KV-cache element bytes if different from the weight dtype (e.g. fp8 KV).
- `"weight_dtype_bytes"` — **served WEIGHT element bytes** when they differ from the model dtype
  (1 = FP8, 2 = BF16). Absent => the model dtype. This applies to every weight, including
  a tied embedding/lm_head table, unless an untied lm_head explicitly declares the exception
  below.
  Decode is weight-HBM-bound, so this is the single biggest decode lever for a model whose
  checkpoint is already quantized.
- `"lm_head_dtype_bytes"` — optional served element bytes for a separate, **untied** lm_head
  when it intentionally differs from the other weights. Absent => `"weight_dtype_bytes"`.
  A tied model rejects a conflicting value because embedding and lm_head are the same object.
- `"compute_precision"` — `"bf16"` (default) or `"fp8"`. `"fp8"` scales the tensor-engine peak by
  `"fp8_te_peak_multiplier"` (default 2.0), which is what moves **prefill** (compute-bound).
  Set these two together: halving the weight bytes without also modeling the FLOP speedup
  understates an FP8 prefill by ~2x. Both are modeled INSIDE the engine — never apply a
  precision factor to the engine's output by hand.
  Worked example: copy a run-spec and set `"weight_dtype_bytes": 1` together with
  `"compute_precision": "fp8"` to A/B an FP8 deployment against its BF16 baseline.
- `"apply_prefill_dcp"` — DCP attention sharding on the prefill pass (false).
- `"activation_io_policy"` — component activation-HBM contract. The default,
  `"component_boundary"`, reads one gathered attention/FFN body input and writes one
  pre-collective body output while allowing ideal internal fusion/residency. The
  sequence-parallel residual/norm bridge is outside the current floor. `"per_operator"`
  replaces the body graph cut with every operator input/output interface for a
  decomposed/eager experiment.
- `"usable_hbm_fraction"` — fraction of per-core HBM the planner may fill with weights + KV +
  activations; the rest is runtime/fragmentation headroom (0.90). Lower it to model a **tighter
  memory budget** — e.g. "I can't fully fit everything yet" — without editing code; it shrinks
  every cell's max-feasible batch (and, if low enough, the fitting grid). Also settable as a
  key in the hardware-constants JSON, or overridden per run via the `--usable-fraction` CLI
  flag (CLI > run-spec > hardware JSON > 0.90).

The grid choices (`TP_CHOICES`, `CP_CHOICES`, `DP_CHOICES`, `MAX_CORES`), bucket count
(`N_BUCKETS`), and batch granularity are constants in `scripts/planner/plan.py` — edit those
only to retune the search, not to change the model.

**Validate the engine against the oracle first.** Before trusting a plan for a new model,
confirm the analytic aggregation matches the meta-device oracle on that model's shapes:

```bash
python3.12 validate_oracle.py <model-name-or-run-spec-path>
```

Weight bytes should match `corrected_weight_bytes` to well under 1% and decode FLOPs within
a few %; a large gap localizes a missing term. For prefill, `validate_oracle.py` reports the
raw meta SDPA count separately from the analytic causal active-pair count.

**If the architecture differs from the worked example** (a new attention variant beyond
GQA/MHA/MLA, a routing scheme the parser does not recognize, an op that lives only in a kernel
language), **read [references/oracle-construction.md](references/oracle-construction.md)**.

### Parallelism axes

- **TP** shards each layer; capped at the query-head count. **CP** is carved from TP
  (`TP % CP == 0`, adds no cores) and shards the KV cache / prefill sequence. **DP** replicates
  (dense: free throughput scale-out; MoE: widens expert sharding, `world = TP·DP`).
- **EP** shards routed experts across `world`; swept only for MoE models (a dense model has no
  routed experts, so EP is a no-op).
- **PP is out of scope** — not implemented by the Neuron inference stack; the grid pins PP=1.

## Reading the output

`python3.12 -m planner <run-spec>` prints, per context bucket:
- the **selected config** (colocated: one config meeting both TTFT and ITL SLAs, max per-node
  throughput), with its SLA verdicts, and
- the **full decode landscape** — every fitting config at max-feasible batch (per-node
  throughput, ITL, binding term) — so the pick is visible against everything it was chosen
  from.

For a per-config deep view (per-(component, layer-type) load/flop/time, comparable to a
profile), call `planner.build.cell(...)` and read its `by_type` field, or use the
`--cell` CLI mode (see `samples/granite4-8b-walkthrough.md` for a worked write-up).
The same cell exposes `activation_boundaries` with ordered graph-cut tensors.

## Key concepts (read the references as needed)

- **Two "best configs."** Max-batch cells answer the *throughput-capacity* question (low
  TP, small replica, scale via DP). *Interactivity* (low ITL) lives below max batch and
  wants high TP. Do not conflate them. The `--visualize` **`pareto_per_node` figure is the
  interactivity Pareto**: per-node throughput vs ITL swept *below* max batch (one gray curve
  per config over its batch ladder, the DP-normalized frontier bold and batch-labeled) — the
  concrete artifact for this distinction, since the max-batch landscape alone shows a flat,
  ~uninformative ITL. The batch sweep is also saved to `agent_artifacts/tmp/sweep_results.json`
  under the project root you run from — or the one you name with `--project-root` /
  `$NEURON_PROJECT_ROOT` (the path is printed on every run)
  (`decode_bs_sweep`) and lives on the plan dict as `decode_bs_grid`.
- **KV does not shard past the KV-head count — it replicates.** The TP ceiling is the
  *query*-head count; per-device KV divides by `min(TP, n_kv_heads)`. This drives both
  long-context decode KV-binding and prefill prefix-cache waste. **If planning TP for any
  GQA/MQA model, read [references/tp-feasibility-and-kv-replication.md](references/tp-feasibility-and-kv-replication.md).**
- **Prefill is its own decision** — matmul-bound, ~100% parallel-efficient, PP=1, scale via
  CP; gated by a TTFT SLA on the same colocated replica that serves decode.
- **Speculative decoding** is not a sharding axis — knobs are γ (draft length) and α
  (acceptance rate); the caller must supply the draft model's requirements explicitly.

### References

- **[references/methodology.md](references/methodology.md)** — the full sweep methodology:
  pipeline, sweep space, oracle validation, the four findings, the two serving objectives,
  prefill + prefix-cache, and caveats. Read when you need to understand or defend a
  recommendation.
- **[references/parallelism-cost-model.md](references/parallelism-cost-model.md)** — how
  TP/PP/CP/DP/EP each scale the four per-device cost terms as a function of degree `d`, in
  table form, plus the speculative-decoding cost model. Read when reasoning about a
  specific axis or tradeoff.
- **[references/tp-feasibility-and-kv-replication.md](references/tp-feasibility-and-kv-replication.md)**
  — why the TP ceiling is the query-head count, how KV replication works, and the
  per-group `kv_bytes_per_device` fix. Read before trusting any high-TP number.
- **[references/oracle-construction.md](references/oracle-construction.md)** — how the
  entitlement oracle is built and why it is a hand-written `torch` module rather than the
  production vLLM/Neuron model, what to do when an op lives only in a kernel language, and
  the step-by-step for building + validating an oracle for a NEW model. **Read before
  adding a model whose architecture differs from the gemma-4-31B example.**
- **[references/design.md](references/design.md)** — the design of the
  `scripts/planner/` engine: the three-layer split (aggregate → shard → cost), the segment
  model, per-layer-type / per-class divisors (TP/CP/DP/EP, decode vs prefill-DCP), and the
  per-component roofline-then-serialize cost model. Read before changing how weights / KV /
  FLOPs are counted or sharded, or when adding a new parallelism axis or layer/FFN type.

## Files

The **engine** (`scripts/planner/`) — the aggregate → shard → cost → sweep+select pipeline
(see [references/design.md](references/design.md)):

- `scripts/planner/segments.py` — §1 aggregation: `build_weight/kv/flop_segments(arch)` →
  per-`(class, layer_type)` `Segment` lists.
- `scripts/planner/shard.py` — §2 sharding: `weight_divisor` / `kv_cache_divisor` /
  `flop_divisor` — the single owner of every TP/CP/DP/EP divisor rule.
- `scripts/planner/cost.py` — §3: `per_type_totals` (segments × divisors × operating point →
  dynamic per-(component, variant) records), `apply_activation_records` (split and recompress
  only cost-equivalent component instances), `roll_up_components` for aggregate work
  reporting, and `step_time` (roofline every cost record, serialize, then report one summary
  per component variant without recomputing the roofline).
- `scripts/planner/activation.py` — ordered component DAG, graph-cut boundary tensors,
  component/per-operator activation policies, and operator-interface shape authority shared
  with Hardware Entitlement.
- `scripts/planner/plan.py` — §4 sweep + selection: `build_grid`, `max_feasible_bs`,
  `select_buckets`, `max_cores` (the single-node core cap), and the colocated
  gate-then-rank selection.
- `scripts/planner/build.py` — run-spec wiring: `build_bundle` → `(Arch, HW, segments, meta,
  serving, SLAs)`; `cell(...)` for a single (config, bucket, bs, regime) entitlement (its
  `by_type` field is the per-layer-type breakdown).
- `scripts/planner/__main__.py` — the CLI (`python3.12 -m planner <run-spec>`).
- `scripts/planner/test_planner.py` — regression + unit tests.
- `scripts/entitlement_formulas.py` — the model/hardware layer the engine is built on: the
  parsed model (`Arch`/`AttnGroup`/`MLAParams`, `arch_from_hf`, `load_runspec`), `HW`, the
  accounting primitives the segments consume, and the topology-aware `comm_total_ms`.

**Validation** — the meta-device oracle the engine's counts are ground-truthed against:

- `scripts/oracle_model.py` — the shape-exact `torch` oracle, built straight from the same
  `Arch` (so config facts can't drift between the two sides).
- `scripts/entitlement_counter.py` — the FLOP/weight/KV-byte counter (subclasses
  `torch.utils.flop_counter`) + the speculative-decoding cost model.
  `scripts/test_entitlement_counter.py` — its tests.
- `scripts/validate_oracle.py` — diffs the engine's analytic totals against the oracle; run
  it when adding a new model.

**Inputs & examples:**

- **No `configs/` directory.** The skill ships no run-specs: a run-spec names YOUR model and
  hardware, and the code never defaults to one. A complete worked example (granite-4.1-8B on trn2)
  is written out in `samples/granite4-8b-walkthrough.md`; the run-specs the package's tests drive
  live in the repo's `tests/runspecs/` (granite, gemma-4-31B, gemma-4-26B-A4B, Devstral-2-123B —
  dense/tied, mixed-attention, MoE, and untied+fp8 respectively).
- `samples/granite4-8b-walkthrough.md` — a full worked walkthrough (granite-4.1-8B on trn2):
  architecture derivation + enablement, oracle validation of the engine, and the sweep/select
  step. **If a user asks for a planning report on a model, follow this as the template.**
