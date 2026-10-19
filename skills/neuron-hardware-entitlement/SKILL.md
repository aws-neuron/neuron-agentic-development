---
name: neuron-hardware-entitlement
description: Computes per-operator mandatory compute/HBM work and component-boundary activation HBM for a single transformer-LLM inference parallelism config on AWS Neuron (Trainium trn1/trn2/trn3), and attributes a deployed graph's actual resident HBM back to those operator floors. Use when localizing which operator or component binds on tensor-engine versus HBM work within a chosen {TP, PP, CP, bucket, bs, prefill} cell, auditing ideal component fusion, routing optimization effort to regions with real headroom, or diffing actual resident HBM against its floor. Builds on neuron-parallelism-planner (imports its analytic engine + meta-device oracle); driven by a run-spec (HF config.json + hardware constants), ships a worked example.
---

# Neuron Hardware Entitlement

## Overview

This skill computes the **per-operator hardware-entitlement floor** for one parallelism
config: for every matmul-class operator a model owns, the theoretical minimum per-device
time it MUST pay, split into a **compute floor** (op FLOPs ÷ tensor-engine peak) and a
**operator HBM floor** ((weight + KV bytes) ÷ HBM bandwidth), and the binding constraint.
The model floor then adds one external body input read and one body output write exactly
once; interfaces internal to the body remain ideally fused or resident. The default is
this component graph cut. A run-spec may select
`"activation_io_policy": "per_operator"` to replace it with every operator interface for an
explicit decomposed/eager experiment.

The certified component-engine scope is **TE and HBM**, emitted as
`"engine_scope": ["te", "hbm"]` in every operator report, config report, and config cell.
VE and SC are unmodeled in this floor and excluded from its binding maximum. They may still
contribute to, or bind, measured execution; exclusion does not mean zero work. The result is
therefore a scoped floor: valid, but potentially looser, because adding another certified
mandatory engine term can only keep or raise the lower bound.

It is the **operator-level companion** to the model-level entitlement in
`neuron-parallelism-planner`. Where the planner answers *which config to
deploy*, this skill answers *which operator binds on what within a chosen config*. The two
are the same bound at different resolutions: summing the per-operator floors reproduces the
planner's TE, weight, and KV totals, while the planner-owned component records reconcile the
activation graph cut. This proves both views use the same engine rather than parallel
re-derivations.

This skill **reuses the planner's engine and oracle** — it imports them via `_planner` and
never copies them — so a single config parse and a single analytic model serve both skills
and cannot drift.

The operator report takes attention-projection, FFN/router, and lm-head FLOPs and parameter
shapes from the oracle. Its SDPA row uses the planner's analytic causal/windowed formula,
and activation interfaces use the sharding formulas owned by `planner.shard`.

## Component definition

For this skill, an attention or FFN **component** is its compute body:

- A collective is a **boundary**, not part of a component. A component never contains an
  all-reduce, reduce-scatter, all-gather, or other collective.
- Under sequence-parallel prefill, a body begins after its input all-gather and ends before
  its output reduce-scatter. Its activation boundary is one gathered hidden input and one
  pre-reduce-scatter output.
- The residual/norm interval between reduce-scatter and all-gather is an SP bridge. It is
  outside the current certified component floor and can be added as a serialized component
  when its HBM/VE/SC contract is established.
- In the current tensor-parallel transformer graph, this produces ordered attention and FFN
  component instances, followed by the terminal `lm_head` component. Different attention or
  FFN variants remain different component instances.
- Repeated instances may be compressed into one cost-equivalent reporting record. This does
  not merge their positions in the execution graph or create overlap between them.

All operators and mandatory TE/HBM work inside the interval belong to that component.
Internal operator interfaces may remain ideally fused or resident. Only unique tensors
crossing the component boundary contribute component activation HBM. Collective latency is
modeled separately and added after the component floors.

## Why the model floor serializes components

Operator rows are the evidence used to construct a component floor; their individual
latencies are not added. For each dynamic component record `c`, the engine assumes ideal
fusion/residency within that component and computes:

```text
component_floor(c) = max(TE_work(c) / TE_peak,
                         mandatory_HBM_bytes(c) / HBM_bandwidth)
model_floor         = sum(component_floor(c)) + collective_time
```

The sum across components follows the current execution contract. Tensor-parallel attention
and FFN intervals end at an all-reduce or reduce-scatter whose result is consumed by the next
interval; the final lm_head similarly depends on the completed hidden state. These collective
and data-dependency boundaries put the component records in sequence, so a single model-wide
`max(TE, HBM)` would assume cross-boundary overlap that the graph does not permit.

The same boundary is useful for optimization: attention variants, FFN variants, and the
embedding/lm_head path are independently replaceable or tunable units. Reporting a floor at
that granularity shows the minimum latency available to the exact region an optimization can
change, while still exposing its operator-level work. If compiled evidence later proves
fusion or overlap across one of these boundaries, the component decomposition must change
rather than silently keeping the serialized policy.

## Three capabilities

1. **Per-operator floor** (`scripts/operator_entitlement.py`) — one `{TP, PP, CP, bucket,
   bs, prefill}` cell → per-operator compute / weight-read / KV-read floors, activation
   interface metadata, and MUST-binding, rolled up per (operator, attention-group) for one
   device. It also emits serialized component floors with their unique boundary
   reads/writes.
2. **Single-config entitlement** (`scripts/config_entitlement.py`) — one `{TP, PP, CP}`
   choice over a whole bucket-regime → per-cell engine decomposition
   (weight_load / kv_load / activation_load / matmul, plus additive comm), TE/HBM binding
   constraint, decode-tps or prefill-ms bound, per-device peak HBM, and reconciliation.
3. **Actual-vs-floor HBM attribution** (`scripts/hbm_attribution.py`) — a deployed graph's
   per-device resident manifest → each resident tensor attributed to an operator class by
   its shape signature, diffed against the floor → a ranked overage table answering "which
   operators use more HBM than their floor, and by how much."

## Quick start (the shipped worked example)

The scripts default to the parallelism-planner's shipped run-spec (a mixed-attention dense
model on trn2), so they run with no arguments from `scripts/`:

```bash
SKILL=$NEURON_AGENTIC_DEVELOPMENT_ROOT/skills/neuron-hardware-entitlement/scripts   # run from your PROJECT dir
python3.12 $SKILL/operator_entitlement.py --tp 8 --bucket 8192 --bs 8     # decode floor
python3.12 $SKILL/operator_entitlement.py --tp 32 --cp 4 --bucket 131072 --bs 1 --prefill
python3.12 $SKILL/config_entitlement.py                                   # single-config, all cells
python3.12 $SKILL/hbm_attribution.py --entry-hlo <manifest>.hlo --floor-json agent_artifacts/tmp/operator_entitlement_decode.json --runspec <run-spec>.json   # the path each script prints
```

**The project root defaults to the cwd; name it only if the cwd is not your project.** Every
script writes to `<project root>/agent_artifacts/tmp/` and `hbm_attribution.py` reads the floor
JSON back from there, so one analysis keeps one evidence trail. Pass `--project-root <dir>` (or
set `$NEURON_PROJECT_ROOT`, the same variable every skill in this family honors) when the
run-spec lives somewhere other than the project you want the artifacts in — a test harness
driving a shared, maintained run-spec is the case this exists for. The named root must already
exist. There is no separate "write them exactly here" variable: the project root is the only
knob, so to send a run somewhere specific, point the root there.

**Interpreter:** `operator_entitlement.py` and `config_entitlement.py` run the meta-device
oracle, so they need an interpreter with `torch` (CPU/meta is enough) — use `python3.12`.
`hbm_attribution.py` only parses text + the floor JSON, so it needs only the standard library.

`operator_entitlement.py` and `config_entitlement.py` confirm reconciliation on every run:
weight-read and KV-read match the planner cell to ~0%, and compute matches within the known
oracle-vs-analytic delta (a few %). A larger gap means the floor and the planner cell
disagree — investigate before trusting the breakdown.

## Analyzing a NEW model

Everything is driven by the **same run-spec JSON the planner uses** (HF `config.json` +
hardware constants). To analyze a different model/hardware, **do not edit the scripts** —
point them at a run-spec:

```bash
# --runspec takes a path, or the NAME of a model shipped with the install (which resolves to
# $NEURON_AGENTIC_DEVELOPMENT_ROOT/runspecs/<model>-run.json). No variable names a run-spec:
# a floor computed for whatever model was last exported looks exactly like a real one.
python3.12 $SKILL/operator_entitlement.py --runspec granite4-8b --tp 8 --bucket 8192 --bs 8
python3.12 $SKILL/config_entitlement.py --runspec /path/to/my-model-run.json
```

If that run-spec is read in place from outside your project (a shared, maintained copy), add
`--project-root <your project dir>` so the floors still land in your evidence trail.

The run-spec format, the `arch_from_hf` parser (flat or `text_config`-nested, MoE, mixed
sliding/global attention, tied embeddings), and the hardware-constants file are all owned
and documented by `neuron-parallelism-planner`. This skill locates that skill's
`scripts/` directory as `$NEURON_AGENTIC_DEVELOPMENT_ROOT/skills/neuron-parallelism-planner/scripts` (the install root is derived from this
skill's own location when the variable is unset) and imports its engine. **Validate the planner's engine against its oracle for the new model
first** (the planner's `validate_oracle.py`) — if the model-level counts don't reconcile,
the operator-level floors won't either.

If the model has an architecture the planner's `Arch`/oracle does not yet cover (a new
attention variant, an op that lives only in a kernel language), extend the oracle in the
planner skill — see its `references/oracle-construction.md`. This skill inherits the fix
for free because it imports that oracle.

**Architectures the engine currently covers:** GQA and MLA softmax attention, mixed
sliding/global attention (`layer_types`), MoE FFN (`enable_moe_block`), tied embeddings,
and `attention_k_eq_v`. GDN / recurrent linear attention is unsupported. If your model's
structure is **not** in the supported list, it needs an engine extension before the numbers
mean anything — do not scope down to the parts that happen to parse.

**Before trusting any entitlement for a new model (learned from two silent-drop
failures — see the planner's `oracle-construction.md`):**
1. **Read `arch_from_hf` and check the model's config field names against what it parses.**
   MoE configs using `moe_num_experts`/`moe_top_k` (vs `n_routed_experts`/`num_experts_per_tok`)
   have fallen through to the dense branch and been modeled as dense.
2. **Assert `sum(group.count) == num_hidden_layers`.** An unrecognized `layer_types` value
   maps to zero groups and silently drops those layers (a 64-layer hybrid was once parsed
   as 16). Validate the layer count against the known model before proceeding.
3. **Sanity-check one magnitude.** A 288-expert MoE `moe_gate_proj` weight read must be ~10×
   the dense-intermediate scale; if it looks dense-sized, the MoE structure was missed.
4. **Treat any reconciliation gap > ~1% as a STOP.** Chase it to zero before reporting —
   duplicated KV/weight formulas across `entitlement_formulas.py` / `planner/cost.py` /
   `operator_entitlement.py` have caused large gaps that a warning-line (not a halt) let
   slip through.
5. **Run ALL scenarios in the run-spec**, not just the first that produces output —
   declaring success after one clean run has hidden per-scenario errors.

## Reading the output

- **Decode is memory-bound; prefill is compute-bound.** Small-batch decode amortizes each
  weight read over few tokens, so projections land at low arithmetic intensity (below the
  ridge point = tensor-engine peak ÷ HBM bandwidth) and bind on memory; `sdpa` binds lowest
  of all at operator resolution because it has no weights to amortize — its operator HBM
  term is pure KV. Component records additionally include their body input/output graph cut.
  Whole-prompt prefill pushes ordinary projection arithmetic intensity into the
  hundreds–thousands, so those ops bind on compute. Weight-only and zero-work structural
  exceptions remain visible rather than being made memory-bound by activation interfaces.
- **The binding is the routing signal.** A memory-bound op cannot be helped by a faster
  matmul kernel — the lever is fewer/larger HBM reads, lower-precision weights, or more
  batch. A compute-bound op wants tiling/occupancy work. An op already at its floor
  (actual ÷ floor ≈ 1.0) warrants no campaign.
- **Read `engine_scope` before interpreting latency.** The current component maximum contains
  TE and mandatory HBM only. VE/SC work may increase the eventual certified floor; the
  current artifact makes no binding claim for those engines.
- **Row names tell you which FFN path you are reading.** A layer's feed-forward is the sum of
  two parallel paths and either may be absent: the always-on dense MLP carries the bare op
  names (`gate_proj` / `up_proj` / `down_proj`, at `dense_intermediate`, including any shared
  experts folded in by `arch_from_hf`), and the routed pool carries `moe_`-prefixed names at
  `moe_intermediate`. A model with both (gemma4-A4B) shows both sets; a dense model shows only
  the bare ones. A MoE layer also shows `moe_router` — the gate. It is the one row that is
  REPLICATED rather than sharded, so its per-device weight is flat in TP while every other row
  falls: ~0.4% of routed-FFN weight at TP=8 but ~1.5% at TP=32, and linear in expert count.
- **Actual-vs-floor weights match apples-to-apples; KV does not.** Resident weight bytes are
  config-independent and diff cleanly against the floor. The resident KV figure is the
  paged-pool reservation (governed by the memory-utilization policy, config-independent
  within a run), while the floor KV is the analytic per-request KV for the cell's bucket/bs —
  a sizing/policy gap, not a per-operator inefficiency. The attribution flags every KV row.

## Key concepts (read the references as needed)

- **What a floor is, why FLOPs/weights come from the oracle while KV/attention corrections
  and activation-interface shapes are analytic, and how reconciliation proves the
  decomposition** —
  [references/operator-floors.md](references/operator-floors.md). Read to understand or
  defend any per-operator number.
- **Capturing a deployed graph's resident manifest and attributing it to operator floors** —
  [references/actual-vs-floor-hbm.md](references/actual-vs-floor-hbm.md). Read before running
  `hbm_attribution.py` or interpreting an overage table — it covers where the resident bytes
  actually live (ENTRY-input constants, not allocation dumps) and the fused-vs-split fold.
- **Calibrating the hardware constants by microbenchmarking** —
  [references/microbench-calibration.md](references/microbench-calibration.md). Read when a
  floor looks wrong and the suspect is a hardware constant (peak FLOP/s, HBM bandwidth,
  dispatch overhead) rather than the model shapes.
- **The shipped worked example, end to end** —
  [references/worked-example.md](references/worked-example.md). A concrete walk-through of
  the decode and prefill breakdowns and the actual-vs-floor table for one model on trn2.

## Scripts

- `scripts/operator_entitlement.py` — per-operator floor engine. One cell → per-instance
  rows (one per layer×op) + a by-(op, group) rollup for one device, each with compute /
  weight-read / KV-read floors, activation-interface metadata, arithmetic intensity over
  the selected row policy, and MUST-binding. Reconciles against the planner's
  `cell_entitlement` and exposes planner-owned component activation boundaries.
- `scripts/config_entitlement.py` — end-to-end single-config entitlement over a whole
  bucket-regime: each cell's engine decomposition + binding + bound + peak HBM, with the
  operator reconciliation attached.
- `scripts/hbm_attribution.py` — parses a deployed graph's ENTRY-signature resident manifest,
  attributes each tensor to an operator class by its (model- and TP-derived) shape signature,
  and diffs actual vs floor into a ranked overage table.
- `scripts/_planner.py` — locates the `neuron-parallelism-planner` skill's
  `scripts/` directory and puts its engine + oracle on the path; loads the run-spec. The
  single seam between the two skills.
