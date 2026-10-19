# Operator floors: definition, data sources, and reconciliation

## Contents

- [What an operator floor is](#what-an-operator-floor-is)
- [Oracle and analytic data sources](#oracle-and-analytic-data-sources)
- [Per-device sharding](#per-device-sharding)
- [Rollup and the MUST-binding](#rollup-and-the-must-binding)
- [Why component floors serialize](#why-component-floors-serialize)
- [Reconciliation against the deployed bound](#reconciliation-against-the-deployed-bound)
- [Caveats](#caveats)

## What an operator floor is

For one operator the engine computes, **per device, for the given config**:

- **certified engine scope** = `["te", "hbm"]`. This field appears at the operator-report,
  config-report, and config-cell levels. An engine outside this scope is unmodeled by this
  floor and excluded from its binding maximum; it may still bind measured execution and is
  not zero work.
- **compute floor** = op FLOPs ÷ tensor-engine peak. The minimum time the matmul takes if
  the tensor engine runs at its rated bf16 throughput.
- **operator HBM floor** = (weight bytes + KV bytes) ÷ HBM bandwidth by default.
  - **weight bytes** — the op's parameters, streamed once. Zero for attention score/context
    (`sdpa`), which owns no weights. Every row uses the served weight precision; a separate,
    untied lm_head may explicitly declare `lm_head_dtype_bytes`.
  - **KV bytes** — K + V cache operands. Only `sdpa` reads these; every other op is a *true*
    zero, not an unmeasured one.
- **activation interface bytes** = input plus output activation volume, per token × device
  tokens. Under the default `component_boundary` policy these rows are diagnostic: interfaces
  internal to a component are ideally fused or resident. The component record instead charges
  each unique graph-cut input read and output write once. Under the explicit `per_operator`
  policy, the row interfaces replace the component graph cut and enter the row HBM term.
- **arithmetic intensity** = FLOPs ÷ mandatory HBM bytes. The bf16 **ridge point** is
  `tensor-engine peak ÷ HBM bandwidth` (FLOP/byte): an op below it is bandwidth-bound, above
  it is compute-bound.
- **MUST-binding** = `compute` if the compute floor ≥ mandatory HBM floor, else `memory`.
  Communication is a per-collective / per-layer property, not a per-op one — it stays in the
  model-level cell's `comm` term.

The floor is a **lower bound**: real per-op time is higher (dispatch, sync, small-tile
inefficiency). It is the minimum permitted by the modeled mandatory work, not a latency
prediction. The gap between a measured kernel and its floor is the optimization headroom.

## Oracle and analytic data sources

The meta-device **oracle** (a shape-exact `torch` module run on the `meta` device, driven
through a FLOP counter — owned by `neuron-parallelism-planner`) is exact for the
things the op graph encodes: **matmul shapes and learned-parameter sizes**. Hardware
Entitlement reads attention-projection, FFN/router, and lm-head FLOPs and weight bytes from
it.

The oracle is blind to two floor corrections, both sourced **analytically**:

1. **KV-cache bytes.** On `meta`, scaled-dot-product attention decomposes to `bmm` +
   `softmax`; no fused-attention op fires, so the counter's KV hook never triggers and
   reports zero. The per-group analytic KV formula (shared with the planner's
   `kv_bytes_total`) is the authority instead.
2. **Causal/windowed SDPA.** The raw meta prefill count covers a dense score/value rectangle.
   The planner and the operator `sdpa` row instead charge causal active pairs and apply the
   sliding-window cap. Decode still uses one query over the effective cache length.

Activation input/output shapes are derived analytically from operator orientation,
attention shape, FFN path width, and the row/width divisors owned by `planner.shard`. The
ordered component DAG separately derives one gathered body input and one pre-collective
body output after TP/CP/DP/EP placement. This prevents internal q/k/v/MLP edges from being
charged repeatedly. The sequence-parallel residual/norm bridge is outside this floor.

Same split, same reasons, as the model-level oracle validation in the planner skill.

## Per-device sharding

Sharding is applied per-device for the given `{TP, PP, CP}`:

- **TP** shards projection/MLP compute and weights 1/TP. It shards KV only down to
  `min(TP, n_kv_heads)` heads, then **replicates** (KV heads cannot split past the KV-head
  count). Column-parallel ops (q/k/v/gate/up/lm_head) read the full input and write a 1/TP
  shard; row-parallel ops (o/down) read a 1/TP shard and write the full hidden.
- **CP** shards prefill compute and the KV sequence 1/CP. In decode only one token per
  sequence is produced per step, so CP does not divide projection FLOPs.
- **PP** holds `num_layers/PP` of each layer type; the by-type rollup scales the instance
  count accordingly.

Change the config → rerun.

## Rollup and the MUST-binding

Per-instance rows (one per layer × op) roll up into per-(op, attention-group) totals for
**one device**. Attention groups are labeled by kind — `full` (uncapped KV) or `sliding`
(windowed KV) — with an index appended only when two groups share a kind, so rollup keys
never collide. The lm_head sits in its own `head` group.

The rollup sorts by the larger of (compute floor, mandatory HBM floor) so the binding operators
surface first.

## Why component floors serialize

The per-operator rows describe mandatory work. They are not independent kernel latencies to
sum.

An attention or FFN **component** is the compute body between its gathered input and
pre-collective output. A collective is excluded from the body: no component contains an
all-reduce, reduce-scatter, all-gather, or other collective. Under sequence-parallel
prefill, the residual/norm interval from reduce-scatter to all-gather is a separate SP
bridge and is outside the current certified floor. The graph therefore produces ordered
attention and FFN body instances followed by the terminal `lm_head` component.

The engine groups each interval's TE work and mandatory HBM bytes and applies one ideal
roofline to that component:

```text
component_floor(c) = max(TE_work(c) / TE_peak,
                         mandatory_HBM_bytes(c) / HBM_bandwidth)
model_floor         = sum(component_floor(c)) + collective_time
```

For the default component policy:

```text
mandatory_HBM_bytes(c) =
    weight_bytes(c) + KV_or_state_bytes(c)
    + unique_component_input_bytes(c) + unique_component_output_bytes(c)
```

The graph cut is built on ordered instances first. Every attention or FFN body reads one
gathered hidden tensor and writes one pre-collective partial. Instances with equal model
variant and byte geometry are compressed; compression is a reporting optimization and does
not change their ordered execution positions.

This policy uses two properties of the component boundary:

1. **It is an execution boundary.** In the current tensor-parallel graph, attention and FFN
   intervals end at an all-reduce or reduce-scatter. The following interval consumes that
   collective's result, so the intervals lie sequentially on the critical path. The lm_head
   likewise consumes the completed final hidden state. A model-wide engine maximum would
   incorrectly overlap work across these dependencies.
2. **It is an optimization boundary.** An attention variant, FFN implementation, or
   embedding/lm_head path can be replaced and tuned independently. A component floor
   therefore measures the minimum cost of the same region an optimizer can change, while
   the operator rows retain the work breakdown needed to explain that cost.

Within one component, ideal fusion and TE/HBM overlap are allowed; across the current
collective-delimited components, latency is serialized. Compiled evidence of cross-component
fusion or overlap would require a different decomposition rather than a different arithmetic
rollup of the same records.

## Reconciliation against the deployed bound

Summing the per-operator work over all operators a device owns reproduces the planner cell's
TE, weight, and KV totals:

- **weight-read** vs cell `weight_load` and **KV-read** vs cell `kv_load` match **exactly**
  (0%): the oracle's weight bytes are the same numbers the planner's corrected-weight term
  uses, and the analytic KV formula is shared.
- **compute** vs cell `matmul` matches within a small **known oracle-vs-analytic delta** (a
  few %): the oracle counts the attention `bmm`s and the lm_head logit projection exactly,
  which the analytic `matmul` term approximates.

So the breakdown is **exact when the deployed cell is memory-bound, and within the
compute-counting delta when it is compute-bound** — a faithful decomposition of the same
entitlement the planner selects against. Every run prints these residuals; a residual above
a few percent means the floor and the planner cell disagree (a missing engine term, a model
the oracle doesn't yet cover) and the breakdown should not be trusted until it closes.

Activation reconciliation depends on the selected policy. `per_operator` sums row interfaces
exactly. `component_boundary` reconciles through the planner-owned component records because a
sum of operator interfaces is intentionally not the component graph cut.

## Caveats

- **Lower bound, not a prediction.** Measured time must be at least the scoped mandatory-work
  floor and may be higher because of implementation overhead or unmodeled engines.
- **The current floor is scoped to TE/HBM.** VE and SC do not bind it yet. Use the emitted
  `engine_scope` rather than inferring coverage from fields elsewhere in the artifact.
- **MUST-binding is compute-vs-memory only.** Communication is excluded per-op (a
  collective/layer property); the planner cell's `comm` term is the authority for it.
- **Precision re-ranks bindings.** The floors above assume the model's stored dtype. Lower
  precision (e.g. FP8 weights) halves every op's weight-read floor and can flip decode
  bindings — recompute under the target precision.
- **At high TP the achieved time is bound by collective WALL-TIME, not the per-op floor.**
  The per-operator compute/memory floor deliberately excludes communication, but on a
  collective-heavy graph that excluded term becomes the dominant reality. When reconciling
  a deployed bound at high TP, expect a large gap between the operator-floor rollup and the
  measured device time, and attribute it to **collectives**, not to operator inefficiency,
  before concluding a kernel is slow. Validated on `generic-encoder-only-model` (8B,
  prefill-only embedding, trn2, TP=16): per-op compute/memory floors accounted for the TE
  work, but live system-trace analysis showed **collectives were 43% of the device
  execution window (7.8ms of 18ms), running ~99.9% serialized with compute** (the
  sequence-parallel `all_gather → matmul → reduce_scatter` dependency admits no overlap on
  the current trn2 path). To localize this gap, compare the planner's `comm_ms` with
  collective wall time from the system trace (`cc_running`) before attributing it to
  operator inefficiency.
- **Whether a kernel can approach the compute floor depends on whether fusion activates.**
  Reachability of the fused matmul/MLP path is gated by
  tiling constraints at the chosen TP. Example (vllm-neuron NKI MLP, `functional/mlp.py`):
  the fused kernel needs `ceil(I_per_rank / 512) ≤ 8` and (for I-sharding) `H ≥ 7168`. For
  the generic-encoder-only-model 8B (H=4096, I=14336): TP=1/2/4/8 fail the tiling gate and fall back to the
  slower PyTorch path, while **TP=16 (I_per_rank=896 → ceil=2) activates the fused kernel**.
  The mathematical floor is unchanged; kernel reachability determines how closely the
  implementation can approach it.
