# Entitlement-sweep methodology

How the planner turns a model + hardware spec into parallelism/bucket recommendations.
This is a **theoretical lower-bound (entitlement)** analysis: it picks where to point the
first on-hardware sweep — it is not a measurement. The worked numbers below are
Granite-4.1-8B (dense) on trn2 LNC2 (158 TFLOP/s bf16, 716 GB/s HBM,
24 GiB / logical NeuronCore), written out in
`samples/granite4-8b-walkthrough.md`.

## Contents
- [Pipeline](#pipeline)
- [Sweep space](#sweep-space)
- [Oracle validation](#oracle-validation-of-the-analytic-engine)
- [The four findings](#the-findings-that-decide-config-selection)
- [Two "best configs"](#two-different-best-configs--pick-by-serving-objective)
- [Prefill and the prefix-cache lever](#prefill-bs1--a-separate-footprint-driven-decision)
- [Caveats](#caveats-theoretical-bound)

## Pipeline

```
run-spec JSON ──► planner/segments ──► planner/shard ──► planner/cost ──► planner/plan
(HF config +      (per-(class,type)    (divisor per      (roofline step   (grid×bucket
 hw constants)     segments)            key)              time)            sweep + select)
```

The pipeline is the `planner/` package (see [design.md](design.md)): aggregate the model into
per-(class, layer-type) segments, shard each by its TP/CP/DP/EP divisor, cost it as a
per-component roofline using weight, KV/state, and unique component-boundary activation
HBM, then sweep the config × bucket grid at each cell's **max-feasible batch** and select
under the serving objective. `python3.12 -m planner
<run-spec>` runs it end to end. Everything is driven by the run-spec; change the model and the
grid, buckets, and recommendation all re-derive.

> This doc gives the methodology and physics (the four findings, the two serving objectives,
> prefill + prefix-cache). Some historical worked counts below (e.g. the PP-enabled 44-config
> grid) predate the current engine, where **PP is out of scope** and the grid is TP/CP/DP; the
> reasoning is unchanged. The engine's structure and selection are specified in
> [design.md](design.md).

## Sweep space

- **Parallelism grid.** TP ∈ {4,6,8,12,16,24,32,48}, CP ∈ {1,2,4,8} (carved from TP), and
  DP is swept for MoE (EP widens with `world = TP·DP`). The engine rejects dimensions that
  are not evenly shardable, configs whose per-rank weights exceed usable HBM, and any
  `TP·DP` footprint above the single-node core count. The **functional-minimum footprint**
  for gemma-4-31B is 3 cores (61.4 GB weights / 21.6 GB usable = 2.84 → ≥3-way weight
  sharding just to load).
- **PP is out of scope** (not supported by the Neuron inference stack); the grid pins PP=1.
- **Buckets — config-derived, not hand-picked.** Candidates are every 4k multiple from 8k
  up to the model's `max_position_embeddings` (read from the config). `select_buckets`
  keeps N=6 by best **ITL–otps tradeoff**: the full context is pinned as the last bucket,
  the rest log-spread, each snapped to the 4k-multiple whose best config has the strongest
  local per-node-throughput / ITL ratio. gemma-4-31B (256k context) → 8k, 16k, 32k, 72k,
  180k, 256k.
- **Batch — max-feasible per cell.** The largest batch whose peak HBM fits, solved in
  closed form (peak HBM is affine in bs: `weights + per_token·bs`) and snapped down to a
  multiple of 2 (`BS_GRANULARITY`, keeps batch even for TP sharding). Every cell is thus a
  **memory-saturated, throughput-capacity** operating point. Consequence: ITL is uniformly
  high — these cells answer "how much throughput can this footprint hold," not "what's the
  lowest latency." Interactivity-optimal points sit *below* max batch.

## Oracle validation of the analytic engine

The analytic engine (`entitlement_formulas.py`) is validated against a **meta-device
FLOP-counter oracle**: a shape-exact `nn.Module` (`oracle_model.py`) run on the `meta`
device under `entitlement_counter.py` (which subclasses `torch.utils.flop_counter`), with
no accelerator. `validate_oracle.py` diffs the three totals; run it before trusting a sweep
for a new model. The oracle is built straight from the same `Arch` the engine uses, so a
config fact (head_dim, layer split, `k_eq_v`, tied embeddings) cannot silently diverge
between the two sides — the only thing that *can* drift is arithmetic, which shapes pin.

**Why the oracle is a hand-written torch module and not the production vLLM/Neuron model**
(custom kernels are invisible to the aten-level FLOP counter; the vendor model needs a live
process group + paged-KV metadata to run; it bakes in non-entitlement work), **what to do
when an op lives only in a kernel language**, and **how to build/validate an oracle for a
new model**, are all in
[oracle-construction.md](oracle-construction.md). For gemma-4-31B:

| Metric | Decode | Verdict |
|---|---|---|
| FLOPs | analytic within +0.6% of oracle | trustworthy |
| Weight bytes | analytic −4.4% | **corrected** (below) |
| KV bytes | oracle reports 0 (meta SDPA decomposes, no fused-op hook) | analytic is authority |

**Two oracle-driven weight corrections** (`entitlement_formulas.py::corrected_weight_bytes`), Gemma-specific
but harmless elsewhere:
- **(+) tied embedding / lm_head table** — counted once (vocab × hidden). The analytic
  engine omitted it; for gemma-4-31B it is 2.82 GB bf16 and dominates the memory verdict
  (it streams every decode step as the logit projection).
- **(−) phantom global v_proj** — the engine counts a V-projection on `attention_k_eq_v`
  global layers, but V aliases K there (no v_proj in the checkpoint). Zero for models
  without k_eq_v layers.

Corrected total weight: **61.4 GB** (matches the oracle to the byte).

**lm_head projection — an entitlement-definition choice.** A real serving stack projects
logits only for the *sampled* positions (~1 token/sequence), not every prefill token. The
oracle defaults to that (last token only), which keeps it comparable to the analytic engine
(whose `flops_per_token` omits the lm_head matmul entirely — a small, known additive
residual). The oracle can charge a full-vocab projection per position
(`project_all_tokens=True`) if your entitlement definition wants it, but the sweep does not.
This is the one place to decide what "prefill FLOPs" means for your model.

## The findings that decide config selection

**(a) Memory is the capacity dial, never a fitting wall.** With batch filled to the HBM
ceiling, peak HBM clusters tight against the budget by construction (18.3–21.6 GB of 21.6
usable for gemma-4-31B). The functional minimum to merely hold weights is 3 cores. So a
model like this needs no memory-headroom kernel campaign and no fallback config — memory
headroom is what gets *spent on batch*.

**(b) The binding constraint shifts matmul → KV-read as context grows** (at max batch):
short context is matmul(TE)-bound (wide decode matmul rides TE peak); long context goes
KV-read-bound (the per-device replication-aware KV cache dominates HBM reads). Weight-read
binds only the smallest footprints (re-streaming all weights/token when batch can't
amortize it).

**(c) Communication can become *a* bottleneck at large TP** (not *the* bottleneck — the comm
terms are serialized between components and added, not overlapped). The comm model is
topology-aware (`comm_total_ms`): TP AllReduce + CP K/V ring + PP P2P, with the trn2 rack
modeled as a torus so rings map natively to mesh axes. Since a ring AllReduce pays a per-hop
latency over `2(TP−1)` hops, comm grows with TP and can dominate at high TP with small batches
(where the per-hop latency floor, not bandwidth, sets the cost). The EP term — the increment
from widening the MoE layers' FFN AllReduce to the expert grid — is dense-zero for
gemma-4-31B, and zero for any MoE model whose expert grid is just its TP group.

**(d) TP is capped at the query-head count** (32 for gemma-4-31B), and KV heads replicate
rather than shard past their count — which is why high-TP goes KV-bound early. See
[tp-feasibility-and-kv-replication.md](tp-feasibility-and-kv-replication.md).

## Two different "best configs" — pick by serving objective

At max-feasible batch every cell is throughput-saturated, so the sweep answers the
**throughput-capacity** question. **Interactivity (low ITL) is a different regime that lives
below max batch.** The two objectives are distinct and must not be conflated.

### Throughput-capacity (max batch, DP-normalized per node)

Footprint isn't a cost if you can fill a node with independent DP replicas — it's a
divisor. Metric: **per-node otps = `otps × cores_per_node / cores`**. Maximize under a
loose ITL ceiling, ties to min cores. For gemma-4-31B the winners cluster at **small
footprint, low TP** (5 of 6 buckets land at 8 cores; 256k needs CP=2 / 16 cores). Low TP
wins because per-node throughput rewards packing the most weight-sharing replicas per node,
each at max batch. **Deploy the smallest replica that holds weights, max batch, scale via DP.**

### Interactivity (below max batch)

Trade batch for latency: **TP wins** because ITL scales as `1/(TP·PP_eff)` via the matmul
term, and the highest feasible TP = query-head count. Structural facts: PP never helps
single-request ITL (sequential stages); CP helps ITL only by relieving a KV-read bind; high
TP goes KV-bound at long context unless paired with CP.

## Prefill (bs=1) — a separate, footprint-driven decision

Prefill is selected at **bs=1** (the single-request TTFT a user waits on) — the TTFT gate on
the colocated replica (see [design.md](design.md) §4.2). Structural facts:

- **Prefill is matmul/compute-bound at every bucket** — a full prompt is S tokens of dense
  compute even at bs=1; never weight/KV-read-bound the way decode is.
- **Parallel efficiency is ~100%** — TTFT halves every time core count doubles. So a naive
  min-TTFT objective trivially picks the largest footprint in the grid (one full node)
  everywhere → useless.
- The meaningful objective is the **smallest footprint meeting a TTFT SLA**. Because prefill
  cost is linear in prompt length, the SLA is a constant **input-token-throughput floor**
  (ITPS, prefill's counterpart to decode's OTPS) → a TTFT ceiling linear in context
  (`ttft_ceiling_ms = S / floor`). The floor is a **policy knob** (`--prefill-itps-floor`,
  default 30000), not an entitlement result — the same caveat class as `USABLE_FRACTION`.
- **PP=1 always wins prefill** (sequential stages = pure latency penalty at bs=1); **CP is
  the scaling lever** (shards prefill compute and sequence with no sequential-stage cost).

### The prefix-cache / KV-replication lever

The replica's spare HBM holds a **prefix cache** (reusing computed KV across requests).
But the peak-HBM tie-break
prefers high TP (only TP·PP shards weights; CP does not) — and high TP **replicates** the
KV cache (factor `TP / min(TP, n_kv_heads)`). For gemma-4-31B the 10 global
(context-growing) layers have only 4 KV heads, so **any TP>4 replicates the
prefix-cache-relevant KV** (TP8=2×, TP16=4×, TP32=8×) — zero cache benefit, multiplied
footprint.

So the selector **excludes KV-replicating configs by default** (`TP ≤ min_kv_heads`) and
reports a `prefix_cache` headroom block per target:
`cacheable_full_ctx_seqs = (usable_HBM − weights/(TP·PP) − bs1_activations) / per-device-per-seq-KV`
(already replication-aware).

The non-replicating set caps at `min_kv_heads × CP_max` cores (PP doesn't help prefill
TTFT) — so it meets the SLA for free at short/medium context but **misses it at long
context**, where only a higher-core replicating config holds TTFT under the ceiling. That
is the explicit tradeoff: max prefix-cache capacity (non-replicating, TTFT floor) vs. lower
TTFT (replicating, less cache, more cores). The comparison is **not
monotone** — at very long context a heavily weight-sharded replicating config can free
enough HBM to hold *more* cache than the non-replicating small-footprint pick. The metric
captures this directly: the rule is "smallest footprint meeting the SLA, with the most
prefix cache," not a blanket "low TP."

**Scope caveat:** the metric is cache *capacity*, not cache *hit benefit* (which depends on
traffic prefix-sharing). Serving is single-node and colocated; splitting prefill and decode
across pools or nodes is out of scope, so there is no KV-handoff cost to model.

## Caveats (theoretical bound)

- Entitlement is a **lower bound**; real ITL/TTFT will be higher (dispatch, sync, idle
  gaps, small-tile inefficiency). Matmul-bound cells assume TE-peak utilization, which is
  optimistic. The first on-hardware sweep should check both a throughput-capacity target
  and an interactivity target.
- **Max-feasible bs** is the closed-form HBM ceiling snapped to a multiple of 2 — the memory
  ceiling, not a serving prescription. The DP-normalization is optimistic (real DP adds
  load-balancing / shared-prefix coordination).
- **Buckets are config-derived**, tuned to the ITL–otps tradeoff; a different objective
  (equal log-spacing, traffic-weighted) would move the medium buckets. The short floor and
  pinned max are stable.
- **Activation working set is a coarse 4×-hidden estimate**; the compile + smoke-run is the
  real **capacity** arbiter. This is separate from the model-floor activation traffic, which
  uses exact ordered component input/output views. Because batch is
  packed near the ceiling, reported max-bs remains sensitive to the coarse capacity estimate.
- **bf16 throughout.** FP8 weights would halve the weight-read bound and free HBM for larger
  batch — worthwhile follow-up, especially for long-context KV-bound cells.
