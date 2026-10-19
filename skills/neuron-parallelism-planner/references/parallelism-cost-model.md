# Per-device parallelism cost model

How each parallelism axis (degree `d`) divides — or fails to divide — the four
per-device cost terms the entitlement engine tracks: **compute** (matmul FLOPs on the
tensor engine), **weight memory** (bytes of parameters held + streamed per device),
**KV-cache memory** (bytes of K/V cache held + streamed per device), and
**communication** (collective traffic the axis introduces). Latency vs. throughput is
called out separately where they differ.

This is the conceptual backing for the planner engine: the divisor rules in
`planner/shard.py` and the weight corrections in `entitlement_formulas.py`
(`corrected_weight_bytes`). The
worked numbers in [methodology.md](methodology.md) and [tp-feasibility-and-kv-replication.md](tp-feasibility-and-kv-replication.md)
are this model applied to gemma-4-31B on trn2.

## The one non-obvious term: KV does not divide by TP

Every other axis divides its terms cleanly by `d`. **Tensor parallelism is the
exception for the KV cache.** The vendor sharding shards KV heads down to one per rank
and then *replicates*: a layer-group's per-device KV cache divides by
`min(TP, n_kv_heads)`, **not** by `TP`. Once `TP > n_kv_heads`, extra ranks hold
redundant copies of the cache — replication factor `TP / min(TP, n_kv_heads)`. This is
why high-TP decode goes KV-read-bound early and why a high-TP config wastes prefix-cache
HBM on the prefill pass. See [tp-feasibility-and-kv-replication.md](tp-feasibility-and-kv-replication.md).

## Cost as a function of degree d

`d` = the degree of the named axis (e.g. TP=8 → d=8). "1/d" = the term shrinks
proportionally; "1" = unchanged (replicated); "×d" = grows.

### Tensor parallelism (TP) — shard each layer across d devices

| Term | Per-device cost | Note |
|---|---|---|
| Compute (matmul) | **1/d** | each rank does its head/hidden shard |
| Weight memory | **1/d** | params split across ranks |
| KV-cache memory | **1/min(d, n_kv_heads)** | shards to KV-head count, then **replicates** |
| Communication | **+ AllReduce/layer** | 2 collectives/layer, ring-optimal on a torus axis |
| Latency (ITL/TTFT) | **1/d** | shards the critical path; the latency lever |

> **Caveat — the compute 1/d does NOT continue to net-reduce latency once
> collective traffic dominates.** The "Latency 1/d" row is the *compute* term only; the
> communication term grows with d. On a collective-bound graph the two cross over, and the
> latency/throughput optimum becomes an **interior TP degree**, not the maximum: past the
> crossover, per-rank matmul compute has shrunk to where the added all-gather/reduce-scatter
> cost (which grows with d) exceeds the marginal matmul savings. Practical rule: when the
> planner predicts a collective-heavy prefill config, expect an interior TP optimum and
> validate the top one or two degrees empirically rather than assuming monotonic 1/d
> latency. *(Annotate before applying to decode/generative workloads.)*


### Pipeline parallelism (PP) — split layers into d sequential stages

| Term | Per-device cost | Note |
|---|---|---|
| Compute (matmul) | **1/d** | fewer layers per device |
| Weight memory | **1/d** | fewer layers per device |
| KV-cache memory | **1/d** | fewer layers per device |
| Communication | **+ P2P** | (d−1) neighbor sends, cheap/contention-free on a torus |
| Throughput | **1/d** rate | pipelines microbatches across stages concurrently |
| Latency (ITL/TTFT) | **×d** (no benefit) | a single token traverses all d stages **sequentially** |

PP is the throughput-vs-latency trap: it divides the steady-state per-token *rate* by d
(good for throughput) but a single request's latency does **not** benefit — it pays
`step_ms × PP`. Never use PP to chase low ITL/TTFT.

> **PP is not yet supported by the Neuron inference stack** and is **out of scope** for the
> planner (the grid pins PP=1). This row documents PP's cost behavior for reference; the
> engine does not sweep it. See [design.md](design.md) §2 (scope) and §4.

### Context / sequence parallelism (CP) — shard the sequence across d devices

| Term | Per-device cost | Note |
|---|---|---|
| Compute (matmul) | **1/d** (prefill) | shards prefill/sequence compute; decode token is single |
| Weight memory | **1** (replicated) | every CP rank holds full weights |
| KV-cache memory | **1/d** | shards the sequence dimension of the cache |
| Communication | **+ K/V ring** | `2bsh(d−1)/d · 1/g` per layer, divided by GQA ratio g=n_q/n_kv |
| Latency | **1/d** *only when KV-bound* | relieves a KV-read bind; otherwise just adds cores |

CP is the long-context lever: it shards the KV cache (no replication ceiling, unlike TP)
and prefill compute with no sequential-stage penalty. The cost is replicated weights.

### Data parallelism (DP) — d independent replicas

| Term | Per-device cost | Note |
|---|---|---|
| Compute | **1** (replicated) | each replica does full work for its own requests |
| Weight memory | **1** (replicated) | full weights per replica |
| KV-cache memory | **1** (replicated) | full cache per replica |
| Communication | **0** (inference) | no cross-replica traffic for independent decode |
| Aggregate throughput | **×d** | d× the requests, zero comm |

DP is "free" footprint scaling for throughput: collapse footprint into a per-node
throughput metric (`otps × cores_per_node / cores`) and the smallest replica that holds
the weights wins. Real DP adds load-balancing and (shared-prefix) coordination overhead.


### Expert parallelism (EP) — shard MoE experts across d devices

| Term | Per-device cost | Note |
|---|---|---|
| Compute (expert matmul) | **1/d** | each rank runs only the experts it hosts |
| Weight memory (experts) | **1/d** | expert params split |
| Weight memory (router) | **1** | the gate is replicated on every rank, so it never shrinks with sharding — its per-device share grows with TP and with num_experts |
| KV-cache memory | **1** | attention/KV is not expert-sharded |
| Communication | **wider FFN AllReduce** | the MoE layers' output reduction spans the expert grid (`d`) instead of TP; ring-optimal, no bisection term |

EP only applies to MoE models (dense models like gemma-4-31B have no experts → EP is a
no-op). Tokens are **not** routed to a core subset: every rank in the grid holds all tokens,
runs its local experts, and one AllReduce sums the partial FFN outputs (design.md §2.2). So
there is no all-to-all to budget for — the only EP communication cost is the *increment* from
widening a collective the TP term already charges, which is zero at `d = TP`. A token-dispatch
implementation would instead cost an all-to-all dispatch+combine and divide expert compute by
TP alone; that is a different machine from the one this model prices.

## Choosing an axis by objective

- **Throughput / batch / offline decode** → smallest replica that holds weights (low
  TP, optionally +PP for batch headroom), max batch, scale out with **DP**.
- **Interactive / low-ITL decode** → high **TP** (up to the query-head count), batch
  tuned below max, **+CP** at long context to keep KV off the critical path. Never PP.
- **Prefill / TTFT** → **PP=1** always; scale cores via **CP** as context grows.
- **Long context (KV-bound)** → **CP** shards the cache without TP's replication ceiling.
- **MoE** → **EP** for the expert weights; its only comm cost is widening the MoE layers'
  FFN AllReduce from TP to the expert grid (zero when the grid is the TP group).

## Speculative decoding — not a sharding axis

Speculative decoding amortizes one **target** verify pass plus γ **draft** forward passes
into multiple accepted tokens. It is orthogonal to the parallelism axes above: it changes
the *per-accepted-token* cost, not how work is split across devices. Its two knobs are
**γ** (draft length — how many tokens the draft proposes per round) and **α** (acceptance
rate — the probability the target accepts a proposed token).

Implemented in `entitlement_counter.py` (`speculative_report`, `expected_accepted_tokens`,
`SpeculativeDraft`). **Design rule: nothing about the draft is inferred.** The caller MUST
supply the draft's per-forward requirements — weight bytes, KV bytes, FLOPs — either via
`SpeculativeDraft(...)` directly, from a measured `EntitlementReport`
(`SpeculativeDraft.from_report(profile_module(draft, ...))`), or from the draft's HF config
(`weights_total(arch_from_hf(cfg)) * dtype_bytes`).

### Expected accepted tokens per verify pass

`N(γ, α) = (1 − α^(γ+1)) / (1 − α)` — truncated geometric. Bounds:
- `= γ + 1` at α = 1 (every proposed token plus the bonus token accepts)
- `→ 1` as α → 0 (only the bonus token ever accepts)
- capped at **both** `γ + 1` and `1/(1 − α)` (saturates as γ grows)

### Per-accepted-token cost

Each verify iteration reads the target weights **once** and runs γ draft forwards, then
amortizes over N accepted tokens:

| Term | Per-iteration | Per-accepted-token |
|---|---|---|
| Weight bytes | `target_w + γ·draft_w` | `(target_w + γ·draft_w) / N` |
| KV bytes | `target_kv + γ·draft_kv` | `(target_kv + γ·draft_kv) / N` |
| FLOPs | `target_flops + γ·draft_flops` | `(target_flops + γ·draft_flops) / N` |

### Weight-read speedup vs. plain decode

Decode is weight-read-bound, so the headline speedup is on the weight term:

```
weight_read_speedup = N / (1 + γ · draft_weight / target_weight)
```

- approaches **N** when the draft is cheap (`draft_weight ≪ target_weight`)
- sits **below N** when the draft is heavy — `draft_weight_fraction = γ·draft_w/target_w`
  is the diagnostic. A draft too large for its acceptance rate erodes the win.

### Interaction with the prefix cache

The draft model's KV state (`draft_kv`) competes with the target's **prefix cache** for
HBM on the colocated replica — the same budget analyzed in
[methodology.md](methodology.md) §prefix-cache. Budget for it explicitly.
