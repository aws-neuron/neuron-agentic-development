# Granite-4.1-8B — Parallelism Planner Walkthrough (trn2)

Planning a serving config for Granite-4.1-8B on trn2 with the
`neuron-parallelism-planner` skill: derive the model architecture, validate the
analytic engine against the meta-device oracle, then sweep configs to pick an operating point.
On-hardware benchmarking/profiling of the chosen config follows in
`GRANITE_BENCHMARK_WALKTHROUGH.md`.

Run-spec: written out in full at the end of this document (see
[The example, in full](#the-example-in-full)). Save those three blocks into a `./granite`
directory under their own filenames, then run the commands below from your project directory with
`SKILL=<skill>/scripts`. Interpreter: `python3.12` with `torch` (meta device, CPU) for the oracle.

---

## Step 1 — Architecture: verify supported, enable if not

Following the skill's "build an oracle for a new model" procedure, four places can need
adjustment: (1) `arch_from_hf` parse, (2) `entitlement_formulas.py` weight/FLOP terms,
(3) `oracle_model.py` structure, (4) `validate_oracle.py` cross-check. Granite needed a change
in only (1); (2)-(3) were already covered, and (4) came out clean.

### 1a. Parse the architecture (`arch_from_hf`)

The run-spec points at the pristine HF config and `arch_from_hf` parses it. Granite's
config (`.artifacts/models/granite-4.1-8b/config.json`):

| HF field | value |
|---|---|
| `architectures` | `GraniteForCausalLM` |
| `hidden_size` | 4096 |
| `intermediate_size` | 12800 |
| `num_hidden_layers` | 40 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 |
| `head_dim` | **absent** |
| `vocab_size` | 100352 |
| `max_position_embeddings` | 131072 |
| `tie_word_embeddings` | true |
| `torch_dtype` | bfloat16 |
| `layer_types` / `num_experts` | absent |

Granite omits `head_dim`, so `arch_from_hf` needed one addition: derive it when absent
(`hidden_size // num_attention_heads` = 128) — the only enablement change required.

With no `layer_types`, routed experts, or `kv_lora_rank`, Granite parses as one uniform
full-attention group with a dense FFN (GQA). Derived `Arch`:

| property | value |
|---|---|
| layers | 40 (one full-attention group) |
| hidden size | 4096 |
| query / KV heads | 32 / 8 (head_dim 128) |
| attention | full context (no sliding window), no MLA |
| FFN | dense, intermediate 12800 (no routed experts) |
| vocab | 100352, tied embeddings |
| dtype | bf16 |
| **total weights** | **16.76 GB** (7.97 B block params × 2 B + 0.82 GB tied embed/lm_head) |

### 1b. Confirm the oracle implements this arch

`OracleModel` builds generically from `Arch.attn_groups` + the dense/MoE switch — a uniform
dense-GQA arch is already covered, so no oracle module change was needed for Granite.

### 1c. Verify engine correctness against the oracle

To confirm the engine's counts before trusting a sweep, `validate_oracle.py` diffs the
analytic engine against the **meta-device oracle** (a shape-exact `torch` module built from the
same `Arch`, run under a per-aten FLOP counter) per segment.

```bash
cd <parallelism-planner>/scripts
python3.12 $SKILL/validate_oracle.py ./granite/granite4-8b-run.json
```

Output:

```
model: granite-4.1-8B (dense)  layers=40  dtype_bytes=2  k_eq_v=False  by=type

=== decode S4096 bs8 ===
segment                     oracle FLOP    analytic FLOP      d%    oracle wt(MB)  analytic wt(MB)      d%
attention/global         48,318,382,080   48,318,382,080   +0.0%           3355.5           3355.4   +0.0%
ffn/dense               100,663,296,000  100,663,296,000   +0.0%          12582.9          12582.9   +0.0%
lm_head                   6,576,668,672    6,576,668,672   +0.0%            822.1            822.1   +0.0%
TOTAL                   155,558,346,752  155,558,346,752   +0.0%          16760.5          16760.4   +0.0%

=== prefill S2048 bs1 ===
segment                     oracle FLOP    analytic FLOP      d%    oracle wt(MB)  analytic wt(MB)      d%
attention/global       9,620,726,743,040 8,247,008,296,960  +16.7%           3355.5           3355.4   +0.0%
ffn/dense              25,769,803,776,000 25,769,803,776,000   +0.0%          12582.9          12582.9   +0.0%
lm_head                     822,083,584      822,083,584   +0.0%            822.1            822.1   +0.0%
TOTAL                  35,391,352,602,624 34,017,634,156,544   +4.0%          16760.5          16760.4   +0.0%
```

### Reading the result

- **Weights: exact (+0.0%)** on every segment. The analytic `corrected_weight_bytes` matches
  the oracle (residual is RMSNorm params the engine omits — negligible at this hidden size).
- **Projection/FFN/lm_head FLOPs:** exact (+0.0%).
- **SDPA FLOPs:** decode is exact. Prefill's raw meta SDPA count is 16.7% above the analytic
  attention segment because the floor charges causal active pairs.
- **KV bytes:** the analytic `kv_bytes_total` is the authority; it is not sourced from the
  raw meta SDPA count.

**Verdict: clean.** Projection/FFN/lm_head weights and FLOPs match, decode SDPA matches, and
the prefill attention difference equals the classified dense-versus-active pair count.

> The oracle validates **engine ⇄ oracle** consistency (both built from the same `Arch`), not
> **arch ⇄ reality**. Because they share one parse, a mis-parsed arch would read +0.0% while
> still being wrong. Guard for that separately: the corrected weight total (16.76 GB, ~8.4 B
> params incl. tied embeddings) is consistent with the model card's "8B", so the parse is sound.

---

## Step 2 — Sweep and select a config

### 2.1 — Decisions

Choices this plan commits to (the rest is engine-derived):

- **Serving mode:** colocated on a single node — one replica serves both prefill and decode,
  gated by both the TTFT and ITL ceilings. (Disaggregated / multi-node serving is out of scope.)
- **SLAs:** operator policy knobs (not arch-derived), left at defaults — prefill TTFT floor
  30000 ITPS, decode ITL ceiling 50 ms. Both are applied to the SAME config: a colocated
  replica must clear the TTFT ceiling AND the ITL ceiling at once.
- **Parallelism axes** (dense model):
  - **DP = 1** in the grid — with no MoE, DP replicas are independent, so DP is a free
    multiplier used to fill the node (folded into the per-node throughput metric), not swept.
  - **PP = 1** — not implemented by vllm-neuron.
  - **CP** — swept for completeness (the engine costs it correctly), **but not selectable
    here**: vllm-neuron supports DCP, yet our Granite model.py path is TP+SP only, so a CP>1
    pick can't actually be run. Kept in the landscape, excluded from the final choice.
  - **EP** — N/A (no routed experts).
  - Leaves **TP** (and CP, for completeness) as the swept axes.
- **Objective:** not yet defined — so rather than commit to a single pick, look at the
  **ITL / per-node-throughput Pareto frontier** and choose from it once an objective is set.
- **Batch / memory:** the same curve lets us vary batch size along the frontier, trading
  throughput for latency and potentially relieving memory pressure (smaller batch → less KV).

### 2.2 — Sweep the 8k landscape (ITL vs throughput vs batch)

Full grid at 8k, each config at its max-feasible batch (per-node throughput = otps ×
cores_per_node / cores; ITL = decode step time). CP configs shown but **not selectable**
(see 2.1); batch-size evolution is left to the visualization.

| config | cores | bs (max) | ITL ms | tput/node | binding |
|---|--:|--:|--:|--:|---|
| TP4 | 4 | 56 | 33.4 | 26,853 | kv_load |
| TP4 CP2 / CP4 | 4 | 56 | 33.4 | ~26,850 | kv_load |
| TP8 | 8 | 124 | 35.1 | **28,275** | kv_load |
| TP8 CP2 / CP4 / CP8 | 8 | 124 | 35.1 | ~28,266 | kv_load |
| TP16 | 16 | 130 | 37.2 | 13,986 | kv_load |
| TP16 CP2 / CP4 / CP8 | 16 | 262 | 39.3 | ~26,680 | kv_load |
| TP32 | 32 | 134 | 41.4 | 6,480 | kv_load |
| TP32 CP4 / CP8 | 32 | 538 | 47.5 | ~22,650 | kv_load |

Reading the max-batch table:
- **Per-node throughput peaks at TP8 (28,275), then falls with more cores** — TP4 close behind
  (26,853), TP16 and TP32 much lower (13,986 / 6,480). Past TP8 = `n_kv` (8), the KV cache
  **replicates** (TP16 → 2×, TP32 → 4×), so extra cores hold redundant KV instead of buying
  throughput — the runnable picks favor the small replica.
- **ITL grows with TP** at max batch (33.4 → 35.1 → 37.2 → 41.4 ms across TP4→32), because
  max-feasible batch grows with cores.
- **CP rows match their TP row** (TP8 CP2/4/8 ≈ 28,266 ≈ TP8's 28,275) — CP appears to buy no
  throughput at 8k. This is counter-intuitive and unexpected; **WIP — needs a closer look**
  (also not runnable regardless).
- **Binding:** all cells bind `kv_load` at their max-feasible batch — the batch-scaled KV read
  dominates once the batch is large. (`comm` overtakes only at low/moderate batch, where the fixed
  per-step all-reduce is large relative to the KV read: TP32, for instance, binds `comm` through
  ~bs32 and flips to `kv_load` by bs64.)

All cells clear the 50 ms ITL ceiling, so among the configs that also meet the TTFT ceiling the
max-per-node-throughput pick is **TP8** — the point the benchmark step probes on hardware.

### 2.3 — Visualization: the interactivity Pareto frontier

`PYTHONPATH=$SKILL python3 -m planner ./granite/granite4-8b-run.json --visualize` writes the per-node interactivity
Pareto (`pareto_per_node_<model>.png`) — ITL vs per-node throughput, one facet per bucket, each
config swept over batch size below its max:

![per-node interactivity Pareto](granite_runs/planner_viz/pareto_per_node_granite_4_1_8b_dense.png)

Reading the 8k facet (top-left): the frontier (colored, bs-labeled) is **all TP8** — it
dominates TP4 (red, only touches the frontier at its max-bs corner) and the higher-TP configs
(grey, well inside). Two things the walkthrough relies on are visible directly:
- **The frontier is the ITL ↔ per-node-throughput trade** — no single point is "best" until an
  objective is set; you pick where on the curve to sit.
- **Moving along the frontier = varying batch** (bs1 top-left → bs124 bottom-right on the TP8
  curve): lower batch → lower ITL and less KV (memory relief), higher batch → more throughput.
  This is the same curve referenced in 2.1.

(Grey curves are the other configs' batch sweeps; TP16/TP32 sit inside the TP8 frontier —
consistent with the KV-replication falloff past TP8 noted above.)

## Step 3 — Cost-model breakdown for a fixed config

### 3.1 — Decisions

- **Config:** pick a Pareto-optimal point from the 2.3 frontier — **TP8 × bs32**. It sits on
  the TP8 frontier but below the max-batch corner (max-feasible bs at TP8/8k is 124), so it
  leaves real memory headroom: peak HBM ≈ **7.5 GB/core of 23.2 usable (32%)**, vs 99% at
  bs124. Reasonable throughput without pinning the KV budget. (bs above 16 did not immediately work
  for profiling, so the following sections use bs16 — though we would typically pick a higher
  value along the Pareto frontier.)
- **Regime:** decode (the bound phase).
- **Granularity:** compressed per-(component, variant) records from the planner's `--cell`
  `by_type` output. Each record carries its layer multiplicity. The engine applies
  `max(record_TE, record_HBM)` before any display roll-up, serializes the records, and adds
  communication separately.

### 3.2 — Breakdown

```bash
PYTHONPATH=$SKILL python3 -m planner ./granite/granite4-8b-run.json --cell --tp 8 --cp 1 --bs 16 --ctx 8192
```

`cores=8  step_ms=8.57  comm_ms=1.86  binding=attention:kv_load` → decode **1,867 otps/replica**,
per-node (×8 DP) **14,938**.

Granite has just three keys: uniform full-attention + a uniform dense FFN + the lm_head
singleton (no sliding/global split, no MoE, no MLA — so no extra attention- or ffn-types).
Weight, KV, and activation columns show mandatory HBM bytes; `time` = max(load, flop).
Each attention or FFN body reads one full hidden input and writes one pre-collective output.
The three reported rows represent 81 serialized component instances: 40 attention bodies,
40 FFN bodies, and one lm_head.

**Per component instance** (reported row divided by its multiplicity; lm_head is ×1):

| component / type | weight | KV | activation | FLOP (compute) | time ms | binds |
|---|--:|--:|--:|--:|--:|---|
| attention / global | 10.5 MB | 67.1 MB | 0.3 MB | 0.436 G (0.003 ms) | 0.109 | **kv_load** |
| ffn / dense | 39.3 MB | 0 | 0.3 MB | 0.629 G (0.004 ms) | 0.055 | **weight_load** |
| lm_head (×1) | 102.8 MB | 0 | 0.5 MB | 1.644 G (0.010 ms) | 0.144 | **weight_load** |

**Reported component records** (full-model quantities after exact cost records are costed):

| component / type | layers | weight | KV | activation | FLOP (compute) | time ms | binds |
|---|--:|--:|--:|--:|--:|--:|---|
| attention / global | 40 | 419.4 MB | 2,684.4 MB | 10.5 MB | 17.45 G (0.110 ms) | 4.350 | **kv_load** |
| ffn / dense | 40 | 1,572.9 MB | 0 | 10.5 MB | 25.17 G (0.16 ms) | 2.211 | **weight_load** |
| lm_head / – | 1 | 102.8 MB | 0 | 0.5 MB | 1.64 G (0.01 ms) | 0.144 | **weight_load** |
| comm | | | | | | 1.86 | — |
| **step** | | 2,095.0 MB | 2,684.4 MB | 21.5 MB | 44.26 G | **8.57** | attention:kv_load |

The certified TE/HBM component floor is
`4.350 + 2.211 + 0.144 = 6.705 ms`; adding 1.864 ms of communication produces the
8.569 ms step. The global engine diagnostic is also 6.705 ms in this cell because all records
are HBM-bound. That equality is incidental: a model with TE-bound and HBM-bound component variants
must roofline their cost records before the reporting roll-up.

Reading it:
- **Attention KV-read dominates** — its 3.75 ms KV load is the bulk of the 4.36 ms attention
  time and of the 8.57 ms step. Per-device KV is 2.68 GB,
  streamed every decode step; the 0.42 GB of attention weights (0.59 ms load) are minor beside
  it. This is the KV-bound regime the frontier lives in.
- **FFN** stays at 2.21 ms (weight-load bound, weights fixed with batch); **lm_head** negligible.
- **comm** 1.86 ms (~22% of the step).
- vs bs16 (step 8.57 ms, 1,867 otps): bs32 remains a throughput/latency tradeoff —
  diminishing return, since decode is already KV-bound past bs16.

## Step 4 — Compare against a real run

The entitlement above is a lower bound; this step checks it against hardware.

> Config sidenote: §3 selected **TP8 × bs64** from the frontier, but a >bs16 run wasn't
> immediately workable on hardware (bs64 exceeds the KV pool at 8k, and forcing it larger hit a
> compile wall), so this comparison **starts at bs16** — the point that runs and profiles
> cleanly. Higher-batch points to follow.

### 4.1 — Benchmark + DP node scaling

TP8/bs16/8k, `vllm bench serve` against a live `vllm serve` instance. Decode-only via TPOT
(= request latency − TTFT). Single replica, then scaled across the node via independent DP
replicas (dense model → no cross-replica collectives).

Common env (default 10s RPC timeout is too short for the uncached 40-layer compile):

```bash
export VLLM_RPC_TIMEOUT=1800000 NEURON_SKIP_EFA_AFFINITY=1
CKPT=~/GemmaTune/.artifacts/models/granite-4.1-8b
```

Serve (DP1 shown; DP3/DP8 add `--data-parallel-size 3` / `8` and widen NEURON_VISIBLE_DEVICES to
24 / 64 logical cores):

```bash
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve $CKPT --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 16 \
  --tensor-parallel-size 8 --port 8000 \
  --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": "true"}, "num_batched_tokens_buckets": [8192], "num_seqs_buckets": [16]}}'
```

Benchmark (concurrency = 16 × replicas: 16 / 48 / 128; prompts = 8× concurrency to stay
saturated):

```bash
vllm bench serve --backend vllm --host localhost --port 8000 --model $CKPT \
  --dataset-name random --random-input-len 7680 --random-output-len 512 \
  --max-concurrency 16 --num-prompts 128 \
  --save-result --result-dir ~/GemmaTune/granite_runs/benchserve --result-filename dp1.json
```

| deployment | cores | measured tok/s | planner entitlement | measured / planner | median TPOT (meas / planner step) |
|---|--:|--:|--:|--:|--:|
| 1 replica (DP1) | 8 | 618 | 1,874 otps | 0.33× | 20.25 ms / 8.54 ms |
| DP3 | 24 | 1,847 | 5,622 (3×) | 0.33× | 20.29 ms |
| DP8 (full node) | 64 | 4,489 | 14,991 (8×) | 0.30× | 20.81 ms |

- **Throughput:** measured ~0.30–0.33× the planner entitlement at every DP degree — the same
  ratio across the node, so DP scaling itself is faithful (DP8 = 7.26× the single replica, ~9%
  under ideal 8×); the offset is per-replica.
- **TPOT** stays flat as replicas are added (per-replica decode unchanged) at ~20 ms, vs the
  planner's 8.54 ms step.

> Method note: the live-server `/start_profile` + `vllm bench serve` path used here is newer than
> the `neuron-framework-profiling-vllm-neuron` skill's current NEFF-capture
> walkthrough — **WIP to update the skill with this method**.

### 4.2 — Planner at reduced layers (to match the profiled proxy)

The full 40-layer model + profiler buffers exceed HBM, so the device profile is captured on a
**reduced-layer proxy**. The planner has no layer-count override — a reduced-layer model is a
different model, so it needs its **own run-spec** (`granite4-8b-4L-run.json`, an HF config with a
reduced `num_hidden_layers`):

```bash
PYTHONPATH=$SKILL python3 -m planner ./granite/granite4-8b-4L-run.json --cell --tp 8 --cp 1 --bs 16 --ctx 8192
```

Expected per-device workload (TP8/bs16/8k, 4-layer):

| component / type | layers | weight | KV | activation | FLOP (compute) | time ms | binds |
|---|--:|--:|--:|--:|--:|--:|---|
| attention / global | 4 | 42.0 MB | 268.4 MB | 1.0 MB | 1.745 G (0.011 ms) | 0.435 | **kv_load** |
| ffn / dense | 4 | 157.3 MB | 0 | 1.0 MB | 2.517 G (0.016 ms) | 0.221 | **weight_load** |
| lm_head / – | 1 | 102.8 MB | 0 | 0.5 MB | 1.644 G (0.010 ms) | 0.144 | **weight_load** |
| comm | | | | | | 0.186 | — |
| **step** | | 302.0 MB | 268.4 MB | 2.6 MB | 5.906 G | **0.987** | attention:kv_load |

Here the compressed component floor is
`0.435 + 0.221 + 0.144 = 0.800 ms`; the 0.186 ms communication term brings the
step to 0.987 ms.

### 4.3 — Profile vs expectation (per-section)

Device profile on the reduced-layer proxy (TP8, bs16, 8k, steady-state decode), sectioned by
collective boundaries, per-segment average vs the per-layer entitlement.

Serve with the profiler enabled (`delay_iterations` skips warmup+prefill; `max_iterations` bounds
the capture to a few steady-state decode steps so the NTFF stays small):

```bash
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve $CKPT --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 16 \
  --tensor-parallel-size 8 --port 8000 \
  --hf-overrides '{"num_hidden_layers": 6}' \
  --profiler-config '{"profiler": "cuda", "delay_iterations": 40, "max_iterations": 5}' \
  --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": "true"}, "num_batched_tokens_buckets": [8192], "num_seqs_buckets": [16]}, "neuron_profiler": {"activities": ["device_profile","system_profile"], "output_dir": ".../nrt_profile"}}'
```

Profile a decode-saturated window, then stop:

```bash
curl -X POST http://localhost:8000/start_profile
vllm bench serve --backend vllm --host localhost --port 8000 --model $CKPT \
  --dataset-name random --random-input-len 7680 --random-output-len 512 \
  --max-concurrency 16 --num-prompts 16
curl -X POST http://localhost:8000/stop_profile
```

#### 4.3a — Sequential segment-by-segment breakdown

The decode NEFF in execution order: each measured region followed by the collective that ends it
(embedding AR, then per layer the attention/global AR + ffn/dense AR, then the lm_head
AllGathers). Segments are labeled by **component × layer_type** — here `attention/global`,
`ffn/dense`, `lm_head/-` (Granite has one attention-type and one ffn-type). Same shape as the
§3.2 planner breakdown — DMA bytes (with load time), matmul FLOP (with compute time):

Decode's completed AllReduce does not expose the later body-start boundary in the collective
trace. These measured attention/FFN regions therefore include the residual/norm preparation
before the body. The body floor is compared against this larger composite interval for strict
containment, not standalone body tightness.

| # | segment (component/type) | DMA (load) | FLOP (compute) | dur ms | exp dur ms |
|--:|---|--:|--:|--:|--:|
| 0 | embed | 8.76 MB (0.012 ms) | 0.02 G (0.000 ms) | 0.296 | — |
| | AllReduce[0] 131072B | | | 0.032 | 0.023 |
| 1 | L0 attention/global | 72.62 MB (0.101 ms) | 0.49 G (0.003 ms) | 0.215 | 0.109 |
| | AllReduce[1] 131072B | | | 0.023 | 0.023 |
| 2 | L0 ffn/dense | 47.21 MB (0.066 ms) | 0.66 G (0.004 ms) | 0.131 | 0.055 |
| | AllReduce[2] 131072B | | | 0.027 | 0.023 |
| 3 | L1 attention/global | 74.73 MB (0.104 ms) | 0.49 G (0.003 ms) | 0.202 | 0.109 |
| | AllReduce[3] 131072B | | | 0.024 | 0.023 |
| 4 | L1 ffn/dense | 47.26 MB (0.066 ms) | 0.66 G (0.004 ms) | 0.130 | 0.055 |
| | AllReduce[4] 131072B | | | 0.028 | 0.023 |
| 5 | L2 attention/global | 74.73 MB (0.104 ms) | 0.49 G (0.003 ms) | 0.202 | 0.109 |
| | AllReduce[5] 131072B | | | 0.023 | 0.023 |
| 6 | L2 ffn/dense | 47.26 MB (0.066 ms) | 0.66 G (0.004 ms) | 0.132 | 0.055 |
| | AllReduce[6] 131072B | | | 0.026 | 0.023 |
| 7 | L3 attention/global | 74.73 MB (0.104 ms) | 0.49 G (0.003 ms) | 0.202 | 0.109 |
| | AllReduce[7] 131072B | | | 0.024 | 0.023 |
| 8 | L3 ffn/dense | 65.11 MB (0.091 ms) | 0.65 G (0.004 ms) | 0.146 | 0.055 |
| | AllReduce[8] 131072B | | | 0.035 | 0.023 |
| 9 | lm_head/- | 66.19 MB (0.092 ms) | 14.89 G (0.094 ms) | 0.234 | 0.144 |
| | AllGather[9] 401408→3211264B | | | 0.037 | — |
| 10-12 | sampling tail | 8.03 MB (0.011 ms) | 0.00 G (0.000 ms) | 0.053 | — |
| | **decode step total** | **586.6 MB** | **19.47 G** | **2.239** | **0.987** |

#### 4.3b — Per-segment-type mean vs expected

Average the like segments (all `attention/global`, all `ffn/dense`) into a per-layer mean and lay
it beside the per-layer entitlement (§3.2), in the same column shape as the §3.2 tables —
weight/KV/FLOP with their load/compute times. `exp` = planner expected, `meas` = profiled mean.

> TODO (formatting): fold the two tables below into one — expected and measured side by side per
> column (weight/KV/FLOP/time) with the ratio inline — rather than two separate tables. Cleaner to
> read the divergence per cell.

**Expected** (per segment type, from §4.2's reduced-layer `--cell`):

| component / type | weight | KV | activation | mandatory load | FLOP (compute) | time ms | binds |
|---|--:|--:|--:|--:|--:|--:|---|
| attention / global (mean) | 10.5 MB | 67.1 MB | 0.26 MB | 77.9 MB (0.109 ms) | 0.436 G (0.003 ms) | 0.109 | **kv_load** |
| ffn / dense | 39.3 MB | 0 | 0.26 MB | 39.6 MB (0.055 ms) | 0.629 G (0.004 ms) | 0.055 | **weight_load** |
| lm_head / – | 102.8 MB | 0 | 0.53 MB | 103.3 MB (0.144 ms) | 1.644 G (0.010 ms) | 0.144 | **weight_load** |

**Measured** (per-segment-type mean):

| component / type | DMA bytes (peak-BW time) | FLOP (compute) | DMA active ms | TE active ms | time ms | meas/exp (DMA) | meas/exp (FLOP) | meas/exp (time) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| attention / global | 74.2 MB (0.104 ms) | 0.49 G (0.003 ms) | 0.160 | 0.072 | 0.207 | 0.95× | 1.12× | 1.90× |
| ffn / dense | 51.7 MB (0.072 ms) | 0.66 G (0.004 ms) | 0.110 | 0.047 | 0.135 | 1.31× | 1.05× | 2.44× |
| lm_head / – | 66.2 MB (0.092 ms) | 14.89 G (0.094 ms) | 0.145 | 0.203 | 0.234 | 0.64× | 9.1× | 1.62× |

> Note: the planner assumes a full-context KV read (eff_ctx = the full bucket S) for every
> decode token. With causal masking a token at an earlier position attends only to the tokens
> before it, so it loads a bit less KV than the full bucket — the planner does not discount this,
> so its attention load is an upper bound for any token before the last position.

> Note: at bs16 attention and ffn are memory-bound (DMA active > TE active: attention
> 0.160 > 0.072, ffn 0.110 > 0.047 ms); lm_head is compute-bound (TE 0.203 > DMA 0.145). A
> memory-bound segment is limited by achieved memory bandwidth (MBU), which is harder to drive
> to peak than tensor-engine utilization — so these segments are the less-efficient ones.

> Method note: per-section attribution follows the
> `neuron-framework-profile-analysis-vllm-neuron` skill — **WIP to get the
> collective-segmented (attention / MLP / lm_head) breakdown wired into the skill**.

The values above are bs16. **Higher batch sizes are still blocked**: the serve-time profiler
(`vllm serve --profiler-config`) fails to compile the reduced-layer proxy at bs>16 with
`[NCC_IGCA037] ... too big for SB, requires 393216 bytes` — an SBUF tile overflow in
`norm_tkg_utils.py:397` via `moe_block_tkg_wrapper`. So bs32/bs64 profiles (the higher points on
the frontier §3.1 would normally pick) are not yet captured.

To avoid that bug, the bs16 profile above used a **different profiling flow** — NEFF-capture
(`neuron-framework-profiling-vllm-neuron`, Mode A) instead of the serve-time
profiler: emit the NEFFs with `vllm bench throughput`, then replay+capture each NEFF with
`neuron-explorer capture` (instruments the already-compiled NEFF, so it does not hit the
compile-time SBUF overflow). Exact commands:

```bash
# 1. emit prefill+decode NEFFs (bf16, greedy; no fp8/MTP)
NEURON_RT_ENABLE_DGE_NOTIFICATIONS=0 NEURON_VISIBLE_DEVICES=<8 LNCs> \
  ./references/run_profile.sh granite_bf16_bs16_8k_4layer_tp8 \
    --model $CKPT --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 16 \
    --tensor-parallel-size 8 --hf-overrides '{"num_hidden_layers": 4}' \
    --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": "true"}, "num_batched_tokens_buckets": [8192], "num_seqs_buckets": [16]}}' \
    --dataset-name random --random-input-len 7680 --random-output-len 512 --num-prompts 16

# 2. capture the device trace for each NEFF (TP=8 collectives)
for neff in $PROFILE_DIR/capture/*.neff; do
  ID=$(basename "$neff" | sed 's/neff_\(.*\)_vnc_0\.neff/\1/'); mkdir -p $PROFILE_DIR/capture/$ID
  NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 neuron-explorer capture -n "$neff" \
    -s "$PROFILE_DIR/capture/$ID/profile.ntff" --profile-nth-exec=2 --enable-dge-notifs \
    --collectives-workers-per-node=8 --collectives-worker-count=8
done

# 3. ingest, then segment by collective boundaries (per neuron-framework-profile-analysis-vllm-neuron)
neuron-explorer view -d $PROFILE_DIR/capture_minimal --data-path ~/.local/share/neuron-profile \
  --display-name granite_bf16_bs16_8k_4layer_tp8 --ingest-only
```

#### 4.3c — Planner per-segment type at bs64 (is ffn more compute-bound?)

Higher batch was not profilable (see above), but the planner shows how the load/compute balance
shifts. Per component instance at TP8/bs64/8k decode (`--cell --bs 64`);
`load ms` = (weight + KV + component activation)/BW, `flop ms` = FLOP/TE, `time` = max:

| component / type | weight | KV | activation | FLOP (compute) | load ms | flop ms | time ms | AI (FLOP/byte) | binds |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| attention / global | 10.5 MB | 268.4 MB | 1.0 MB | 1.745 G | 0.391 | 0.011 | 0.391 | 6.23 | **kv_load** |
| ffn / dense | 39.3 MB | 0 | 1.0 MB | 2.517 G | 0.056 | 0.016 | 0.056 | 62.34 | **weight_load** |
| lm_head / – | 102.8 MB | 0 | 2.1 MB | 6.577 G | 0.146 | 0.042 | 0.146 | 62.70 | **weight_load** |

`step_ms=20.46  comm_ms=2.41  binding=attention:kv_load` → 3,129 otps/replica, per-node 25,029.

Arithmetic intensity (FLOP / loaded byte) per segment, bs16 vs bs64:

| component / type | AI bs16 | AI bs64 |
|---|--:|--:|
| attention / global | 5.60 | 6.23 |
| ffn / dense | 15.89 | 62.34 |
| lm_head / – | 15.92 | 62.70 |

FFN and lm_head AI rise by slightly less than 4× with the 4× batch: FLOPs scale with batch,
weights stay fixed, and their mandatory activation traffic also scales with batch. Attention
AI is nearly flat because KV and activation traffic grow alongside its FLOPs. FFN remains
weight-load-bound at bs64 (load 0.058 ms > flop 0.016 ms).

---

## The example, in full

Everything this walkthrough runs is written out here, so the example cannot drift from the prose
that explains it. Save the three blocks into one directory under the filenames shown and point the
planner at the run-spec:

```bash
mkdir -p granite    # then save the three blocks below into it
SKILL=$NEURON_AGENTIC_DEVELOPMENT_ROOT/skills/neuron-parallelism-planner/scripts
PYTHONPATH=$SKILL python3.12 -m planner ./granite/granite4-8b-run.json
```

They must share a directory because a run-spec resolves `hf_config` and `hardware` relative to
itself. The HF config is the pristine `config.json` from the model card, unmodified -- every
serving choice (precision, KV dtype, memory budget) lives in the run-spec instead. There is no
default model: the planner requires a run-spec argument.

**`granite4-8b-run.json`**

```json
{
  "_comment": "Run-spec for Granite-4.1-8B oracle validation. Dense GQA model (32 query heads, 8 KV heads, head_dim 128, tied embeddings, 40 layers, full attention only). Points at the pristine HF config.json (granite4-8b-hf.json, unmodified) + trn2 constants. TP <= 32 (32 query heads); KV does not shard past 8 KV heads. No mixed attention, no MoE, no MLA.",
  "model": "granite-4.1-8B (dense)",
  "hardware_name": "trn2",
  "hf_config": "granite4-8b-hf.json",
  "hardware": "trn2_constants.json",
  "configs": [
    {
      "label": "TP=8",
      "regime": "decode_heavy",
      "TP": 8,
      "PP": 1,
      "CP": 1,
      "bucket": 8192,
      "bs": 8
    },
    {
      "label": "TP=16",
      "regime": "decode_heavy",
      "TP": 16,
      "PP": 1,
      "CP": 1,
      "bucket": 8192,
      "bs": 16
    },
    {
      "label": "TP=32",
      "regime": "decode_heavy",
      "TP": 32,
      "PP": 1,
      "CP": 1,
      "bucket": 8192,
      "bs": 16
    },
    {
      "label": "TP=32 CP=2",
      "regime": "decode_heavy",
      "TP": 32,
      "PP": 1,
      "CP": 2,
      "bucket": 65536,
      "bs": 2
    },
    {
      "label": "TP=16",
      "regime": "prefill_heavy",
      "TP": 16,
      "PP": 1,
      "CP": 1,
      "bucket": 8192,
      "bs": 8
    },
    {
      "label": "TP=32",
      "regime": "prefill_heavy",
      "TP": 32,
      "PP": 1,
      "CP": 1,
      "bucket": 8192,
      "bs": 16
    }
  ],
  "serving": "colocated"
}
```

**`granite4-8b-hf.json`**

```json
{
  "architectures": [
    "GraniteForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attention_multiplier": 0.0078125,
  "bos_token_id": 100257,
  "embedding_multiplier": 12.0,
  "eos_token_id": 100257,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.1,
  "intermediate_size": 12800,
  "logits_scaling": 16.0,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "granite",
  "num_attention_heads": 32,
  "num_hidden_layers": 40,
  "num_key_value_heads": 8,
  "pad_token_id": 100256,
  "residual_multiplier": 0.22,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.53.3",
  "use_cache": true,
  "vocab_size": 100352
}
```

**`trn2_constants.json`**

```json
{
  "_comment": "trn2.48xl constants for one LNC2 LOGICAL NeuronCore (two physical NeuronCore-v3s). TE peak and rank count follow the physical-core geometry. HBM bandwidth is a direct per-LNC2 HBM target, and HBM capacity is the 96 GiB device pool divided across four LNC2 ranks; neither HBM value is derived from a per-physical-core quantity. Bandwidths are bytes/s and the TE rate is the rounded profiler-metadata BF16 ceiling used by NPE.",
  "_lnc2_derivation": {
    "physical_cores_per_logical_core": 2,
    "logical_cores_per_device": 4,
    "te_peak_flops_per_physical_core": 79000000000000.0,
    "hbm_bw_bytes_per_s_per_logical_core": 716000000000.0,
    "physical_cores_per_node": 128
  },
  "_te_comment": "A measured trn2 Metadata.parquet reports a 128x128 Tensor Engine at 2.4 GHz and num_physical_cores=2 for LNC2: 128 * 128 * 2 FLOP/MAC * 2.4e9 * 2 = 157.2864e12 FLOP/s, rounded here to 158e12. The 79e12 field is therefore a per-physical-core profiler rate, not an architecture-guide ownership claim.",
  "_capacity_comment": "Trainium2 provides 96 GiB of HBM per device. LNC2 exposes four logical ranks per device, so the per-rank budget is 96 GiB / 4 = 24 GiB. HBM capacity is not a per-physical-core property.",
  "_hbm_bw_comment": "A measured trn2 Metadata.parquet reports hbm_ddr_bandwidth=716 GB/s directly for an LNC2 logical rank. Do not derive this by doubling a physical-core value: physical cores expose DMA engines, while the LNC2 HBM rate is bounded by the shared HBM subsystem and is below the aggregate DMA ceiling.",
  "_link_bw_comment": "The link rates are retained only for the planner's analytical collective diagnostic. They are topology/collective-axis contracts, not per-physical-core rates, are not multiplied by lnc, and are outside the certified TE/HBM component floor. Metadata.parquet contains no link-bandwidth field, so 1.28 TB/s is not hardware-certified here.",
  "_topology_comment": "Trainium rack is a torus: TP/CP/PP rings map to mesh axes. Every collective the current cost model charges is a ring. Collective modeling remains a separate analytical diagnostic/contract from the certified TE/HBM component floor.",
  "te_peak_flops": 158000000000000.0,
  "hbm_bw_bytes_per_s": 716000000000.0,
  "link_bw_within_node_bytes_per_s": 1280000000000.0,
  "link_bw_cross_node_bytes_per_s": 400000000000.0,
  "cores_per_node": 64,
  "hbm_capacity_bytes_per_core": 25769803776,
  "topology": "torus",
  "lnc": 2
}
```
