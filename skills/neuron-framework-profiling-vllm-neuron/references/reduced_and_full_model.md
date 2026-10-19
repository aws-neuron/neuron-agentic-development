# Benchmark, reduced→full projection, and A/B

Practices around running the full-depth model, projecting from a cheap
reduced-layer capture, and comparing A/B optimization arms. SKILL.md's
"Benchmark & analyze" summary points here.

## Benchmark the full model

Use `run_bench.sh` (no profiling overhead) to establish baseline throughput:

```bash
./references/run_bench.sh <run_name> [vllm bench throughput args]
```

## Profiling the full model

You **can** profile the full model directly — earlier guidance that "the full
model cannot be profiled" was wrong. On a near-full-HBM model the profiler simply
can't allocate its buffers unless you make room: **lower `--gpu-memory-utilization`**
(≈0.6 dense, ≈0.3 MoE) to free the ~3.4 GB pool + 128 MB/core. See
[serve_profiling.md → Making room for the profiler](serve_profiling.md#making-room-for-the-profiler-on-large-models)
for the full signature to watch for.

## Two-stage profiling: reduced-model first, then full model

Profile in two passes. A **reduced-layer build** (cut the model's layer count to a small
`k` via `--hf-overrides '{"num_hidden_layers": <k>}'` — the layer key is at top level or
under `text_config` / `llm_config` depending on the model) is a cheap first pass that
surfaces every **config-agnostic** problem, followed by the full-depth capture that the SLA
numbers rest on. Keep everything else identical between the two builds (TP/DP, precision, KV
dtype, bucketing) so only depth changes.

**Why it works:** a pathology that lives inside a single layer / kernel / op repeats
identically at any depth. A misbehaving kernel (bad tile size, transpose blowup,
DMA-inefficiency, a NKI kernel silently falling back to a slow path), an attention
mask/positional/GQA-sharding bug, a per-layer collective pattern, or an architectural
compile error (`_deconcat_` SB Memloc, `NCC_INLA001` activation-table-limit) shows up at
`k=2` just as at the full depth — for a fraction of the compile+capture cost. The reduced
pass also **de-risks the profiler tooling itself** (does capture arm, do the `.pb` system
traces flush) before you commit to a long full-depth compile.

**Portable reduced→full** (use the reduced pass FOR these):
- per-kernel correctness & efficiency: tile size, initiation interval, transpose fraction,
  DMA transfer-size efficiency
- a custom/NKI kernel not firing or falling back to a default path
- attention issues (causal/mask, positional/RoPE, GQA/KV sharding pattern)
- the per-layer collective pattern and TE/collective overlap (~0 on SP data deps)
- architectural compile errors independent of depth

**NOT portable** (require the full config — never conclude these from the reduced pass):
- absolute TTFT / ITL / aggregate tok/s
- weight-HBM / KV-HBM totals, HBM-capacity and `num_gpu_blocks` / `max_num_seqs` packing effects
- collective TOTAL wall-time and its fraction of E2E at full depth
- the interior-optimum TP verdict

**Project reduced → full with a two-point linear per-layer model.** Capture at TWO reduced
depths `k1 < k2` (e.g. 2 and 4), read the per-step device time from the live
`nc_exec_running` window at each, then:

```python
per_layer_cost = (t_k2 - t_k1) / (k2 - k1)
fixed_overhead = t_k1 - per_layer_cost * k1          # embeddings, lm_head, launch, one-time CC
t_full_estimate = fixed_overhead + per_layer_cost * L_full   # L_full = real num_hidden_layers
```

The two points separate the depth-invariant intercept from the per-layer slope; a single
reduced depth cannot — it folds fixed overhead into the slope and over-predicts. Fit prefill
and decode separately (different slope and intercept).

**Caveats / guardrails:**
- Changing the layer count changes `num_dynamic_blocks` / bucket geometry → a **different
  compiled graph** that can independently fail to compile; a reduced-model compile success is
  necessary-not-sufficient for the full graph.
- Reduced-model numbers are **estimates for localization, not certification**. Any SLA
  verdict, feasibility resolution, or aggregate-throughput figure must come from the
  full-depth run (and, for aggregates, a concurrent N-replica run).
- Cross-check the projection against the full-depth capture once available; a large
  divergence flags a depth-dependent effect the linear model doesn't capture (bucket-geometry
  graph change, HBM pressure) — investigate rather than trust the projection.

## Plan B: derive latency arithmetically

When device profiling is impossible (e.g. MoE where every capture path fails),
derive per-step decode latency from `vllm bench throughput` tok/s across a batch
sweep (b1/b2/…/b256), and read HBM / scratchpad usage from the compiler memory
tables. This is the fallback when no NTFF can be produced.

## A/B profiling for optimization

When comparing an optimization against a baseline, capture **same-commit** A/B
profiles differing **only** in the gated change (env flag, `--optimization-level`,
etc.), and re-baseline on the same commit. Comparing against a stale baseline
misattributes deltas to the wrong change. Apply statistical significance before
concluding a win.
