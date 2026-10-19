# Profile Analysis: Gemma4 31B, 4 layers, TP16

**Config:** 4 layers, TP16, max_model_len=1024, max_num_seqs=64, kv-fp8, num_gpu_blocks_override=8192
**Hardware:** trn2.48xlarge, LNC=2

---

## 0. Hardware Constants

| Constant | Value |
|----------|-------|
| HBM BW | 716 GB/s |
| TE peak | 157.3 TFLOPS |
| num_physical_cores | 2 |

## 1. Session Workload

| Metric | Value |
|--------|-------|
| Total wall clock | 18.2 s |
| Total executions | 578 |
| Distinct NEFFs | 2 |
| Model switches | 12 |

### NEFFs identified

| NEFF | model_id | Role | Count | Mean dur (us) | Total (ms) |
|------|----------|------|-------|---------------|------------|
| A | 374305408507786 | prefill | 65 | 6,750 | 438.7 |
| B | 315622240876207 | decode | 513 | 11,053 | 5,670.4 |

### High-level timeline

| Phase | Duration | Dispatches |
|-------|----------|------------|
| A×1 | 7 ms | 1 prefill |
| gap | 3.4 s | |
| B×1 | 10 ms | 1 decode |
| gap | 1.9 s | |
| A×1 | 7 ms | 1 prefill |
| B×1 | 11 ms | 1 decode |
| A×63 | ~600 ms | 63 prefills |
| B×511 | ~12.3 s | 511 decodes |

### Execution pattern

`A×1, B×1, A×1, B×1, A×63, B×511`

### Inter-execution gaps (steady state)

| Transition | Count | Gap mean (us) | Gap/dur ratio |
|-----------|-------|---------------|---------------|
| A→A | 62 | 2,582 | 0.38 |
| A→B | 3 | 1,162,763 | 171.26 |
| B→A | 2 | 928,495 | 96.01 |
| B→B | 510 | 12,903 | 1.17 |

### Throughput estimate from trace

| Metric | From trace | bench_results.json |
|--------|-----------|-------------------|
| Output tok/s | 64 / 11,053 us = 5,790 | 2,537 |

---

## 2. Performance Bounds

### NEFF B: 315622240876207 (decode)

#### Bounds

| Bound | Value (us) |
|-------|-----------|
| total_time | 9,724 |
| memory_bound | 4,772 |
| memory_bound_ideal | 936 |
| compute_bound | 1,826 |
| compute_bound_ideal | 342 |
| compute_no_transposes | 163 |
| perfect_pipeline (GpSimd) | 6,716 |

#### Engine active times

| Engine | Active time |
|--------|------------|
| GpSimd | 6,716 us (69.1%) |
| DMA | 4,772 us (49.1%) |
| Tensor | 1,826 us (18.8%) |
| Scalar | 1,742 us (17.9%) |
| Vector | 505 us (5.2%) |

Bottleneck: GpSimd

#### Memory family gaps

| Gap | Value (us) | % of total |
|-----|-----------|------------|
| DMA idle | 4,952 | 50.9% |
| DMA inefficiency | 3,836 | 39.4% |

#### Compute family gaps

| Gap | Value (us) | % of total |
|-----|-----------|------------|
| TE idle | 7,899 | 81.2% |
| TE underutil | 1,483 | 15.3% |
| Transpose | 179 | 1.8% (52.3% of flops) |

#### Key metrics

| Metric | Value |
|--------|-------|
| DMA transfer bytes | 670 MB |
| Achieved BW | 140 GB/s (20% of peak) |
| hw_flops | 53.9 GFLOPS |
| transpose_flops | 28.1 GFLOPS (52.3%) |

---

### NEFF A: 374305408507786 (prefill)

#### Bounds

| Bound | Value (us) |
|-------|-----------|
| total_time | 6,539 |
| memory_bound | 4,103 |
| memory_bound_ideal | 1,852 |
| compute_bound | 3,049 |
| compute_bound_ideal | 1,966 |
| compute_no_transposes | 1,628 |
| perfect_pipeline (DMA) | 4,103 |

#### Engine active times

| Engine | Active time |
|--------|------------|
| DMA | 4,103 us (62.8%) |
| Tensor | 3,049 us (46.6%) |
| GpSimd | 1,960 us (30.0%) |
| Scalar | 1,916 us (29.3%) |
| Vector | 1,656 us (25.3%) |

Bottleneck: DMA

#### Memory family gaps

| Gap | Value (us) | % of total |
|-----|-----------|------------|
| DMA idle | 2,436 | 37.2% |
| DMA inefficiency | 2,251 | 34.4% |

#### Compute family gaps

| Gap | Value (us) | % of total |
|-----|-----------|------------|
| TE idle | 3,490 | 53.4% |
| TE underutil | 1,083 | 16.6% |
| Transpose | 338 | 5.2% (17.2% of flops) |

#### Key metrics

| Metric | Value |
|--------|-------|
| DMA transfer bytes | 1,326 MB |
| Achieved BW | 323 GB/s (45% of peak) |
| hw_flops | 309.2 GFLOPS |
| transpose_flops | 53.2 GFLOPS (17.2%) |

---

## 3. Collective Partitioning — Decode NEFF

### Code trace (decode, 4 layers)

```
Embedding → AR[0]
Per layer (×4): attn → AR[2i+1], MLP → AR[2i+2]
Final (lm_head + sampling) → AG[9], AG[10], AG[11] (logits gather)
```

Expected: 9 AllReduce + 3 AllGather = 12 collectives. Verified: 12. ✓

### Per-segment breakdown

| Segment | dur (us) | DMA (us) | TE (us) | GP (us) | MB | GFLOPS |
|---------|----------|----------|---------|---------|------|--------|
| Embedding | 340 | 75 | 2 | 81 | 16.1 | 0.02 |
| L0 attn | 328 | 107 | 155 | 128 | 20.5 | 1.64 |
| L0 MLP | 161 | 123 | 50 | 34 | 56.1 | 3.04 |
| L1 attn | 345 | 149 | 160 | 170 | 29.1 | 1.67 |
| L1 MLP | 163 | 126 | 58 | 32 | 56.1 | 3.04 |
| L2 attn | 344 | 148 | 162 | 171 | 29.1 | 1.67 |
| L2 MLP | 160 | 123 | 55 | 31 | 56.1 | 3.04 |
| L3 attn | 344 | 151 | 167 | 175 | 29.1 | 1.67 |
| L3 MLP | 189 | 152 | 59 | 44 | 73.9 | 2.99 |
| Final | 6,139 | 2,770 | 947 | 5,819 | 165.0 | 34.56 |
| Post-logits | 720 | 682 | 9 | 21 | 138.4 | 0.54 |
| Collectives (×12) | 491 | — | — | — | — | — |

### Time allocation

| Category | us | % of total |
|----------|------|------------|
| Final (lm_head + sampling + logits) | 6,859 | 70.5% |
| Attention (×4 layers) | 1,361 | 14.0% |
| MLP (×4 layers) | 673 | 6.9% |
| Collectives | 491 | 5.0% |
| Embedding | 340 | 3.5% |

---

## 4. Hotspot: Final section (6,859 us = 70.5% of decode)

The final section (lm_head + on-device sampling + logits all_gather) dominates
decode time. GpSimd is the bottleneck engine at 5,819 us (84.7% of the section).

This section includes:
- `lm_head`: `[64, 5376] × [5376, 16384]` matmul (34.56 GFLOPS)
- `softcap`: `tanh(logits / 30.0) * 30.0`
- On-device sampling (argmax over 16384 vocab per rank)
- AllGather of logits: 4,194,304 → 67,108,864 bytes (434 us)

The GpSimd dominance suggests on-device sampling or the logit softcapping
operation is the primary cost, not the lm_head matmul itself.

---

## 5. Full model benchmark

| Config | Output tok/s |
|--------|-------------|
| 4-layer (profiled) | 2,537 |
| 60-layer (full model) | 451 |

Estimated 60-layer decode step time from per-layer costs:
- Per layer (attn + MLP + collectives): ~490 us × 60 = 29,400 us
- Final section: ~6,859 us (constant across layer count)
- Embedding: ~340 us (constant)
- **Estimated total: ~36,600 us = 27.3 ms per decode step**
- **Estimated output tok/s: 64 / 36,600 us = 1,749 tok/s**

Measured: 451 tok/s. The 3.9× discrepancy suggests significant overhead not
captured in the 4-layer profile (likely num_gpu_blocks_override scaling with
60 layers, or scheduling gaps that scale with step count).
