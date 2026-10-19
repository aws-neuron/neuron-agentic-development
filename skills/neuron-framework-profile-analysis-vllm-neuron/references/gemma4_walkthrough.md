# Gemma4 31B Profile Analysis Walkthrough

Continues from the profiling skill's [gemma4_walkthrough.md](../../neuron-framework-profiling-vllm-neuron/references/gemma4_walkthrough.md).

Input: `$NEURON_PROJECT_ROOT/agent_artifacts/profiles/gemma4_kvfp8_bs64_4layer/` (4-layer, kv-fp8, BS64, 8192 blocks, TP16)

## Step 1a: Stage minimal capture directory

```bash
export NEURON_PROJECT_ROOT=~/GemmaTune
PROFILE_DIR=$NEURON_PROJECT_ROOT/agent_artifacts/profiles/gemma4_kvfp8_bs64_4layer
FIRST_WORKER=$(find $PROFILE_DIR -mindepth 1 -maxdepth 1 -type d -name "i-*" | sort | head -1)
SESSION_DIR=$(find $FIRST_WORKER -mindepth 1 -maxdepth 1 -type d)

mkdir -p $PROFILE_DIR/capture_minimal
cp $SESSION_DIR/ntrace.pb $SESSION_DIR/cpu_util.pb $SESSION_DIR/host_mem.pb $SESSION_DIR/trace_info.pb $PROFILE_DIR/capture_minimal/
cp $SESSION_DIR/*.neff $PROFILE_DIR/capture_minimal/
for neff_dir in $PROFILE_DIR/capture/*/; do
    NEFF_ID=$(basename "$neff_dir")
    mkdir -p $PROFILE_DIR/capture_minimal/$NEFF_ID
    cp "$neff_dir/profile_rank_0_exec_2.ntff" $PROFILE_DIR/capture_minimal/$NEFF_ID/
done
```

**Output:**
```
capture_minimal/
├── ntrace.pb
├── cpu_util.pb
├── host_mem.pb
├── trace_info.pb
├── neff_315622240876207_vnc_0.neff
├── neff_374305408507786_vnc_0.neff
├── 315622240876207/
│   └── profile_rank_0_exec_2.ntff
└── 374305408507786/
    └── profile_rank_0_exec_2.ntff
```

## Step 1b: Ingest

```bash
NE=$NEURON_PROJECT_ROOT/agent_artifacts/data/gemma4-kvfp8-bs64-4layer   # --data-path: parquet joins the run's evidence trail
neuron-explorer view \
    -d $PROFILE_DIR/capture_minimal \
    --data-path $NE \
    --display-name gemma4-kvfp8-bs64-4layer \
    --ingest-only
```

**Output** at `$NE/profiles/global/`:
```
gemma4-kvfp8-bs64-4layer@latest                                          ← system profile
gemma4-kvfp8-bs64-4layer_i-0e1d1489ce83f0eff_pid_1834775_nc_0_session_0@latest  ← NEFF 315622240876207
gemma4-kvfp8-bs64-4layer_i-0e1d1489ce83f0eff_pid_1834995_nc_0_session_0@latest  ← NEFF 374305408507786
```

## Profile Analysis
### 1. Session Workload

#### 1a. Load and filter

```python
import pandas as pd, numpy as np, os, json
from itertools import groupby

NE = os.path.join(os.environ.get("NEURON_PROJECT_ROOT", os.getcwd()),
                  "agent_artifacts/data/gemma4-kvfp8-bs64-4layer/profiles/global")   # the --data-path used at ingest
sys_d = f"{NE}/gemma4-kvfp8-bs64-4layer@latest"

events = pd.read_parquet(f"{sys_d}/SystemProfileEvents.parquet")
nc_exec = events[events['name'] == 'nc_exec_running'].copy()
nc_exec['model_id'] = nc_exec['extra_attributes_json'].apply(
    lambda x: json.loads(x).get('model_id') if pd.notna(x) else None)

pcore = nc_exec['pcore_idx'].min()
pc = nc_exec[nc_exec['pcore_idx'] == pcore].sort_values('start_ts').reset_index(drop=True)

print(f"nc_exec_running events: {len(pc)} (pcore {pcore})")
print(f"Distinct model_ids: {pc['model_id'].nunique()}")
print(f"Total span: {(pc.iloc[-1]['end_ts'] - pc.iloc[0]['start_ts'])/1e9:.1f} s")
```

**Result:**
```
nc_exec_running events: 578 (pcore 4)
Distinct model_ids: 2
Total span: 18.2 s
```

### 1b. Identify NEFFs

```python
id_map = {}
for mid in pc['model_id']:
    if mid not in id_map:
        id_map[mid] = chr(ord('A') + len(id_map))

for mid, grp in pc.groupby('model_id'):
    dur_us = grp['duration_ns'] / 1e3
    print(f"{id_map[mid]} ({mid}): count={len(grp)}, mean={dur_us.mean():.0f} us, total={dur_us.sum()/1e3:.1f} ms")
```

**Result:**
```
A (374305408507786): count=65, mean=6750 us, total=438.7 ms
B (315622240876207): count=513, mean=11053 us, total=5670.4 ms
```
Since we are using a batch size of 64 with 1 request per prefill batch, 
we can infer that A is the prefill NEFF while the ~512 NEFF b executions correspond
to each output token generated (decode).  
### 1c. Execution pattern

```python
pattern = [id_map[row['model_id']] for _, row in pc.iterrows()]
runs = [(k, len(list(g))) for k, g in groupby(pattern)]
print(", ".join(f"{label}×{count}" for label, count in runs))
```

**Result:**
```
A×1, B×1, A×1, B×1, A×63, B×511
```

### 1d. Gaps

```python
pc['label'] = pc['model_id'].map(id_map)
pc['next_start'] = pc['start_ts'].shift(-1)
pc['gap_ns'] = pc['next_start'] - pc['end_ts']
pc['next_label'] = pc['label'].shift(-1)
pc['transition'] = pc['label'] + '→' + pc['next_label'].fillna('')

for trans in sorted(pc['transition'].unique()):
    if trans.endswith('→'):
        continue
    subset = pc[pc['transition'] == trans]
    gap_mean = subset['gap_ns'].mean() / 1e3
    dur_mean = subset['duration_ns'].mean() / 1e3
    print(f"{trans}: n={len(subset)}, gap_mean={gap_mean:.0f} us, gap/dur={gap_mean/dur_mean:.4f}")
```

**Result:**
```
A→A: n=62, gap_mean=2582 us, gap/dur=0.3827
A→B: n=3, gap_mean=1162763 us, gap/dur=171.2604
B→A: n=2, gap_mean=928495 us, gap/dur=96.0132
B→B: n=510, gap_mean=12903 us, gap/dur=1.1664
```

### 1e. Throughput estimate

| Metric | From trace | bench_results.json |
|--------|-----------|-------------------|
| Output tok/s | 64 / 11,053 us = 5,790 | 2,537 |

### 2. Performance Bounds

Hardware: trn2, hbm_ddr_bandwidth = 716 GB/s, TE peak = 157.3 TFLOPS (LNC=2).

#### NEFF B: 315622240876207 (decode)

##### Bounds

| Bound | Value (us) |
|-------|-----------|
| total_time | 9,724 |
| memory_bound | 4,772 |
| memory_bound_ideal | 1,541 |
| compute_bound | 1,826 |
| compute_bound_ideal | 685 |
| compute_no_transposes | 327 |
| perfect_pipeline (GpSimd) | 6,716 |

##### Engine active times

| Engine | Active time |
|--------|------------|
| GpSimd | 6,716 us (69.1%) |
| DMA | 4,772 us (49.1%) |
| Tensor | 1,826 us (18.8%) |
| Scalar | 1,742 us (17.9%) |
| Vector | 505 us (5.2%) |

Bottleneck: GpSimd

| Metric | Value |
|--------|-------|
| DMA transfer bytes | 670 MB |
| hw_flops | 53.9 GFLOPS |
| transpose_flops | 28.1 GFLOPS (52.3% of hw_flops) |

##### Memory family gaps

| Gap | Value (us) | % of total |
|-----|-----------|------------|
| DMA idle | 4,952 | 50.9% |
| DMA inefficiency | 3,231 | 33.2% |

##### Compute family gaps

| Gap | Value (us) | % of total |
|-----|-----------|------------|
| TE idle | 7,898 | 81.2% |
| TE underutil | 1,141 | 11.7% |
| Transpose | 358 | 3.7% (52.3% of flops) |

#### NEFF A: 374305408507786 (prefill)

##### Bounds

| Bound | Value (us) |
|-------|-----------|
| total_time | 6,539 |
| memory_bound | 4,103 |
| memory_bound_ideal | 3,049 |
| compute_bound | 3,049 |
| compute_bound_ideal | 3,932 |
| compute_no_transposes | 3,256 |
| perfect_pipeline (DMA) | 4,103 |

##### Engine active times

| Engine | Active time |
|--------|------------|
| DMA | 4,103 us (62.8%) |
| Tensor | 3,049 us (46.6%) |
| GpSimd | 1,960 us (30.0%) |
| Scalar | 1,916 us (29.3%) |
| Vector | 1,656 us (25.3%) |

Bottleneck: DMA

| Metric | Value |
|--------|-------|
| DMA transfer bytes | 1,326 MB |
| hw_flops | 309.2 GFLOPS |
| transpose_flops | 53.2 GFLOPS (17.2% of hw_flops) |

##### Memory family gaps

| Gap | Value (us) | % of total |
|-----|-----------|------------|
| DMA idle | 2,436 | 37.3% |
| DMA inefficiency | 1,054 | 16.1% |

##### Compute family gaps

| Gap | Value (us) | % of total |
|-----|-----------|------------|
| TE idle | 3,490 | 53.4% |
| TE underutil | -883 | — |
| Transpose | 676 | 10.3% (17.2% of flops) |

---

### 3. Collective Partitioning

#### 3a. Code trace

**Decode NEFF (B = 315622240876207):**

```
Embedding: embed_tokens(scatter_tokens=False) → all_reduce [64×5376×2 = 688,128 bytes]

Per layer (×4, all sliding_attention):
  Attention: QKV, RoPE, KV cache, SDPA, O-proj → all_reduce [688,128 bytes]
  MLP: NF.mlp(gate, up, down) → all_reduce [688,128 bytes]

Final: norm → index_select → lm_head → softcap → sampling
  all_gather [256 → 4096] (×2, sampling-related)
  all_gather [4,194,304 → 67,108,864] (logits gather: [64, 16384] fp32 → [64, 262144] fp32)
```

Expected: 9 AllReduces + 2 small AllGathers + 1 logits AllGather = 12 collectives.

**Prefill NEFF (A = 374305408507786):**

```
Embedding: embed_tokens(scatter_tokens=True) → reduce_scatter [11,010,048 → 688,128]

Per layer (×4):
  Attention: all_gather [688,128 → 11,010,048] → compute → reduce_scatter [11,010,048 → 688,128]
  MLP: all_gather [688,128 → 11,010,048] → compute → reduce_scatter [11,010,048 → 688,128]

Final: all_gather [688,128 → 11,010,048] → norm → index_select → lm_head → sampling
  all_gather [4 → 64] (×2, sampling-related)
  all_gather [65,536 → 1,048,576] (logits gather: [1, 16384] fp32 → [1, 262144] fp32)
```

Expected: 9 ReduceScatter + 9 AllGather + 2 small AllGather + 1 logits AllGather = 21 collectives.

#### 3b. Verify against CcOp

```python
# Decode NEFF
d = f"{NE}/gemma4-kvfp8-bs64-4layer_i-0e1d1489ce83f0eff_pid_1834775_nc_0_session_0@latest"
ccop = pd.read_parquet(f"{d}/CcOp.parquet")
valid = ccop[ccop['operation'] != 'Invalid'].sort_values('start_ts').reset_index(drop=True)
print(f"Decode CcOps: {len(valid)}")
print(valid[['operation', 'input_size', 'output_size']].to_string())
```

**Result (decode):** 12 collectives — 9 AllReduce (688,128 B) + 2 AllGather (256→4096) + 1 AllGather (4,194,304→67,108,864). ✓

```python
# Prefill NEFF
d = f"{NE}/gemma4-kvfp8-bs64-4layer_i-0e1d1489ce83f0eff_pid_1834995_nc_0_session_0@latest"
ccop = pd.read_parquet(f"{d}/CcOp.parquet")
valid = ccop[ccop['operation'] != 'Invalid'].sort_values('start_ts').reset_index(drop=True)
print(f"Prefill CcOps: {len(valid)}")
print(valid[['operation', 'input_size', 'output_size']].to_string())
```

**Result (prefill):** 21 collectives — 9 ReduceScatter + 9 AllGather (SP pattern) + 2 AllGather (small) + 1 AllGather (logits). ✓

#### 3c. Algorithmic minimum load per section (decode)

Constants: B=64, H=5376, D_swa=256, Nh=2/rank, Nkv=1/rank, block_size=16,
max_blocks_per_seq=64, I_per_rank=1344, V_per_rank=16384, KV cache dtype=fp8.

**Attention (per layer):**

| Data | Size | Notes |
|------|------|-------|
| W_qkv [5376, 1024] bf16 | 11.01 MB | (Nh+2×Nkv)×D = 1024 |
| W_o [512, 5376] bf16 | 5.51 MB | Nh×D = 512 |
| KV cache read (K+V) fp8 | 33.55 MB | B×64×Nkv×16×256×1B×2 |
| KV cache write fp8 | 0.03 MB | B×1×Nkv×1×256×1B×2 |
| Activations in+out bf16 | 1.38 MB | [B,H]×2B×2 |
| Norm weights (3) | 0.03 MB | |
| **Total** | **51.51 MB** | |

**MLP (per layer):**

| Data | Size | Notes |
|------|------|-------|
| W_gate [5376, 1344] bf16 | 14.45 MB | |
| W_up [5376, 1344] bf16 | 14.45 MB | |
| W_down [1344, 5376] bf16 | 14.45 MB | |
| Activations in+out bf16 | 1.38 MB | |
| Norm weights (2) | 0.02 MB | |
| **Total** | **44.75 MB** | |

**Final (lm_head):**

| Data | Size | Notes |
|------|------|-------|
| lm_head weight [5376, 16384] bf16 | 176.16 MB | Full vocab shard |
| Input [B, H] bf16 | 0.69 MB | |
| Output [B, V_per_rank] fp32 | 4.19 MB | |
| **Total** | **181.04 MB** | |

**Summary (4 layers):**

| Section | Minimum (MB) | Ideal at 716 GB/s (us) | Measured (us) | Ratio |
|---------|-------------|----------------------|---------------|-------|
| Attention (×4) | 206.04 | 288 | 1,361 | 4.7× |
| MLP (×4) | 179.00 | 250 | 673 | 2.7× |
| Final | 181.04 | 253 | 6,859 | 27.1× |
| Embedding | 0.69 | 1 | 340 | — |
| **Total** | **566.77** | **792** | **9,724** | **12.3×** |

Measured total DMA: 670 MB (1.18× algorithmic minimum).

#### 3d. Per-segment breakdown

Apply the collective-partitioning workflow in
[collective-partitioning.md](collective-partitioning.md): segment the step at the collective
boundaries, label each segment by its dominant kernel source (§3c), and roll up into the
per-component table (attention / MLP / collective / other) with per-component HBM read and
memory floor.
