# Collective Partitioning

Detailed methodology for step 3 of the profile analysis workflow.

## Overview

The compiled NEFF executes the full forward pass as a single graph. Collectives
are the only synchronization points visible in the profile — they mark boundaries
between logical sections of the model (attention, MLP, embedding, lm_head).

> **Labeling is content-based, not positional.** Segments are assigned to a component
> (attention / MLP / embedding / lm_head / …) from the *profile itself* — the dominant
> kernel source of the Tensor-engine work inside each segment (§3c) — so this workflow is
> model-generic and needs no per-model editing. The `forward()` trace in §3a is an
> *optional* cross-check on the collective count and the attention↔MLP alternation, not the
> source of truth for labels.

## 3a. (Optional) Trace model source for collectives

Walk through `forward()` in `vllm_neuron/model/<model>/model.py` for each mode
(prefill, decode). At each collective call, note:

- Which operation (all_reduce, reduce_scatter, all_gather)
- What it operates on (post-attention hidden states, post-MLP, embedding scatter, etc.)
- The expected data size (e.g. `[B, H] × 2 bytes`)

The expected collective sequence for a standard autoregressive model with TP:

**Decode (no sequence parallelism):**
```
AR[0]: embedding all_reduce
Per layer (×N):
  AR[2i+1]: attention O-proj all_reduce
  AR[2i+2]: MLP all_reduce
AR[final]: lm_head (may be all_gather or part of on-device sampling)
```

**Prefill (with sequence parallelism):**
```
RS[0]: embedding reduce_scatter
Per layer (×N):
  AG[4i+1]: attention all_gather (SP → full)
  RS[4i+2]: attention reduce_scatter (full → SP)
  AG[4i+3]: MLP all_gather (SP → full)
  RS[4i+4]: MLP reduce_scatter (full → SP)
AG[final]: final all_gather before lm_head
```

These patterns vary by model. Always verify against the actual source code.

## 3b. Load CcOp and verify

```python
ccop = pd.read_parquet(f"{d}/CcOp.parquet")
valid_ccop = ccop[ccop['operation'] != 'Invalid'].sort_values('start_ts').reset_index(drop=True)
print(f"Valid collectives: {len(valid_ccop)}")
print(valid_ccop[['operation', 'start_ts', 'end_ts', 'input_size', 'output_size']].head(20))
```

Compare the count against what the code trace predicts. If there's a mismatch:
- XLA may trace both branches of a Python conditional (dead-code collectives)
- The model may have collectives you missed (e.g. norm-related gathers)

Use `input_size`/`output_size` to verify the mapping (e.g. an AllReduce of
43,008 bytes = B × H × 2 = 4 × 5376 × 2 confirms it's a hidden-state reduction).

## 3c. Measure and label per-segment

**Step window.** Prefer `[t0_step, t1_step]` from the live `nc_exec_running` event for the
step (Step 1e) — the authoritative device window, always available. When the capture was
taken with DGE notifications on, `dma_agg['start_ts'].min()/max()` is an equally valid
window. If neither is available, fall back to
`Instruction['start_ts'].min()/end_ts.max()`. (See the window options in the code below.)

**Tables.** `active` below is the per-engine active-interval table if your ingest has one;
on a fresh ingest that lacks it, derive per-engine active-wall by interval-merging
`Instruction` rows grouped by `engine` (merge over BOTH pcores for LNC=2). Per-segment HBM
bytes come from `Instruction['hbm_read_bytes']` (always present). `DmaPacket` /
`DmaPacketAggregated` MB is available only when DGE notifications were on at capture — guard
the `dma_mb` / `dma_us` columns on its presence and omit them otherwise.

Define segments as intervals between consecutive collectives:

```python
def interval_merge_ns(starts, ends, t0, t1):
    starts = np.maximum(starts, t0)
    ends = np.minimum(ends, t1)
    valid = starts < ends
    starts, ends = starts[valid], ends[valid]
    if len(starts) == 0:
        return 0
    order = np.argsort(starts)
    total, cur_end = 0, 0
    for s, e in zip(starts[order], ends[order]):
        if s >= cur_end:
            total += e - s
            cur_end = e
        elif e > cur_end:
            total += e - cur_end
            cur_end = e
    return total

def engine_active_ns(active_df, engine_name, t0, t1):
    eng = active_df[active_df['engine'].str.lower() == engine_name.lower()]
    return interval_merge_ns(eng['start_ts'].values, eng['end_ts'].values, t0, t1)

# Preferred: the live nc_exec_running step window (Step 1e) — always valid.
exec_start, exec_end = t0_step, t1_step
# Option (DGE-on captures, where DmaPacketAggregated is populated):
#   exec_start = int(dma_agg['start_ts'].min()); exec_end = int(dma_agg['end_ts'].max())
# Fallback (no system trace and no DGE):
#   exec_start = int(inst['start_ts'].min()); exec_end = int(inst['end_ts'].max())

# Build segment boundaries from CcOp
boundaries = [exec_start]
for _, row in valid_ccop.iterrows():
    boundaries.append(int(row['start_ts']))
    boundaries.append(int(row['end_ts']))
boundaries.append(exec_end)

# Segments alternate: compute, collective, compute, collective, ...
segments = []
for i in range(0, len(boundaries) - 1, 2):
    t0, t1 = boundaries[i], boundaries[i+1]
    if t1 <= t0:
        continue

    kernel_pkts = dma_pkts[(dma_pkts['queue_type'] != 'instruction') & (dma_pkts['transfer_bytes'] > 4)]
    window_pkts = kernel_pkts[(kernel_pkts['start_ts'] >= t0) & (kernel_pkts['start_ts'] < t1)]
    dma_mb = window_pkts['transfer_bytes'].sum() / 1e6

    matmuls = inst[(inst['engine'] == 'Tensor') & (inst['opcode'] == 'MATMUL')
                   & (inst['start_ts'] >= t0) & (inst['start_ts'] < t1)]
    gflops = matmuls['adjusted_flops'].sum() / 1e9

    segments.append({
        'dur_us': (t1 - t0) / 1e3,
        'dma_us': interval_merge_ns(dma_pkts['start_ts'].values, dma_pkts['end_ts'].values, t0, t1) / 1e3,
        'te_us': engine_active_ns(active, 'tensor', t0, t1) / 1e3,
        'vec_us': engine_active_ns(active, 'vector', t0, t1) / 1e3,
        'sc_us': engine_active_ns(active, 'scalar', t0, t1) / 1e3,
        'gp_us': engine_active_ns(active, 'gpsimd', t0, t1) / 1e3,
        'dma_mb': dma_mb,
        'gflops': gflops,
    })
```

### Label each segment by component — source + IR debug info as ground truth

A segment's component (attention / MLP / MoE / embedding / lm_head / norm / sample) is
decided by *what ran in it*, resolved from the compiler/framework debug info baked into the
NEFF and — where the debug info is ambiguous — from the model's own source code. Never use a
hardcoded per-model position map, and never trust a bare backend/kernel name token (a
decode/prefill backend token such as a `*_tkg`/`*_cte` suffix appears in *both* attention and
MLP paths, so substring-matching it mislabels the split).

**The three attribution columns on `Instruction`, in decreasing authority:**

1. **`layer`** — the framework **module scope** path, e.g.
   `…/<AttentionModule>[self_attn]/…` or `…/<MLPModule>[mlp]/…`. It *names* the owning
   `nn.Module`, so it is the strongest, most model-generic signal. Populated only when
   `NEURON_FRAMEWORK_DEBUG=1` was set at compile; otherwise often `"Unknown"`.
2. **`nki_source_location` / `bir_debug_info_source_location`** — the source `file:line` of
   the kernel or framework op (needs `XLA_IR_DEBUG=1` / `XLA_HLO_DEBUG=1` at compile). NKI
   kernel paths carry the component in the *directory* (`…/nki/kernels/<component>/…`,
   `…/nkilib/core/<component>/…`); framework ops point into the model's `model.py`.
3. **`hlo_name`** — last resort for ops with no source line.

**Ground-truth procedure (deterministic and auditable):**

1. If `layer` is populated, read the component from its module-scope path — done.
2. Else group the segment's Tensor-engine time by unique `file:line` and sort by time. For
   each top contributor:
   - **NKI kernel path** → component = the kernel directory (`/attention/`, `/mlp/`,
     `/moe/`, `output_projection`, `embedding`, `sampling`). Unambiguous.
   - **Framework `model.py:<line>`** → open the model source at that line ONCE and read what
     the op computes; the enclosing `nn.Module`/function is the component (a QKV/O matmul →
     attention; a gate/up/down matmul → MLP; a vocab matmul → lm_head; an RMSNorm → norm).
3. Record the resolved `{file:line → component}` map (derived from *this* model's source, so
   correct by construction) and apply it to every instruction. It is explicit and reviewable
   — no guessing on shared backend tokens.

```python
# Authority order: framework module scope > resolved source line > NKI kernel directory.
SRC = ['layer', 'bir_debug_info_source_location', 'nki_source_location', 'hlo_name']

# Filled ONCE from step 2 by reading the model source at the file:line the IR points to.
# Model-specific by construction (read from this model's code) — NOT a guessed heuristic.
FRAMEWORK_LINE_MAP = {}   # e.g. {'model.py:1930': 'attention', 'model.py:3031': 'sample'}

def component_of(row):
    # 1. framework module scope (`layer`) names the nn.Module directly — strongest signal
    ls = str(row.get('layer') or '').lower()
    if ls and ls != 'unknown':
        for key, comp in (('norm','norm'), ('attention','attention'), ('attn','attention'),
                         ('mlp','mlp'), ('moe','mlp'), ('expert','mlp'),
                         ('lm_head','lm_head'), ('embed','embed')):
            if key in ls:
                return comp
    src = ' '.join(str(row.get(c) or '') for c in SRC).lower()
    # 2. framework source lines resolved against model source (step 2)
    for key, comp in FRAMEWORK_LINE_MAP.items():
        if key.lower() in src:
            return comp
    # 3. NKI kernel DIRECTORY is the component (unambiguous path segment)
    if '/mlp/' in src or '/moe/' in src or 'expert' in src:
        return 'norm' if 'norm' in src else 'mlp'
    if '/attention/' in src or 'output_projection' in src or 'oproj' in src or 'qkv' in src:
        return 'attention'
    if 'lm_head' in src: return 'lm_head'
    if 'sampl' in src or 'topk' in src or 'logit' in src or 'softcap' in src:
        return 'sample'
    if 'embed' in src: return 'embed'
    if 'norm' in src: return 'norm'
    return 'other'

te = inst[inst['engine'] == 'Tensor']
def label_segment(t0, t1):
    w = te[(te['start_ts'] >= t0) & (te['start_ts'] < t1)].copy()
    if len(w) == 0:
        return 'other'
    w['component'] = w.apply(component_of, axis=1)
    return w.groupby('component')['duration_ns'].sum().idxmax()   # dominant kernel wins
```

**How to build `FRAMEWORK_LINE_MAP` (the one-time source read):**

```python
# Rank the framework (non-NKI) source lines that carry real Tensor-engine time, then open
# each in the model source and classify by the enclosing module/op. This is a read of the
# model's own code — the source of truth — not a substring guess.
fw = te[te['bir_debug_info_source_location'].fillna('').str.contains('/model/')]
print(fw.groupby('bir_debug_info_source_location')['duration_ns'].sum()
        .sort_values(ascending=False).head(10) / 1e3)   # us; inspect each file:line once
```

**Validate:** with a correct map, labeled segments alternate the way §3a predicts and the
attention/MLP segment counts each equal the layer count (± boundary segments). If a large
share lands in `other`, resolve those segments' top `file:line`s against the source before
trusting the roll-up (verification rule 7).

## 3d. Report

Present as a per-segment table:

| Segment | dur (us) | DMA (us) | TE (us) | Vec (us) | Sc (us) | GP (us) | MB | GFLOPS |
|---------|----------|----------|---------|----------|---------|---------|------|--------|
| Embedding | ... | ... | ... | ... | ... | ... | ... | ... |
| AR[0] | ... | ... | ... | ... | ... | ... | ... | ... |
| L0 attn | ... | ... | ... | ... | ... | ... | ... | ... |
| AR[1] | ... | ... | ... | ... | ... | ... | ... | ... |
| L0 MLP | ... | ... | ... | ... | ... | ... | ... | ... |
| AR[2] | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | | | | | | | | |

Plus a time allocation summary:

| Category | us | % of total |
|----------|------|------------|
| Attention (×N layers) | ... | ... |
| MLP (×N layers) | ... | ... |
| Final (lm_head + sampling) | ... | ... |
| Collectives | ... | ... |
| Embedding | ... | ... |
