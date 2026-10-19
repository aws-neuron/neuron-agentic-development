# System Trace Queries

Detailed queries for step 1 of the profile analysis workflow.

## 1a. Load and filter nc_exec_running events

Each NEFF dispatch appears as one `nc_exec_running` event per physical core.
Filter to one pcore to get one row per dispatch. Extract `model_id` to
distinguish between different compiled graphs.

```python
import json
from itertools import groupby

events = pd.read_parquet(f"{sys_d}/SystemProfileEvents.parquet")
nc_exec = events[events['name'] == 'nc_exec_running'].copy()
nc_exec['model_id'] = nc_exec['extra_attributes_json'].apply(
    lambda x: json.loads(x).get('model_id') if pd.notna(x) else None)

# Deduplicate to one pcore (LNC=2 produces 2 rows per dispatch)
pcore = nc_exec['pcore_idx'].min()
pc = nc_exec[nc_exec['pcore_idx'] == pcore].sort_values('start_ts').reset_index(drop=True)
```

## 1b. Identify NEFFs and per-NEFF stats

Label each distinct `model_id` (A, B, ...) in order of first appearance.
Report count, mean duration, and total time per NEFF. The count and duration
distinguish roles.

```python
id_map = {}
for mid in pc['model_id']:
    if mid not in id_map:
        id_map[mid] = chr(ord('A') + len(id_map))

for mid, grp in pc.groupby('model_id'):
    dur_us = grp['duration_ns'] / 1e3
    print(f"{id_map[mid]} ({mid}): count={len(grp)}, mean={dur_us.mean():.0f} us, total={dur_us.sum()/1e3:.1f} ms")
```

## 1c. Execution pattern

Show the dispatch sequence as contiguous runs. This reveals the workload
structure — initial dispatches (warmup/compilation), then the steady-state
pattern of the benchmark.

```python
pattern = [id_map[row['model_id']] for _, row in pc.iterrows()]
runs = [(k, len(list(g))) for k, g in groupby(pattern)]
print("Pattern:", ", ".join(f"{label}×{count}" for label, count in runs))
```

## 1d. Inter-execution gaps by transition type

Measure the time between the end of one dispatch and the start of the next,
grouped by transition (A→A, A→B, B→B, etc.). The gap/duration ratio shows
how much time is spent in scheduling overhead vs execution. Large gaps between
same-NEFF transitions (e.g. B→B) indicate host-side bottlenecks.

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

## 1e. Throughput estimate from trace

Per-step device time is the steady-state `nc_exec_running` window (§1f). From it:

- **Decode output tok/s** ≈ `steady_state_batch_size / decode_step_device_time_s` (one token
  per sequence per decode step; use the median steady-state window, excluding
  warmup/compile outliers).
- **Prefill TTFT** ≈ the prefill `nc_exec_running` window (plus host overhead not captured
  in the device trace — see the device-vs-E2E rule).
- Cross-check against `bench_result_json` (`output_throughput`, `total_token_throughput`);
  the trace estimate is device-only and runs slightly ahead of the end-to-end bench number
  by the host overhead.

## 1f. Collective wall-time inside the execution window (collective-bound detection)

The single most useful cross-check for a TP>1 workload: how much of each `nc_exec_running`
window is actually spent in collectives. Query `cc_running` events that fall inside one
steady-state `nc_exec_running` window and overlap-merge their intervals (they may be many
small ops).

```python
ev = pd.read_parquet(f"{sys_d}/SystemProfileEvents.parquet")

# One steady-state device-execution window (exclude warmup/compile outliers)
nc = ev[(ev.name == 'nc_exec_running') & (ev.trace_event_source == 'neuron_hw')].copy()
nc['dur_ms'] = nc.duration_ns / 1e6
w = nc[(nc.dur_ms > 15) & (nc.dur_ms < 22)].sort_values('start_ts').iloc[0]  # tune bounds
t0, t1 = w.start_ts, w.end_ts

cc = ev[(ev.name == 'cc_running') & (ev.start_ts >= t0) & (ev.end_ts <= t1)
        & (ev.process_id == w.process_id)]

def union_ms(starts, ends):
    iv = sorted(zip(starts, ends)); tot = 0; cs = ce = None
    for s, e in iv:
        if cs is None: cs, ce = s, e
        elif s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    if cs is not None: tot += ce - cs
    return tot / 1e6

cc_union = union_ms(cc.start_ts.values, cc.end_ts.values)
print(f"window={w.dur_ms:.2f}ms  collective ops={len(cc)}  collective wall={cc_union:.2f}ms "
      f"({100*cc_union/w.dur_ms:.0f}% of window)")
# Measured on generic-encoder-only-model TP=16: 18.0ms window, 130 ops, 7.8ms (43%) collective.
```

**Critical cross-check — the isolated NEFF replay undercounts collectives.** A standalone
`neuron-profile capture -n <neff>` reports a device `total_time` (~12.9ms in the example)
that is **lower** than the live `nc_exec_running` median (~18.0ms), because the standalone
replay does not reproduce the full 16-rank collective data movement the live run pays.
Always treat the live-trace `nc_exec_running` as the honest per-request device time, and
attribute the delta to real collective wall-time — not to host overhead or measurement
error.

## 1g. TE / collective overlap (is the collective time hideable?)

If collectives are a large fraction of the window, the next question is whether they run
concurrently with TensorEngine compute or serialize against it. This needs the **device**
profile (NEFF+NTFF ingest), not the system trace — the `Instruction` (engine=='Tensor')
and `CcInstruction` tables. Measure the merged-interval overlap:

```python
d = f"{NE}/<neff_device_profile>@latest"
inst = pd.read_parquet(f"{d}/Instruction.parquet")
cc   = pd.read_parquet(f"{d}/CcInstruction.parquet")
te   = inst[inst.engine == 'Tensor']

def merge(iv):
    out = []; cs = ce = None
    for s, e in sorted(iv):
        if cs is None: cs, ce = s, e
        elif s <= ce: ce = max(ce, e)
        else: out.append((cs, ce)); cs, ce = s, e
    if cs is not None: out.append((cs, ce))
    return out

A, B = merge(zip(te.start_ts, te.end_ts)), merge(zip(cc.start_ts, cc.end_ts))
i = j = ov = 0
while i < len(A) and j < len(B):
    lo, hi = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
    if lo < hi: ov += hi - lo
    if A[i][1] < B[j][1]: i += 1
    else: j += 1
print(f"TE-CC overlap = {ov/1e6:.3f}ms")
# Measured on generic-encoder-only-model TP=16: 0.004ms = 0.1% -> essentially fully serialized.
```

**Interpreting a ~0% overlap:** confirm whether it's a genuine data dependency or a
scheduler miss by reading the model's forward. A sequence-parallel pattern
(`all_gather → matmul → reduce_scatter`) is a **true dependency** — the gather output *is*
the matmul input, so no overlap is possible without windowed/collective-matmul fusion.
Note also that a "time-aware post-scheduler" flag like `--policy=3` can *shorten the
startup barrier* (measurable device-time win) while producing **zero** steady-state TE/CC
overlap — verify with this query rather than trusting the flag's name.

**Why it's a true dependency, mechanistically (check this before concluding "blocked"):**
identify which tensor dimension the all-gather reconstructs, and which dimension of the
following matmul is the **reduction** (contraction) dimension. If they're the same
dimension — e.g. the all-gather reconstructs the full feature/hidden dimension (sharded
`H/TP` per rank) and the matmul reduces over that same hidden dimension — then **every**
output element of the matmul needs the *full* gathered vector, so there is no way to start
computing on a partial gather result. This is unlike a sequence-parallel split on a
**non-reduction** dimension (e.g. splitting the token/sequence axis), where a matmul on
the first completed chunk of tokens genuinely doesn't need the other chunks and partial
overlap is architecturally possible. Confirm by checking the model source
(`model/<model>/model.py`) for which axis is sharded before the gather and which axis the
following `nn.Linear`/matmul contracts over — if they match, the 0% overlap you measured
is correct and expected, not a compiler/scheduler bug to chase further. If they *don't*
match, a genuine scheduler-level overlap opportunity may exist and is worth escalating.

Also check whether a windowed-einsum / collective-matmul compiler pass could tile around
this dependency (start the matmul on the first gathered chunk while later chunks are still
in flight) — this exists in the compiler binaries (`AwsNeuronCollectiveMatmul` /
`run-collective-matmul` in `hlo-opt`/`hlo2penguin`) but, as of the current SDK, is **not
reachable for the trn2 target** through any flag exposed by the vllm-neuron compile path:
`XLA_FLAGS` windowed-einsum thresholds are GPU-backend-only and never reach the
`neuronx-cc` invocation; `--internal-backend-options` has no
collective-matmul option; `--internal-hlo2tensorizer-options` (hlo2penguin) only exposes
`--drop-collective-tokens`. Empirically tested (XLA_FLAGS windowed-einsum thresholds): no
change in device time or TE/CC overlap, and collective op count can even *increase*
without any compute-hiding benefit. Do not re-spend compile cycles rediscovering this —
it is a real compiler capability gated behind an interface not currently exposed for trn2,
not a flag-tuning problem.


## 1h. Query hygiene — two footguns that produce plausible-but-wrong numbers

These are the two mistakes that most often make a *computed* system-trace number wrong
even when the query "runs fine" and returns a believable value (see the SKILL.md
verification-discipline rule on distrusting computed/aggregated metrics). Both were hit
for real on `generic-encoder-only-model` TP=16.

**(1) Exclude warmup / compile execs before any steady-state stat.** The first several
`nc_exec_running` events per `model_id` include compilation, cache population, and cold
DMA — they are multi-second or high-variance and will inflate a naive mean. Drop them
before computing mean/median/percentiles. A robust filter is either "drop the first N per
group" or "keep only rows within a sane duration band":

```python
nc['dur_ms'] = nc.duration_ns / 1e6
# per-model_id, drop the first 3-12 execs (warmup) then take steady state:
steady = nc.sort_values('start_ts').groupby('model_id').apply(
    lambda g: g.iloc[3:]).reset_index(drop=True)
# or band-filter around the expected device time to exclude cold outliers.
```
Report **median** alongside mean — a large mean-vs-median gap is the tell that warmup
(or a straggler rank) is still in the sample.

**(2) For TP>1, pin to ONE rank (or explicitly aggregate) — do not blend all ranks.**
The trace holds one set of events per rank (`worker_gid` / `process_id` / `nc_idx`).
Overlap-merging `cc_running` across *all* ranks at once double-counts collectives and
inflates the fraction; naively averaging `nc_exec_running` across ranks mixes a straggler
rank into the number. Compute the per-exec breakdown on a **single** `worker_gid` (rank 0
is fine — all ranks are equivalent for a symmetric TP graph), then, if you want an
all-rank view, report the distribution across ranks rather than a blended pool:

```python
wg = nc['worker_gid'].value_counts().index[0]   # one rank
nc_r  = nc[nc.worker_gid == wg]
cc_r  = cc[cc.worker_gid == wg]                  # merge collectives WITHIN this rank only
```
Real incident: merging `cc_running` across streams/ranks (instead of within one
`worker_gid`) plus halving the op count reported the collective fraction as **21% / 65
ops**; the correct single-rank overlap-merge gave **43.5% / 130 ops**. The device-time
and byte-size raw reads from the same analysis were fine — only the merged/aggregated
metric was wrong. Always show the query and its raw output for a computed metric that
drives a decision.

## 1i. Confirming a host-side device→host readback bottleneck (`nrt_tensor_read`)

When an E2E-latency SLA is missed but the live `nc_exec_running` device time is well
under budget, the gap is host overhead that the device window does **not** span. The most
common actionable culprit is an oversized device→host tensor readback. Read the
`nrt_tensor_read` event sizes directly — this is a raw-field read, so it's trustworthy:

```python
tr = ev[ev.name == 'nrt_tensor_read'].copy()
tr['size'] = tr.extra_attributes_json.apply(lambda s: json.loads(s).get('size'))
tr['dur_ms'] = tr.duration_ns / 1e6
print(tr['size'].value_counts().head())          # distinct readback sizes + counts
print(tr[tr['size'] > 1_000_000][['size','dur_ms']].describe())  # the big reads
```
Real incident (`generic-encoder-only-model` TP=16): **156 reads of exactly 4,194,304 bytes**
(= `512 × 4096 × 2`, the full `[T,H]` bf16 hidden states) at ~2.0ms each — the model
returned full hidden states for CPU-side pooling instead of the pooled `[num_seqs,H]`.
After moving pooling on-device, the same query showed **zero** 4MB reads and only
`262144`-byte reads (`16 × 4096 × 4` = pooled fp32 output). This is the canonical
"integration adaptation created the perf bug" signature: a Step-0 framework-fit choice
(return `[T,H]`, pool on host) that only shows up as host overhead in profiling, never in
the device window. The device compute/collective time is unchanged by the fix (verify
that as a non-regression cross-check).
