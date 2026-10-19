---
name: neuron-framework-profile-analysis-vllm-neuron
description: |
  Ingests vllm-neuron model execution artifacts into parquet using 
  neuron-explorer and guides profile analysis: identifies
  prefill/decode NEFFs, segments by collective boundaries,
  computes per-engine bounds and DMA efficiency, and produces
  structured workload reports.
  Use when encountering "analyze profile", "ingest profile",
  "query system trace", "what's the bottleneck",
  "decode step time", "prefill breakdown",
  "segment by layer", "DMA efficiency".
---

# Analyzing vllm-neuron Profiles

> **Note:** The analysis methodology (steps 2–5) is adapted
> from the NKI profiling skill and is not fully validated for framework-compiled NEFFs.
> Performance bounds, collective partitioning, and investigation results should be
> treated as approximate. Step 1 (ingestion) is stable.

## Verification discipline (BLOCKING — read and satisfy before any analysis)

Profile analysis is where confident-sounding wrong answers are easiest to produce. These
rules are a **hard gate**, not advice — distilled from repeated real failures on this
workflow:

0. **Device time comes from the LIVE `nc_exec_running` system trace, never from an isolated
   `neuron-profile` / `neuron-explorer capture` NEFF replay.** A standalone replay
   **undercounts** the real multi-rank collective traffic (measured: replay ~12.9 ms vs
   live ~18 ms for the same TP=16 graph; another run mis-quoted a 14.8 ms replay as "the
   device floor" and built a wrong "serving overhead" conclusion on it). Always cross-check
   against the live trace ([references/system-trace-queries.md](references/system-trace-queries.md) §1f).
1. **Query already-ingested profile data before writing any custom timing script.** The
   parquet tables already hold per-op, per-engine, ns-resolution timestamps. If you catch
   yourself about to write `time.perf_counter()` around a model call, or a benchmark/timing
   harness for a device question — STOP and query the profile. Reach for a hand-rolled timer
   only when the question is genuinely host-side and never touches the device — and even
   then, check the system trace first.
2. **Never compare two numbers captured from different runs.** Subtracting a device
   `total_time` from one capture and a wall-clock from another ("19.5 − 12.9 = 6.6ms
   overhead") is arithmetic dressed up as measurement. Numbers compared must come from the
   **same profile / same execution**.
3. **Validate concurrency claims with overlapping measurement windows, not overlapping
   process lifetimes.** Two replicas whose *processes* run concurrently can still have
   *timed windows* that don't overlap (compile-time skew). Use an explicit barrier and
   confirm the windows overlap before reporting a scaling number.

   **Concrete mechanism and fix (real incident, DP=2×TP=1 scaling measurement):** two
   replica processes were launched at the same instant and their *process lifetimes*
   clearly overlapped (confirmed by wall-clock start/end of each process). But each
   replica's own **timed 64-request measurement burst** only started *after* that
   replica finished its own independent compile+warmup — and compile time varied by
   **±11s** between the two processes (normal JIT/cache variance). This shifted the two
   timed windows apart by **5.7s with zero overlap**, even though the processes
   themselves ran concurrently the whole time. The resulting "99.6% DP efficiency" number
   was silently invalid — it was actually measuring two **sequential** solo runs.

   **Fix:** add an explicit file-based barrier so both replicas signal "warmup done" and
   then wait for every other replica's signal before starting the timed burst:
   ```python
   import os, time
   BARRIER_DIR = "/tmp/dp_barrier"  # shared across replica processes
   os.makedirs(BARRIER_DIR, exist_ok=True)
   open(f"{BARRIER_DIR}/ready_{replica_id}", "w").close()
   while len(os.listdir(BARRIER_DIR)) < num_replicas:
       time.sleep(0.05)
   # Now start the timed measurement burst — guaranteed synchronized start.
   ```
   **Verify the fix, don't just trust it**: log each replica's own `[window_start,
   window_end]` timestamps and compute the actual overlap
   (`min(end_a, end_b) - max(start_a, start_b)`); it should be ≈ each replica's own
   elapsed time (measured: 5.451s overlap out of a 5.489s/5.492s window — 99.3%,
   confirming genuine concurrency). Only then compare `concurrent_measured_aggregate`
   against `solo_baseline × N` for the efficiency number. In the real incident this
   correction changed the reported DP scaling efficiency from an invalid 99.6% down to a
   genuinely-verified **97.7%** — a real number, not just a more conservative guess.

4. **Spot-check any subagent / delegated analysis against real on-disk artifacts before
   trusting it.** A subagent once returned a fully detailed host-overhead breakdown table
   with plausible numbers and methodology narrative and **zero** backing artifacts — caught
   only by grepping the raw trace for the claimed event names (absent) and searching disk
   for the claimed script (absent). If an analysis claims a capture/measurement, verify the
   artifact exists and contains what's claimed. See
   [references/fabricated-analysis-case-study.md](references/fabricated-analysis-case-study.md).
5. **Confirm the *binding* op before proposing a fix.** A correct optimization for a
   secondary op is wasted effort. Example: on this 8B embedding model the MLP is at its
   compute floor and collectives are the wall — a (correct, validated) non-causal attention
   kernel moved nothing E2E because attention was secondary. Rank ops by measured time and
   optimize the top one.
6. **Do not stop at "found a smaller/different number, flagged uncertainty" — finish the
   investigation.** A real incident: an initial gap analysis was corrected from "6.6ms IPC
   overhead" down to "1.9ms driver overhead," and a second, larger, genuinely unexplained
   5.2ms gap was found in the same pass — then left open as a flagged hypothesis instead of
   being pursued to a real answer, and the overall optimization task was abandoned mid-step
   as a result. Treat "I found a smaller number than before" as a sign to keep going with
   the correct tool (live system trace, direct source read), not as a stopping point —
   either resolve the question or explicitly exhaust the available tooling and say so, then
   continue to the next actionable item. Do not silently drop the task.
7. **Distrust *computed/aggregated* subagent numbers harder than raw-field reads — error
   likelihood scales with analytical complexity, not plausibility.** A delegated analysis's
   accuracy correlates with how much the metric is *derived*, not how confident it sounds.
   Direct field reads (an op `duration_ns`, a `tensor_read` byte `size`, an event count)
   are usually right. Anything requiring **interval-overlap-merge, cross-stream or
   cross-rank aggregation, or steady-state/warmup filtering** must be independently
   recomputed against the parquet before you build on it — those are where sincere-but-wrong
   answers come from, and they are a *distinct* failure mode from the fabrication in rule 4
   (the number is real, the computation is wrong). Real incident (`generic-encoder-only-model`,
   TP=16): a subagent reported the collective fraction as **21% / 65 CC-ops per exec**;
   independently re-querying the same parquet with a proper overlap-merge across all
   collective intervals on a single `worker_gid` gave the true **43.5% / 130 CC-ops** —
   the subagent had counted one stream instead of merging, and had halved the per-exec op
   count. The device-time and byte-size numbers from the *same* subagent were correct. When
   a computed metric matters to a decision, show your own query and its raw output, per the
   query-hygiene rules in [references/system-trace-queries.md](references/system-trace-queries.md).
8. **"Device time meets the SLA" is NOT "the request meets the SLA" — state which quantity
   the SLA is defined on.** The live `nc_exec_running` trace is authoritative for **device
   execution time only**. An end-to-end request-latency SLA also includes host overhead
   (tokenization, dispatch, device↔host tensor transfers, framework/Python) that
   `nc_exec_running` does **not** span. Do not silently equate the two: if the SLA is on
   E2E latency, report device time as the trustworthy *lower bound* and name the host
   portion as real-but-not-captured-by-this-trace, rather than declaring the SLA met on
   device time alone. (Corollary: a device→host readback — e.g. returning full `[T,H]`
   hidden states for CPU-side pooling instead of the pooled `[num_seqs,H]` — shows up as
   host overhead, is invisible in `nc_exec_running`, yet can be the binding contributor to
   an E2E-latency miss. Confirm it via the `nrt_tensor_read` byte sizes in the system
   trace, not the device window.)
9. **A single-replica capture is DP-invariant for kernel attribution; a node aggregate is NOT
   `single-replica × N`.** DP replicas are independent (no cross-replica collectives), so per-op
   device time, engine bounds, the collective pattern, and per-step latency read **identically at
   every DP degree** — analyze ONE rank of ONE replica (rank 0; all ranks are equivalent) and do
   **not** attempt to sum or scale device-time across replicas from the trace to get a node number.
   Conversely, never turn a single-replica per-step device time into node throughput by multiplying
   by the replica count: concurrent replicas contend for shared host CPU, PCIe / host↔device DMA,
   the serving stack, the interconnect fabric, and the power/thermal envelope that a lone replica
   never stresses, so `× N` can only **over-estimate** the true aggregate. A node aggregate is a
   **MEASURED** quantity — a concurrent N-replica run (one client per replica) with **verified
   overlapping measurement windows** (rule 3) and the **worst percentile latency across replicas**.
   A projection may bound or motivate the aggregate; it may never certify it. (This restates, for
   analysis, the profiling skill's "profile ONE replica — DP>1 is not a capture input" note.)

## Known dead-ends for collective-bound vllm-neuron models (do NOT re-test — prune at decision time)

Empirically confirmed non-levers for short-sequence / collective-bound prefill models on
the current trn2 SDK. Each one below was measured to have ~no effect (or to regress); do
not spend compile cycles re-discovering them:

- **Replicated-TP / RS+AG→AllReduce fusion** — an all_reduce moves the same bytes as the
  reduce_scatter+all_gather it replaces, so it cuts collective *count* but not
  bandwidth/wall-time on the sequence-parallel data dependency; also easily breaks
  correctness (the RS/AG are not a matched pair — residual add + RMSNorm sit between them).
  Not a latency win.
- **Windowed-einsum / `AwsNeuronCollectiveMatmul` (TE/collective overlap)** — the enabling
  flags are XLA-GPU-backend-only and are not wired to the trn2 vllm-neuron path
  (`XLA_FLAGS`, `--internal-backend-options`, `--internal-hlo2tensorizer-options` all
  tested). No effect. The SP `all_gather → matmul → reduce_scatter` is a genuine data
  dependency (gather output *is* the matmul input) → ~0.1% overlap is expected, not a
  scheduler miss.
- **`-O3` / `--model-type=transformer` at TP>1** — borderline-noise (±2%) on
  collective-heavy graphs; O3 can even raise DMA activity. (O3 *does* help the TP=1
  single-partition / nomodular / large-bucket regime — regime-specific, does not generalize.)
- **vLLM async scheduling** — a generative-decode-step feature (draft/output-token
  bookkeeping); architecturally N/A to prefill-only pooling/embedding models.

## Collective-bound workloads (analysis pattern)

For TP>1 graphs, first establish whether the workload is **collective-bound** before
chasing per-engine or DMA inefficiencies:

- Measure collective wall-time inside the execution window and TE/collective overlap using
  [references/system-trace-queries.md](references/system-trace-queries.md) §1f–1g.
- Interpretation guide (validated on an 8B embedding model at TP=16: 43% of device time in
  collectives, 0.1% TE/CC overlap): if collectives dominate and overlap is ~0 due to a
  genuine sequence-parallel data dependency, then kernel fusion / DMA-efficiency work has
  little leverage — the levers are TP degree (expect an interior optimum) or
  collective-compute overlap (windowed einsum / collective matmul, which may not be
  reachable on the current trn2 compiler path — verify empirically).
- Remember the isolated-NEFF-replay undercount (§1f): a standalone `neuron-profile` device
  `total_time` can be several ms below the live per-request device time because it omits
  realistic multi-rank collective traffic.


## Reduced-model profiles: what transfers, and projecting to full model

A reduced-layer capture (small `num_hidden_layers = k`; see the profiling skill's two-stage
method) is analyzed with the SAME queries as a full profile — but be explicit about which
conclusions are depth-invariant and which are not, and project per-layer quantities to the
real depth with a two-point linear fit.

**Depth-invariant (safe to conclude from a reduced profile):** per-kernel bounds and gaps
(Steps 2/4) for a single layer's ops — TE underutil, DMA inefficiency, transpose fraction,
initiation interval; a kernel firing vs falling back to a default path; the per-layer
collective pattern and TE/CC overlap (Step 3); source-attributed hotspots within a layer
(Step 5). These live inside one layer and repeat, so the reduced profile localizes them
faithfully.

**Depth-dependent (do NOT conclude from a reduced profile):** whole-NEFF `total_time`,
absolute per-step latency and tok/s, collective TOTAL wall-time and its fraction of the step,
HBM-capacity / KV-pressure effects, and the interior-optimum TP verdict. Report these only
from the full-depth profile.

**Projecting a per-step time to full depth (two-point linear fit).** Capture at two reduced
depths `k1 < k2` (e.g. 2 and 4), measure per-step device time from the live `nc_exec_running`
window (Step 1e) at each, then:

```python
per_layer_cost  = (t_k2 - t_k1) / (k2 - k1)
fixed_overhead  = t_k1 - per_layer_cost * k1              # embeddings, lm_head, launch, one-time CC
t_full_estimate = fixed_overhead + per_layer_cost * L_full   # L_full = real num_hidden_layers
```

Fit prefill and decode separately (different slope and intercept). Report the projected
full-depth number as an ESTIMATE with the two measured points and the fit shown — it is a
localization / planning aid, never an SLA certification (which requires the full-depth run,
and for aggregates a concurrent N-replica run; see verification rules 0/8 above). Once the
full-depth capture exists, cross-check the estimate against it: a large divergence means a
depth-dependent effect (bucket-geometry graph change, HBM pressure) the linear model doesn't
capture — investigate rather than trust the projection.

## Where outputs go (the shared project-root contract)

`$NEURON_PROJECT_ROOT` names the PROJECT directory this analysis belongs to; everything this
skill generates lands under `<root>/agent_artifacts/`, and it defaults to the cwd, so an ordinary
interactive run needs no extra input. It is the SAME input the rest of this family takes — the
parallelism-planner and hardware-entitlement skills accept it as `--project-root`, the
vllm-neuron profiling skill writes its profiles under it — which is what lets one run's plan,
per-operator floor, profile and ingested parquet sit in a single evidence trail instead of four
unrelated directories.

```
$SESSION_DIR / $PROFILE_DIR   (INPUT profile locations handed over by the capture step)
  else $NEURON_PROJECT_ROOT/agent_artifacts/data/<run-id>/   (ingested parquet, reports)
    else $PWD/agent_artifacts/data/<run-id>/   (ingested parquet, reports)
```

Ingestion reads a profile from wherever it was captured — `$PROFILE_DIR` / `$SESSION_DIR` handed over by the profiling skill, frequently on a Trainium box's local disk — and that stays an explicit input. What this contract fixes is where the *derived* artifacts go: parquet, queries and the written report follow the project root, so the analysis lands beside the floor it is being compared against.

## Step 1: Ingest

After profiling (see `neuron-framework-profiling-vllm-neuron`),
ingest the artifacts into queryable parquet tables.

### 1.0. Read the manifest first (if present)

The profiling skill's isolation recipes (`run_profile_decode.sh`,
`run_profile_prefill.sh`) emit `$PROFILE_DIR/manifest.json` at end-of-run.
It resolves the two nested-glob paths (`i-<instance>_pid_<pid>/<timestamp>/`
on the raw side, NEFF hashes from `trace_info.pb`) so you can jump straight
to files instead of `find`ing them:

```python
import json
manifest = json.load(open(f"{PROFILE_DIR}/manifest.json"))
device_ntff       = manifest["profile"]["device_ntff"]
system_trace_pb   = manifest["profile"]["system_trace_pb"]
trace_info_pb     = manifest["profile"]["trace_info_pb"]
session_dir       = manifest["profile"]["session_dir"]  # for the ingest step below
neff_dir          = manifest["profile"]["neff_dir"]
neff_hashes       = manifest["neff_hashes"]             # list[str]
workload          = manifest["workload"]                # isl, osl, conc, ...
bench_result_json = manifest["bench_result_json"]       # ground-truth metrics
```

Every path is absolute, so nothing needs to be resolved. `cpu_util_pb` and
`host_mem_pb` are populated whether or not they contain data (the 0-byte
caveat is documented in the profiling skill) — check `os.path.getsize` if
you need to know.

**If manifest.json is absent** (legacy capture, older skill version, or a
run that died before teardown), fall back to `find`/`ls` as in 1a below.
The manifest is a shortcut, not a requirement.

### 1a. Stage a minimal capture directory

From the profiling output, stage only what's needed for ingestion — system
traces, NEFFs, and one rank's NTFF per NEFF:

```bash
PROFILE_DIR=$NEURON_PROJECT_ROOT/agent_artifacts/profiles/<run_name>
FIRST_WORKER=$(find $PROFILE_DIR -mindepth 1 -maxdepth 1 -type d -name "i-*" | sort | head -1)
SESSION_DIR=$(find $FIRST_WORKER -mindepth 1 -maxdepth 1 -type d)

mkdir -p $PROFILE_DIR/capture_minimal
# System traces
cp $SESSION_DIR/ntrace.pb $SESSION_DIR/cpu_util.pb $SESSION_DIR/host_mem.pb $SESSION_DIR/trace_info.pb $PROFILE_DIR/capture_minimal/
# NEFFs
cp $SESSION_DIR/*.neff $PROFILE_DIR/capture_minimal/
# One rank's NTFF per NEFF (from capture step)
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
├── neff_<id_a>_vnc_0.neff
├── neff_<id_b>_vnc_0.neff
├── <id_a>/
│   └── profile_rank_0_exec_2.ntff
└── <id_b>/
    └── profile_rank_0_exec_2.ntff
```

### 1b. Ingest

```bash
# --data-path is where the ingested parquet lands. Point it under the project root so the
# parquet joins THIS run's evidence trail (beside the plan and the entitlement floor it is
# compared against), rather than the neuron-explorer default of ~/.local/share/neuron-profile.
NE=$NEURON_PROJECT_ROOT/agent_artifacts/data/<run_name>
neuron-explorer view \
    -d $PROFILE_DIR/capture_minimal \
    --data-path $NE \
    --display-name <run_name> \
    --ingest-only
```

**Output** at `$NE/profiles/global/` (= `$NEURON_PROJECT_ROOT/agent_artifacts/data/<run_name>/profiles/global/`):
```
<run_name>@latest/                              ← system profile
    ├── SystemProfileEvents.parquet
    ├── SystemProfileMetadata.parquet
    ├── SystemProfileHbmUsage.parquet
    ├── DeviceProfileList.parquet
    └── SchemaFields.parquet

<run_name>_<instance>_pid_<A>_nc_0_session_0@latest/   ← NEFF A device profile
    ├── Summary.parquet
    ├── Instruction.parquet
    ├── DmaPacket.parquet
    ├── DmaPacketAggregated.parquet
    ├── ActiveTime.parquet
    ├── CcOp.parquet
    ├── CcInstruction.parquet
    ├── Metadata.parquet
    ├── TensorInfo.parquet
    ├── Flow.parquet
    ├── HbmUsage.parquet
    ├── NeffHeader.parquet
    ├── ... (34 tables total)

<run_name>_<instance>_pid_<B>_nc_0_session_0@latest/   ← NEFF B device profile
    └── ... (same tables)
```

One system profile + one device profile per NEFF (rank 0 only — all ranks are equivalent).

**If ingesting the full capture dir** (all 16 ranks per NEFF instead of `capture_minimal`):
```
<run_name>@latest/                                     ← system profile
<run_name>_<instance>_pid_<A>_nc_0_session_0@latest/   ← NEFF A, rank 0
<run_name>_<instance>_pid_<A>_nc_1_session_0@latest/   ← NEFF A, rank 1
...                                                     (×16 ranks)
<run_name>_<instance>_pid_<B>_nc_0_session_0@latest/   ← NEFF B, rank 0
...                                                     (×16 ranks)
Total: 33 (2 NEFFs × 16 ranks + 1 system)
```

**If ingesting a session worker dir** (Mode B from profiling skill):
```
<run_name>@latest/                                     ← system profile
<run_name>_<instance>_pid_<N>_nc_0_session_0@latest/   ← device profile (full session, all NEFFs)
Total: 2
```

### 1c. Verify quality

For best profile quality, ensure these were set during the profiling and capture steps:
- `NEURON_FRAMEWORK_DEBUG=1` — source attribution in instructions
- `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1` — DMA packet data (set during capture)
- `XLA_IR_DEBUG=1` — IR-level debug info in NEFFs
- `XLA_HLO_DEBUG=1` — HLO-level debug info in NEFFs

## Step 2: Load and query parquet

```python
import pandas as pd, numpy as np, os

NE = os.path.join(os.environ.get("NEURON_PROJECT_ROOT", os.getcwd()),
                  "agent_artifacts/data/<run_name>/profiles/global")   # the --data-path used at ingest
profile = "<run_name>_<instance>_pid_<A>_nc_0_session_0"
d = f"{NE}/{profile}@latest"

# Core tables
metadata = pd.read_parquet(f"{d}/Metadata.parquet").iloc[0]
summary  = pd.read_parquet(f"{d}/Summary.parquet").iloc[0]
inst     = pd.read_parquet(f"{d}/Instruction.parquet")
active   = pd.read_parquet(f"{d}/ActiveTime.parquet")
dma_pkts = pd.read_parquet(f"{d}/DmaPacket.parquet")
dma_agg  = pd.read_parquet(f"{d}/DmaPacketAggregated.parquet")
ccop     = pd.read_parquet(f"{d}/CcOp.parquet")
tensors  = pd.read_parquet(f"{d}/TensorInfo.parquet")
flow     = pd.read_parquet(f"{d}/Flow.parquet")
```

### Key tables

| Table | Contents |
|-------|----------|
| `Summary` | Top-level metrics (total_time, engine active %, HBM bytes, MFU) |
| `Metadata` | Hardware constants (HBM BW, TE dimensions, clock freq) |
| `Instruction` | Every instruction executed (engine, opcode, timestamps, flops, source location) |
| `ActiveTime` | Per-engine active intervals, one row set **per `pcore_idx`** (engines lowercase: tensor/vector/scalar/gpsimd). Merged within a pcore's stream, NOT across pcores — interval-merge both `pcore_idx` for an LNC=2 wall (see the active-wall recipe in Step 2). |
| `DmaPacket` | Individual DMA transfers (start/end timestamps, bytes, queue type) |
| `DmaPacketAggregated` | Per-instruction DMA stats (variable, source/dest, transfer bytes) |
| `CcOp` | Collective operations (all-reduce, all-gather, reduce-scatter) |
| `TensorInfo` | I/O tensor metadata (name, type IN/OUT/WEIGHT, size, shape) |
| `Flow` | Links between tables (Instruction → DmaPacketAggregated, etc.) |

### System profile tables

```python
sys_profile = "<run_name>"
sys_d = f"{NE}/{sys_profile}@latest"

events = pd.read_parquet(f"{sys_d}/SystemProfileEvents.parquet")
```

| Table | Contents |
|-------|----------|
| `SystemProfileEvents` | NEFF execution events (nc_exec_running), model switches, gaps |
| `SystemProfileMetadata` | Trace time range, HBM capacity |
| `SystemProfileHbmUsage` | HBM usage over time |
| `DeviceProfileList` | Maps device profiles to instance/pid/nc |

## Profile Analysis

Systematically analyze a full model's execution on VLLM Neuron. 

### 0: Identify hardware constants

```python
metadata = pd.read_parquet(f"{d}/Metadata.parquet").iloc[0]

# DMA bandwidth
# dma_ddr_bandwidth: peak for a single NeuronCore (all 16 DMA engines)
# hbm_ddr_bandwidth: peak for one HBM stack (shared across cores in LNC=2)
dma_bw = metadata['dma_ddr_bandwidth']        # single-core profiles use this
hbm_bw = metadata['hbm_ddr_bandwidth']        # LNC=2 profiles use this

# Tensor Engine peak FLOPS (per physical core)
te_peak_single = (metadata['tensor_engine_num_rows']
                  * metadata['tensor_engine_num_cols']
                  * 2.0 * metadata['tensor_engine_clock_freq'] * 1e9)  # flops/s

# LNC=2: profile captures both cores of the logical NeuronCore.
# TE peak and HBM BW are already the combined values.
num_physical_cores = metadata['num_physical_cores']  # 2 for LNC=2
te_peak = te_peak_single * num_physical_cores

print(f"dma_ddr_bandwidth:  {dma_bw/1e9:.0f} GB/s (single core)")
print(f"hbm_ddr_bandwidth:  {hbm_bw/1e9:.0f} GB/s (HBM stack)")
print(f"TE peak (1 core):   {te_peak_single/1e12:.1f} TFLOPS")
print(f"TE peak (LNC=2):    {te_peak/1e12:.1f} TFLOPS")
print(f"num_physical_cores: {num_physical_cores}")
```

For LNC=2 (default for vllm-neuron), use `hbm_ddr_bandwidth` and `te_peak` (both cores).
For LNC=1, use `dma_ddr_bandwidth` and `te_peak_single`.

### 1: Profile session workload using system trace

The system profile (`<run_name>@latest`) records every NEFF dispatch during
the benchmark. This tells you which compiled graphs executed, how many times,
in what order, and the time between dispatches. Use this to identify workload
phases and per-step cost before diving into device-level analysis.

Steps: 1a (load/filter), 1b (identify NEFFs), 1c (execution pattern),
1d (gap stats), 1e (throughput estimate).

See [references/system-trace-queries.md](references/system-trace-queries.md)
for the detailed queries.

Report the results using the template in section 1 below.

### 2: Run per NEFF performance bounds

For each NEFF identified in step 1, compute bounds from its device profile.

Follow the methodology in [references/performance-bounds-nki-reference.md](references/performance-bounds-nki-reference.md)
with these adaptations for vllm-neuron LNC=2 profiles:

- Use `hbm_ddr_bandwidth` (not `dma_ddr_bandwidth`) for memory_bound_ideal
- Use `te_peak = te_peak_single × num_physical_cores` for compute_bound_ideal
- Skip `memory_bound_no_reloads` — not well-defined for framework NEFFs
  (KV cache tensors are runtime-allocated, TensorInfo sizes don't reflect
  algorithmic minimum)

Report per NEFF:
- **2a.** Bounds table (total_time, memory_bound, memory_bound_ideal, compute_bound, compute_bound_ideal, compute_no_transposes, perfect_pipeline)
- **2b.** Engine active times + bottleneck
- **2c.** Memory family gaps (DMA idle, DMA inefficiency) and compute family gaps (TE idle, TE underutil, transpose)
- **2d.** Key metrics: achieved BW, DMA transfer bytes, hw_flops, transpose fraction 

> **"Active time" means active-WALL, not summed instruction time.** Report the wall-clock
> time each engine was busy — the interval-merge of its active intervals — never
> `Instruction.duration_ns.sum()` (or `ActiveTime.duration_ns.sum()`) per engine. The raw
> sum double-counts (a) the two physical cores of an LNC=2 logical core and (b) overlapping
> issue slots on one engine, so it can exceed the step (e.g. a 44 ms TE "busy" figure on a
> 21 ms step is the sum, not the wall). Always interval-merge, and for LNC=2 merge across
> BOTH `pcore_idx` values into one timeline.
>
> ```python
> def active_wall_ns(df, engine):   # df = ActiveTime or Instruction; merges across pcores
>     e = df[df['engine'].str.lower() == engine.lower()]   # ActiveTime engines are lowercase
>     total = cur_s = cur_e = 0
>     for s, x in sorted(zip(e['start_ts'].tolist(), e['end_ts'].tolist())):
>         if s > cur_e:
>             total += cur_e - cur_s; cur_s, cur_e = s, x
>         else:
>             cur_e = max(cur_e, x)
>     return total + (cur_e - cur_s)
> # % of step = active_wall_ns(...) / step_ns   (step from nc_exec_running, Step 1e)
> ```
>
> Cross-check against `Summary`'s `*_engine_active_time` / `*_active_time_percent` fields —
> those are already active-wall. `ActiveTime` rows are per-`pcore_idx`; a fresh ingest does
> **not** pre-merge them across pcores, so the sum of both pcores' rows is ~2× the wall.

### 3: Partition each NEFF by collectives

Collectives (all_reduce, reduce_scatter, all_gather) are the only synchronization
points in the compiled graph. Use them as segment boundaries to isolate per-module
costs (attention, MLP, embedding, lm_head).

Steps:
- **3a.** Load `CcOp`, drop `Invalid`, sort by `start_ts`; the collective op type
  (`AllReduce` vs `ReduceScatter`+`AllGather`) tells you replicated-TP vs sequence-parallel
  without reading any model code.
- **3b.** Build segments as the intervals between consecutive collectives, over the live
  `nc_exec_running` step window (Step 1e) — not `dma_agg` min/max, which is empty when DGE
  is off.
- **3c.** Measure per-segment: duration, per-engine active-wall (interval-merged over both
  pcores), HBM read bytes, FLOPS.
- **3d.** **Label each segment by its dominant component**, resolved from ground truth: the
  framework module scope (`layer`), then the IR debug source `file:line`
  (`bir_debug_info_source_location` / `nki_source_location`) — NKI kernel paths name the
  component by directory, and ambiguous framework `model.py:<line>` ops are resolved by
  reading the model source once. Do not hand-label by position or match bare backend tokens.
  Optionally cross-check segment count/alternation against a `forward()` trace.
- **3e.** Report per-segment table + component roll-up (report template §3 above).

See [references/collective-partitioning.md](references/collective-partitioning.md)
for the detailed methodology, the classifier, and the queries.

### 4: Per-segment performance bounds

Run the same bounds methodology (step 2) windowed to the dominant segments
identified in step 3. Focus on segments that consume the most time.

For each hotspot segment `[t0, t1]`:
- Recompute all bounds using only data within that window
- Identify the segment-level bottleneck engine (may differ from whole-NEFF)
- Compute segment-level gaps

This localizes the performance problem from "the decode NEFF is DMA-bound"
to "layer N attention is DMA-bound due to X" — actionable for optimization.

### 5: Run investigations on hotspot segments

Once a segment's bottleneck engine and dominant gap are identified, run the
matching investigation from the NKI skill's investigation library, windowed
to that segment's `[t0, t1]`:

- **DMA inefficiency gap** → [DMA efficiency investigation](references/performance-bounds-nki-reference.md) (transfer sizes, achieved BW per source line)
- **Excess traffic gap** → redundant DMA transfers investigation (reload decomposition, spill detection)
- **TE underutil gap** → TE inefficiency investigation (tile sizes, initiation intervals)
- **Transpose gap** → redundant TE transposes investigation

Each investigation has a detect step (quantify the gap) and a localize step
(trace to source lines via Flow → DmaPacketAggregated or Instruction source attribution).

---

## Report Template

```markdown
# Profile Analysis: <model_name>, <num_layers> layers, <parallelism_config>

**Config:** <layers>, <parallelism_config>, max_model_len=<X>, max_num_seqs=<B>, <quantization>
**Hardware:** <instance_type>, LNC=<N>

---

## 0. Hardware Constants

| Constant | Value |
|----------|-------|
| HBM BW | <X> GB/s |
| TE peak | <X> TFLOPS |
| num_physical_cores | <N> |

## 1. Session Workload

| Metric | Value |
|--------|-------|
| Total wall clock | <X> s |
| Total executions | <N> |
| Distinct NEFFs | <N> |
| Model switches | <N> |

### NEFFs identified

| NEFF | model_id | Role | Count | Mean dur (us) | Total (ms) |
|------|----------|------|-------|---------------|------------|
| A | <id> | <role> | <N> | <X> | <X> |
| B | <id> | <role> | <N> | <X> | <X> |

### High-level timeline

| Phase | Duration | Dispatches |
|-------|----------|------------|
| <phase_1> | <X> ms | <N> (<description>) |
| <gap> | <X> s | |
| <phase_2> | <X> ms | <N> (<description>) |
| <gap> | <X> s | |
| <phase_3> | <X> s | <N> (<N> A + <N> B) |

### Execution pattern (real execution)

`<pattern>`

### Inter-execution gaps (steady state)

| Transition | Count | Gap mean (us) | Gap/dur ratio |
|-----------|-------|---------------|---------------|
| A→A | <N> | <X> | <X> |
| A→B | <N> | <X> | <X> |
| B→A | <N> | <X> | <X> |
| B→B | <N> | <X> | <X> |

### Throughput estimate from trace

| Metric | From trace | bench_results.json |
|--------|-----------|-------------------|
| Output tok/s | <batch_size> / <decode_step_us> | <X> |
| Total tok/s | — | <X> |

## 2. Performance Bounds (per NEFF)

### NEFF <label>: <model_id> (<role>)

#### Bounds

| Bound | Value (us) |
|-------|-----------|
| total_time | <X> |
| memory_bound | <X> |
| memory_bound_ideal | <X> |
| compute_bound | <X> |
| compute_bound_ideal | <X> |
| compute_no_transposes | <X> |
| perfect_pipeline (<engine>) | <X> |

#### Engine active times

Active-wall (interval-merged over both pcores; see the Step 2 recipe), not summed
instruction time. `%` is of the step.

| Engine | Active-wall (us) | % of step |
|--------|------------------|-----------|
| <engine_1> | <X> | <X>% |
| <engine_2> | <X> | <X>% |
| ... | | |

Bottleneck: <engine> (highest active-wall)

#### Memory family gaps

| Gap | Value (us) | % of total |
|-----|-----------|------------|
| DMA idle | <X> | <X>% |
| DMA inefficiency | <X> | <X>% |

#### Compute family gaps

| Gap | Value (us) | % of total |
|-----|-----------|------------|
| TE idle | <X> | <X>% |
| TE underutil | <X> | <X>% |
| Transpose | <X> | <X>% (<X>% of flops) |

#### Key metrics

| Metric | Value |
|--------|-------|
| DMA transfer bytes | <X> MB |
| Achieved BW | <X> GB/s (<X>% of peak) |
| hw_flops | <X> GFLOPS |
| transpose_flops | <X> GFLOPS (<X>% of hw_flops) |

## 3. Collective Partitioning

Wall-time is taken from the live `nc_exec_running` window for the step (Step 1e); every
row's time is an interval-merge (over both pcores for LNC=2), so the component roll-up
sums to the step.

### 3.1 Per-segment table

One row per inter-collective compute segment and per collective, in execution order:

| # | Segment | Component | dur (us) | % step | DMA (us) | TE (us) | Vec (us) | GP (us) | HBM rd (MB) | GFLOPS |
|---|---------|-----------|----------|--------|----------|---------|----------|---------|-------------|--------|
| 0 | pre | <embed/other> | <X> | <X> | <X> | <X> | <X> | <X> | <X> | <X> |
| 1 | CC[0] <op> | collective | <X> | <X> | — | — | — | — | — | — |
| 2 | seg | <attention> | <X> | <X> | <X> | <X> | <X> | <X> | <X> | <X> |
| 3 | CC[1] <op> | collective | <X> | <X> | — | — | — | — | — | — |
| 4 | seg | <mlp> | <X> | <X> | <X> | <X> | <X> | <X> | <X> | <X> |
| ... | | | | | | | | | | |

`Component` is assigned by the content-based classifier in
[references/collective-partitioning.md](references/collective-partitioning.md) §3c — do
not hand-label by position alone.

### 3.2 Component roll-up (sums to the step)

| Component | wall (us) | % step | HBM rd (MB) | mem floor (us) |
|-----------|-----------|--------|-------------|----------------|
| Attention | <X> | <X> | <X> | <X> |
| MLP / FFN | <X> | <X> | <X> | <X> |
| Collective (<op> ×N) | <X> | <X> | — | — |
| Other (norm / embed / lm_head / sample / idle) | <X> | <X> | <X> | <X> |
| **TOTAL** | **= step** | 100% | <X> | <X> |

`mem floor = HBM_rd_bytes / hbm_ddr_bandwidth` (the memory-bound lower bound for that
component's reads; see Step 0 for the LNC bandwidth choice).

### 3.3 Interpretation checklist

- **Binding component:** rank components by wall time; that is where optimization effort
  goes (verification rule 5). Report it explicitly.
- **Why it binds:** compare each component's wall to its mem floor.
  - `wall ≈ mem floor` → memory-bound (lever: read fewer bytes — quantization, KV dtype,
    smaller weights — or raise effective BW).
  - `wall ≫ mem floor` → kernel/compute-bound (lever: the kernel itself; check which
    engine dominates its active-wall — TE vs Vector/Scalar/GpSimd).
  - collective % high with ~0 compute overlap → collective-bound (levers per the
    collective-bound section: TP degree, compute/collective overlap — verify the collective
    is transfer- vs sync-bound first via CcInstruction, since a sync-bound collective has no
    bandwidth lever).
- **Efficiency:** report `Σ mem floor / step`; a low ratio quantifies the headroom that is
  NOT explained by data movement.
```

---

## Backlog

- [ ] Adapt performance bounds from NKI to vllm-neuron models. Reference doc copied to [references/performance-bounds-nki-reference.md](references/performance-bounds-nki-reference.md). Key differences to address: LNC=2 bandwidth choice, collectives included in DMA bytes, `no_reloads` bound not well-defined for framework NEFFs (runtime-allocated KV cache, TP weight shards), multi-engine bottlenecks common in framework NEFFs.
- [ ] Identify how other parallelism strategies supported by vllm-neuron (sequence parallelism, expert parallelism, data parallelism, pipeline parallelism) manifest in profiles — additional NEFFs, different collective patterns, different NEFF counts per dispatch, and how that affects the profiling/capture/analysis workflow.
- [ ] Figure out what `nc_model_switch` events in SystemProfileEvents represent and when they occur.

---

## Sweep-campaign cost discipline (analysis side) — cross-reference

When this skill runs inside a multi-config **optimization sweep**, ingest/analysis must not
become its own multi-hour gate between the capture and the next stage. The query-hygiene and
one-rank/bounded-ingest mechanics are covered above; this is the campaign-level framing. Canonical
playbook: `../experimental-neuron-framework-optimization/references/SWEEP_ACCELERATION.md` (Tiers 2 & 3).

- **Cheap ingest is the default, not the fallback.** Pin ONE rank, skip warm-up with an iteration
  delay so the window is decode-only and a handful of steps, ingest to parquet and query specific
  tables — never serialize a whole multi-rank trace (it buffers hundreds of GB with zero output
  and can silently yield empty tables). Bound every ingest with a timeout and confirm the output
  grows within ~1 min, or kill and re-capture smaller. When you re-capture smaller, change ONLY
  the window (one-variable hygiene applies to captures too).
- **Kill a lingering `neuron-explorer` or a re-profile server by its RECORDED process group,
  never with `pkill -f` (KERNEL D16).** A bare `neuron-explorer view` (without `--ingest-only`)
  writes the parquet and then blocks forever bound to a viewer port — finished work that looks
  like a hang. Start any such long-running command (an ingest you must be able to abort, or a
  server you re-launch to re-capture) through `references/safe_serve.py launch`, and stop it with
  `references/safe_serve.py teardown` (a self-guarded SIGKILL of that command's own
  `setsid`-isolated group). NEVER use `pkill -9 -f vllm` / `pkill -9 -f neuron-explorer` or a
  bare `kill -9 -<PGID>`: a `nohup cmd &` child shares the launcher's group, so those killed the
  agent's own session and unrelated users' runs in a real incident. Confirm with
  `safe_serve.py status` and `fuser /dev/neuron*`.
- **Apportion an axis from two points instead of demanding a capture per point (KERNEL B13).**
  From one capture at two batch sizes, fit `step_time = intercept + slope × batch`: the intercept
  is the depth-/batch-invariant term (weight read + collectives + fixed overhead), the slope is
  the per-item term. That characterizes the whole batch axis with no additional device time —
  valid only where compiled width tracks the swept parameter (at fixed padded width the slope is
  an artifact).
- **Reduced-layer profiles have a sharp transfer boundary.** They localize per-op structure,
  per-kernel efficiency, the per-layer collective/overlap pattern, and relative rankings (project
  to full depth with a two-point linear per-layer model). They do **not** transfer for absolute
  SLA numbers or capacity/HBM-budget-dominated quantities (cache-block count, max concurrent
  sequences, batch at the capacity boundary, KV headroom) — a reduced model has a completely
  different memory budget. Never let a reduced-model number carry a feasibility/calibration verdict.
- **Trust ordering: the device profile outranks a quick system-trace estimate on a >2× disagreement,
  and a mechanism claim ("X binds") always cites the ingested artifact.** A cheap serving-log read
  observes which phase occupies the device; it never licenses a *because*.
