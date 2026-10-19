---
name: neuron-framework-profiling-vllm-neuron
description: |
  Guides profiling vllm-neuron models on Trainium: capturing device (NTFF) and
  system traces from a live server via vLLM's /start_profile and /stop_profile
  endpoints (the primary method), or offline via `vllm bench throughput` +
  `neuron-explorer capture` (fallback). Covers the neuron_profiler config,
  steady-state decode capture, and the common traps.
  Use when encountering "profile model", "benchmark vllm-neuron",
  "profile running server", "start_profile", "stop_profile",
  "profile live vllm serve", "capture NEFF from vllm", "get system trace",
  "capture NTFF from model run", "get profiling artifacts", "profile decode step",
  "profile MoE model", "per-rank profiling".
---

# Profiling vllm-neuron Models

Use this skill on its own or with a campaign controller. The serve-based and offline
methods below can be used directly; the bundled driver is a convenience for automating
prefill/decode captures. A controller, campaign grant and attempt ID are not prerequisites.

Use the installed vLLM Neuron profiling interface. Record package versions for
reproducibility; a different version number or unfamiliar private API is not a reason
to reject profiling. To inspect configuration compatibility without loading a model:

```bash
python3 references/profile_run.py --run-spec run_spec.json --check-capabilities <run_name>
```

This prints a diagnostic without changing existing run files. During a normal capture,
the driver saves the same diagnostic in `capabilities.json`. Missing version metadata,
unfamiliar interfaces and an unavailable probe produce warnings; the installed server
validates its configuration. Successful diagnostics or HTTP responses do not establish
capture success: check the requested traces. The driver records new nonempty trace files
in `capture.json` and fails when a requested, recognized Neuron trace is missing. Inspect
unfamiliar profiler formats with the tools supplied by that profiler.

Keep requested and observed weight, compute and KV precision separate. `auto` and
import-time FP8 banners are not runtime observations. Optionally supply
`--runtime-config <raw-runtime-observation.json>` containing `effective_config` from the
runtime, rather than a copy of requested settings. `precision.json` records checkpoint
metadata and available observations; without runtime evidence, actual precision remains
unknown and trace collection can still proceed. If an attempt ID is supplied, it must
match in the run specification and runtime observations.

`measurement.json` saves benchmark results and return code before profiler stop/cleanup.
`cleanup.json` records teardown separately; cleanup cannot erase results or make them pass.
The driver refuses an existing runfile: reconcile/resume the native child and existing output.
Use `safe_serve.py status --runfile <path> --json` for `live|terminal|unverified` process
state and separate cleanup receipt. Do not infer gains from process state.

**When used with a campaign controller:** preserve its granted baseline/configuration and
allowed tuning axes. Propagate `attempt_id` in the run spec or pass
`--attempt-id <task.attempt_id>`. Submit observed `runtime_config.json` as a PRIMARY_MEASURED
`runtime_config` artifact when the campaign requires it. Preserve raw evidence; the
controller normalizes aliases and validates witnessed defaults. A user-required precision
mismatch remains a blocker, and scalar token limits and bucket lists remain distinct.
Diagnose tooling problems before changing serving source or the campaign's baseline.

> **Read device time from the LIVE system trace (`nc_exec_running`), not from an isolated
> NEFF replay.** A standalone `neuron-explorer capture -n <neff>` / `neuron-profile` replay
> undercounts real multi-rank collective traffic (measured: ~12.9 ms replay vs ~18 ms live
> for the same TP=16 graph). Use the isolated replay only for per-op *structure*; quote
> per-request device time from the live `nc_exec_running` window. Do not build
> "serving-overhead" or "device-floor" conclusions on a replay number.

## Prerequisites

**Hardware:** trn (preferably trn2) instance running Ubuntu 24.04 DLAMI.

**Environment:** Follow the [NxDI-Env setup guide](https://github.com/aws-neuron/private-vllm-neuron/wiki/vLLM-Neuron:-Setup-and-Development)
to provision a venv and install system packages. This setup tracks Neuron SDK
alpha releases.

After setup, verify these are installed:

**System debs** (via `sudo apt install ./dep/apt/aws-neuronx-*.deb`):
- `aws-neuronx-dkms` — Neuron kernel driver
- `aws-neuronx-runtime-lib` — Neuron runtime
- `aws-neuronx-collectives` — collective communication library
- `aws-neuronx-tools` — provides `neuron-explorer`, `neuron-ls`, `neuron-top`

**Python packages** (see `src/AGENTS.md` for the single install command):
- `nki` — NKI kernel language
- `neuronx-cc` — Neuron compiler
- `torch-xla` — PyTorch XLA backend
- `libtorch-neuronx-lite` — Neuron libtorch bindings (exposes the NRT profiling
  bindings on `torch.classes.neuron.Runtime`)
- `nki-cpu-simulator` — NKI CPU simulator
- `nki-library` — NKI kernel library
- `nki-library-testing` — NKI test utilities
- `vllm-neuron` — vLLM Neuron backend plugin (editable install)

**Model weights:** Download the model checkpoint locally (e.g. via `huggingface-cli download`).
The path is passed as `--model` to `vllm serve` / `vllm bench`.

**Configuration** — set these variables before using this skill:

| Variable | Description |
|----------|-------------|
| `VLLM_NEURON_VENV_PATH` | Path to the venv with Neuron SDK packages |
| `VLLM_NEURON_DIR` | Path to the private-vllm-neuron repo |
| `NEURON_PROJECT_ROOT` | The PROJECT directory this run belongs to (default: the cwd). ALL outputs land under `<root>/agent_artifacts/` — `models/`, `profiles/`, `bench/` |
| `HF_HOME` | Model weights location (default: `$NEURON_PROJECT_ROOT/agent_artifacts/models`) |

**Where outputs go (the shared project-root contract).** `NEURON_PROJECT_ROOT` is the same input
every Neuron skill in this family takes — the parallelism-planner and hardware-entitlement skills
accept it as `--project-root` too — so one export puts a profile, the entitlement floor it is
compared against, and the plan that chose the config in ONE evidence trail:

```
$NEURON_PROJECT_ROOT/agent_artifacts/{profiles,bench,models}/<run_name>
  else $PWD/agent_artifacts/{profiles,bench,models}/<run_name>
```

**There is one knob and no override.** `ARTIFACTS_DIR`, `PROFILE_OUTPUT_DIR` and
`BENCH_OUTPUT_DIR` were removed: four names for one idea meant a campaign that set some of them
scattered its plan, floors, profiles and reports across unrelated directories, and no reader
could tell which run produced what. Profiling on a box whose local disk is not your project?
Point the project root AT that disk (`export NEURON_PROJECT_ROOT=/local/scratch/$USER`) — same
variable, same structure, and the paths still read the same way everywhere.

> **Environment note:** The commands below assume a single interactive trn2 box.
> If you run on a managed cluster (Slurm, etc.), the launch wrapper differs
> (job submission, whole-node allocation) but everything about *how profiling
> works* — the `neuron_profiler` config, the HTTP endpoints, HBM/util tuning,
> and the traps — is identical. Adapt the launch step to your scheduler.

## Which method?

There are two ways to capture a device trace. **Prefer Method 1 (serve-based).**

| Situation | Method |
|-----------|--------|
| Default. Any model, including **MoE / expert-parallel / large TP (>16)**. Profile a running server, target steady-state decode, select ranks. | **1 — Serve + HTTP endpoints** |
| You need one NEFF's device trace in isolation, or a scripted offline capture on a dense model where the serve path isn't convenient. | **2 — Offline throughput + capture** (fallback) |

Method 1 is what works for the hard cases. `neuron-explorer capture` (Method 2)
crashes on large-TP NEFFs with baked-in collectives, and `NEURON_RT_INSPECT_*`
device profiling crashes MoE NKI kernels (see [Traps](#traps--recovery)). All
methods produce the same artifacts (NTFF + NEFF + `ntrace.pb`), so any of them
feeds the same [analysis skill](#step-3-analyze).

---

## Method 1: Serve + HTTP endpoints (primary)

Runtime NRT profiling is toggled on a **live `vllm serve` instance** via vLLM's
`/start_profile` and `/stop_profile` endpoints — no restart, no separate capture
step. The traces (NTFF) and the NEFFs needed to render them are written to the
configured output directory. Implemented by `NeuronProfiler`
(`vllm_neuron/vllm/worker/neuron_profiler.py`), which subclasses vLLM's
`WorkerProfiler` and calls the NRT inspect bindings.

### 1. Launch the server with profiling enabled

```bash
source "$VLLM_NEURON_VENV_PATH/bin/activate"
cd "$VLLM_NEURON_DIR"

vllm serve "$HF_HOME/<model>" \
  --tensor-parallel-size <TP> \
  --max-model-len <N> --max-num-seqs <B> \
  --additional-config '{"neuron_config": {...}, "neuron_profiler": {"output_dir": "'"$NEURON_PROJECT_ROOT/agent_artifacts/profiles"'/<run_name>"}}' \
  --profiler-config '{"profiler": "cuda", "delay_iterations": 3, "max_iterations": 5}' \
  > /tmp/<run_name>.log 2>&1 &
```

- **`--profiler-config '{"profiler": "cuda"}'`** mounts the `/start_profile` and
  `/stop_profile` routes. `"cuda"` is a deliberate misnomer on Neuron — it is the
  only `ProfilerKind` slot a plugin can reinterpret (`Literal["torch", "cuda"]`);
  on Neuron it dispatches to `NeuronProfiler`, not CUDA. Any CUDA-related warnings
  are harmless. (`"torch"` gives a CPU-side torch profiler instead — rarely what
  you want here.)
- **`neuron_profiler` block** lives inside `--additional-config` alongside
  `neuron_config`. Fields (parsed by `NeuronProfilerConfig`):

  | Field | Type | Default | Description |
  |-------|------|---------|-------------|
  | `activities` | list[str] | `["device_profile", "system_profile"]` | What to capture. Valid: `device_profile`, `system_profile`, `host_memory`, `cpu_util`, `all` |
  | `neuron_cores` | list[int] \| null | `null` → rank 0 only | Which **worker ranks** (`local_rank`) profile. NOT physical cores — do not confuse with `NEURON_VISIBLE_DEVICES`. Each rank owns one NeuronCore. |
  | `output_dir` | str | `./neuron_profiles` | Where traces + NEFFs land |
  | `sys_trace_max_events_per_nc` | int \| null | `null` (NRT default) | Ring-buffer event cap per NeuronCore. Raise for long system-profile sessions to avoid event loss. |

- **`delay_iterations` / `max_iterations`** (in `--profiler-config`) control the
  capture window. One iteration = one `execute_model` call = one batched
  forward pass. See
  [serve_profiling.md → Capturing steady-state decode](references/serve_profiling.md#capturing-steady-state-decode)
  for how to size these — in particular the `max_iterations=2` CcOp lower bound.
- **Debug env vars at compile time** — set these three before `vllm serve`
  so the NEFFs carry the debug info the analysis skill needs for source
  attribution (mapping instructions back to HLO / framework source lines).
  They must be set at *compile* time, not at capture time, because the info
  is baked into the NEFF; a warm cache only carries it if the original
  compile had them set. Both isolation scripts export these defensively:

  | Variable | Default | Purpose |
  |---|---|---|
  | `XLA_IR_DEBUG` | `1` | Debug info in compiled NEFFs |
  | `XLA_HLO_DEBUG` | `1` | HLO-level debug info |
  | `NEURON_FRAMEWORK_DEBUG` | `1` | Framework-level source attribution |

- **Model-specific env vars must match the target workload EXACTLY.** The
  compile cache hashes the full neuronx-cc argv, and workload-affecting env
  vars alter that argv. Before invoking the recipe, identify every such env
  var the target run uses, export them explicitly in this shell, and keep
  the exact same set across cold/warm reruns — otherwise the warm cache
  silently misses and you pay a full recompile. The recipe scripts
  intentionally do not set these; profile the exact workload you care about.

- **DGE notifications (opt-in per capture)** —
  `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1` adds per-DMA-packet metadata to the
  device trace so the `DmaPacket` / `DmaPacketAggregated` parquet tables
  populate (needed for DMA-efficiency bounds). NRT reads the env var **once
  at init**, so it must be set before `vllm serve` starts — the
  `neuron_profiler` config block cannot toggle it. The isolation scripts
  gate it behind `ENABLE_DGE=1` (default off). Keep it off on MoE decode
  graphs and on dense models with tight KV configs; see
  [traps.md → DGE notification overflow](references/traps.md#dge-notification-overflow-on-moe-session-capture).

### 2. Warm up, then bracket the profiled window

Wait for compilation + warmup to finish (`tail -f /tmp/<run_name>.log` until the
server is serving — see [monitoring](#monitoring-and-hangs)). Then:

```bash
curl -X POST http://localhost:8000/start_profile
# ... drive requests (see below) ...
curl -X POST http://localhost:8000/stop_profile
```

If you set `delay_iterations`/`max_iterations`, the profiler arms and disarms
itself automatically around those iteration counts — `start_profile` still gates
when the state machine begins counting.

**Driving requests:**
- **Instruction-tuned models:** use `/v1/chat/completions` (not `/v1/completions`),
  so the chat template applies — otherwise the model emits end-of-turn
  immediately and you capture empty generations. Keep `prompt_tokens + max_tokens
  ≤ max-model-len` or you get a 500 / IndexError.
- **Batch / TTFT profiling:** `vllm bench serve --profile` drives a serving
  instance and auto-brackets `/start_profile`→`/stop_profile` for you:
  ```bash
  vllm bench serve --backend vllm --model "$HF_HOME/<model>" \
    --dataset-name random --random-input-len 512 --random-output-len 512 \
    --num-prompts 8 --max-concurrency 4 --profile \
    --percentile-metrics ttft,e2el
  ```

### Method 1 sub-recipes

All the practical details of Method 1 live in
[references/serve_profiling.md](references/serve_profiling.md):

- **[Capturing steady-state decode](references/serve_profiling.md#capturing-steady-state-decode)**
  — `delay_iterations` / `max_iterations` sizing, the `max_iterations=2` CcOp
  lower bound, and why a naive bracket captures only prefill.
- **[Isolating decode via log-triggered `/start_profile`](references/serve_profiling.md#isolating-decode-via-log-triggered-start_profile)**
  — deterministic BS=N decode capture; runnable
  [`run_profile_decode.sh`](references/run_profile_decode.sh).
- **[Isolating prefill via idle serve + explicit `/stop_profile`](references/serve_profiling.md#isolating-prefill-via-idle-serve--explicit-stop_profile)**
  — one-shot workload isolation; runnable
  [`run_profile_prefill.sh`](references/run_profile_prefill.sh).
- **[System-only capture (device profile disabled)](references/serve_profiling.md#system-only-capture-device-profile-disabled)**
  — `activities: ["system_profile"]` to skip the NTFF and profiler HBM pool.
- **[Making room for the profiler on large models](references/serve_profiling.md#making-room-for-the-profiler-on-large-models)**
  — `--gpu-memory-utilization` guidance, allocation-failure signature.
- **[Output layout (Method 1)](references/serve_profiling.md#output-layout-method-1)**
  — the per-worker directory tree and the `neffs/graph_<hash>.neff`
  auto-copy behaviour.

Both isolation recipe `.sh` are thin back-compat shims over a single driver,
[`references/profile_run.py`](references/profile_run.py) (`--phase {decode,prefill}`).
The shared launch / wait-ready / teardown goes through
[`references/safe_serve.py`](references/safe_serve.py) (isolated `setsid` group,
recorded-PGID kill-after-capture, self-guard) — no bare `vllm serve &`, no box-wide
`pkill`. The `<run_name>` interface, output layout, and compile key (serve env +
`--additional-config` / `--profiler-config` JSON) are unchanged; prove parity with
`profile_run.py --dry-run --phase <p> <run>`, which prints the resolved serve argv,
env delta, and both config JSONs without touching a device.

**Two input modes.** Consuming a master-authored **`run_spec.json` is the default** whenever
one is available (located via `--run-spec PATH` > `$PROFILE_RUN_SPEC` >
`<profiles>/<run_name>/run_spec.json`); the spec is the source of truth for
`serve_config` / `workload` / `model_env`, and `model_env` is replayed into the child's
environment. **Env-var resolution is the fallback** when no spec is found (unchanged from the
original recipes; `PROFILE_IGNORE_RUN_SPEC=1` forces it). Only host infra
(`VLLM_NEURON_VENV_PATH`, `VLLM_NEURON_DIR`, `NEURON_PROJECT_ROOT`) always comes from env.

**Handoff contract (env in → `run_spec.json` + `manifest.json` out).** The two artifacts are
split by lifecycle: **`run_spec.json`** (the INPUT contract) is written at *launch* — the
structured `serve_config` (the launch source of truth; the exact argv is derived from it),
the `workload` knobs, and `model_env` — so it survives a compile/launch failure;
**`manifest.json`** (the OUTPUT) is written before cleanup with the profile paths + NEFF
hashes and a `run_spec` back-reference. Model-specific compile flags (kernel routing,
backend toggles, dedup) live OUTSIDE `resolve_config` and reach `vllm serve` by inheritance;
declare their NAMES in `PROFILE_MODEL_ENV` (and/or use the `GEMMA4_` / `VLLM_NEURON_BACKEND_`
auto-capture prefixes) so the driver snapshots them into `run_spec.json`'s `model_env`,
making the compile key self-documenting and a run reproducible. See the invocation example
and full samples under *run_spec.json (input) + manifest.json (output)* /
[`references/run_spec.example.json`](references/run_spec.example.json) +
[`references/manifest.example.json`](references/manifest.example.json).

`serve_profiling.md` also holds the `neuron_profiler` /
`--profiler-config` field reference, the profiler call chain for debugging
when the endpoints don't mount, and the offline `LLM` API variant.

---

## Method 2: Offline throughput + capture (fallback)

Runs `vllm bench throughput` with profiling env vars to emit NEFFs + system
traces, then captures each NEFF's device trace standalone with
`neuron-explorer capture`. Useful for isolating one NEFF's execution on a
**dense** model when the serve path is inconvenient. **Does not work for MoE
or large-TP models** — see
[traps.md → Method 2 fails on MoE / large-TP](references/traps.md#method-2-fails-on-moe--large-tp-models).

Basic form:

```bash
./references/run_profile.sh <run_name> [vllm bench throughput args...]
# then loop `neuron-explorer capture` over the emitted NEFFs
```

Full recipe (three steps, `Mode B` inline session variant, Method-2-only
env vars, and the `neuron-explorer inspect` system-trace gotchas) lives in
[references/offline_capture.md](references/offline_capture.md).

---

## Traps & recovery

Find your symptom in the table below and follow the link into
[references/traps.md](references/traps.md). Ranked by how often they bite.

| Symptom | Fix / detail |
|---|---|
| `No available shared memory broadcast block found in 60 seconds` on a long run | Usually benign compilation, not a hang — see [traps.md](references/traps.md#no-available-shared-memory-broadcast-block-is-usually-not-a-hang) |
| Generation succeeds but the profile output dir is tiny | Profiler buffers couldn't allocate — [lower `--gpu-memory-utilization`](references/traps.md#profiler-buffers-cant-allocate-silent-no-ntff) |
| `neuron-explorer view --ingest-only` runs for hours / emits 0-byte parquet | [Session NTFF too large](references/traps.md#session-ntff-too-large-to-ingest); keep `max_iterations` small |
| `NCC_EOOM002 Maximum peak HBM usage ... exceeds HBM limit` at compile | [Stale cache entry, oversized I/O, or `gpu_memory_utilization`](references/traps.md#compile-time-hbm-oom-ncc_eoom002) |
| `neuron-explorer capture` crashes on large-TP NEFF (`group_id < replica_groups.size()`) or `NCC_INLA001` on MoE | [Method 2 doesn't work on MoE / large-TP — use Method 1](references/traps.md#method-2-fails-on-moe--large-tp-models) |
| `notification queue overflow` in session capture of MoE | [Disable DGE for session/serve capture on MoE](references/traps.md#dge-notification-overflow-on-moe-session-capture) |
| Next launch fails with `nrta status=5` / `WorkerProc initialization failed` | [Orphaned device-holding workers](references/traps.md#orphaned-device-holding-workers-after-a-killedfailed-run) — tear down by recorded PGID via [`safe_serve.py`](references/traps.md#safe-launch-and-teardown), never `pkill -f` |
| Byte-identical source re-fails with the same error after a failed compile | [Compile-cache poisoning](references/traps.md#compile-cache-poisoning-after-a-failed-compile) — purge before retest |
| Instruction-tuned model captures empty generations | [Use `/v1/chat/completions`](references/traps.md#instruction-tuned-models-capture-empty-generations) so the chat template applies |

## Monitoring and hangs

Always redirect output to a log file — runs can appear to hang during compilation
or on a scheduler issue:

```bash
vllm serve ... > /tmp/<run_name>.log 2>&1 &
tail -f /tmp/<run_name>.log
```

> **Prefer launching through [`references/safe_serve.py`](references/traps.md#safe-launch-and-teardown)**
> rather than a bare `&`: it isolates the server in its own `setsid` process group (so teardown
> can never hit the agent's session), polls readiness, and gives you the self-guarded
> kill-after-capture SIGKILL for the `empty_cache` teardown-hang. A bare `nohup ... &` shares the
> launcher's process group and is the setup that makes `pkill -f` / `kill -9 -<PGID>` dangerous.

Before deciding a run is hung, apply the
[compile-vs-hang heuristic](references/traps.md#no-available-shared-memory-broadcast-block-is-usually-not-a-hang).

## Device pinning (single interactive box)

`NEURON_VISIBLE_DEVICES` pins a run to specific Logical NeuronCores (LNCs) so you
can run several in parallel on one box. This is **device allocation** and is
distinct from Method 1's `neuron_cores` config (which selects *worker ranks* to
profile). In LNC=2 config, each LNC is 2 physical cores sharing an HBM.

> **trn2.48xlarge:** 16 physical devices × 8 pcores each = 128 pcores. In LNC=2:
> 64 LNCs (indices 0–63). A TP=16 run uses 16 LNCs — up to 4 parallel runs.

Check `neuron-ls` for free devices (cores with PID = NA) and pick a contiguous
range matching your TP size. The number of LNCs needed = `tensor_parallel_size`.
On a managed cluster this is handled by the scheduler; `NEURON_VISIBLE_DEVICES`
is the single-box fallback.

## Data parallelism and the capture: profile ONE replica — DP>1 is not a capture input

A device/kernel capture is a **per-replica** object, and DP replicas are **independent**: each
runs the same compiled graph on its own TP shard group and exchanges **no** collectives with the
other replicas. So the decode/prefill graph, the per-op instruction mix, the intra-replica
collective pattern, and the per-step device time are **identical at every DP degree**. You do
**not** need `data-parallel-size > 1` to profile — bring up a single replica (or profile one rank
of one replica inside a multi-replica server) and that capture fully characterizes the kernel
behaviour of every replica. Raising DP for a capture only multiplies identical traces and spends
extra HBM (profiler pool + per-core notification buffer) and host budget for no additional
information. This is why the capture pins ONE rank (`neuron_cores: [0]`).

**This is a different question from aggregate throughput, which DP>1 *does* change — and which is
exactly why an aggregate must be MEASURED concurrently, never projected as `single-replica × N`:**

- Independent replicas still contend for **shared node resources a lone replica never stresses**:
  host CPU (scheduling, on-device-sampling glue, detokenization), PCIe / host↔device DMA, the
  serving/HTTP stack, the interconnect fabric shared by every replica's collectives, and the node
  power/thermal envelope. Contention can only make the true aggregate **≤** the projection, so
  `× N` is a one-sided over-estimate, not a measurement.
- A joint latency+throughput target must hold in **one** finite-rate run: under concurrent load
  per-replica latency degrades from that same contention, and the figure to report is the **worst
  percentile across replicas**, not the isolated single-replica latency. Projecting throughput from
  one run and quoting latency from another stitches two different operating points together.

So: **profile at DP=1 / one rank — the kernel artifact is DP-invariant; certify a node aggregate
only from a concurrent N-replica measured run** (one client per replica, with the overlapping
measurement-window check the analysis skill's verification rules require). A projection is fine to
bound or motivate the answer, never to certify it.

---

## Step 2: Benchmark & project

Use `run_bench.sh` (no profiling overhead) to establish baseline throughput:

```bash
./references/run_bench.sh <run_name> [vllm bench throughput args]
```

The full model **can** be profiled directly — lower `--gpu-memory-utilization`
so the profiler pool fits. For iteration speed on kernel work, profile a
reduced-layer build first (`--hf-overrides '{"num_hidden_layers": k}'`) and
project to full depth with a two-point linear per-layer model. Full recipe,
what is/isn't portable across depths, and the A/B and Plan-B (no-NTFF) fallbacks
live in
[references/reduced_and_full_model.md](references/reduced_and_full_model.md).

## Step 3: Analyze

All methods produce NTFF + NEFF + `ntrace.pb`. Ingest into queryable parquet with
`neuron-explorer view --ingest-only` and analyze with the
`neuron-framework-profile-analysis-vllm-neuron` skill.

## Walkthroughs

- [references/serve_profiling.md](references/serve_profiling.md) — Method 1
  (serve-based) worked example, `neuron_profiler` field reference, offline-`LLM`
  variant.
- [references/gemma4_walkthrough.md](references/gemma4_walkthrough.md) — Method 2
  (offline throughput + capture) worked example: Gemma4 31B, reduced layers,
  device capture, full-model benchmark.
- [references/profiling_attempts.md](references/profiling_attempts.md) — empirical
  log of what env-var / capture combinations produce which artifacts.

---

## Sweep-campaign cost discipline (capture side) — cross-reference

When this skill is driven inside a multi-config **optimization sweep**, the capture itself is
minutes but a fresh server bring-up (compile + multi-rank weight load) is not — so *how the
capture is scheduled across configs* dominates the campaign's wall-clock. The per-capture
mechanics below are covered above; this section is the campaign-level framing. Canonical
playbook: `../experimental-neuron-framework-optimization/references/SWEEP_ACCELERATION.md` (Tiers 1 & 3).

- **Capture INSIDE the one warm baseline server, not as a separate launch.** Profile-capture
  on/off, in-flight concurrency/batch, and arrival rate do **not** change the compiled-graph
  cache key, so they need no recompile — take the capture and all such variations off a single
  warm server. Only a change to shard degree, block count, bucket, KV dtype, quantization,
  optimization level, or the memory-util knob forces a new bring-up. The two isolation
  recipes ([decode](references/serve_profiling.md#isolating-decode-via-log-triggered-start_profile),
  [prefill](references/serve_profiling.md#isolating-prefill-via-idle-serve--explicit-stop_profile))
  point their `vllm serve` at the sweep's `VLLM_CACHE_ROOT`; a warm cache skips the
  ~4 min NEFF-load + bucket-warmup entirely, so each isolation capture is
  bench-time-only.
- **Budget the profiler's HBM into the FIRST compile.** Lowering the memory-util knob to fit the
  profiler's pool + per-core notification buffer changes the cache key and forces a recompile; do
  it once up front so it does not recur per sweep point. A suspiciously tiny trace dir is a
  FAILED capture, not a small workload — verify the artifact is non-trivially sized.
- **One device job at a time; full teardown after ANY abnormal exit.** A cancelled/interrupted
  turn does **not** kill the server it spawned. Stacking a new capture onto stale or contending
  server state makes every request time out and yields empty/garbage traces (which then get
  re-captured). Tear down by the recorded process group with `references/safe_serve.py teardown`
  (setsid-isolated, self-guarded) — never `pkill -f` or a bare `kill -9 -<PGID>` (see
  [Safe launch and teardown](references/traps.md#safe-launch-and-teardown)). Verify from the
  device listing that nothing holds the device, then relaunch. Never assert "nothing is running"
  from memory — check the process table.
- **Deadline + pollable progress, never a big sleep.** Size readiness timeouts to measured load ×
  replicas; detect a wedge by absent log progress, not an expiring timeout. A profile *viewer*
  bound to a port is finished work waiting for a client, not a running capture — never block a
  step on it.

These are the same D4/D7/F4/D15/D16/D10 rules the orchestrator enforces; keeping them here means
the capture agent applies them without being told per delegation.
