# Serve-Based Profiling (Method 1) — Worked Reference

Runtime NRT profiling toggled on a live `vllm serve` instance through vLLM's
`/start_profile` and `/stop_profile` endpoints. This is the primary profiling
method — it works for MoE, expert-parallel, and large-TP models where the offline
`neuron-explorer capture` flow fails.

Implemented by `NeuronProfiler` (`vllm_neuron/vllm/worker/neuron_profiler.py`),
which subclasses vLLM's `WorkerProfiler` and drives the NRT inspect bindings on
`torch.classes.neuron.Runtime`.

---

## `neuron_profiler` field reference

Configured inside `--additional-config` as `{"neuron_profiler": {...}}`. Parsed by
`NeuronProfilerConfig`; defaults below are the code defaults.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `activities` | list[str] | `["device_profile", "system_profile"]` | Activity types to capture. Valid: `device_profile`, `system_profile`, `host_memory`, `cpu_util`, `all`. |
| `neuron_cores` | list[int] \| null | `null` | Which **worker ranks** (`local_rank`) participate in profiling. `null` → only rank 0 profiles (minimizes overhead). Each rank owns exactly one NeuronCore. This is rank selection, **not** physical-device pinning (that is `NEURON_VISIBLE_DEVICES`). |
| `output_dir` | str | `./neuron_profiles` | Output directory for traces + NEFFs. |
| `sys_trace_max_events_per_nc` | int \| null | `null` (NRT default) | Max system-trace events stored across ring buffers per NeuronCore. Raise for long sessions to avoid event loss. |

The `--profiler-config` object (separate from the block above) carries:

| Field | Purpose |
|-------|---------|
| `profiler` | `"cuda"` → `NeuronProfiler` (device/system NRT profiling). `"torch"` → CPU-side torch profiler. Anything else → `ValueError`. `"cuda"` is a misnomer reused because vLLM's `ProfilerKind` is `Literal["torch","cuda"]` and a plugin cannot add a new kind — on Neuron it does NOT invoke CUDA. |
| `delay_iterations` | Skip N `execute_model` calls before `nrt_inspect_begin`. Used to arm mid-decode. |
| `max_iterations` | Auto-stop after N iterations. Keep small (~5) to bound NTFF size. |

`warmup_iterations` / `wait_iterations` / `active_iterations` are PyTorch-profiler
schedule fields and do not apply to the Neuron path.

One iteration = one `execute_model` call = one batched forward pass. The
`WorkerProfiler.step()` state machine (driven from `NeuronWorker.execute_model`)
implements the delay/max counting.

---

## Full worked example

### 1. Launch the server

```bash
source "$VLLM_NEURON_VENV_PATH/bin/activate"
cd "$VLLM_NEURON_DIR"

RUN=decode_profile
PROFILE_DIR="$NEURON_PROJECT_ROOT/agent_artifacts/profiles/$RUN"

vllm serve "$HF_HOME/<model>" \
  --tensor-parallel-size 16 \
  --max-model-len 1024 --max-num-seqs 4 \
  --gpu-memory-utilization 0.6 \
  --additional-config '{
    "neuron_config": {
      "on_device_sampling_config": {"all_greedy": "true"},
      "num_batched_tokens_buckets": [1024],
      "num_seqs_buckets": [4]
    },
    "neuron_profiler": {
      "activities": ["device_profile", "system_profile"],
      "neuron_cores": [0],
      "output_dir": "'"$PROFILE_DIR"'"
    }
  }' \
  --profiler-config '{"profiler": "cuda", "delay_iterations": 3, "max_iterations": 5}' \
  > /tmp/$RUN.log 2>&1 &
```

- `--gpu-memory-utilization 0.6` leaves HBM for the profiler pool (~3.4 GB + 128
  MB/core). Raise it if the model is small; lower it (≈0.3) for MoE.
- `neuron_cores: [0]` profiles only rank 0. Add more ranks to compare across TP
  ranks (e.g. to check collective imbalance).
- `delay_iterations: 3` / `max_iterations: 5` capture a short steady-state decode
  window. Scale `delay` up with batch size (≈10 at batch 8).

### 2. Wait for warmup

```bash
tail -f /tmp/$RUN.log      # wait until the server is actually serving
```

Long "No available shared memory broadcast block" stretches during this phase are
benign compilation — see
[traps.md → shm broadcast is (usually) not a hang](traps.md#no-available-shared-memory-broadcast-block-is-usually-not-a-hang).
Do not kill it.

### 3. Bracket the profiled requests

```bash
curl -X POST http://localhost:8000/start_profile

# Instruction-tuned model — use the chat endpoint so the template applies:
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<model>", "messages": [{"role":"user","content":"Write a short story."}], "max_tokens": 256}'

curl -X POST http://localhost:8000/stop_profile
```

With `delay_iterations`/`max_iterations` set, the profiler arms/disarms itself
around the configured iteration counts; `start_profile` gates when counting
begins.

### Alternative: `vllm bench serve --profile`

For batch / per-request TTFT profiling with automatic start/stop bracketing:

```bash
vllm bench serve --backend vllm --model "$HF_HOME/<model>" \
  --dataset-name random --random-input-len 512 --random-output-len 512 \
  --num-prompts 8 --max-concurrency 4 \
  --profile --percentile-metrics ttft,e2el
```

### 4. Analyze

```bash
# --data-path sends the ingested parquet into this run's evidence trail under the project
# root, not the neuron-explorer default (~/.local/share/neuron-profile). Same location the
# analysis skill then reads from.
neuron-explorer view -d "$PROFILE_DIR" --ingest-only --display-name "$RUN" \
    --data-path "$NEURON_PROJECT_ROOT/agent_artifacts/data/$RUN"
```

Then use the `neuron-framework-profile-analysis-vllm-neuron` skill.
Keep `max_iterations` small — a multi-GB NTFF can take 1–2 h / ~40 GB RAM to
ingest and may silently emit 0-byte parquet.

---

## Capturing steady-state decode

**This is the single most important technique.** A naive
`start_profile → send requests → stop_profile` records the heavy **prefill
first**, which overflows the device inspect ring buffer so it stops **before**
decode — leaving the decode NEFF with **zero** engine/DMA/FLOP data (only `CcOp`
collective ops survive). To capture decode:

- Use **`delay_iterations`** to skip past prefill/warmup so `nrt_inspect_begin`
  fires **mid-decode** with an empty buffer, and **`max_iterations`** to auto-stop.
- **`delay_iterations` must scale with batch size** — decode dispatches more work
  per iteration at higher batch. Observed working values: `delay_iterations=3` at
  batch 4; `delay_iterations=10` at batch 8. At batch 8 with `delay=3` the buffer
  still overflowed and decode coverage was zero.
- **Keep `max_iterations` small (≈5, not 25/50).** A large window produces a
  multi-GB NTFF that can take 1–2 h and ~40 GB RAM to ingest — and can silently
  emit 0-byte parquet. See
  [traps.md → session NTFF too large to ingest](traps.md#session-ntff-too-large-to-ingest).
- **Do not set `max_iterations=1` either.** The NTFF is emitted, but the analysis
  skill's `CcOp.parquet` (aggregated collective-op table) is empty — pairing
  CcInstruction BEGIN/END events across an all-reduce/all-gather requires ≥2
  recorded iterations. `max_iterations=2` is the smallest value that preserves
  collective-op analysis. Verified on Gemma4-31B TP=16 (max=1 → no `CcOp.parquet`;
  max=2 → aggregation works).

## Isolating decode via log-triggered `/start_profile`

The `delay_iterations` counter above is a static heuristic — it assumes prefill's
iteration cost is predictable, but under `vllm bench serve` the scheduler mixes
prefill chunks and decode iterations opportunistically, so the count-to-steady-state
varies by batch size, arrival pattern, and warmup requests. When you need the
device NTFF to contain **only** decode iterations at a specific batch size (e.g.
for kernel-level analysis of the decode NEFF), drive `/start_profile` off the
server log instead:

1. Launch the server with `--profiler-config '{"profiler": "cuda",
   "delay_iterations": 0, "max_iterations": 2}'` — no delay; the profiler arms on
   the very next `execute_model` call after `/start_profile`, and auto-stops after
   two iterations (the CcOp lower bound from above).
2. Kick off `vllm bench serve --max-concurrency N ...` in the background at the
   target batch size `N`.
3. Poll `serve.log` for the regex
   `Avg prompt throughput: 0\.0.*Running: N reqs, Waiting: 0 reqs`. vLLM emits
   both fields on the same scheduler-summary line every ~10 s; when it matches,
   all N requests are past prefill and the last window contained no prefill
   dispatch — the scheduler is guaranteed to run BS=N decode on the next
   iteration. Use a 300 s poll timeout.
4. `curl -X POST http://localhost:8000/start_profile`. The very next
   `execute_model` is BS=N decode; `max_iterations=2` stops the profiler two
   iterations later.
5. Wait for the bench workload to finish; `curl -X POST .../stop_profile` (the
   auto-stop already fired — this call is idempotent).

Because the trigger comes from an event the scheduler emits, this is robust to
warmup variability and different prompt distributions — the profiler cannot arm
until steady state is *observed*, not counted-for.

Runnable script: [run_profile_decode.sh](run_profile_decode.sh) — a thin shim over
[`profile_run.py`](profile_run.py) `--phase decode`. Server launch / wait-ready /
teardown is delegated to [`safe_serve.py`](safe_serve.py) (isolated `setsid` group,
recorded-PGID kill-after-capture, self-guard — no bare `vllm serve &`, no box-wide
`pkill`); the `curl` / steady-state-poll methodology above is unchanged.

## Isolating prefill via idle serve + explicit `/stop_profile`

Prefill is one iteration long, so `delay_iterations` cannot help — arming after
the prefill dispatch means you've missed it. Isolate prefill by triggering the
profiler while the server is **idle**, then firing a one-shot workload:

1. Launch the server as above (`delay_iterations: 0`, `max_iterations: 2` — the
   `max` here is defensive, not the stop trigger). Wait for READY. **Do NOT send
   any warmup requests.**
2. `curl -X POST http://localhost:8000/start_profile`. The profiler is armed;
   the iteration counter stays at 0 because no `execute_model` is running.
3. Fire ONE synchronous request with `vllm bench serve --num-warmups 0
   --num-prompts 1 --max-concurrency 1 --random-output-len 1`. The next
   `execute_model` after the request lands is the prefill dispatch — the counter
   ticks to 1, the profiler records the iteration, and the request returns
   (~250 ms for ISL=8192 on Gemma4).
4. `curl -X POST http://localhost:8000/stop_profile` **immediately** when the
   bench command returns. The stop is now the trigger — the max-counter is only
   a safety net.

Why OSL=1 and not OSL=0: OSL=0 is rejected by vLLM. OSL=1 is exactly one
iteration — the prefill pass emits the first output token itself via on-device
sampling; there is no separate decode step for the sampled token. Post-hoc,
`trace_info.pb` will list both the prefill NEFF and the BS=1 decode NEFF hashes
(the BS=1 graph gets warmed by the request even if no decode iteration executes);
filter by the prefill NEFF hash when ingesting.

Runnable script: [run_profile_prefill.sh](run_profile_prefill.sh) — a thin shim over
[`profile_run.py`](profile_run.py) `--phase prefill` (same `safe_serve.py`-managed
lifecycle as the decode recipe).

## System-only capture (device profile disabled)

Set `activities: ["system_profile"]` in the `neuron_profiler` block to emit only
the system-trace `.pb` files (`ntrace.pb`, `trace_info.pb`, plus
`cpu_util.pb` / `host_mem.pb` if listed) with **no NTFF and no profiler HBM
pool**. Use this when you want end-to-end dispatch, collective, and host-side
timing without the device profiler's ~3.4 GB + 128 MB/core buffer footprint —
e.g. on a full-HBM model where the profiler pool won't fit, or when you only
need cross-rank correlation across a longer session. `activities:
["device_profile"]` is the symmetric option (NTFF only, no system trace).
The two isolation recipes above always request both.

## Making room for the profiler on large models

The device profiler needs headroom — roughly a **~3.4 GB pool + ~128 MB
notification buffer per profiled core**. On a near-full-HBM model there may be no
room, and **profiler start is best-effort**: generation succeeds while **no NTFF
is written** (the output dir stays tiny, ~a few hundred KB). Watch for this
signature in the log:

```
NRT:nrt_profile_notification_subscribe_nc failed to subscribe to notification type N
TDRV:log_dev_mem Failed to allocate 128.000MB (usage: profiler buffers)
Failed to start profiler ... status=4 message=Allocation Failure
```

**Fix: lower `--gpu-memory-utilization`** to carve out the pool (≈0.6 for dense,
≈0.3 for MoE). This is why a full model *can* be profiled directly (see
[reduced_and_full_model.md → Profiling the full model](reduced_and_full_model.md#profiling-the-full-model)).
Note that changing `gpu_memory_utilization` (or block counts) changes
`num_gpu_blocks` → a new KV shape → a **new decode-graph hash → a cold
recompile**; profile from a warm cache where possible, or accept the one-time
recompile.

## Output layout (Method 1)

```
$NEURON_PROJECT_ROOT/agent_artifacts/profiles/<run_name>/
├── manifest.json                        ← resolved paths + NEFF hashes (isolation recipes)
├── output/                              (= neuron_profiler.output_dir)
│   ├── i-<instance>_pid_<pid>/<timestamp>/
│   │   ├── profile_nc_0_session_0.ntff  ← device trace (NTFF)
│   │   ├── ntrace.pb                    ← system trace
│   │   ├── trace_info.pb                ← trace metadata
│   │   ├── cpu_util.pb                  ← only if "cpu_util" in activities
│   │   └── host_mem.pb                  ← only if "host_memory" in activities
│   └── neffs/
│       ├── graph_<hash1>.neff           ← NEFFs auto-copied from compile cache
│       └── graph_<hash2>.neff
└── bench_c<N>.json (or bench_prefill.json)  ← workload metrics from vllm bench serve
```

### run_spec.json (input) + manifest.json (output) — the two handoff artifacts

Input and output are **two separate files** on purpose — they have different lifecycles and
owners:

- **`run_spec.json` — the handoff-IN artifact, written at LAUNCH** (before the server
  starts). It is the caller's contract: the resolved serve configuration plus the
  compile-affecting model flags. Because it is written up front, it **survives a
  compile/launch failure** — the case where you most need to see which flags were used (a
  wrong compile flag usually kills the server before any output could be written). It is
  also the diffable, reproducible record of a run. Full example:
  [`run_spec.example.json`](run_spec.example.json).

  ```json
  {
    "schema_version": "1.0",
    "run_name": "<passed to the script>", "phase": "decode" | "prefill",
    "recipe": "decode_isolation" | "prefill_isolation",
    "workload": {"isl": ..., "osl": ..., "conc": ..., "nprompts": ...,
                 "steady_state_bs": ...},   // ONLY non-serve driving knobs (bench + arming)
    "serve_config": {                        // structured launch spec = the compile-key source of truth
      "model": "<path>", "tensor_parallel_size": 16, "dtype": "bfloat16",
      "kv_cache_dtype": "fp8_e4m3", "max_model_len": ..., "max_num_batched_tokens": ...,
      "max_num_seqs": ..., "num_gpu_blocks_override": ... | null, "block_size": 16,
      "optimization_level": 2, "port": 8041,
      "additional_config": { /* neuron_config (buckets, num_seqs_buckets, max_top_k) + neuron_profiler */ },
      "profiler_config": { "profiler": "cuda", "delay_iterations": 0, "max_iterations": 2 }
    },
    "model_env": {"GEMMA4_DECODE_BACKEND": "tkg", "VLLM_NEURON_BACKEND_TKG_CAST_WIDEN": "0", ...}
  }
  ```

  `serve_config` is the **structured, single source of truth** for the launch — the form a
  master agent authors and hands over (not a fragile argv string). The exact `vllm serve`
  argv is **derived from it** by the driver (`build_serve_argv(serve_config)` — the same call
  `launch_server` uses), so the launch matches the record and there is no second restatement
  to drift; the argv is regenerable from `serve_config` and therefore not stored redundantly.

  **`workload` and `serve_config` are deduplicated by the compile-key rule:** anything that
  becomes a `vllm serve` flag / `--additional-config` value — `tensor_parallel_size`,
  `max_model_len`, `max_num_batched_tokens`, `max_num_seqs`, `num_seqs_buckets` — defines the
  **compiled graph** and lives in `serve_config` only (change it → recompile). `workload`
  holds ONLY what `vllm bench serve` sends (`isl`/`osl`/`conc`/`nprompts`) plus the
  profiler-arming trigger (`steady_state_bs`) — vary these and you drive different traffic at
  the *same* compiled server. So no value is duplicated across the two blocks.
  (`--dry-run` prints both `serve_config` and the derived argv.)

  `model_env` is the crux of the handoff: the **compile-affecting, model-specific env that
  lives OUTSIDE `resolve_config`** (kernel routing, backend toggles, dedup, device pinning)
  and reaches `vllm serve` only by process inheritance. The driver snapshots it — from an
  explicit `PROFILE_MODEL_ENV="NAME1 NAME2 ..."` list unioned with a `GEMMA4_` /
  `VLLM_NEURON_BACKEND_` prefix allowlist — **before** applying its own defaults, so the
  compile key is self-documenting. Two runs' `model_env` can be diffed to explain a
  warm-cache miss or a behavioural delta. (`--dry-run` echoes `model_env` too, so a master
  can verify the flags it intends to hand off before spending a device.)

- **`manifest.json` — the handoff-OUT artifact, written at TEARDOWN.** It **resolves the two
  nested-glob paths** every downstream consumer would otherwise expand:
  `i-<instance>_pid_<pid>/<timestamp>/` on the raw side (the `<timestamp>` is a unix-ns int
  the harness cannot know in advance) and the per-run NEFF hashes from `trace_info.pb`. It
  carries a `run_spec` back-reference to the input file. Full example:
  [`manifest.example.json`](manifest.example.json).

  ```json
  {
    "schema_version": "1.1",
    "recipe": "decode_isolation" | "prefill_isolation",
    "run_name": "<passed to the script>",
    "workload": { ... },                       // kept for schema-1.0 consumers
    "run_spec": "/absolute/.../run_spec.json",  // back-reference to the input artifact
    "profile": {
      "session_dir": "/absolute/path/output/i-.../<ts>/",
      "device_ntff": "/absolute/.../profile_nc_0_session_0.ntff",
      "system_trace_pb": "/absolute/.../ntrace.pb",
      "trace_info_pb": "/absolute/.../trace_info.pb",
      "cpu_util_pb": "...",       // known 0-byte caveat, path present regardless
      "host_mem_pb": "...",       // same
      "neff_dir": "/absolute/.../output/neffs"
    },
    "neff_hashes": ["<hex>", ...],
    "bench_result_json": "/absolute/.../bench_c<N>.json"
  }
  ```

Downstream consumers (the analysis skill, human operators, the eval harness) should **read
`manifest.json` first** for the profile artifacts, and follow its `run_spec` back-reference
when they need the config/flags. Missing files come out as `null` rather than crashing the
emission, so a partially-completed run still tells you what did land; and a run that died at
compile still leaves a `run_spec.json` even when there is no manifest. Legacy captures
without either file still work — consumers can compute the same paths by globbing.

### Invoking / handing off to this skill

There are **two input modes**. Consuming a master-authored `run_spec.json` is the **default**
whenever one is available; env-var resolution is the **fallback**.

**INPUT mode (default) — a master hands over a `run_spec.json`.** The master authors the
structured `serve_config` + `workload` + `model_env` (see
[`run_spec.example.json`](run_spec.example.json)) and points the driver at it. Only host infra
stays in the environment; the model path + all compile knobs come from the spec, and
`model_env` is replayed into the child's environment. The driver locates the spec via
`--run-spec PATH` > `$PROFILE_RUN_SPEC` > `<profiles>/<run_name>/run_spec.json` (if present).

```bash
# host-specific infra only (NOT part of the portable spec):
export VLLM_NEURON_VENV_PATH=/path/to/venv VLLM_NEURON_DIR=/path/to/private-vllm-neuron
export NEURON_PROJECT_ROOT=/path/to/campaign
export VLLM_CACHE_ROOT=/path/to/warm/cache        # optional: warm-cache reuse (a host path)

# master writes spec.json (serve_config + workload + model_env), then:
./run_profile_decode.sh --run-spec /path/to/spec.json decode_c16
# or drop it at profiles/decode_c16/run_spec.json and just: ./run_profile_decode.sh decode_c16
python3 references/profile_run.py --run-spec /path/to/spec.json --dry-run decode_c16   # verify first
```

**ENV-fallback mode — no spec.** The config is built from environment variables, unchanged
from the original recipes. Export the shared context + the model-specific compile flags,
declare those flags via `PROFILE_MODEL_ENV` so they land in `model_env`, then invoke:

```bash
export VLLM_NEURON_VENV_PATH=/path/to/venv VLLM_NEURON_DIR=/path/to/private-vllm-neuron
export TARGET_MODEL=/path/to/model NEURON_PROJECT_ROOT=/path/to/campaign
# model-specific compile flags (the caller OWNS these; declare names so they are recorded):
export GEMMA4_DECODE_BACKEND=tkg GEMMA4_SWA_DECODE_BACKEND=tkg
export GEMMA4_TKG_FUSE_OPROJ=1 GEMMA4_TKG_DIRECT=0 GEMMA4_TKG_FUSE_OPROJ_KERNEL=partitioned
export VLLM_NEURON_BACKEND_TKG_CAST_WIDEN=0 VLLM_NEURON_DISABLE_INPUT_DEDUP=1
export NEURON_VISIBLE_DEVICES=0-15 NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export PROFILE_MODEL_ENV="VLLM_NEURON_DISABLE_INPUT_DEDUP NEURON_VISIBLE_DEVICES NEURON_PLATFORM_TARGET_OVERRIDE"
# (GEMMA4_* and VLLM_NEURON_BACKEND_* are auto-captured by prefix; list the rest above.)
export TP=16 ISL=8192 OSL=1024 MAXLEN=10240 SEG=8192 NUM_SEQS_MAX=48 NUM_GPU_BLOCKS=79872 CONC=16
./run_profile_decode.sh decode_c16    # builds run_spec.json from env, then runs
```

`PROFILE_IGNORE_RUN_SPEC=1` forces the env path even if a spec exists (use it to re-run with
changed env vars, so a stale in-dir `run_spec.json` is not silently reused). Either way,
`run_spec.json` lands the instant the server is launched (see
[`run_spec.example.json`](run_spec.example.json)) and `manifest.json` lands after a clean
capture+teardown (see [`manifest.example.json`](manifest.example.json)) and points back at
it — together they give the analysis skill (or a re-run) everything without re-reading the
caller's launch script.

NEFFs are auto-copied from the local compile cache
(`get_neuron_compile_cache_dir()`, default `~/.cache/vllm/neuron/compile_cache`,
or `/tmp/vllm_neuron_wdir_$USER/neuron/compile_cache` when `$HOME` is on NFS) so
Neuron Explorer can correlate device profiles with NEFF instruction data. The
Method-1 layout uses `neffs/graph_<hash>.neff` — the `neff_<id>_vnc_0.neff`
naming and the capture `sed` regex in Method 2 do **not** apply here.

> Note: only the NTFF + `ntrace.pb` reliably carry the real data;
> `cpu_util.pb` / `host_mem.pb` sometimes come back 0 bytes in serve captures.

---

## Offline `LLM` API variant

The design doc shows `vllm serve`, but the same profiler works through the offline
`LLM` API — useful for scripted, reproducible captures without a server:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="<model>",
    tensor_parallel_size=16,
    max_model_len=1024,
    gpu_memory_utilization=0.6,
    additional_config={
        "neuron_config": {"on_device_sampling_config": {"all_greedy": "true"}},
        "neuron_profiler": {
            "activities": ["device_profile", "system_profile"],
            "output_dir": "/path/to/profile",
        },
    },
    profiler_config={"profiler": "cuda", "delay_iterations": 3, "max_iterations": 5},
)

llm.start_profile()
llm.generate(["prompt"] * 4, SamplingParams(max_tokens=256))
llm.stop_profile()
```

Same `delay/max_iterations` decode-capture semantics apply. Set
`gpu_memory_utilization` low enough to leave room for profiler buffers.

---

## Disaggregated inference (DI) — OUT OF SCOPE, retained for reference only

> **Do not use this section to plan, configure, or recommend a deployment.**
> Serving is single-node and colocated: one replica serves both prefill and decode
> within one node. Disaggregated inference, split prefill/decode pools, xPyD
> topologies, and any multi-node or multi-instance arrangement are out of scope and
> must never be suggested. This section is kept only to explain the multi-backend
> profile-toggle mechanism should you encounter a pre-existing DI deployment; it is
> not a supported configuration to create.

In DI, prefill and decode run as separate `vllm serve` instances (possibly on
different hosts), each with its own `/start_profile` / `/stop_profile`. There is
no built-in atomic cross-server profile toggle.

The DI proxy server (which routes requests to the prefill/decode backends) is
extended to forward `/start_profile` and `/stop_profile` to **all** backends in
parallel, returning success only when all have responded. Each server writes its
own profiles locally.

```
Client            Proxy              Prefill        Decode
  │  start_profile  │                  │              │
  ├────────────────>├─ start_profile ─>│              │
  │                 ├─ start_profile ─────────────────>│
  │  <── 200 (after all backends ack) ─┤              │
  │      ... requests profiled on both servers ...    │
  │  stop_profile   │                  │              │
  ├────────────────>├─ stop_profile ──>│              │
  │                 ├─ stop_profile ──────────────────>│
```

If your deployment's proxy doesn't support pass-through, hit `/start_profile` and
`/stop_profile` on each DI server manually.

---

## Profiler call chain (for debugging)

If the endpoints don't mount or profiling does nothing, trace the path:

```
POST /start_profile
  → AsyncLLM.start_profile()
  → EngineCore.profile(is_start=True)
  → ModelExecutor.profile(is_start=True)
  → collective_rpc("profile", args=(True, prefix))
  → NeuronWorker.profile(is_start=True)        # gated by _should_profile()
  → NeuronProfiler.start() → _start()
  → torch.classes.neuron.Runtime().start_profiling(
        output_dir, activities, None, sys_trace_max_events, neff_cache_dir)
```

- The routes mount **only** when `--profiler-config.profiler` is set at startup.
- `NeuronWorker._should_profile()` returns True on rank 0 by default, or on ranks
  in `neuron_cores`. Ranks that shouldn't profile no-op.
- The NRT binding takes **no DGE parameter** — the serve path does not enable DGE
  notifications (avoids the MoE notification-queue overflow seen with DGE-on
  session capture).

## Open questions to verify against your installed version

- Whether the debug env vars used by Method 2 (`XLA_IR_DEBUG`, `XLA_HLO_DEBUG`,
  `NEURON_FRAMEWORK_DEBUG`) are needed for source attribution in serve-captured
  NEFFs, or whether the compile cache already carries that info.
- Exact behavior of `sys_trace_max_events_per_nc` tuning for very long
  system-profile sessions.
