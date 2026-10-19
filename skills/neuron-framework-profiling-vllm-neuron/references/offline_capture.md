# Method 2: Offline throughput + capture (fallback)

Runs `vllm bench throughput` with profiling env vars to emit NEFFs + system
traces, then captures a single execution of each NEFF in isolation with
`neuron-explorer capture`. Useful for isolating one NEFF's device trace on a
**dense** model. **Does not work for MoE or large-TP models** — see
[traps.md → Method 2 fails on MoE / large-TP models](traps.md#method-2-fails-on-moe--large-tp-models).

Prefer Method 1 (serve-based) — see SKILL.md → *Which method?*. Reach for
Method 2 only when the serve path is unusable on your target.

## 1. Run the profiling benchmark

```bash
./references/run_profile.sh <run_name> [vllm bench throughput args...]
```

> Invoke `references/*.sh` by their path **alongside this skill** — they are not
> assumed to exist in the target repo's cwd.

**Alternative:** the same NEFFs can be obtained by wrapping `run_bench.sh` with
`neuron-explorer inspect`:

```bash
export NEURON_RT_INSPECT_DEVICE_PROFILE=1
neuron-explorer inspect -o <output_dir> -- ./references/run_bench.sh <run_name> [args...]
```

## 2. Set up capture directory

Copy from the first worker:

```bash
PROFILE_DIR=$NEURON_PROJECT_ROOT/agent_artifacts/profiles/<run_name>
FIRST_WORKER=$(find $PROFILE_DIR -mindepth 1 -maxdepth 1 -type d -name "i-*" | sort | head -1)
SESSION_DIR=$(find $FIRST_WORKER -mindepth 1 -maxdepth 1 -type d)
mkdir -p $PROFILE_DIR/capture
cp -r "$SESSION_DIR"/* $PROFILE_DIR/capture/
```

## 3. Capture device traces for each NEFF

```bash
for neff in $PROFILE_DIR/capture/*.neff; do
    NEFF_ID=$(basename "$neff" | sed 's/neff_\(.*\)_vnc_0\.neff/\1/')
    mkdir -p $PROFILE_DIR/capture/$NEFF_ID
    NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 neuron-explorer capture \
        -n "$neff" \
        -s "$PROFILE_DIR/capture/$NEFF_ID/profile.ntff" \
        --profile-nth-exec=2 \
        --enable-dge-notifs \
        --collectives-workers-per-node=<TP_SIZE> \
        --collectives-worker-count=<TP_SIZE>
done
```

Replace `<TP_SIZE>` with the tensor parallel size used in the run (e.g. 16). In a
standard vllm autoregressive run there are 2 NEFFs (1 prefill, 1 decode).

## Mode B variant: full session trace (inline NTFFs)

Captures an NTFF for the entire session inline — every NEFF execution is traced,
no follow-up capture step. Larger output, higher overhead.

```bash
NEURON_RT_INSPECT_DEVICE_PROFILE=session ./references/run_profile.sh <run_name> [args...]
```

This produces the same `profile_nc_0_session_0.ntff` as Method 1's SESSION-mode
capture — the difference is only the trigger (env var over the whole run vs. HTTP
endpoints over a window). `NEURON_RT_INSPECT_DEVICE_PROFILE=session` maps to NRT's
`NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION`; MODEL mode (`=1`) is incompatible with
async execution.

## Method 2 profiling env vars

`run_profile.sh` defaults these but respects any value already set by the caller.
**These apply to Method 2 only** — Method 1 configures everything through the
`neuron_profiler` block instead.

| Variable | Default | Purpose |
|----------|---------|---------|
| `XLA_IR_DEBUG` | `1` | Debug info in compiled NEFFs |
| `XLA_HLO_DEBUG` | `1` | HLO-level debug info |
| `NEURON_FRAMEWORK_DEBUG` | `1` | Framework-level source attribution |
| `NEURON_RT_INSPECT_ENABLE` | `1` | Enable runtime inspection |
| `NEURON_RT_INSPECT_SYSTEM_PROFILE` | `1` | System-level trace (`ntrace.pb`, `trace_info.pb`) |
| `NEURON_RT_INSPECT_DEVICE_PROFILE` | `1` | Device mode (`1` = NEFFs only, `session` = inline NTFFs) |
| `NEURON_RT_INSPECT_OUTPUT_DIR` | `$PROFILE_DIR` | Where artifacts land |
| `NEURON_RT_ENABLE_DGE_NOTIFICATIONS` | `1` | DMA packet data in traces (but see the [DGE hang trap](traps.md#dge-notification-overflow-on-moe-session-capture)) |

## System-trace capture requirements (gotchas)

Getting the system-trace `.pb` files (`ntrace.pb`, `cpu_util.pb`, `host_mem.pb`,
`trace_info.pb`) to actually appear cost ~9 failed capture attempts to work out. If you
run `neuron-explorer inspect` directly (instead of `run_profile.sh`), these are the
non-obvious requirements:

- **Set `VLLM_NEURON_WORKER_TERMINATION_TIMEOUT=60` (or higher).** This is the single most
  important one. At the default (5s) the TP workers are killed before the runtime flushes
  the system-trace buffers, so the `.pb` files come out **empty (0 bytes) or missing** even
  though `NEURON_RT_INSPECT_SYSTEM_PROFILE=1` is set. vllm-neuron logs an explicit warning
  about this ("Neuron profiling is enabled but VLLM_NEURON_WORKER_TERMINATION_TIMEOUT is at
  default (5s). Profile data may be incomplete.") that is easy to miss in the noise.
- **`neuron-explorer inspect` attaches to the LIVE process; `neuron-profile capture -t
  systrace -n <neff>` does NOT.** The latter replays the NEFF standalone in an isolated
  harness and captures **zero** host-side events (no `nrta_execute_schedule`, `nc_exec_running`,
  `cc_running`, etc.). If you need host-side dispatch/collective timing, you must use
  `inspect` on the running workload.
- **Invocation form:** `neuron-explorer inspect -o <dir> -- python3 <script>` — the `--`
  separator followed by a real launched command. Passing the command as a single quoted
  string (`"python3 x.py"`) fails with `fork/exec ... no such file or directory`.
- **Full env set required** (in addition to `VLLM_NEURON_WORKER_TERMINATION_TIMEOUT`):
  `NEURON_RT_INSPECT_ENABLE=1`, `NEURON_RT_INSPECT_SYSTEM_PROFILE=1`,
  `NEURON_RT_INSPECT_DEVICE_PROFILE=1`, `NEURON_RT_INSPECT_OUTPUT_DIR=<dir>`,
  `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1`, `XLA_IR_DEBUG=1`, `XLA_HLO_DEBUG=1`,
  `NEURON_FRAMEWORK_DEBUG=1`.

A successful capture produces one `i-<instance>_pid_<N>/<timestamp>/` directory per worker,
each containing the four `.pb` files plus the NEFFs. See
[profiling_attempts.md](profiling_attempts.md) for the full attempt-by-attempt log and
additional model-specific gotchas.
