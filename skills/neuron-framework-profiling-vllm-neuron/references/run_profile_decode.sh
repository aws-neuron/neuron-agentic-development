#!/bin/bash
# Usage: ./run_profile_decode.sh <run_name>
#
# Isolate a decode NEFF+NTFF from a live vllm serve, using log-poll steady-state
# detection instead of the static delay_iterations heuristic. The profiler arms
# only after vLLM's scheduler-summary log line proves the workload has reached
# BS=$STEADY_STATE_BS decode-only steady state, so no prefill iterations
# contaminate the trace. max_iterations=2 auto-stops after two decode iterations
# — the smallest value that preserves CcOp aggregation (see SKILL.md).
#
# Requires: VLLM_NEURON_VENV_PATH, VLLM_NEURON_DIR, TARGET_MODEL
# Optional: NEURON_PROJECT_ROOT (default: $PWD). Outputs go to
#           <root>/agent_artifacts/profiles/<run_name>/.
# Optional: VLLM_CACHE_ROOT (default: unset — vllm picks its own cache dir). If
#           you point this at the warm cache from an earlier sweep, the server
#           skips ~4 min of NEFF load + bucket warmup.
#
# Workload knobs (env overrides):
#   TP=16 ISL=8192 OSL=1024 MAXLEN=10240 SEG=8192
#   NUM_SEQS_MAX=48 NUM_SEQS_BUCKETS='[1,2,4,8,16,24,32,48]'
#   NUM_GPU_BLOCKS=<override or unset for auto> MAX_TOP_K=64 PORT=8041
#   CONC=8       — vllm bench --max-concurrency, drives target batch size
#   NPROMPTS=50  — enough decode iterations for the profiler to see steady state
#   STEADY_STATE_BS=$CONC — the "Running: N reqs" batch size to wait for
#
# Model-specific env vars belong OUTSIDE this script. This recipe is
# workload-agnostic; anything that changes the compile key (kernel routing
# flags, dedup toggles, etc.) is the caller's responsibility. Export them
# in the same shell that invokes this script — the compile cache hashes them
# into the NEFF key, so an inconsistent value between two runs will produce
# two different NEFFs and silently miss the warm cache.
#
# Optional: PROFILE_MODEL_ENV="NAME1 NAME2 ..." — declare the compile-affecting
#   model flag NAMES so they are snapshotted into run_spec.json's `model_env`
#   (making the compile key self-documenting). GEMMA4_* and VLLM_NEURON_BACKEND_*
#   are auto-captured by prefix; list anything else (e.g. VLLM_NEURON_DISABLE_INPUT_DEDUP,
#   NEURON_VISIBLE_DEVICES, NEURON_PLATFORM_TARGET_OVERRIDE) here.
#
# Artifacts: run_spec.json (INPUT: workload + serve_config + model_env) is written at
#   launch and survives a failed compile; manifest.json (OUTPUT: profile paths + NEFF
#   hashes + a run_spec back-reference) is written after teardown.
#
# Thin shim → profile_run.py. Server launch/wait/teardown is handled by
# safe_serve.py (isolated setsid group, recorded-PGID kill-after-capture,
# self-guard) — NO bare `vllm serve &`, NO box-wide `pkill -f`. See
# profile_run.py and PROFILE_RUN_REFACTOR_PLAN.md.
exec python3 "$(dirname "$0")/profile_run.py" --phase decode "$@"
