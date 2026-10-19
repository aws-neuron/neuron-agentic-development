#!/bin/bash
# Usage: ./run_profile_prefill.sh <run_name>
#
# Isolate a prefill NEFF+NTFF from a live vllm serve. Workload isolation, not
# iteration-counter isolation: /start_profile fires while the server is idle,
# then a single OSL=1 one-shot request runs the prefill dispatch, then
# /stop_profile fires the moment the bench command returns. No decode iters
# make it into the trace (aside from possibly one BS=1 decode NEFF-warmup step,
# which is trivially filterable by NEFF hash in trace_info.pb).
#
# Requires: VLLM_NEURON_VENV_PATH, VLLM_NEURON_DIR, TARGET_MODEL
# Optional: NEURON_PROJECT_ROOT (default: $PWD). Outputs go to
#           <root>/agent_artifacts/profiles/<run_name>/.
# Optional: VLLM_CACHE_ROOT — point at the warm sweep cache to skip recompile.
#
# Workload knobs (env overrides):
#   TP=16 ISL=8192 MAXLEN=10240 SEG=8192
#   NUM_SEQS_MAX=48 NUM_SEQS_BUCKETS='[1,2,4,8,16,24,32,48]'
#   NUM_GPU_BLOCKS=<override or unset for auto> MAX_TOP_K=64 PORT=8041
#
# OSL, CONC, NPROMPTS are pinned (OSL=1 CONC=1 NPROMPTS=1) — the whole point of
# this mode is a single-shot prefill dispatch. Do not override.
#
# Model-specific compile flags belong OUTSIDE this script (caller's responsibility).
# Optional: PROFILE_MODEL_ENV="NAME1 NAME2 ..." — declare those flag NAMES so they are
#   snapshotted into run_spec.json's `model_env`. GEMMA4_* / VLLM_NEURON_BACKEND_* are
#   auto-captured by prefix; list anything else (dedup toggle, device pinning, platform
#   target) here. run_spec.json (INPUT) is written at launch; manifest.json (OUTPUT, with a
#   run_spec back-reference) after teardown.
#
# Thin shim → profile_run.py. Server launch/wait/teardown is handled by
# safe_serve.py (isolated setsid group, recorded-PGID kill-after-capture,
# self-guard) — NO bare `vllm serve &`, NO box-wide `pkill -f`. See
# profile_run.py and PROFILE_RUN_REFACTOR_PLAN.md.
exec python3 "$(dirname "$0")/profile_run.py" --phase prefill "$@"
