#!/bin/bash
# Usage: ./run_profile.sh <run_name> [vllm bench throughput args...]
#
# Requires: VLLM_NEURON_VENV_PATH, VLLM_NEURON_DIR
# Optional: NEURON_PROJECT_ROOT (default: $PWD). Outputs go to <root>/agent_artifacts/profiles/<run_name>
#
# Example:
#   ./run_profile.sh llama3_baseline \
#       --model $HF_HOME/Llama-3.1-8B-Instruct \
#       --max-model-len 256 --max-num-batched-tokens 256 --max-num-seqs 4 --tensor-parallel-size 8 \
#       --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": true}, "num_batched_tokens_buckets": [256], "num_seqs_buckets": [4]}}' \
#       --dataset-name random --random-input-len 200 --random-output-len 50 --num-prompts 8

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <run_name> [vllm bench throughput args...]"
    echo ""
    echo "  run_name:  Name for this profiling run (output goes to \$OUTPUT_ROOT/<run_name>)"
    echo "  args:      All remaining arguments are passed directly to 'vllm bench throughput'"
    exit 1
fi

: "${VLLM_NEURON_VENV_PATH:?Set VLLM_NEURON_VENV_PATH to the Neuron SDK venv}"
: "${VLLM_NEURON_DIR:?Set VLLM_NEURON_DIR to the private-vllm-neuron repo}"

# Where outputs go: ONE knob, the PROJECT ROOT ($NEURON_PROJECT_ROOT, default the cwd) -- the same
# input the parallelism-planner and hardware-entitlement skills take as --project-root. Everything
# lands under <root>/agent_artifacts/, so a profile, the entitlement floor it is compared against
# and the plan that chose the config share one evidence trail.
#
# There is deliberately NO separate ARTIFACTS_DIR / PROFILE_OUTPUT_DIR any more. They were four names for one
# idea, and a campaign that set some of them scattered its evidence across unrelated directories.
# Profiling on a box whose local disk is not your project? Point the project root AT that disk
# (export NEURON_PROJECT_ROOT=/local/scratch/<you>) -- one variable, same structure.
PROJECT_ROOT="${NEURON_PROJECT_ROOT:-$PWD}"

OUTPUT_ROOT="$PROJECT_ROOT/agent_artifacts/profiles"

RUN_NAME="$1"
shift

PROFILE_DIR="$OUTPUT_ROOT/$RUN_NAME"

mkdir -p "$PROFILE_DIR"

source "$VLLM_NEURON_VENV_PATH/bin/activate"
cd "$VLLM_NEURON_DIR"

# Timeouts
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"
export VLLM_NEURON_WORKER_TERMINATION_TIMEOUT="${VLLM_NEURON_WORKER_TERMINATION_TIMEOUT:-60}"

# Neuron runtime
export NEURON_SKIP_EFA_AFFINITY="${NEURON_SKIP_EFA_AFFINITY:-1}"

# Profiling
export XLA_IR_DEBUG="${XLA_IR_DEBUG:-1}"
export XLA_HLO_DEBUG="${XLA_HLO_DEBUG:-1}"
export NEURON_FRAMEWORK_DEBUG="${NEURON_FRAMEWORK_DEBUG:-1}"
export NEURON_RT_INSPECT_ENABLE="${NEURON_RT_INSPECT_ENABLE:-1}"
export NEURON_RT_INSPECT_SYSTEM_PROFILE="${NEURON_RT_INSPECT_SYSTEM_PROFILE:-1}"
export NEURON_RT_INSPECT_DEVICE_PROFILE="${NEURON_RT_INSPECT_DEVICE_PROFILE:-1}"
export NEURON_RT_INSPECT_OUTPUT_DIR="${NEURON_RT_INSPECT_OUTPUT_DIR:-$PROFILE_DIR}"
export NEURON_RT_ENABLE_DGE_NOTIFICATIONS="${NEURON_RT_ENABLE_DGE_NOTIFICATIONS:-1}"

echo "=== Profiling run: $RUN_NAME ==="
echo "Output: $PROFILE_DIR"
echo "Command: vllm bench throughput $*"
echo ""

vllm bench throughput "$@" --output-json "$PROFILE_DIR/bench_results.json"
