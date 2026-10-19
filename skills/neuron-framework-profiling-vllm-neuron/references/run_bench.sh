#!/bin/bash
# Usage: ./run_bench.sh <run_name> [vllm bench throughput args...]
#
# Runs vllm bench throughput without profiling env vars.
#
# Requires: VLLM_NEURON_VENV_PATH, VLLM_NEURON_DIR
# Optional: NEURON_PROJECT_ROOT (default: $PWD). Outputs go to <root>/agent_artifacts/bench/<run_name>
#
# Example:
#   ./run_bench.sh gemma4_full \
#       --model $HF_HOME/gemma-4-31B-it \
#       --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 4 \
#       --tensor-parallel-size 16 \
#       --additional-config '{"neuron_config": {"quantization": "bf16", "on_device_sampling_config": {"all_greedy": "true"}, "num_batched_tokens_buckets": [4096], "num_seqs_buckets": [4]}}' \
#       --dataset-name random --random-input-len 3500 --random-output-len 500 --num-prompts 4

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <run_name> [vllm bench throughput args...]"
    echo ""
    echo "  run_name:  Name for this benchmark run (output goes to \$OUTPUT_ROOT/<run_name>)"
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
# There is deliberately NO separate ARTIFACTS_DIR / BENCH_OUTPUT_DIR any more. They were four names for one
# idea, and a campaign that set some of them scattered its evidence across unrelated directories.
# Profiling on a box whose local disk is not your project? Point the project root AT that disk
# (export NEURON_PROJECT_ROOT=/local/scratch/<you>) -- one variable, same structure.
PROJECT_ROOT="${NEURON_PROJECT_ROOT:-$PWD}"

OUTPUT_ROOT="$PROJECT_ROOT/agent_artifacts/bench"

RUN_NAME="$1"
shift

BENCH_DIR="$OUTPUT_ROOT/$RUN_NAME"

mkdir -p "$BENCH_DIR"

source "$VLLM_NEURON_VENV_PATH/bin/activate"
cd "$VLLM_NEURON_DIR"

# Timeouts
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"
export NEURON_SKIP_EFA_AFFINITY="${NEURON_SKIP_EFA_AFFINITY:-1}"

echo "=== Benchmark run: $RUN_NAME ==="
echo "Output: $BENCH_DIR"
echo "Command: vllm bench throughput $*"
echo ""

vllm bench throughput "$@" --output-json "$BENCH_DIR/bench_results.json"
