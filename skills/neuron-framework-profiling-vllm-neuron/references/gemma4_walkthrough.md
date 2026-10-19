# Gemma4 31B Profiling Walkthrough (Method 2: offline throughput + capture)

> This walks through the **offline** `vllm bench throughput` + `neuron-explorer
> capture` flow (Method 2 / fallback). For the primary serve-based flow, see
> [serve_profiling.md](serve_profiling.md). Method 2 works for dense models like
> Gemma4; it does **not** work for MoE or large-TP models.


## Config

| Arg | Value | Why |
|-----|-------|-----|
| `--model` | `$HF_HOME/gemma-4-31B-it` | Model checkpoint path |
| `--tensor-parallel-size` | `16` | Gemma4 31B requires TP=16 to fit in HBM |
| `--hf-overrides` | `'{"text_config": {"num_hidden_layers": 4}}'` | Reduce layers as a convenient approximation. (The full model *can* also be profiled directly by lowering `--gpu-memory-utilization` to free the profiler pool — see SKILL.md "Profiling the full model". Note reduced layers change the compiled graph geometry, so the per-step profile is an approximation, not identical.) |
| `--max-model-len` | `1024` | Maximum sequence length (matches sliding window size) |
| `--max-num-batched-tokens` | `1024` | Prefill bucket size (must match max_model_len for single-bucket config) |
| `--max-num-seqs` | `64` | Batch size |
| `--num-gpu-blocks-override` | `8192` | 2× minimum (64 seqs × 64 blocks_per_seq = 4096 min). Tight pool reduces DMA amplification. |
| `--kv-cache-dtype` | `fp8` | FP8 KV cache (halves cache memory, enables larger batch) |
| `--additional-config` | see below | Neuron-specific config |

### `--additional-config`

```json
{
  "neuron_config": {
    "on_device_sampling_config": {"all_greedy": "true"},
    "num_batched_tokens_buckets": [1024],
    "num_seqs_buckets": [64]
  }
}
```

| Field | Value | Controls |
|-------|-------|----------|
| `on_device_sampling_config.all_greedy` | `"true"` | Deterministic sampling on device |
| `num_batched_tokens_buckets` | `[1024]` | Single prefill bucket → 1 prefill NEFF |
| `num_seqs_buckets` | `[64]` | Single decode bucket → 1 decode NEFF |

### Workload

| Arg | Value |
|-----|-------|
| `--dataset-name` | `random` |
| `--random-input-len` | `512` |
| `--random-output-len` | `512` |
| `--num-prompts` | `64` |

This produces: 64 prefill steps + ~512 decode steps (batch=64, so 512 decode dispatches).

---

## Step 1: Profile (4 layers)

> **Note:** `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1` causes hangs with this config
> (kv-fp8 + BS64 + 8192 blocks). Disable it during the profiling run and enable
> it only during the `neuron-explorer capture` step instead.

```bash
export VLLM_NEURON_VENV_PATH=~/NxDI-Env/venv
export VLLM_NEURON_DIR=~/GemmaTune/private-vllm-neuron
export NEURON_PROJECT_ROOT=~/GemmaTune

NEURON_RT_ENABLE_DGE_NOTIFICATIONS=0 \
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  ./references/run_profile.sh gemma4_kvfp8_bs64_4layer \
    --model ~/GemmaTune/.artifacts/models/gemma-4-31B-it \
    --max-model-len 1024 --max-num-batched-tokens 1024 --max-num-seqs 64 \
    --tensor-parallel-size 16 \
    --hf-overrides '{"text_config": {"num_hidden_layers": 4}}' \
    --num-gpu-blocks-override 8192 \
    --kv-cache-dtype fp8 \
    --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": "true"}, "num_batched_tokens_buckets": [1024], "num_seqs_buckets": [64]}}' \
    --dataset-name random --random-input-len 512 --random-output-len 512 --num-prompts 64 \
    > /tmp/gemma4_kvfp8_bs64_4layer.log 2>&1
```

**Result:** 2537 output tok/s

**Output per worker (×16):**
```
i-<instance>_pid_<N>/<session>/
├── neff_315622240876207_vnc_0.neff
├── neff_374305408507786_vnc_0.neff
├── ntrace.pb
├── cpu_util.pb
├── host_mem.pb
└── trace_info.pb
```

## Step 2: Set up capture directory

```bash
PROFILE_DIR=$NEURON_PROJECT_ROOT/agent_artifacts/profiles/gemma4_kvfp8_bs64_4layer
FIRST_WORKER=$(find $PROFILE_DIR -mindepth 1 -maxdepth 1 -type d -name "i-*" | sort | head -1)
SESSION_DIR=$(find $FIRST_WORKER -mindepth 1 -maxdepth 1 -type d)
mkdir -p $PROFILE_DIR/capture
cp -r "$SESSION_DIR"/* $PROFILE_DIR/capture/
```

**Output:**
```
capture/
├── neff_315622240876207_vnc_0.neff
├── neff_374305408507786_vnc_0.neff
├── ntrace.pb
├── cpu_util.pb
├── host_mem.pb
└── trace_info.pb
```

## Step 3: Capture device traces

```bash
for neff in $PROFILE_DIR/capture/*.neff; do
    NEFF_ID=$(basename "$neff" | sed 's/neff_\(.*\)_vnc_0\.neff/\1/')
    mkdir -p $PROFILE_DIR/capture/$NEFF_ID
    NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 neuron-explorer capture \
        -n "$neff" \
        -s "$PROFILE_DIR/capture/$NEFF_ID/profile.ntff" \
        --profile-nth-exec=2 \
        --enable-dge-notifs \
        --collectives-workers-per-node=16 \
        --collectives-worker-count=16
done
```

**Output:**
```
capture/
├── neff_315622240876207_vnc_0.neff
├── neff_374305408507786_vnc_0.neff
├── ntrace.pb, cpu_util.pb, host_mem.pb, trace_info.pb
├── 315622240876207/
│   ├── profile_rank_0_exec_2.ntff
│   └── ... (×16)
└── 374305408507786/
    ├── profile_rank_0_exec_2.ntff
    └── ... (×16)
```

## Step 4: Benchmark full model (60 layers)

Same config without `--hf-overrides`:

```bash
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  ./references/run_bench.sh gemma4_kvfp8_bs64_60layer \
    --model ~/GemmaTune/.artifacts/models/gemma-4-31B-it \
    --max-model-len 1024 --max-num-batched-tokens 1024 --max-num-seqs 64 \
    --tensor-parallel-size 16 \
    --num-gpu-blocks-override 8192 \
    --kv-cache-dtype fp8 \
    --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": "true"}, "num_batched_tokens_buckets": [1024], "num_seqs_buckets": [64]}}' \
    --dataset-name random --random-input-len 512 --random-output-len 512 --num-prompts 64
```

**Result:** 451 output tok/s
