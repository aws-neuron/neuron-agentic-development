# Profiling Attempts

## Common env

```bash
export VLLM_NEURON_VENV_PATH=~/NxDI-Env/venv
export VLLM_NEURON_DIR=~/GemmaTune/private-vllm-neuron
export NEURON_PROJECT_ROOT=~/GemmaTune
```

## Common args

```
--model ~/GemmaTune/.artifacts/models/gemma-4-31B-it
--max-model-len 1024 --max-num-batched-tokens 1024 --max-num-seqs 4
--tensor-parallel-size 16
--hf-overrides '{"text_config": {"num_hidden_layers": 4}}'
--additional-config '{"neuron_config": {"quantization": "bf16", "on_device_sampling_config": {"all_greedy": "true"}, "num_batched_tokens_buckets": [1024], "num_seqs_buckets": [4]}}'
--dataset-name random --random-input-len 512 --random-output-len 512 --num-prompts 4
```

---

## 1. `NEURON_RT_INSPECT_DEVICE_PROFILE=session`

**Command:**
```bash
NEURON_VISIBLE_DEVICES=16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  ./references/run_profile.sh gemma4_profile_test [common args]
```

**Output dir:** `~/.artifacts/profiles/gemma4_profile_test/`

**Throughput:** 838 tok/s output

**Files per worker (16 workers):**
```
i-0e1d1489ce83f0eff_pid_1595451/1779469455688895880/
├── cpu_util.pb
├── host_mem.pb
├── neff_768830514966607_vnc_0.neff
├── neff_955504708622379_vnc_0.neff
├── ntrace.pb
├── profile_nc_0_session_0.ntff
└── trace_info.pb
```

**Total files:** 113

---

## 2. `NEURON_RT_INSPECT_DEVICE_PROFILE=1`

**Command:**
```bash
export VLLM_NEURON_VENV_PATH=~/NxDI-Env/venv
export VLLM_NEURON_DIR=~/GemmaTune/private-vllm-neuron
export NEURON_PROJECT_ROOT=~/GemmaTune
export NEURON_RT_INSPECT_DEVICE_PROFILE=1
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  ./references/run_profile.sh gemma4_device1_test [common args]
```

**Output dir:** `~/.artifacts/profiles/gemma4_device1_test/`

**Throughput:** 860 tok/s output

**Files per worker (16 workers):**
```
i-0e1d1489ce83f0eff_pid_1622440/1779473458189368791/
├── cpu_util.pb
├── host_mem.pb
├── neff_768830514966607_vnc_0.neff
├── neff_955504708622379_vnc_0.neff
├── ntrace.pb
└── trace_info.pb
```

**Total files:** 97

---

## 3. `NEURON_RT_INSPECT_DEVICE_PROFILE=0`

**Command:**
```bash
export NEURON_RT_INSPECT_DEVICE_PROFILE=0
NEURON_VISIBLE_DEVICES=16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  ./references/run_profile.sh gemma4_device0_test [common args]
```

**Output dir:** `~/.artifacts/profiles/gemma4_device0_test/`

**Throughput:** 885 tok/s output

**Files per worker (16 workers):**
```
i-0e1d1489ce83f0eff_pid_1626046/1779473560857252143/
├── cpu_util.pb
├── host_mem.pb
├── ntrace.pb
└── trace_info.pb
```

**Total files:** 65

---

## 4. `neuron-explorer inspect` wrap + `NEURON_RT_INSPECT_DEVICE_PROFILE=0`

**Command:**
```bash
export NEURON_RT_INSPECT_DEVICE_PROFILE=0
export NEURON_RT_INSPECT_ENABLE=0
source $VLLM_NEURON_VENV_PATH/bin/activate
cd $VLLM_NEURON_DIR
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NEURON_SKIP_EFA_AFFINITY=1 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
  neuron-explorer inspect -o ~/GemmaTune/.artifacts/profiles/gemma4_inspect_test \
  -- vllm bench throughput [common args] \
  --output-json ~/GemmaTune/.artifacts/profiles/gemma4_inspect_test/bench_results.json
```

**Output dir:** `~/.artifacts/profiles/gemma4_inspect_test/`

**Throughput:** 1095 tok/s output

**Files per worker (16 workers):**
```
i-0e1d1489ce83f0eff_pid_1629392/1779473664557168673/
├── cpu_util.pb
├── host_mem.pb
├── ntrace.pb
└── trace_info.pb
```

**Total files:** 65

---

## 5. `neuron-explorer inspect` wrap + no profiling env vars (run_bench.sh)

**Command:**
```bash
source $VLLM_NEURON_VENV_PATH/bin/activate
cd $VLLM_NEURON_DIR
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NEURON_SKIP_EFA_AFFINITY=1 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
  neuron-explorer inspect -o ~/GemmaTune/.artifacts/profiles/gemma4_inspect_test2 \
  -- ./references/run_bench.sh gemma4_inspect_bench [common args]
```

**Output dir:** `~/.artifacts/profiles/gemma4_inspect_test2/`

**Throughput:** 1066 tok/s output

**Files per worker (16 workers):**
```
i-0e1d1489ce83f0eff_pid_1632873/1779473834720516420/
├── cpu_util.pb
├── host_mem.pb
├── ntrace.pb
└── trace_info.pb
```

**Total files:** 64

---

## 6. `neuron-explorer inspect` wrap + `NEURON_RT_INSPECT_DEVICE_PROFILE=1` (run_bench.sh)

**Command:**
```bash
export NEURON_RT_INSPECT_DEVICE_PROFILE=1
source $VLLM_NEURON_VENV_PATH/bin/activate
cd $VLLM_NEURON_DIR
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NEURON_SKIP_EFA_AFFINITY=1 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
  neuron-explorer inspect -o ~/GemmaTune/.artifacts/profiles/gemma4_inspect_test3 \
  -- ./references/run_bench.sh gemma4_inspect_bench3 [common args]
```

**Output dir:** `~/.artifacts/profiles/gemma4_inspect_test3/`

**Throughput:** 1075 tok/s output

**Files per worker (16 workers):**
```
i-0e1d1489ce83f0eff_pid_1636173/1779473986855949280/
├── cpu_util.pb
├── host_mem.pb
├── neff_768830514966607_vnc_0.neff
├── neff_955504708622379_vnc_0.neff
├── ntrace.pb
└── trace_info.pb
```

**Total files:** 96

---

## 7. `neuron-explorer inspect` wrap + `NEURON_RT_INSPECT_DEVICE_PROFILE=1` + `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1` (run_bench.sh)

**Command:**
```bash
export NEURON_RT_INSPECT_DEVICE_PROFILE=1
export NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1
source $VLLM_NEURON_VENV_PATH/bin/activate
cd $VLLM_NEURON_DIR
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NEURON_SKIP_EFA_AFFINITY=1 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
  neuron-explorer inspect -o ~/GemmaTune/.artifacts/profiles/gemma4_inspect_test4 \
  -- ./references/run_bench.sh gemma4_inspect_bench4 [common args]
```

**Output dir:** `~/.artifacts/profiles/gemma4_inspect_test4/`

**Throughput:** 857 tok/s output

**Files per worker (16 workers):**
```
i-0e1d1489ce83f0eff_pid_1644380/1779474858466406221/
├── cpu_util.pb
├── host_mem.pb
├── neff_768830514966607_vnc_0.neff
├── neff_955504708622379_vnc_0.neff
├── ntrace.pb
└── trace_info.pb
```

**Total files:** 96

---

## 8. Benchmark only (run_bench.sh, no profiling)

**Command:**
```bash
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  ./references/run_bench.sh gemma4_bench_test [common args]
```

**Output dir:** `~/.artifacts/bench/gemma4_bench_test/`

**Throughput:** 1112 tok/s output

**Files:**
```
bench_results.json
```

**Total files:** 1

---

## 9. Empty `.pb` files → set `VLLM_NEURON_WORKER_TERMINATION_TIMEOUT`

**Symptom:** `neuron-explorer inspect` runs cleanly and writes per-worker directories, but
the system-trace `.pb` files (`ntrace.pb`, `cpu_util.pb`, `host_mem.pb`, `trace_info.pb`)
are **0 bytes or absent** — only the NEFFs land. Reproduced across multiple env-var
combinations before the cause was found.

**Cause:** the default worker termination timeout (5s) kills the TP workers before the
runtime flushes the system-trace buffers on shutdown.

**Fix:** `VLLM_NEURON_WORKER_TERMINATION_TIMEOUT=60`. With it set (all other
`NEURON_RT_INSPECT_*` env vars unchanged), all four `.pb` files populate correctly
(`ntrace.pb` ~4-5MB/worker). vllm-neuron logs a warning about this at startup that is easy
to miss. This was a model-independent capture bug, not specific to any model.

Also confirmed here: `neuron-profile capture -t systrace -n <neff>` replays the NEFF
standalone and captures **zero** host-side events — it does not observe a live process.
Use `neuron-explorer inspect` on the running workload for host-side traces.

---

## 10. `vllm bench throughput` unpicklable-config failure (model-specific)

**Symptom:** `neuron-explorer inspect -- vllm bench throughput ...` on
`generic-encoder-only-model` crashes during worker spawn:
`AttributeError: Can't pickle local object '_register_llama_bidirec_hf_config.<locals>._LlamaBidirecHFConfig'`.

**Cause:** the model registers a **function-local** HF config class, which cannot be
pickled by `vllm bench`'s spawn-based multiprocessing.

**Workaround:** drive inference from a plain `LLM()` script (not the `vllm bench` CLI) with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, wrapped by `neuron-explorer inspect`. Model-specific;
noted so others hitting it don't chase the profiler.

---

## 11. Non-working overlap flags on the trn2 path (dead ends)

Attempting collective-compute overlap (windowed einsum / collective matmul) to hide
collective wall-time:

- `XLA_FLAGS=--xla_gpu_operand_bytes_threshold_for_windowed_einsum=... --xla_gpu_multi_streamed_windowed_einsum=true`
  → **no effect** on device time or TE/CC overlap. These `xla_gpu_*` flags target XLA's GPU
  backend, which the Neuron path does not use; they never reach the neuronx-cc invocation.
- `AwsNeuronCollectiveMatmul` / `run-collective-matmul` **exists** in the compiler binaries
  (`hlo-opt`/`hlo2penguin`) but is **not reachable** for the trn2 target via `XLA_FLAGS`,
  `--internal-backend-options`, or `--internal-hlo2tensorizer-options`
  (hlo2penguin only exposes `--drop-collective-tokens`). Empirically produced no overlap.

Conclusion: on this SDK version, collective-compute overlap is not enable-able for trn2
from the vllm-neuron compile path. Recorded so the same flags aren't re-tried.

---

## Serve-mode profiling attempts

Empirical findings from iterating on the two isolation recipes
(`references/run_profile_decode.sh`, `references/run_profile_prefill.sh`) on
Gemma4-31B at 60L / TP=16 / trn2. Method-1 defaults in `neuron_profiler`:
`activities: ["device_profile", "system_profile"]`, `neuron_cores: [0]`.

### 12. `max_iterations=1` → NTFF present, `CcOp.parquet` empty

Two-iteration decode capture with `delay_iterations=0 max_iterations=1`, log-triggered.

- NTFF: emitted, ingests cleanly.
- `Instruction.parquet`, `DmaPacket.parquet`, `CcInstruction.parquet`: populated.
- **`CcOp.parquet`: 0 bytes** — the aggregation step needs to pair CcInstruction
  BEGIN/END events across at least 2 recorded iterations. Downstream analysis
  (collective-partitioning stage in the analysis skill) fails.

Fix: `max_iterations=2`. The two-iteration run produces a valid `CcOp.parquet`
with one row per collective op. This is the recommended lower bound for any
serve-mode capture. Do not go smaller.

### 13. Prefill isolation — `max_iterations` alone doesn't work

Attempt: `delay_iterations=0 max_iterations=2` with `OSL=10` and no explicit
`/stop_profile`.

- NTFF: 3.6 GB.
- Actual iterations recorded: 11 (1 prefill + 10 decode).

The `WorkerProfiler.step()` counter increments on every `execute_model` call
regardless of workload origin — auto-stop fires at `> max_iterations`, but the
scheduler happily keeps dispatching iters until the whole OSL=10 request
completes. `max_iterations` does not bound iteration count on a workload with a
predetermined output length shorter than the profiler's ring buffer.

Fix: workload isolation. Send `OSL=1 CONC=1 --num-prompts 1 --num-warmups 0`
and follow the request completion with `curl POST /stop_profile` (the working
recipe in `run_profile_prefill.sh`). The NTFF then contains exactly the prefill
iteration.

### 14. `delay_iterations` vs log-poll for decode steady state

Attempt: `delay_iterations=N` with N chosen to skip prefill (`3` at BS=4, `10`
at BS=8) — the SKILL's original guidance.

- At small BS, prefill iterations are cheap and consistent; `delay_iterations=3`
  usually works.
- At production BS (8+) with `vllm bench serve` warmup + steady-state phasing,
  the "iteration to reach BS=N decode-only" is not stable across runs — warmup
  requests, arrival-rate jitter, and scheduler groupings all shift it. Empirical
  under-arming caused prefill contamination; over-arming missed the decode
  window entirely.

Fix: log-poll. vLLM emits
`Avg prompt throughput: 0.0 tokens/s ... Running: N reqs, Waiting: 0 reqs`
on a single scheduler-summary line every ~10 s once all N requests are past
prefill. Grep for the joint regex, then `curl POST /start_profile` with
`delay_iterations=0` and `max_iterations=2`. This is deterministic — the trigger
is an observed event, not a counted heuristic.

### 15. MODEL-mode / async-off for per-NEFF NTFFs — not viable

Attempt: turn off `--async-scheduling` and set NRT device profile mode to MODEL
(`NRT_INSPECT_DEVICE_PROFILE_MODE_MODEL`) hoping to get per-NEFF NTFFs at
per-iteration granularity.

- MODEL mode is not selectable from the libtorch profiling bindings — SESSION is
  hard-coded in `torch.classes.neuron.Runtime.start_profiling(...)`.
- Disabling `--async-scheduling` breaks compilation on the current
  vllm-neuron path (async-scheduling is required by the trn2 backend's
  scheduler).

Conclusion: SESSION mode with the two isolation recipes is the way to get
phase-pure NTFFs. Recorded so the same experiments aren't re-attempted.

### 16. Standalone `neuron-explorer capture` OOMs on 60L production shape

Attempt: Method 2 capture on Gemma4-31B TP=16 60L. The tool allocates every
NEFF ENTRY input independently — 40 K/V tensors at ~624 MB each = 22 GB per NC,
exceeding the 24 GB HBM budget once profiler buffers are added.

- vLLM runtime pools those 40 KV tensors into 10 shared HBM buffers via
  `_shared_dtype_view` (`vllm_neuron/vllm/worker/neuron_model_runner.py`), so at
  runtime the KV footprint is ~21 GB — fits with room to spare.
- `neuron-explorer capture` has no mechanism to receive the framework's pool-
  sharing declaration. `.artifact_metadata_v0.json.io_map` handles output→input
  aliasing (verifiable via `neuron-explorer capture -v 2`) but does NOT collapse
  the 40 input allocations.

Workaround: use Method 1 (serve-based). Long-term fix documented as a feature
request for `neuron-explorer capture --input-pool-manifest` (external file:
`/mnt/data/neuron_explorer_feature_request/neuron_explorer_input_pool_manifest.md`).

For 12L Gemma4 the 40-input allocation still fits standalone — Method 2 works
at reduced layer count, which is what `gemma4_walkthrough.md` uses.

