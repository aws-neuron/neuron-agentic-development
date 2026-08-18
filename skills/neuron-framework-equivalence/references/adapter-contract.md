# Adapter Contract

The equivalence methodology (R-ratio, 3-tensor, 8 stages) is platform-agnostic. Only 5 integration points differ per serving stack. Each adapter implements these 5 methods.

## Adding a New Platform

1. Create `scripts/adapters/{stack_name}.py`
2. Subclass `StackAdapter` from `scripts/adapters/base.py`
3. Decorate with `@register("{stack_name}")`
4. Implement all 5 methods
5. Add `target_stack: "{stack_name}"` to `equiv_config_template.json`
6. Create `references/{stack_name}-adaptation.md` documenting stack-specific details

## The 5 Methods

### `init_distributed(tp_degree: int)`

Set up distributed process groups for CPU-mode testing. Called once before `create_model()`.

| Stack | Implementation |
|---|---|
| NxDI | `torch.distributed.init_process_group("gloo")` + `neuronx_distributed.parallel_state.initialize_model_parallel(tp)` |
| vLLM-Neuron (0.24.0) | `torch.distributed.init_process_group("gloo")` + `vllm_neuron.parallel.neuron_parallel_state.initialize_neuron_parallel_state(tp_global_ranks=..., local_rank=0)` — delegates `VllmConfig`/`set_current_vllm_config`/`initialize_model_parallel`. Do NOT call vLLM's `init_distributed_environment` first (pre-creates `_WORLD`). |

### `create_model(target_module_file, target_class_name, target_config_name, hf_model_path)`

Instantiate the target model in CPU mode. Returns an unweighted model for tree building and component mapping.

| Stack | Implementation |
|---|---|
| NxDI | `NeuronConfig(on_cpu=True)` → `ConfigClass.from_pretrained(path, neuron_config=...)` → `InnerClass(config)` |
| vLLM-Neuron | `NXD_CPU_MODE=1` → `AutoConfig.from_pretrained()` → `ConfigClass.from_configs(hf_config)` → `ModelClass(config)` |

### `load_weights(model, hf_model_path, dtype)`

Load HuggingFace weights into the target model with all stack-specific transforms.

| Stack | Key transforms |
|---|---|
| NxDI | `model.load(path)` or standard state_dict loading |
| vLLM-Neuron | All linear weights transposed (`.t()`), Q/K/V fused (`cat([Q.t(), K.t(), V.t()], dim=-1)`), weight names `_weight` not `.weight` |

### `forward(model, input_ids)`

Run a forward pass, return logits as float32 tensor.

| Stack | Signature handled internally |
|---|---|
| NxDI | `model(input_ids, attention_mask, position_ids)` → logits |
| vLLM-Neuron | `model(input_ids, positions, attn_metadata, sampling_positions)` → logits. Adapter constructs `attn_metadata` dict-of-dicts, allocates KV caches, passes `sampling_positions`. |

### `device_inference(model_id, tp_size, prompts, max_tokens)`

Run inference on actual Neuron hardware. Returns list of `{"text": ..., "tokens": [...]}`.

| Stack | Implementation |
|---|---|
| NxDI | `HuggingFaceGenerationAdapter` + compiled-model loading via `scripts/nxdi_compiled_loader.py` |
| vLLM-Neuron | `vllm.LLM(model=id, tensor_parallel_size=tp)` + `SamplingParams(temperature=0.0)` |

## Environment Check (`check_environment()`)

Optional, no-op by default. `get_adapter()` calls it right after constructing the adapter. Override it to pin dependency versions and fail fast with a clear, actionable message + early exit instead of crashing mid-stage.

| Stack | Implementation |
|---|---|
| NxDI | inherits no-op default |
| vLLM-Neuron | verifies `vllm_neuron` is importable and that both `vllm` (`PINNED_VLLM_VERSION`) and the `vllm-neuron` plugin (`PINNED_VLLM_NEURON_VERSION`) are on the pinned `0.24` line; raises `EnvironmentError` otherwise |

Pass `check_environment=False` to `get_adapter()` only for pure introspection that exercises no stack APIs.

## Auto-Detection

If `target_stack` is not specified, the registry reads the target modeling file's imports:
- `from vllm_neuron.*` → `vllm_neuron` adapter
- `from neuronx_distributed_inference.*` → `nxdi` adapter
- Neither → defaults to `nxdi`

## Optional Diagnostic Methods

Beyond the 5 core methods, adapters may implement diagnostic methods that leverage stack-specific debugging tools. These raise `NotImplementedError` by default in the base class. Currently only `VLLMNeuronAdapter` implements them (via the `accuracy_debugger` module from `vllm-neuron`).

### `run_accuracy_analysis(model_id, tp_size, eval_fn, thresholds, ...)`

Task-level accuracy analysis with per-sample deviation tracking. Wraps `accuracy_debugger.run_task_analysis()` + `LmEvalAnalyzer`.

- Runs eval against a vLLM server (or accepts existing results via `input_task_results`)
- Compares per-document results against a reference
- Returns `{passed, scores, thresholds, deviated_prompts, report_path}`
- Used by: Stage 7 (when `--use-accuracy-debugger` flag is set)

> **API change (post source-merge):** `run_task_analysis` was simplified — it now **only analyzes** results and no longer runs the eval itself. Its signature dropped `server_handle`, `eval_fn`, `eval_kwargs`, and `server_cmd`; `input_task_results` (a results dir) is now required. The adapter therefore runs `eval_fn` against a server first, then passes the resulting `results_dir` to `run_task_analysis`. Also, the `run_accuracy_*` eval runners now **default `limit=None` (full dataset)** — so `run_stage7.py` applies a safe default of `200` samples unless you pass `--limit N` (specific count) or `--full-dataset` (whole dataset, `limit=None`). This preserves the historical Stage 7 behavior and avoids an omitted `--limit` silently launching a full-dataset run.

### `run_logit_validation(model_id, tp_size, prompts, output_length, ...)`

Per-token logit validation on compiled device model. Wraps `LogitValPlugin` from accuracy_debugger.

- Compares device logits against FP32 baseline and dtype reference per token
- Returns `{prompts, plugin_results, report_path}`
- Used by: **Stage 5** — but not through this wrapper. `run_teacher_forced_comparison.py`
  imports `LogitValPlugin` and calls `run_prompt_analysis()` directly. This adapter method
  currently has no caller.

### `run_kv_cache_analysis(model_id, tp_size, prompts, output_length, ...)`

Three-way KV cache comparison (FP32 vs HF-dtype vs vLLM). Wraps `KvCachePlugin` from accuracy_debugger.

- Extracts HF reference KV caches (FP32 and dtype) via teacher-forced decoding
- Extracts vLLM paged KV caches and reconstructs contiguous layout
- Compares per-layer, per-position to identify divergent attention layers
- Returns `{prompts, plugin_results, report_path}`
- Used by: Stage 5 (when `--kv-analysis-on-fail` flag is set and R-ratio exceeds threshold)

### `run_prompt_diagnosis(model_id, tp_size, prompts, output_length, plugins, ...)`

Full prompt-level diagnosis combining multiple plugins. Wraps `accuracy_debugger.run_prompt_analysis()`.

- Orchestrates logit validation + KV cache analysis (or custom plugin subset)
- Returns `{prompts, plugin_results, report_path}`
- Used by: Stage 7 (when `--diagnose-failures` flag is set, feeds deviated prompts from task analysis)

## Existing Adapters

| Adapter | File | Stack | Diagnostics |
|---|---|---|---|
| `NxDIAdapter` | `scripts/adapters/nxdi.py` | NeuronX Distributed Inference | Core 5 only |
| `VLLMNeuronAdapter` | `scripts/adapters/vllm_neuron.py` | vLLM-Neuron | Core 5 + all diagnostic methods |
