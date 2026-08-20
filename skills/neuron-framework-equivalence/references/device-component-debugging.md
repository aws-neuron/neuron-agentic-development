# Device Component-Level Debugging

When a component equivalence test passes on CPU but fails on device, the issue lies in the translation from CPU-mode Python to compiled XLA/NEFF execution. This reference covers diagnosing and fixing device-specific divergence at the individual component level.

**Core principle:** "CPU passes, device fails" means the mathematical algorithm is correct but something about XLA tracing, SPMD execution, or weight preprocessing breaks it on device.

---

## Running Component Tests on Device

### Approach A: Compile Individual Component

For simple components (embedding, norm, linear), compile just that module:

```python
import torch
from torch_neuronx import trace

component = TargetComponent(config)
component.eval()

example_input = torch.randn(BS, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)
traced = trace(component, example_input)
device_output = traced(test_input)
```

### Approach B: 1-Layer Wrapper (Preferred for Complex Components)

For components that require model context (attention with KV cache, MoE with routing):

1. Override `config.num_hidden_layers = 1` when creating the model
2. Compile the 1-layer model with `TensorCaptureConfig` (see [dump-tensors.md](dump-tensors.md)) to capture the target component's output
3. Run inference and extract the component's tensor from the captured outputs

Then compare using the 3-tensor method:

```python
ref_fp32_out = hf_component(input_fp32).float()
ref_bf16_out = hf_component(input_bf16).float()
device_bf16_out = device_component(input_bf16).float()

from tensor_compare import compare_3tensors
result = compare_3tensors(ref_fp32_out, ref_bf16_out, device_bf16_out)
```

---

## Device-Specific Root Cause Categories

| Root Cause                                        | Error Magnitude | Symptom                                               | Diagnostic                                                                 |
| ------------------------------------------------- | --------------- | ----------------------------------------------------- | -------------------------------------------------------------------------- |
| **SPMDRank** (Python ints bake as constants)      | 100x-600x       | Embeddings/routing use rank-0 values on all TP cores  | Check if component uses `parallel_state.get_rank()` for slicing            |
| **Code path divergence**                          | 100x+           | Device uses different dispatch path than CPU          | Trace framework dispatch logic; check `sliding_window`, `is_prefill`, etc. |
| **Missing Parameters** (not in compiled NEFF)     | 10x+            | Bias or weight tensor is zero/absent on device        | Check for tensors assigned but not registered as `nn.Parameter`            |
| **Weight preprocessing** (pre_shard_weights_hook) | 1000x+          | Weights laid out differently than expected            | Compare weight checksums before/after shard hook                           |
| **Missing bias flags**                            | 10x+            | `preprocess_checkpoint` removes biases as "redundant" | Check if base class constructor receives `has_bias=True`                   |

---

## XLA-Compatible Patch Patterns

### Pattern 1: SPMDRank Instead of parallel_state

**Problem:** `parallel_state.get_tensor_model_parallel_rank()` returns a Python int. During XLA tracing, this is baked as a **constant 0** — all TP ranks use rank 0's value.

**Fix:** Use `SPMDRank` — a sharded Parameter loaded from checkpoint as `torch.arange(tp)`:

```python
from neuronx_distributed.parallel_layers.layers import SPMDRank

# In model __init__:
self.rank_util = SPMDRank(tp_degree)

# In pre_shard_weights_hook:
model_sd['module.rank_util.rank'] = torch.arange(0, tp_degree, dtype=torch.int32)

# In forward: rank_util.get_rank() returns a tensor, not a Python int
rank = self.rank_util.get_rank()  # [1] tensor, different value per NeuronCore
```

### Pattern 2: torch.index_select Instead of torch.narrow

**Problem:** `torch.narrow(tensor, dim, start, length)` with a **tensor** `start` (from SPMDRank) bakes the trace-time value as a constant in the XLA graph.

**Fix:** Use `torch.index_select()` which creates a dynamic Gather HLO:

```python
# WRONG (bakes constant):
local_slice = tensor.narrow(0, rank * local_size, local_size)

# CORRECT (dynamic Gather HLO):
indices = torch.arange(local_size, device=tensor.device) + (rank * local_size).to(torch.int64)
local_slice = torch.index_select(tensor, 0, indices)
```

### Pattern 3: Framework \_reduce Instead of torch.distributed

**Problem:** `torch.distributed.all_reduce()` is not executable during XLA tracing.

**Fix:** Use the framework's `_reduce()`:

```python
from neuronx_distributed.parallel_layers.mappings import _reduce
output = _reduce(tensor)  # AllReduce HLO on device, torch.distributed on CPU
```

### Pattern 4: Parameters Must Exist Before Tracing

**Problem:** Tensors not registered as `nn.Parameter` before tracing won't be part of the compiled NEFF.

**Fix:** Register as Parameters in `__init__`, inject values in `pre_shard_weights_hook`:

```python
# In __init__:
self._per_expert_bias = nn.Parameter(torch.zeros(num_experts, hidden_size), requires_grad=False)

# In pre_shard_weights_hook:
model_sd['layers.0.mlp.experts._per_expert_bias'] = hf_sd['model.layers.0.mlp.experts.bias']
```

### Pattern 5: Unified CPU+Device Patches

```python
def _patched_forward(self, ...):
    if hasattr(self, 'rank_util') and self.rank_util is not None:
        rank = self.rank_util.get_rank()  # Device: SPMDRank tensor
    else:
        from neuronx_distributed.parallel_layers import parallel_state
        rank = parallel_state.get_tensor_model_parallel_rank()  # CPU: Python int
```

---

## pre_shard_weights_hook Pattern

The hook runs after loading the state dict but before weight sharding:

```python
def pre_shard_weights_hook(model_instance):
    builder = model_instance.get_builder()
    original_loader = builder.checkpoint_loader

    def patched_loader(mmap=False):
        model_sd = original_loader(mmap)

        # 1. Fix weight layouts
        fix_down_proj_layout(model_sd, ...)

        # 2. Inject SPMDRank tensors
        rank_tensor = torch.arange(0, tp_degree, dtype=torch.int32)
        model_sd['embed_tokens.rank_util.rank'] = rank_tensor.clone()

        # 3. Inject per-expert biases from HF state dict
        hf_sd = load_file(os.path.join(model_path, "model.safetensors"))
        for i in range(num_layers):
            model_sd[f'layers.{i}.mlp.experts._per_expert_bias'] = \
                hf_sd[f'model.layers.{i}.mlp.experts.bias']

        return model_sd

    builder.checkpoint_loader = patched_loader
```

---

## Iterative Fix-Compile-Verify Loop

```
Write/update device patch
    │
    v
Copy patch to Docker / apply in venv
    │
    v
Recompile component (or 1-layer wrapper, ~5 min)
    │
    v
Run device inference
    │
    v
Compare output (3-tensor method)
    │
    v
error_ratio <= 1.2? ──yes──> PASS — proceed to next failing component
    │
    no
    │
    v
Revise the patch, return to diagnosis
```

- Name each compiled model distinctly: `compiled_1layer_embfix/`, `compiled_1layer_sinkfix/`, etc.
- When testing a 1-layer wrapper, verify that fixing one component causes the **first-fail to move downstream** in the comparison table — this confirms the fix worked.

---

## Pitfalls

1. **TensorCaptureConfig requires OnDeviceSamplingConfig** — without it, captured tensors are silently not returned.
2. **TP-gathered outputs are N\*vocab_size** — take first `vocab_size` entries: `logits[:, :, :vocab_size]`.
3. **preprocess_checkpoint removes "redundant" biases** — ensure base class receives bias flags (`qkv_bias=True`, `o_bias=True`).
4. **self_attn captures cos_cache, not hidden_states** — use `post_attention_layernorm` as attention quality proxy.
5. **"Removing redundant keys" warning is normal** — framework's preshard hook remaps individual q/k/v into combined qkv.

---

## Escalation to Compiler Debugging

If framework code matches the reference (verified by manual reconstruction on CPU) but device output still diverges, the issue may be in the NeuronX compiler.

**Before escalating**, analyze the compiler log `log-neuron-cc.txt` for errors, warnings, or unexpected optimization passes that may explain the divergence.

Escalate when:

- `log-neuron-cc.txt` has been reviewed and does not reveal an actionable fix
- All patches verified correct on CPU
- Code paths confirmed identical between CPU and device modes
- Weight loading confirmed correct (pre/post shard checksums match)
- Component device output shows divergence unexplainable by code differences

---

Based on: GPT-OSS 20B device component debugging (Feb-Apr 2026)
