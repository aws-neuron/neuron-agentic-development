# Intermediate Tensor Capture

Capture intermediate tensors at named module boundaries from both the reference (HF) and target (NeuronX device) models, saving inputs/outputs to disk for offline comparison.

---

## Key Concepts

### Module Naming

HF wraps the transformer body in `model.`, so HF modules are `model.layers.0.self_attn` while Neuron uses `layers.0.self_attn`. When setting up hooks, strip the `model.` prefix from HF names to get consistent names across both sides:

```python
MODULES_TO_CAPTURE = [
    "embed_tokens",
    "layers.0.input_layernorm",
    "layers.0.self_attn",
    "layers.0.post_attention_layernorm",
    "layers.0.mlp",
    "layers.0",
    "norm",
    "lm_head",
]
```

### File Naming Convention

```
captured_tensors_{phase}_step_{step}_module_{module_name}_output.pt
```

- `phase`: `"cte"` for context encoding (prefill), `"tg"` for token generation
- `step`: generation step number (1 for CTE)
- Tuple outputs: element 0 saved as `_output_0.pt`

### Shape Differences

Device captures pad input to the full `seq_len` (e.g., 128), while HF captures use the actual input length (e.g., 6 tokens). Downstream comparison must slice:

```python
ref_seq_len = hf_tensor.shape[1]
device_aligned = device_tensor[:, :ref_seq_len, :]
```

### Known Quirks

| Module          | Quirk                                                                                                         | Workaround                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `self_attn`     | Device captures `cos_cache` (3rd field of NeuronAttentionBaseOutput) instead of hidden_states                 | Use `post_attention_layernorm` as proxy for attention quality    |
| `embed_tokens`  | FP32 and BF16 embeddings identical (lookup, no computation) → baseline_err = 0 → error_ratio = inf            | Check cosine similarity instead; cosine = 1.0 means PASS         |
| `lm_head`       | Device with on-device sampling outputs `[1, 1, vocab]` (last position only), HF outputs `[1, seq_len, vocab]` | Compare only last position: `hf[:, -1:, :]` vs `device[:, :, :]` |
| Sharded modules | Device captures local TP shard, not global tensor                                                             | Mark as sharded; interpret with care or skip in layer comparison |

---

## Part 1: HF Reference Capture (Hook-Based)

Register PyTorch `forward_hook` on each target module:

```python
import os, torch
from transformers import AutoModelForCausalLM, AutoConfig

def capture_hf_tensors(model_path, modules, save_dir, torch_dtype=torch.float32):
    os.makedirs(save_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch_dtype)
    model.eval()

    hooks = []
    for module_name in modules:
        target = None
        for prefix in ["model.", ""]:
            try:
                target = model.get_submodule(prefix + module_name)
                break
            except AttributeError:
                continue
        if target is None:
            print(f"WARNING: Module '{module_name}' not found, skipping")
            continue

        def make_hook(name):
            def hook_fn(mod, inp, out):
                tensor = out[0] if isinstance(out, tuple) else out
                if isinstance(tensor, torch.Tensor):
                    fpath = os.path.join(save_dir,
                        f"captured_tensors_cte_step_1_module_{name}_output.pt")
                    torch.save(tensor.detach().float().cpu(), fpath)
            return hook_fn

        hooks.append(target.register_forward_hook(make_hook(module_name)))

    with torch.no_grad():
        outputs = model(input_ids)

    for h in hooks:
        h.remove()
    return outputs
```

Run twice: once with `torch_dtype=torch.float32`, once with `torch.bfloat16`.

---

## Part 2: Device Tensor Capture (TensorCaptureConfig)

Device tensor capture requires configuration at **compile time**.

### Compile with TensorCaptureConfig

```python
from neuronx_distributed_inference.models.config import (
    NeuronConfig, TensorCaptureConfig, OnDeviceSamplingConfig,
)

tensor_capture_config = TensorCaptureConfig(
    modules_to_capture=MODULES_TO_CAPTURE,
    max_intermediate_tensors=len(MODULES_TO_CAPTURE),
    capture_inputs=True,
)

# OnDeviceSamplingConfig is REQUIRED for tensor capture to work
on_device_sampling_config = OnDeviceSamplingConfig(
    dynamic=False, do_sample=False, global_topk=1,
)

neuron_config = NeuronConfig(
    tp_degree=TP_DEGREE, batch_size=1, seq_len=SEQ_LEN,
    torch_dtype=torch.bfloat16,
    save_sharded_checkpoint=True,
    enable_cte_modular_flow=True,
    tensor_capture_config=tensor_capture_config,
    on_device_sampling_config=on_device_sampling_config,
)
```

### Extract Captured Tensors

Monkey-patch the model's `forward` to access `result.captured_tensors`:

```python
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM

_original_forward = NeuronBaseForCausalLM.forward

def _capturing_forward(self, *args, **kwargs):
    result = _original_forward(self, *args, **kwargs)
    if hasattr(result, 'captured_tensors') and result.captured_tensors:
        for i, tensor in enumerate(result.captured_tensors):
            if isinstance(tensor, torch.Tensor) and i < len(MODULES_TO_CAPTURE):
                fname = f"captured_tensors_cte_step_1_module_{MODULES_TO_CAPTURE[i]}_output.pt"
                torch.save(tensor.detach().float().cpu(), os.path.join(OUTPUT_DIR, fname))
    return result

NeuronBaseForCausalLM.forward = _capturing_forward
```

---

### Init Patch for Config Propagation

Some model classes ignore `config.neuron_config` when the `neuron_config` kwarg is not explicitly passed. Patch `__init__` to extract it:

```python
_original_init = NeuronModelForCausalLM.__init__

def _patched_init(self, model_path, config=None, neuron_config=None):
    if neuron_config is None and config is not None:
        nc = getattr(config, 'neuron_config', None)
        if nc is not None and getattr(nc, 'tensor_capture_config', None) is not None:
            neuron_config = nc
    _original_init(self, model_path, config, neuron_config)

NeuronModelForCausalLM.__init__ = _patched_init
```

---

## Part 3: Fallback Strategies

When `TensorCaptureConfig` is unavailable:

1. **Monkey-patch forward** to store intermediates in a global dict before compilation (tensor writes become part of the traced graph)
2. **Truncated model compilation** — create a modified model class that ends at the module of interest
3. **CPU-mode device code** — run device code path on CPU (`NXD_CPU_MODE=1`) to capture intermediates without compilation

---

Based on: GPT-OSS 20B device equivalence debugging (Feb-Apr 2026)
