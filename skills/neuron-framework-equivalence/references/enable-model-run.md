# Enable Model Execution on Neuron Devices

Workflow for compiling a target model and enabling it to run on both Neuron devices (production) and CPU (debugging/comparison). This is a prerequisite for device-stage equivalence validation (Stages 5+).

## Templates

| Template                                        | Purpose                                                    |
| ----------------------------------------------- | ---------------------------------------------------------- |
| `templates/neuron_device_validator_template.py` | Validates model runs on Neuron hardware, checks throughput |
| `templates/run_inference_cpu_template.py`       | CPU inference with monkey patches for E2E comparison       |

---

## Compilation Workflow

### 1. Compile the Model

Use the standard NxDI compile path — build the `NeuronConfig`, resolve the inference config
from the HF checkpoint, instantiate the `ForCausalLM` class, then compile:

```python
import torch

# ModelNeuronConfig if the port subclasses it, else NeuronConfig from
# neuronx_distributed_inference.models.config
neuron_config = ModelNeuronConfig(
    tp_degree=TP_DEGREE,
    world_size=TP_DEGREE,
    batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
    save_sharded_checkpoint=True,
)

config = ModelInferenceConfig.from_pretrained(HF_MODEL_PATH, neuron_config=neuron_config)
config.add_derived_config()

model = NeuronModelForCausalLM(HF_MODEL_PATH, config)
model.compile(COMPILED_MODEL_PATH)
```

If `ModelInferenceConfig.from_pretrained` is unavailable on the port, fall back to the
two-arg form used by `scripts/adapters/nxdi.py`:

```python
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
config = ModelInferenceConfig(neuron_config, load_config=load_pretrained_config(HF_MODEL_PATH))
```

**Expected output:**

```
{COMPILED_MODEL_PATH}/
├── model.pt              # NEFF binary
├── neuron_config.json
└── weights/              # Sharded checkpoints
```

### 2. Validate Neuron Execution

Run `templates/neuron_device_validator_template.py` and verify:

- `neuron-ls` shows available devices
- Logs show `CPU Mode: False`
- Throughput meets threshold (small models: >20 tok/s, medium: >10 tok/s)

### 3. Enable CPU Execution

Run `templates/run_inference_cpu_template.py` with `cpu_mode=True`:

- Bypasses NEFF, loads from HF weights
- Use same dtype (bfloat16) for fair comparison
- CPU is typically 10–20x slower — this validates the difference

---

## Compilation Troubleshooting

| Issue                               | Solution                                                                     |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| HLO verification fails              | Set `NEURON_CC_FLAGS='--internal-hlo2tensorizer-options=--verify-hlo=false'` |
| `get_program_sharding_info` missing | Add fallback function in affected files                                      |
| Compilation OOM                     | Reduce `seq_len` or `batch_size`                                             |
| NEFF not found at inference         | Check `output_path` matches `compiled_model_path`                            |
| Low throughput (< 5 tok/s)          | Verify not running on CPU fallback — check `neuron-ls` and logs              |

---

## Logging Requirements

Always capture full output:

```bash
# Compilation — the snippet from "1. Compile the Model" above
python3 <your_compile_script>.py 2>&1 | tee logs/compilation.log
# Device validation — your copy of templates/neuron_device_validator_template.py
python3 <your_device_validator>.py 2>&1 | tee logs/inference_neuron.log
```

---

## Known Issue: BF16 with Gloo Backend (CPU TP>1)

**Problem:** CPU inference with `tp_degree > 1` fails with:

```
"The gloo backend does not natively support bfloat16"
```

**Root cause:** Gloo backend doesn't support BF16 for collective operations (all_reduce, reduce).

**Fix:** Two changes required:

1. **`comm.py`** (NeuronxDistributed) — upcast BF16→FP32 before reduction, cast back after:

```python
def all_reduce(...):
    if cpu_mode():
        for tensor in tensor_bucket:
            if tensor.dtype == torch.bfloat16:
                tensor_fp32 = tensor.float()
                dist.all_reduce(tensor_fp32, op=op_type, group=groups)
                tensor.copy_(tensor_fp32.to(torch.bfloat16))
            else:
                dist.all_reduce(tensor, op=op_type, group=groups)
```

2. **`application_base.py`** (NeuronxDistributedInference) — remove the blocking guard check in `to_cpu()` that raises `NotImplementedError` for BF16+gloo.

**Running CPU TP>1:** Use `torchrun --nproc_per_node={TP_DEGREE}` matching the model's TP degree.

---

Based on: enable-model-run skill from Equivalence-1 (Qwen3-0.6B, Gemma3-1B, GPT-OSS 20B)
