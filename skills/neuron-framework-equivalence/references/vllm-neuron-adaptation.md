# vLLM-Neuron Adaptation Reference

Critical details for writing equivalence tests against vLLM-Neuron models. Based on the TinyLlama adaptation experiment (Apr 2026).

## Weight Transpositions

All linear weights in vLLM-Neuron are **transposed** relative to HuggingFace. This is the single most important difference.

| Weight      | HF Shape  | vLLM-Neuron Shape | Transform                            |
| ----------- | --------- | ----------------- | ------------------------------------ |
| gate_proj   | `[I, H]`  | `[H, I]`          | `.t()`                               |
| up_proj     | `[I, H]`  | `[H, I]`          | `.t()`                               |
| down_proj   | `[H, I]`  | `[I, H]`          | `.t()`                               |
| q_proj      | `[q, H]`  | fused into QKV    | see below                            |
| k_proj      | `[kv, H]` | fused into QKV    | see below                            |
| v_proj      | `[kv, H]` | fused into QKV    | see below                            |
| o_proj      | `[H, q]`  | `[q, H]`          | `.t()`                               |
| QKV (fused) | N/A       | `[H, q+2kv]`      | `cat([Q.t(), K.t(), V.t()], dim=-1)` |
| Norms       | `[H]`     | `[H]`             | direct copy                          |
| Embedding   | `[V, H]`  | `[V, H]`          | direct copy                          |
| LM head     | `[V, H]`  | `[V, H]`          | direct copy                          |

## Weight Naming

HF uses `module.weight` (nn.Parameter inside nn.Linear). vLLM-Neuron uses `module_weight` (bare nn.Parameter). Example: `self_attn.q_proj.weight` → `self_attn.qkv_proj_weight`.

## Forward Signature

vLLM-Neuron's forward is completely different from HF:

```python
model(
    input_ids,              # [seq_len] — no batch dim
    positions,              # [seq_len] — position IDs
    attn_metadata,          # dict[str, dict] — per-layer attention config
    sampling_positions,     # [num_positions] — which positions get logits
)
```

The adapter constructs `attn_metadata` and `sampling_positions` internally.

## KV Cache Shape

`[num_blocks, kv_heads_per_rank, block_size, head_dim]` — heads BEFORE block_size. Getting this wrong wastes ~10 minutes debugging.

## TP Size

**TP=1 causes SIGSEGV on trn2.48xlarge with zero diagnostic information.** Always read the model's example script (`examples/vllm_neuron/models/MODEL/run.py`) for the correct TP size. On trn2.48xlarge, TP=8 is standard.

## `load_weights()` Bypass

vLLM-Neuron's `load_weights()` calls `get_current_vllm_config()` which only works inside the vLLM serving stack. For equivalence testing, bypass it entirely and use manual weight mapping (the adapter's `load_weights()` does this).

## Environment Variables

| Context        | Required                                                                                |
| -------------- | --------------------------------------------------------------------------------------- |
| CPU testing    | `NXD_CPU_MODE=1`, `WORLD_SIZE=1`, `MASTER_ADDR=localhost`, `MASTER_PORT=8099`, `RANK=0` |
| Device testing | `NEURON_SKIP_EFA_AFFINITY=1`, `TOKENIZERS_PARALLELISM=false`                            |
| EP models      | Add `NXDI_SWITCH_CC=1`                                                                  |

## Component Test Differences

| Component      | HF Forward                   | vLLM-Neuron Forward                                         | Weight Setup                         |
| -------------- | ---------------------------- | ----------------------------------------------------------- | ------------------------------------ |
| MLP            | `forward(x)`                 | `forward(x, is_prefill=True)`                               | `.t()` on gate/up/down               |
| Q/K/V          | 3 separate `F.linear()`      | Single `NF.qkv_proj()` with fused weight                    | `cat([Q.t(), K.t(), V.t()], dim=-1)` |
| O Projection   | `F.linear(x, o_proj.weight)` | `NF.o_proj(x, o_proj_weight)`                               | `.t()`                               |
| Full Attention | Independently testable       | NOT independently testable (needs KV cache + attn_metadata) | Test QKV + O projections separately  |

## Shape Alignment

HF outputs `[1, T, H]`, vLLM-Neuron outputs `[T, H]`. Squeeze batch dim before comparison.

## Distributed Init (vLLM 0.24.0)

**Pinned to vLLM `0.24.0`.** In 0.24.0 (as in 0.21.0), `initialize_model_parallel` must run inside a `set_current_vllm_config(VllmConfig(...))` context, or it fails with `AssertionError: Current vLLM config is not set.` The `VllmConfig` / `ParallelConfig` constructor signatures also changed across releases, so do **not** hand-roll the config. (Verified on vLLM-Neuron 0.24: `vllm_neuron.parallel.neuron_parallel_state` still wraps `initialize_model_parallel` in `set_current_vllm_config`, and `ParallelConfig(tensor_parallel_size=...)` / `VllmConfig(parallel_config=...)` are unchanged.)

Instead, delegate to vLLM-Neuron's own bootstrap helper, which builds the minimal `VllmConfig`, sets the context, and calls `initialize_model_parallel` internally — the same path the serving stack and MPExecutor tests use:

```python
import torch.distributed as dist
from vllm_neuron.parallel.neuron_parallel_state import (
    initialize_neuron_parallel_state,
    is_initialized,
)

if not is_initialized():
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo",
                                init_method="tcp://localhost:8099",
                                rank=0, world_size=tp_degree)
    # builds VllmConfig + set_current_vllm_config + initialize_model_parallel
    initialize_neuron_parallel_state(
        tp_global_ranks=list(range(tp_degree)),
        local_rank=0,
    )
```

Do **not** call vLLM's `init_distributed_environment` first — it pre-creates `_WORLD`, causing `initialize_neuron_parallel_state` to early-return before building the TP groups. The adapter (`init_distributed`) already does this correctly. Note the import path is `from vllm.config import set_current_vllm_config` in 0.24.0 (unchanged from 0.21.0; not `vllm.config.vllm`).

## Version Check (Automatic)

`get_adapter()` runs `VLLMNeuronAdapter.check_environment()` on construction. It exits early with an actionable message if `vllm_neuron` can't be imported, or if either the `vllm` framework or the `vllm-neuron` plugin is off the `0.24` line. Two independent pins live in `scripts/adapters/vllm_neuron.py`:

- `PINNED_VLLM_VERSION = "0.24.0"` — upstream vLLM framework (distributed-init, VllmConfig, v1 APIs).
- `PINNED_VLLM_NEURON_VERSION = "0.24.0"` — the vLLM-Neuron plugin (dist name `vllm-neuron`, e.g. version `0.24.0.1.1.0`; model classes + parallel helpers the adapter calls).

Both are matched on the `major.minor` line. Bump them in lockstep with any adapter API changes. The plugin version is read from `importlib.metadata.version("vllm-neuron")`, falling back to `vllm_neuron.__version__`; if neither is available (editable install without metadata) it warns rather than hard-fails, since the import itself already succeeded.

## PYTHONPATH for Editable Installs

`vllm_neuron` is a local editable install, and the vLLM plugin entry point (`vllm_neuron:register`) imports it. When running stage scripts from outside the `vllm-neuron` directory, `import vllm_neuron` raises `ModuleNotFoundError`. Prepend the project root to `PYTHONPATH` on every stage command:

```bash
PYTHONPATH=/path/to/vllm-neuron:$PYTHONPATH python3 scripts/run_stage0.py ...
```

## Hybrid Models (mixed attention) — partial / unverified

Hybrid models interleave full-attention and linear/recurrent-attention layers, so only a subset of layers have a `self_attn` submodule. The original `forward()` assumed every layer had `self_attn` and crashed on these models. The current code only **removes that assumption** — it is **not** full hybrid support:

- Reads `config.layer_types` (or `layers_block_type`) and builds `attn_metadata` **only** for `full_attention` layers, instead of emitting `layers.{i}.self_attn` metadata for every layer.
- Best-effort, **unverified** hook: calls `model.bind_recurrent_state(batch_size=1, device="cpu")` if the model exposes it. The linear-attention layers' recurrent state is otherwise **not** threaded through this forward path.

Non-hybrid models have no `layer_types` and no `bind_recurrent_state`, so both paths degrade to the original "every layer is full attention" behaviour — and that non-hybrid path is the only one validated (Llama-3.2-1B). No hybrid model exists in the repo/weights to exercise the hybrid path, so it remains untested; completing and validating real hybrid forward support is follow-up work.

---

Based on: Equivalence Framework Adaptation experiment (TinyLlama-1.1B, Apr 2026); vLLM 0.21.0 update + homogeneous-layer-assumption fix (Jun 2026); vLLM 0.24.0 pin bump + accuracy_debugger source-merge (Jul 2026)
