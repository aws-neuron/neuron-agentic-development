# Expected Structural Differences: HuggingFace → NxDI Neuron

This reference documents common structural differences for the most common
use case (HuggingFace source → NxDI Neuron target). For other source-target
pairs, adapt accordingly.

## Module Type Changes (TP Sharding)

| HuggingFace    | Neuron Port            | Reason                                  |
| -------------- | ---------------------- | --------------------------------------- |
| `nn.Linear`    | `ColumnParallelLinear` | Tensor parallel sharding (column split) |
| `nn.Linear`    | `RowParallelLinear`    | Tensor parallel sharding (row split)    |
| `nn.Embedding` | `ParallelEmbedding`    | Embedding sharded across TP ranks       |

## Structural Wrappers (Framework)

| HuggingFace                   | Neuron Port                          | Reason                        |
| ----------------------------- | ------------------------------------ | ----------------------------- |
| Flat `q_proj, k_proj, v_proj` | Wrapped in `GroupQueryAttention_QKV` | NxDI attention framework      |
| Flat `o_proj`                 | Wrapped in `GroupQueryAttention_O`   | NxDI attention framework      |
| (none)                        | `SPMDRank` (rank_util)               | TP rank tracking              |
| (none)                        | `KVCacheManager` (kv_mgr)            | Inference KV cache management |

## Normalization Variants

| HuggingFace    | Neuron Port (CPU mode)       | Neuron Port (Neuron HW)      |
| -------------- | ---------------------------- | ---------------------------- |
| `XxxRMSNorm`   | `LlamaRMSNorm` or same class | `CustomRMSNorm` (NKI kernel) |
| `nn.LayerNorm` | `nn.LayerNorm`               | `nn.LayerNorm`               |

## Operator Fusion/Split

| HuggingFace                             | Neuron Port                          | Reason                        |
| --------------------------------------- | ------------------------------------ | ----------------------------- |
| Fused `gate_up_proj` [2*inter, hidden]  | Split `gate_proj` + `up_proj`        | TP requires separate sharding |
| Fused `qkv_proj` [3*hidden, hidden]     | Split `q_proj` + `k_proj` + `v_proj` | TP requires separate sharding |
| Single `RotaryEmbedding` at model level | Per-layer `RotaryEmbedding`          | Implementation choice         |

## Activation Functions

| HuggingFace         | Neuron Port                     | Notes                      |
| ------------------- | ------------------------------- | -------------------------- |
| `NewGELUActivation` | `F.gelu(x, approximate='tanh')` | Same math, different class |
| `SiLU`              | `SiLU`                          | Identical                  |

## Modules with No Counterpart

**HF-only (no Neuron equivalent):**

- `Dropout` layers — disabled during inference (set to 0.0)
- `RotaryEmbedding` at model level (Neuron uses per-layer)

**Neuron-only (no HF equivalent):**

- `SPMDRank` — distributed rank utilities
- `KVCacheManager` — inference KV caching
- `LogitsProcessor` — sampling-time logit manipulation (some models)

## Differences That Indicate BUGS

- Missing modules (e.g., a norm layer absent in the port)
- Extra unexpected modules with no HF counterpart and no framework explanation
- Wrong nesting (e.g., MLP inside attention instead of parallel)
- Mismatched layer counts (e.g., 47 layers instead of 48)
- Missing activation functions
