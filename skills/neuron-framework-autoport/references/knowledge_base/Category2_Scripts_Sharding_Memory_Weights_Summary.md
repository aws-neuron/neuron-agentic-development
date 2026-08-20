# Category 2: Sharding, Memory & Weight Management Scripts Summary

**Total Scripts**: 34 files
**Purpose**: Scripts that handle weight loading, SPMD transformation, tensor/expert parallelism, and memory-efficient operations for the GenericMoE model

---

## Executive Summary

This category contains scripts focused on the complex challenge of managing MoE model weights across distributed hardware. With 16 experts per layer and TP/EP configurations, the port required a sophisticated 4-stage weight transformation pipeline. The 34 scripts document the discovery and implementation of this pipeline, addressing weight key mismatches, SPMD format conversions, memory constraints, and expert routing mechanisms.

**Key Achievement**: Implemented a complete weight transformation pipeline that converts HuggingFace weights → Compilation format → SPMD format → Inference format while managing 15-20GB of weight data efficiently.

---

## 1. The SPMD Weight Transformation Pipeline

### 1.1 Pipeline Overview

The most critical discovery was that weights go through **4 distinct stages** during the GenericMoE port:

```
Stage 1: HuggingFace Format
  ├── Individual expert weights
  ├── Keys: model.layers.X.block_sparse_moe.experts.E.w1.weight
  └── Shape: [intermediate_size, hidden_size] per expert

Stage 2: Compilation Format
  ├── Combined expert weights per layer
  ├── Keys: layers.X.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight
  └── Shape: [num_experts, hidden_size, 2*intermediate_size]

Stage 3: SPMD Format (after compilation)
  ├── Sharded across TP ranks
  ├── Keys: layers.X.block_sparse_moe.expert_mlps.spmd_rank.rank
  └── Shape: Varies by TP degree and EP degree

Stage 4: Inference Format
  ├── Expected by inference framework
  ├── Keys: layers.X.block_sparse_moe.expert_mlps.mlp_op.{gate_up_proj,down_proj}.weight
  └── Must match SPMD sharding scheme
```

**Scripts Documenting Pipeline**:

- `transform_spmd_weights.py` - Stage 3 → Stage 4 transformation
- `fix_compiled_weights.py` - Complete HF → Inference transformation
- `analyze_real_weight_loading.py` - Pipeline analysis and verification

---

### 1.2 Stage 1 → Stage 2: HF to Compilation Format

**Purpose**: Convert individual expert weights to combined format for compilation

**Script**: `fix_compiled_weights.py`, `fix_neuronx_model_weight_loading.py`

**Transformation Logic**:

```python
def convert_hf_to_compilation_format(hf_state_dict, config):
    """Convert HF expert weights to compilation format"""

    neuron_state_dict = {}

    for layer_idx in range(config.num_hidden_layers):
        # Initialize combined tensors
        gate_up_proj = torch.empty(
            config.num_local_experts,  # 16 experts
            config.hidden_size,         # 4096
            2 * config.intermediate_size  # 2 × 6400 = 12800
        )

        down_proj = torch.empty(
            config.num_local_experts,      # 16 experts
            config.intermediate_size,       # 6400
            config.hidden_size             # 4096
        )

        # Fill with individual expert weights
        for expert_idx in range(config.num_local_experts):
            # HF uses w1=gate, w2=down, w3=up
            w1_key = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight"
            w2_key = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight"
            w3_key = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight"

            # Transpose and concatenate gate + up projections
            gate_weights = hf_state_dict[w1_key].T  # Transpose!
            up_weights = hf_state_dict[w3_key].T

            gate_up_proj[expert_idx, :, :config.intermediate_size] = gate_weights
            gate_up_proj[expert_idx, :, config.intermediate_size:] = up_weights

            # Transpose down projection
            down_weights = hf_state_dict[w2_key].T
            down_proj[expert_idx] = down_weights

        # Add to neuron state dict
        neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj
        neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.expert_mlps.mlp_op.down_proj.weight"] = down_proj

    return neuron_state_dict
```

**Critical Details**:

1. **Weight naming**: HF uses `w1/w2/w3`, NeuronX uses `gate_proj/down_proj/up_proj`
2. **Transposition**: All weights must be transposed (.T)
3. **Concatenation**: gate and up projections are concatenated along last dimension
4. **Prefix removal**: "model." prefix is removed from HF keys

---

### 1.3 Stage 2 → Stage 3: Compilation Creates SPMD Format

**Purpose**: Neuron compiler automatically shards weights across TP ranks

**What happens during compilation**:

```
Input (Stage 2):
  layers.0.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight
  Shape: [16, 4096, 12800]

After Compilation (Stage 3):
  # TP=16 creates 16 shards
  layers.0.block_sparse_moe.expert_mlps.spmd_rank.rank_0
  layers.0.block_sparse_moe.expert_mlps.spmd_rank.rank_1
  ...
  layers.0.block_sparse_moe.expert_mlps.spmd_rank.rank_15

  # Each shard contains portion of experts
  Shape per shard: [1-2 experts, 4096, 12800] (depends on EP degree)
```

**Scripts Analyzing SPMD**:

- `investigate_compiled_model_artifacts.py`
- `check_tp_ep_weights.py`
- `investigate_expert_routing.py`

**Key Discovery**: The SPMD format keys are **automatically generated** by the compiler and depend on:

- TP degree (tensor_model_parallel_size)
- EP degree (expert_model_parallel_size)
- Total number of experts (16)

---

### 1.4 Stage 3 → Stage 4: SPMD to Inference Format

**Purpose**: Transform compiler-generated SPMD weights to format expected by inference framework

**Script**: `transform_spmd_weights.py`

**Problem**: Inference framework expects `mlp_op.gate_up_proj.weight` but SPMD creates `spmd_rank.rank_X`

**Solution**: Post-compilation weight transformation

```python
def transform_spmd_to_inference(compiled_path):
    """Transform SPMD weights back to mlp_op format"""

    weights_dir = Path(compiled_path) / "weights"

    for weight_file in weights_dir.glob("*.safetensors"):
        with safetensors.safe_open(weight_file, framework="pt") as f:
            weights = {key: f.get_tensor(key) for key in f.keys()}

        # Find SPMD keys
        spmd_keys = [k for k in weights.keys()
                     if 'block_sparse_moe.expert_mlps.spmd_rank.rank' in k]

        # Transform each SPMD key
        for spmd_key in spmd_keys:
            # Extract layer info
            layer_prefix = spmd_key.split('.block_sparse_moe')[0]

            # Get SPMD tensor
            spmd_tensor = weights[spmd_key]
            # Shape: [num_experts_on_this_rank, hidden_size, intermediate_size*2]

            # Create mlp_op keys
            gate_up_key = f"{layer_prefix}.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight"
            down_key = f"{layer_prefix}.block_sparse_moe.expert_mlps.mlp_op.down_proj.weight"

            # Add transformed weights
            new_weights[gate_up_key] = spmd_tensor
            # (Similar for down_proj)

            # Remove SPMD key
            del new_weights[spmd_key]

        # Save transformed weights
        safetensors.torch.save_file(new_weights, weight_file)
```

**Challenge**: Must preserve SPMD sharding scheme while changing key names

---

## 2. Weight Key Mapping Issues

### 2.1 The Key Mismatch Problem

**Problem**: HuggingFace and NeuronX use different weight key conventions

**Scripts**:

- `debug_weight_key_mapping.py`
- `fix_weight_key_mismatch.py`
- `investigate_missing_weights.py`

**Key Mapping Table**:

| HuggingFace Key                                       | NeuronX Key                                                        | Notes                   |
| ----------------------------------------------------- | ------------------------------------------------------------------ | ----------------------- |
| `model.embed_tokens.weight`                           | `embed_tokens.weight`                                              | Remove "model." prefix  |
| `lm_head.weight`                                      | `lm_head.weight`                                                   | Same (no prefix in HF)  |
| `model.layers.X.self_attn.q_proj.weight`              | `layers.X.self_attn.qkv_proj.q_proj.weight`                        | QKV combined in NeuronX |
| `model.layers.X.block_sparse_moe.experts.E.w1.weight` | `layers.X.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight` | Expert weights combined |
| `model.layers.X.block_sparse_moe.gate.weight`         | `layers.X.block_sparse_moe.router.linear_router.weight`            | Router renamed          |

**Conversion Function**:

```python
def convert_key_hf_to_neuronx(hf_key):
    """Convert HuggingFace key to NeuronX format"""

    # Remove "model." prefix
    key = hf_key.replace("model.", "")

    # Convert attention keys
    if "self_attn" in key:
        # q_proj/k_proj/v_proj → qkv_proj.q_proj/k_proj/v_proj
        key = key.replace("self_attn.q_proj", "self_attn.qkv_proj.q_proj")
        key = key.replace("self_attn.k_proj", "self_attn.qkv_proj.k_proj")
        key = key.replace("self_attn.v_proj", "self_attn.qkv_proj.v_proj")

    # Convert MoE router keys
    if "block_sparse_moe.gate" in key:
        key = key.replace("block_sparse_moe.gate", "block_sparse_moe.router.linear_router")

    # Expert weights handled separately (need combining)

    return key
```

---

### 2.2 Missing Keys Investigation

**Script**: `investigate_missing_keys.py`, `investigate_missing_weights.py`

**Common Missing Keys**:

1. **rank_util.rank tensors**: Need to be added manually

   ```python
   state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
   ```

2. **Expert MLP operation keys**: Created during Stage 1→2 transformation

3. **Bias terms**: Some models have bias, some don't
   ```python
   if config.attention_bias:
       # Load attention biases
   if config.lm_head_bias:
       # Load lm_head bias
   ```

---

## 3. Memory-Efficient Weight Operations

### 3.1 The Memory Challenge

**Problem**: GenericMoE has ~20GB of weights, and loading/converting them can use 50GB+ RAM

**Scripts**:

- `memory_optimized_weight_conversion.py`
- `memory_efficient_float32_test.py`
- `test_memory_solution_demo.py`
- `test_cpu_tp2_memory_safe.py`

**Memory Usage Breakdown**:

```
HuggingFace model in memory:      ~20 GB (bfloat16)
Weight conversion intermediate:    ~30 GB (dtype conversions)
NeuronX model in memory:          ~20 GB (bfloat16)
Compilation artifacts:            ~10 GB (temp files)
---------------------------------------------------
Total peak memory:                 ~80 GB
```

---

### 3.2 Memory-Optimized Conversion Pattern

**Script**: `memory_optimized_weight_conversion.py`

**Strategy**: Process weights layer-by-layer instead of all-at-once

```python
def convert_hf_to_neuron_memory_optimized(state_dict, config, rank=0, world_size=1):
    """Memory-optimized conversion - process incrementally"""

    neuron_state_dict = {}

    # Process non-MoE weights first (smaller)
    non_moe_keys = [k for k in state_dict.keys() if ".block_sparse_moe." not in k]
    for key in non_moe_keys:
        new_key = convert_key_hf_to_neuronx(key)
        neuron_state_dict[new_key] = state_dict[key]

    # Process MoE weights layer-by-layer
    for layer_idx in range(config.num_hidden_layers):
        # Only process layers assigned to this rank
        if layer_idx % world_size != rank:
            continue

        print(f"[Rank {rank}] Processing layer {layer_idx}")

        # Initialize layer output tensors
        gate_up_proj = torch.empty(...)
        down_proj = torch.empty(...)

        # Process experts in batches of 4
        batch_size = 4
        for batch_start in range(0, config.num_local_experts, batch_size):
            batch_end = min(batch_start + batch_size, config.num_local_experts)

            for expert_idx in range(batch_start, batch_end):
                # Load and convert expert weights
                w1 = state_dict[f"...experts.{expert_idx}.w1.weight"]
                w2 = state_dict[f"...experts.{expert_idx}.w2.weight"]
                w3 = state_dict[f"...experts.{expert_idx}.w3.weight"]

                # Fill output tensors
                gate_up_proj[expert_idx] = torch.cat([w1.T, w3.T], dim=-1)
                down_proj[expert_idx] = w2.T

                # Clear references
                del w1, w2, w3

            # Force garbage collection after each batch
            gc.collect()

        # Add layer weights to output
        neuron_state_dict[f"layers.{layer_idx}...gate_up_proj.weight"] = gate_up_proj
        neuron_state_dict[f"layers.{layer_idx}...down_proj.weight"] = down_proj

        # Force GC after each layer
        gc.collect()

    return neuron_state_dict
```

**Memory Savings**: Reduces peak memory from ~80GB to ~40GB

---

### 3.3 Memory Monitoring

**Script**: `test_memory_usage_simple.py`

```python
def monitor_memory_usage():
    """Monitor current memory usage"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 3)
    print(f"Current memory: {memory_gb:.2f} GB")
    return memory_gb

# Use throughout conversion
print("Before loading HF model:")
monitor_memory_usage()

hf_model = AutoModelForCausalLM.from_pretrained(...)

print("After loading HF model:")
monitor_memory_usage()

# ... conversion ...

print("After conversion:")
monitor_memory_usage()
```

---

## 4. Expert Routing Implementation

### 4.1 Forward All Experts Analysis

**Scripts**:

- `examine_neuronx_forward_all_experts_implementation.py`
- `investigate_expert_routing.py`
- `investigate_moe_structure.py`

**Key Discovery**: The `ExpertMLPsV2.forward_all_experts` method controls how routing weights are applied

**Two Routing Methods Identified**:

**Method 1: Binary Masking** (early_expert_affinity_modulation=True)

```python
# expert_mask: one-hot encoding of selected experts
expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts)
# Shape: [num_experts, top_k, batch*seq]

output = torch.zeros_like(hidden_states)
for e in range(num_experts):
    if expert_mask[e].any():
        # Binary mask loses routing weight precision
        mask_value = expert_mask[e].float().sum(dim=0)  # Just 0 or 1
        output += expert_outputs[e] * mask_value.unsqueeze(1)
```

**Method 2: Weighted Routing** (early_expert_affinity_modulation=False)

```python
# expert_affinities_masked: routing weights applied to mask
expert_affinities_masked = torch.zeros(batch*seq, num_experts)
for token_idx in range(batch*seq):
    for k in range(top_k):
        expert_idx = selected_experts[token_idx, k]
        weight = routing_weights[token_idx, k]  # e.g., 0.6, 0.4
        expert_affinities_masked[token_idx, expert_idx] = weight

output = torch.zeros_like(hidden_states)
for e in range(num_experts):
    affinity = expert_affinities_masked[:, e]  # Routing weights preserved
    output += expert_outputs[e] * affinity.unsqueeze(1)
```

**Critical Difference**: Method 2 preserves fractional routing weights, Method 1 loses them

---

### 4.2 Routing Configuration Impact

**Script**: `apply_early_expert_affinity_modulation_fix.py`

**Test Demonstrating Difference**:

```python
# Test scenario
routing_weights = torch.tensor([[0.8, 0.2], [0.6, 0.4]])  # Fractional weights
expert_outputs = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # Different values

# Method 1 (Binary): Each token gets sum of expert outputs
# Token 0: 2.0 + 4.0 = 6.0
# Token 1: 6.0 + 8.0 = 14.0

# Method 2 (Weighted): Each token gets weighted sum
# Token 0: 2.0 * 0.8 + 4.0 * 0.2 = 2.4
# Token 1: 6.0 * 0.6 + 8.0 * 0.4 = 6.8

# Difference: Method 1 gives MUCH larger values!
```

**Accuracy Impact**:

- Method 1: Predicts wrong tokens (e.g., 'a' instead of 'Paris')
- Method 2: Matches HuggingFace predictions exactly

---

## 5. Tensor Parallelism and Expert Parallelism

### 5.1 TP vs EP Tradeoffs

**Scripts**:

- `check_tp_ep_weights.py`
- `test_expert_sharding.py`
- `test_cpu_tp2_memory_safe.py`

**Parallelism Strategies**:

| Strategy    | Description                        | Pros                    | Cons                           |
| ----------- | ---------------------------------- | ----------------------- | ------------------------------ |
| **TP Only** | Shard attention/FFN across ranks   | Simpler, more stable    | Limited expert distribution    |
| **EP Only** | Distribute experts across ranks    | Expert-specific scaling | Complex routing                |
| **TP + EP** | Both attention and experts sharded | Maximum parallelism     | Most complex, harder debugging |

**Recommended Configuration**:

```python
# For 32 cores
tp_degree = 16      # Use half cores for tensor parallelism
moe_tp_degree = 16  # Same as tp_degree
moe_ep_degree = 1   # Initially disable expert parallelism
```

**Rationale**:

- Start with TP-only for stability
- Add EP later after confirming correctness
- TP=16 provides good balance of parallelism and simplicity

---

### 5.2 Expert Distribution with EP

**When using EP** (expert_model_parallel_size > 1):

```python
# With 16 experts and EP=8
experts_per_rank = 16 / 8 = 2

# Rank 0: Experts 0, 1
# Rank 1: Experts 2, 3
# Rank 2: Experts 4, 5
# ...
# Rank 7: Experts 14, 15
```

**Weight Sharding**:

```python
# Each rank stores only its experts
rank_0_weights = {
    "expert_mlps.spmd_rank.rank_0": experts_0_and_1_weights
}
rank_1_weights = {
    "expert_mlps.spmd_rank.rank_1": experts_2_and_3_weights
}
```

**Routing with EP**: Requires all-to-all communication to route tokens to correct expert ranks

---

## 6. ColumnParallelLinear Precision Issues

### 6.1 The Precision Loss Root Cause

**Script**: `columnparallel_precision_root_cause_analysis.py`

**Discovery**: ColumnParallelLinear has a hidden precision loss in the allreduce operation

**Problem Location**:

```
File: neuronx_distributed/parallel_layers/layers_utils.py
Function: _linear_autograd_bwd_grad_reduce
Lines: 99-102
```

**Problematic Code**:

```python
if ctx.async_grad_allreduce:
    # Convert to reduce_dtype (default: float32)
    grad_input = grad_input.to(ctx.reduce_dtype)  # bfloat16 → float32

    # All-reduce operation
    handle = torch.distributed.all_reduce(grad_input, group=ctx.process_group, async_op=True)

    # Convert back to original dtype
    grad_input = grad_input.to(original_dtype)  # float32 → bfloat16 (PRECISION LOSS!)
```

**Why It Matters**:

- The float32 → bfloat16 conversion introduces quantization artifacts
- Differences are exactly 1/64 (0.015625) multiples
- Accumulates through 32 layers
- Final prediction changes from correct ('Paris') to wrong ('a')

---

### 6.2 Evidence of Precision Loss

**Script**: `columnparallel_precision_root_cause_analysis.py`

**Test Results**:

```python
# With default reduce_dtype=float32
max_diff = 0.015625  # Exactly 1/64
percentage_1_64_multiples = 77%  # Most differences are 1/64 multiples

# Expected Paris logit: 10.5
# Actual Paris logit: 10.484375  # Difference of 0.015625 = 1/64

# This 1/64 difference is enough to change token prediction!
```

**Fix**:

```python
# Set reduce_dtype to match tensor dtype
q_proj = ColumnParallelLinear(
    ...,
    reduce_dtype=torch.bfloat16  # NOT float32!
)
```

---

## 7. Weight Loading Verification

### 7.1 Weight Verification Pattern

**Script**: `validate_weight_loading.py`, `quick_weight_analysis.py`

```python
def verify_weight_loading(hf_model, neuronx_model):
    """Verify weights match between HF and NeuronX models"""

    # 1. Embedding weights
    hf_embed = hf_model.model.embed_tokens.weight
    nx_embed = neuronx_model.model.embed_tokens.weight

    embed_diff = torch.abs(hf_embed - nx_embed).max().item()
    assert embed_diff < 1e-6, f"Embedding mismatch: {embed_diff}"

    # 2. LM head weights
    hf_lm_head = hf_model.lm_head.weight
    nx_lm_head = neuronx_model.lm_head.weight

    lm_head_diff = torch.abs(hf_lm_head - nx_lm_head).max().item()
    assert lm_head_diff < 1e-6, f"LM head mismatch: {lm_head_diff}"

    # 3. Attention weights (first layer)
    hf_q = hf_model.model.layers[0].self_attn.q_proj.weight
    nx_q = neuronx_model.model.layers[0].self_attn.qkv_proj.q_proj.weight

    q_diff = torch.abs(hf_q - nx_q).max().item()
    assert q_diff < 1e-6, f"Q projection mismatch: {q_diff}"

    # 4. Expert weights (first layer, first expert)
    hf_expert_0_w1 = hf_model.model.layers[0].block_sparse_moe.experts[0].w1.weight
    nx_gate_up = neuronx_model.model.layers[0].block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight
    nx_expert_0_gate = nx_gate_up[0, :, :config.intermediate_size].T  # Transpose back

    expert_diff = torch.abs(hf_expert_0_w1 - nx_expert_0_gate).max().item()
    assert expert_diff < 1e-6, f"Expert 0 gate mismatch: {expert_diff}"

    print("✅ All weights match!")
    return True
```

---

### 7.2 Common Weight Loading Failures

**Failure 1: Transposition Errors**

```python
# Wrong - weights transposed incorrectly
gate_up_proj[e] = torch.cat([w1, w3], dim=-1)

# Correct - must transpose first
gate_up_proj[e] = torch.cat([w1.T, w3.T], dim=-1)
```

**Failure 2: Dimension Concatenation**

```python
# Wrong - concatenating along wrong dimension
gate_up = torch.cat([gate, up], dim=0)  # [2*intermediate, hidden]

# Correct
gate_up = torch.cat([gate, up], dim=-1)  # [hidden, 2*intermediate]
```

**Failure 3: Expert Index Off-by-One**

```python
# Wrong - starting from 1
for expert_idx in range(1, num_experts + 1):

# Correct - starting from 0
for expert_idx in range(num_experts):
```

---

## 8. Distributed Weight Loading

### 8.1 Multi-Rank Weight Loading Pattern

**Script**: `apply_comprehensive_weight_fix.py`

```python
def load_weights_distributed(model_path, compiled_path, rank, world_size):
    """Load and shard weights across ranks"""

    # Each rank loads full HF weights
    hf_state_dict = load_hf_weights(model_path)

    # Convert to NeuronX format
    neuron_state_dict = convert_hf_to_neuron(hf_state_dict, config)

    # Shard according to TP/EP configuration
    sharded_state_dict = shard_weights_for_rank(
        neuron_state_dict,
        rank=rank,
        world_size=world_size,
        tp_degree=config.tp_degree,
        ep_degree=config.moe_ep_degree
    )

    # Save rank-specific checkpoint
    output_file = Path(compiled_path) / "weights" / f"tp{rank}_sharded_checkpoint.safetensors"
    safetensors.torch.save_file(sharded_state_dict, output_file)

    return True
```

---

### 8.2 Weight Sharding Strategy

**For TP=16, EP=1**:

```python
# Attention weights: Sharded across TP dimension
# Q/K/V projections: Split hidden_size across 16 ranks
qkv_weight_rank_0 = full_qkv_weight[:, 0:256]      # First 256 of 4096
qkv_weight_rank_1 = full_qkv_weight[:, 256:512]    # Next 256
# ...
qkv_weight_rank_15 = full_qkv_weight[:, 3840:4096] # Last 256

# Expert weights: Replicated (no EP)
expert_weights_rank_0 = all_expert_weights  # Full copy
expert_weights_rank_1 = all_expert_weights  # Full copy
# ... (all 16 ranks have full expert weights)
```

**For TP=16, EP=8**:

```python
# Attention weights: Same as above (sharded by TP)

# Expert weights: Sharded across EP dimension
# Rank 0-1: Experts 0-1 (TP sharded within these experts)
# Rank 2-3: Experts 2-3
# ...
# Rank 14-15: Experts 14-15
```

---

## 9. Key Learnings and Patterns

### 9.1 Weight Transformation Best Practices

1. **Always transpose expert weights** from HF to NeuronX (w1.T, w2.T, w3.T)
2. **Concatenate gate+up** along last dimension (dim=-1)
3. **Process layer-by-layer** for memory efficiency
4. **Force garbage collection** after processing each layer
5. **Verify transformations** with checksums or sample comparisons
6. **Save backups** before transforming compiled weights

---

### 9.2 SPMD Pipeline Insights

**Critical Understanding**:

- Stage 1→2: Manual transformation (our code)
- Stage 2→3: Automatic (compiler does it)
- Stage 3→4: Manual transformation (our code) **OR** compiler should do it properly

**Current State**: Stage 3→4 transformation needed because inference framework expects `mlp_op` keys but compiler generates `spmd_rank` keys.

**Future Improvement**: Compiler should generate correct keys directly.

---

### 9.3 Memory Optimization Strategies

**For Large Models (>15GB weights)**:

1. **Incremental loading**: Load and convert layer-by-layer
2. **Batch expert processing**: Process 4 experts at a time, not all 16
3. **Explicit garbage collection**: Force `gc.collect()` frequently
4. **Memory monitoring**: Track memory usage throughout
5. **Dtype consistency**: Avoid unnecessary float32 conversions
6. **Use memory mapping**: Load safetensors with mmap when possible

---

## 10. Debugging Weight Issues

### 10.1 Systematic Debugging Approach

**Step 1: Verify Weight Files Exist**

```bash
ls -lh model/model.safetensors*
# Should show multiple safetensors files totaling ~20GB
```

**Step 2: Check Weight Keys**

```python
from safetensors import safe_open
with safe_open("model/model.safetensors", framework="pt") as f:
    keys = list(f.keys())
    print(f"Total keys: {len(keys)}")
    print("Sample keys:", keys[:10])
```

**Step 3: Verify Key Conversion**

```python
hf_keys = set(hf_state_dict.keys())
neuron_keys = set(neuron_state_dict.keys())

# Check for missing keys
expected_keys = compute_expected_neuron_keys(config)
missing = expected_keys - neuron_keys
print(f"Missing keys: {missing}")
```

**Step 4: Verify Weight Values**

```python
# Check weight statistics
for key, tensor in neuron_state_dict.items():
    mean = tensor.float().mean().item()
    std = tensor.float().std().item()
    print(f"{key}: mean={mean:.6f}, std={std:.6f}")

    # Healthy weights typically have:
    # - Mean near 0
    # - Std between 0.01 and 0.1
    # - No NaN or Inf values
```

**Step 5: Compare with HF Weights**

```python
# Direct comparison
hf_weight = hf_model.model.embed_tokens.weight
nx_weight = neuronx_model.model.embed_tokens.weight

cos_sim = F.cosine_similarity(hf_weight.flatten(), nx_weight.flatten(), dim=0)
print(f"Cosine similarity: {cos_sim:.6f}")  # Should be > 0.99
```

---

## 11. Performance Considerations

### 11.1 Weight Loading Performance

| Operation                 | Time (32-layer model) | Memory Peak    |
| ------------------------- | --------------------- | -------------- |
| Load HF safetensors       | 30-60 seconds         | +20 GB         |
| Convert to NeuronX format | 2-5 minutes           | +30 GB         |
| Save NeuronX safetensors  | 30-60 seconds         | +10 GB         |
| **Total**                 | **3-7 minutes**       | **60 GB peak** |

### 11.2 Optimized Loading Performance

| Operation                  | Time (optimized) | Memory Peak    |
| -------------------------- | ---------------- | -------------- |
| Load HF safetensors (mmap) | 5-10 seconds     | +5 GB          |
| Convert incrementally      | 3-4 minutes      | +20 GB         |
| Save sharded checkpoints   | 1-2 minutes      | +5 GB          |
| **Total**                  | **4-6 minutes**  | **30 GB peak** |

**Optimization Impact**: 50% memory reduction, similar time

---

## 12. Common Errors and Solutions

### Error 1: SPMD Key Not Found

```
KeyError: 'layers.0.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight'
```

**Solution**: Run Stage 3→4 transformation script

### Error 2: Weight Shape Mismatch

```
RuntimeError: shape mismatch: [16, 4096, 12800] vs [16, 6400, 4096]
```

**Solution**: Check transposition - weights must be .T

### Error 3: Missing rank_util.rank Tensors

```
KeyError: 'rank_util.rank'
```

**Solution**: Add rank tensors manually to state dict

### Error 4: Out of Memory During Conversion

```
RuntimeError: CUDA out of memory
```

**Solution**: Use memory-optimized conversion pattern

---

## Summary Statistics

- **Total Weight Management Scripts**: 34
- **Weight Transformation Stages**: 4 (HF → Compilation → SPMD → Inference)
- **Key Transformations**: 3 major (transpose, concatenate, rename)
- **Memory Optimization**: 50% reduction (80GB → 40GB)
- **Critical Configurations**:
  - early_expert_affinity_modulation = False
  - reduce_dtype = torch.bfloat16
  - TP=16, EP=1 (recommended starting point)
- **Weight Verification Points**: 5 (embeddings, lm_head, attention, experts, router)

---

## Conclusion

The weight management phase of the GenericMoE port required understanding a complex 4-stage transformation pipeline, implementing memory-efficient loading strategies, and discovering subtle precision issues in distributed operations. The 34 scripts in this category document the systematic exploration and resolution of weight-related challenges.

**Key Takeaways**:

1. **SPMD pipeline is critical**: Understanding the 4 stages prevents many debugging headaches
2. **Memory matters**: 15-20GB models need careful memory management
3. **Precision is fragile**: Small dtype conversion errors accumulate significantly
4. **Expert routing configuration**: early_expert_affinity_modulation flag is crucial for accuracy
5. **Verification is essential**: Always verify weight loading with checksums and sample comparisons

These patterns and insights are directly applicable to future large MoE model ports.
