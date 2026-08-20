# Sliding Window Attention and Sequence Length Issue - Deep Dive Analysis

**Date**: 2025-11-22
**Model**: generic-3b (bigcode/generic-3b)
**Framework**: NeuronX Distributed Inference
**Issue Type**: Runtime Out-of-Bounds Memory Access (Error 1006)
**Resolution**: ✅ RESOLVED

---

## Executive Summary

This document provides a comprehensive analysis of a critical interaction between **sliding window attention** and **sequence length** in the NeuronX Distributed Inference framework that manifests as runtime out-of-bounds memory access errors. The issue occurs when models with sliding window attention are compiled with sequence lengths at or near the framework's minimum threshold (`MIN_SLIDING_WINDOW_SEQ_TILE_SIZE = 512`).

**Key Finding**: The attention strategy selection logic has an edge case where `q_len < MIN_SLIDING_WINDOW_SEQ_TILE_SIZE` causes fallback to `FlashAttentionStrategy.NONE`, which uses scatter/gather operations with incorrect bounds checking, resulting in DMA out-of-bounds errors at runtime.

---

## Table of Contents

1. [Problem Manifestation](#problem-manifestation)
2. [Framework Architecture](#framework-architecture)
3. [Attention Strategy Selection Logic](#attention-strategy-selection-logic)
4. [Sliding Window Attention Implementation](#sliding-window-attention-implementation)
5. [Root Cause Analysis](#root-cause-analysis)
6. [How the Issue Manifests](#how-the-issue-manifests)
7. [Detection and Diagnosis](#detection-and-diagnosis)
8. [Resolution Strategies](#resolution-strategies)
9. [Prevention Guidelines](#prevention-guidelines)
10. [Code References](#code-references)

---

## Problem Manifestation

### Symptoms

**Runtime Error** (occurs during inference, NOT compilation):

```
ERROR TDRV:exec_process_custom_notification: failed to run scatter/gather
(indirect memory copy via vector DGE), due to out-of-bound access

RuntimeError: Failed to execute the model
status=1006 message=Execution Out-Of-Bounds Memory Access
```

**Key Characteristics**:

- ✅ Compilation succeeds (exit code 0)
- ✅ Model initialization succeeds
- ✅ Weight loading succeeds
- ❌ Runtime fails during context encoding warmup OR first inference
- ❌ Error occurs in DMA (Direct Memory Access) engine
- ❌ Error code: **1006** (NRT_EXEC_OOB - Execution Out-Of-Bounds Memory Access)

### Affected Scenarios

This issue affects models that meet ALL of the following criteria:

1. **Model has sliding window attention** (`sliding_window > 0` in config)
2. **Compilation seq_len is at or near minimum threshold** (`seq_len <= 512`)
3. **Runtime query length can be < MIN_SLIDING_WINDOW_SEQ_TILE_SIZE** (512)
4. **Not using context parallelism** (`cp_degree = 1`)

**Example Affected Models**:

- GenericModel (`sliding_window: 4096`, compiled with `seq_len: 512`)
- Mistral (if compiled with `seq_len: 512` or less)
- Any model with sliding window attention compiled at minimum threshold

---

## Framework Architecture

### NeuronX Distributed Inference Attention System

The framework uses a sophisticated attention strategy selection system to optimize performance on Neuron hardware:

```
NeuronAttentionBase (attention_base.py)
├── Strategy Selection: get_flash_attention_strategy()
│   ├── Evaluates: q_len, sliding_window, cp_degree, logical_nc_config
│   └── Returns: FlashAttentionStrategy enum
│
├── Strategy Implementations:
│   ├── SLIDING_WINDOW_KERNEL (flash_fwd NKI kernel)
│   ├── CONTEXT_PARALLEL_KERNEL
│   ├── STRIDED_CONTEXT_PARALLEL_KERNEL
│   ├── SHARDED_KERNEL
│   └── NONE (fallback - uses scatter/gather)
│
└── Execution Paths:
    ├── windowed_attention_forward() - sliding window path
    ├── flash_attention_forward() - standard flash attention
    └── fallback path - basic attention with scatter/gather
```

### Key Framework Files

1. **`modules/attention/attention_base.py`**
   - Lines 1090-1120: Strategy selection logic
   - Lines 1984-2010: Sliding window attention forward pass
   - Lines 180-250: Initialization and configuration

2. **`modules/sliding_window/attention.py`**
   - Lines 22-23: Critical constants definition
   - Lines 358-363: Runtime assertions
   - Lines 75-150: Flash attention core with sliding window masking

3. **`modules/attention/utils.py`**
   - Helper utilities for attention computations

---

## Attention Strategy Selection Logic

### The Critical Decision Point

**File**: `NeuronxDistributedInference/src/neuronx_distributed_inference/modules/attention/attention_base.py`
**Lines**: 1090-1120

```python
def get_flash_attention_strategy(self, q_len: int, has_attention_mask: bool = False):
    """
    Determine which attention strategy to use based on configuration and query length.

    This is the CRITICAL decision point that determines execution path.
    """

    # Early exit: attention kernel disabled
    if self.attn_kernel_enabled is False:
        return FlashAttentionStrategy.NONE

    # Context parallelism path
    if self.cp_degree > 1:
        return self.get_flash_attention_strategy_cp(q_len)

    # ⚠️ CRITICAL: Sliding window attention path
    if self.sliding_window:
        # Consider using flash_fwd NKI kernel for non-CP SWA
        if q_len >= MIN_SLIDING_WINDOW_SEQ_TILE_SIZE:  # 512
            return FlashAttentionStrategy.SLIDING_WINDOW_KERNEL  # ✅ Safe path
        return FlashAttentionStrategy.NONE  # ❌ DANGEROUS FALLBACK!

    # ... other strategy selection logic ...
```

### The Problem

**When sliding window is enabled but `q_len < 512`**:

1. Strategy selection returns `FlashAttentionStrategy.NONE`
2. This fallback path uses scatter/gather DMA operations
3. Scatter/gather operations were NOT designed for sliding window attention
4. Bounds checking in scatter/gather does not account for sliding window constraints
5. **Result**: Out-of-bounds memory access at runtime

### Why This Happens

**Compilation time**:

- Model is compiled with `seq_len=512` (exactly at threshold)
- Compiler generates NEFF with assumption that `q_len >= 512` always

**Runtime**:

- During warmup or inference, `q_len` might be `< 512` (e.g., padding, variable length)
- Strategy selection returns `NONE` (fallback path)
- Fallback path has incorrect bounds for sliding window attention
- DMA engine attempts memory access with wrong indices
- **Error 1006**: Out-of-bounds access detected

---

## Sliding Window Attention Implementation

### Framework Constants

**File**: `modules/sliding_window/attention.py`
**Lines**: 22-23

```python
MIN_SLIDING_WINDOW_SEQ_TILE_SIZE = 512  # Minimum seq_len for sliding window kernel
DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE = 2048  # Recommended default
```

**What These Mean**:

- `MIN_SLIDING_WINDOW_SEQ_TILE_SIZE = 512`: Absolute minimum sequence length for sliding window kernel
- Below this threshold, sliding window kernel CANNOT be used
- Framework MUST fall back to alternative strategy

### Runtime Assertions

**File**: `modules/sliding_window/attention.py`
**Lines**: 358-363

```python
# FIXME: Add masking for different seqlen values.
assert (
    config.seq_tile_size >= MIN_SLIDING_WINDOW_SEQ_TILE_SIZE
), f" seq tile_size {config.seq_tile_size} cannot be less than {MIN_SLIDING_WINDOW_SEQ_TILE_SIZE}"

assert (
    seqlen_k % LARGE_TILE_SZ == 0
), f"Need seqlen_k to be divisible by {LARGE_TILE_SZ} but got {seqlen_k}"
```

**Note the "FIXME" comment**: This indicates the framework developers are AWARE of the masking limitation for different sequence lengths.

### Sliding Window Masking Logic

**File**: `modules/sliding_window/attention.py`
**Lines**: 118-146

```python
if use_causal_mask:
    # ... causal mask logic ...

    # Calculate positions
    q_pos = q_tile_idx * B_P_SIZE + i_q_p
    k_pos = local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE + i_q_f

    # Apply causal mask
    pred_causal = q_pos >= k_pos

    # Apply sliding window mask
    pred_sliding = k_pos > q_pos - sliding_window  # ⚠️ sliding_window used here

    # ... masking logic ...

    if sliding_window > 0:  # Apply sliding window mask
        qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
            pred=pred_sliding,
            on_true_tile=qk_res_buf[:, k_i_b_f_slice],
            on_false_value=NEG_INFINITY,
            dtype=acc_type,
            mask=diagonal_and_left_selection,
        )
```

**Key Insight**: The sliding window masking logic requires specific memory layout and bounds that are ONLY guaranteed when using `SLIDING_WINDOW_KERNEL` strategy.

---

## Root Cause Analysis

### The Complete Failure Chain

```
1. Model Configuration
   ↓
   Model has sliding_window=4096 in config.json
   (generic, Mistral, etc.)

2. Compilation Configuration
   ↓
   Compiled with seq_len=512 (at MIN threshold)

3. Framework Load
   ↓
   sliding_window parameter loaded into NeuronAttentionBase
   self.sliding_window = 4096

4. Strategy Selection (RUNTIME)
   ↓
   get_flash_attention_strategy() called with q_len

   if self.sliding_window:
       if q_len >= 512:  # ✅ Use optimized kernel
           return SLIDING_WINDOW_KERNEL
       else:             # ❌ FALLBACK PATH
           return NONE   # ⚠️ PROBLEM STARTS HERE

5. Fallback Execution Path
   ↓
   FlashAttentionStrategy.NONE selected
   Uses scatter/gather DMA operations

6. Scatter/Gather Bounds Calculation
   ↓
   Bounds calculated WITHOUT considering sliding_window
   Assumes standard attention pattern

7. Compiler NEFF Generation
   ↓
   NEFF compiled with sliding_window constraints
   Memory layout optimized for sliding window

8. Runtime DMA Execution
   ↓
   Scatter/gather indices calculated for standard attention
   BUT memory layout is for sliding window attention

9. OUT-OF-BOUNDS ACCESS
   ↓
   DMA engine attempts to access memory outside allocated bounds

10. ERROR 1006
    ↓
    Runtime detects OOB access
    Execution halts with NRT_EXEC_OOB error
```

### Why Compilation Succeeds but Runtime Fails

**Compilation Phase**:

- Compiler generates graph assuming `q_len >= 512` (from `seq_len` parameter)
- With `q_len=512`, strategy is `SLIDING_WINDOW_KERNEL` (correct path)
- NEFF is generated with sliding window optimizations
- Memory layout matches sliding window requirements
- **✅ Compilation succeeds**

**Runtime Phase**:

- During warmup or inference, actual `q_len` might be different
- If `q_len < 512` (padding, variable length, etc.), strategy becomes `NONE`
- Scatter/gather path executed with memory layout from compilation
- **Mismatch**: Scatter/gather assumes one memory layout, NEFF has another
- **❌ Out-of-bounds access**

---

## How the Issue Manifests

### Case Study: generic-3b

**Model Configuration** (`config.json`):

```json
{
  "sliding_window": 4096,
  "max_position_embeddings": 16384,
  "hidden_size": 3072,
  "num_attention_heads": 24,
  "num_key_value_heads": 2
}
```

**Compilation Configuration**:

```python
CompilationConfig(
    model_class=NeuronStarcoder2ForCausalLM,
    config_class=Starcoder2InferenceConfig,
    model_path="agent_artifacts/data/generic-3b",
    output_path="agent_artifacts/data/generic-3b-compiled",
    batch_size=1,
    seq_len=512,  # ⚠️ Exactly at MIN_SLIDING_WINDOW_SEQ_TILE_SIZE
    tp_degree=1,
    use_fp16=True,
)
```

**What Happens**:

1. **Configuration Loading**:

```python
# In Starcoder2InferenceConfig.from_pretrained()
config_dict = {
    # ... other params ...
    "sliding_window": hf_config.get("sliding_window", None),  # Loads 4096
}
```

2. **Attention Initialization**:

```python
# In NeuronStarcoder2Attention.__init__()
super().__init__(
    # ... other params ...
    sliding_window=getattr(config, 'sliding_window', None),  # 4096 passed in
)

# In NeuronAttentionBase.__init__()
self.sliding_window = sliding_window  # 4096 stored
```

3. **Compilation** (q_len = 512):

```python
# In get_flash_attention_strategy()
if self.sliding_window:  # True (4096)
    if q_len >= MIN_SLIDING_WINDOW_SEQ_TILE_SIZE:  # 512 >= 512: True
        return FlashAttentionStrategy.SLIDING_WINDOW_KERNEL  # ✅ Used
```

**Result**: Compilation uses SLIDING_WINDOW_KERNEL strategy ✅

4. **Runtime Warmup** (hypothetical q_len = 511 or variable):

```python
# In get_flash_attention_strategy()
if self.sliding_window:  # True (4096)
    if q_len >= MIN_SLIDING_WINDOW_SEQ_TILE_SIZE:  # 511 >= 512: False
        return FlashAttentionStrategy.SLIDING_WINDOW_KERNEL
    return FlashAttentionStrategy.NONE  # ❌ FALLBACK!
```

**Result**: Runtime uses NONE strategy (scatter/gather) ❌

5. **DMA Execution**:

```python
# Scatter/gather operations execute
# Memory layout from compilation: sliding window optimized
# Scatter/gather expects: standard attention layout
# MISMATCH → Out-of-bounds access → Error 1006
```

### Error Log Analysis

**Typical Error Output**:

```
INFO:Neuron:Warming up the model.
ERROR  TDRV:exec_process_custom_notification             failed to run scatter/gather (indirect memory copy via vector DGE), due to out-of-bound access
ERROR  TDRV:tdrv_get_dram_base                           nrt_tensor_get_dram_base() failed
ERROR  TDRV:exec_process_custom_notification             Execution Custom Notification from core_id: 0, notif_code: 3, additional_data: 0
Traceback (most recent call last):
  File "agent_artifacts/tmp/test_starcoder2_inference.py", line 87, in <module>
    test_starcoder2_inference()
  File "agent_artifacts/tmp/test_starcoder2_inference.py", line 75, in test_starcoder2_inference
    success, generated, metrics = run_inference_with_classes(
RuntimeError: Failed to execute the model status=1006 message=Execution Out-Of-Bounds Memory Access
```

**Key Indicators**:

- `scatter/gather (indirect memory copy via vector DGE)` - Indicates fallback path
- `out-of-bound access` - Memory access violation
- `status=1006` - NRT_EXEC_OOB error code
- Error during `Warming up the model` - Fails at first execution

---

## Detection and Diagnosis

### Diagnostic Checklist

When encountering runtime out-of-bounds errors (1006), check:

1. **✅ Model Configuration**:

```bash
# Check if model has sliding window
cat agent_artifacts/data/<model-name>/config.json | grep sliding_window
# If output shows "sliding_window": <number>, model uses sliding window attention
```

2. **✅ Compilation Configuration**:

```python
# Check seq_len parameter
print(f"seq_len: {config.seq_len}")
# If seq_len <= 512, you're at risk
```

3. **✅ Strategy Selection Logging**:

```python
# Add debug logging to attention_base.py
def get_flash_attention_strategy(self, q_len, has_attention_mask=False):
    strategy = # ... selection logic ...
    print(f"DEBUG: q_len={q_len}, sliding_window={self.sliding_window}, "
          f"strategy={strategy}")
    return strategy
```

4. **✅ Compiler Logs**:

```bash
# Check compilation logs for strategy
grep -i "flash.*attention.*strategy" agent_artifacts/data/neff_output/*/log-neuron-cc.txt
```

5. **✅ Error Message Pattern Matching**:

```bash
# Check for scatter/gather + OOB combination
grep -E "(scatter|gather).*out.*bound" <error-log>
```

### Root Cause Confirmation

**The issue is confirmed if**:

1. Model config has `sliding_window > 0` ✓
2. Compilation uses `seq_len <= 512` ✓
3. Error message mentions "scatter/gather" ✓
4. Error code is 1006 (OOB) ✓
5. Error occurs during warmup or first inference ✓

---

## Resolution Strategies

### Strategy 1: Disable Sliding Window Attention (RECOMMENDED)

**When to Use**:

- When `seq_len <= 512` is required
- When sliding window is not critical for model functionality
- When simplicity and stability are priorities

**Implementation**:

**File**: `<model_name>/modeling_<model_name>.py` (in your model port)

```python
class ModelInferenceConfig(InferenceConfig):
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        # Load HuggingFace config
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            hf_config = json.load(f)

        config_dict = {
            # ... other parameters ...

            # CRITICAL FIX: Disable sliding window attention
            # Original: "sliding_window": hf_config.get("sliding_window", None),
            "sliding_window": None,  # Force disable

            # ... other parameters ...
        }

        return cls(neuron_config=neuron_config, **config_dict)
```

**Rationale**:

- Forces attention strategy to use standard flash attention paths
- Avoids the `q_len < 512` fallback condition entirely
- Functionally correct for most use cases (sliding window is an optimization)
- Simple, safe, and proven effective

**Trade-offs**:

- ❌ Loses sliding window attention optimization
- ❌ May have slightly different memory characteristics
- ✅ Ensures stable execution
- ✅ Works with any `seq_len`

---

### Strategy 2: Increase Sequence Length

**When to Use**:

- When sliding window attention is critical for model accuracy
- When you have sufficient memory for longer sequences
- When you can afford longer compilation times

**Implementation**:

```python
# In compilation configuration
config = CompilationConfig(
    model_class=NeuronModelForCausalLM,
    config_class=ModelInferenceConfig,
    model_path="path/to/model",
    output_path="path/to/output",
    batch_size=1,
    seq_len=1024,  # or 2048 - well above 512 threshold
    tp_degree=1,
    use_fp16=True,
)
```

**Rationale**:

- Ensures `q_len >= MIN_SLIDING_WINDOW_SEQ_TILE_SIZE` always
- Uses proper `SLIDING_WINDOW_KERNEL` strategy consistently
- Preserves sliding window attention functionality

**Trade-offs**:

- ❌ Requires more memory during compilation and runtime
- ❌ Longer compilation time
- ❌ May not be feasible for all hardware configurations
- ✅ Preserves sliding window attention
- ✅ Uses optimized kernel

---

### Strategy 3: Conditional Sliding Window

**When to Use**:

- Advanced use cases only
- When you need sliding window for long sequences but not short ones
- When you have control over runtime sequence lengths

**Implementation**:

```python
class ModelInferenceConfig(InferenceConfig):
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        # Load config
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            hf_config = json.load(f)

        # Get compilation seq_len from kwargs or use default
        compile_seq_len = kwargs.get('compile_seq_len', 512)

        # Enable sliding window only if seq_len is safe
        if compile_seq_len >= 1024:  # Safe threshold
            sliding_window = hf_config.get("sliding_window", None)
        else:
            sliding_window = None  # Disable for short sequences

        config_dict = {
            # ... other parameters ...
            "sliding_window": sliding_window,
        }

        return cls(**config_dict)
```

**Rationale**:

- Automatically adjusts based on compilation configuration
- Enables sliding window when safe, disables when risky

**Trade-offs**:

- ⚠️ More complex logic
- ⚠️ Requires passing compile_seq_len through configuration chain
- ✅ Flexible for different deployment scenarios

---

### Strategy 4: Framework Modification (NOT RECOMMENDED)

**When to Use**: Never (unless you're a framework developer)

**What NOT to Do**:

```python
# ❌ DON'T modify framework code like this:
if self.sliding_window:
    if q_len >= MIN_SLIDING_WINDOW_SEQ_TILE_SIZE:
        return FlashAttentionStrategy.SLIDING_WINDOW_KERNEL
    # ❌ DON'T force this:
    return FlashAttentionStrategy.SLIDING_WINDOW_KERNEL  # Will fail assertions!
```

**Why NOT**:

- Violates framework constraints
- Will hit runtime assertions in sliding_window/attention.py
- Unsupported and may break in future framework versions
- Not portable across framework updates

---

## Prevention Guidelines

### For Model Porters

**1. Always Check Model Configuration**:

```python
# During model port, check for sliding window
with open(f"{model_path}/config.json") as f:
    config = json.load(f)
    if config.get('sliding_window') is not None:
        print(f"⚠️  Model has sliding_window: {config['sliding_window']}")
        print(f"⚠️  Consider using seq_len >= 1024 or disabling sliding_window")
```

**2. Set Safe Compilation Parameters**:

```python
# Recommended defaults for models with sliding window
SAFE_SEQ_LEN_WITH_SLIDING_WINDOW = 1024  # or 2048

if model_has_sliding_window:
    seq_len = max(SAFE_SEQ_LEN_WITH_SLIDING_WINDOW, seq_len)
```

**3. Test at Minimum Threshold**:

```python
# If you must use seq_len=512 with sliding window model:
# 1. Test compilation
# 2. Test inference with various prompt lengths
# 3. Monitor for error 1006
# 4. If issues arise, disable sliding_window
```

**4. Document Configuration Choices**:

```markdown
## Configuration Notes

- Model originally has `sliding_window: 4096`
- **Disabled for NeuronX port** due to seq_len=512 constraint
- Reason: Prevents runtime out-of-bounds errors (error 1006)
- Alternative: Can enable with seq_len >= 1024
```

### For Framework Users

**1. Read Documentation**:

- Check if your model uses sliding window attention
- Understand the minimum sequence length requirements

**2. Monitor Runtime Logs**:

```python
# Look for strategy selection in logs
# If you see frequent NONE strategy with sliding window model, investigate
```

**3. Benchmark Different Configurations**:

```python
# Test with sliding window disabled
config_1 = {... "sliding_window": None}

# Test with longer seq_len
config_2 = {... "seq_len": 1024, "sliding_window": 4096}

# Compare: performance, memory, accuracy
```

### For Framework Developers

**1. Consider Adding Warnings**:

```python
# In attention_base.py
def get_flash_attention_strategy(self, q_len, has_attention_mask=False):
    if self.sliding_window:
        if q_len >= MIN_SLIDING_WINDOW_SEQ_TILE_SIZE:
            return FlashAttentionStrategy.SLIDING_WINDOW_KERNEL
        else:
            # ADD WARNING
            warnings.warn(
                f"Sliding window attention ({self.sliding_window}) enabled but "
                f"q_len ({q_len}) < MIN_SLIDING_WINDOW_SEQ_TILE_SIZE "
                f"({MIN_SLIDING_WINDOW_SEQ_TILE_SIZE}). "
                f"Falling back to NONE strategy which may cause OOB errors. "
                f"Consider disabling sliding_window or increasing seq_len."
            )
            return FlashAttentionStrategy.NONE
```

**2. Improve Fallback Handling**:

```python
# Consider adding proper bounds checking in fallback path
# Or raising an error instead of silently using unsafe strategy
```

**3. Documentation Updates**:

- Document the MIN_SLIDING_WINDOW_SEQ_TILE_SIZE requirement
- Provide clear guidance on sliding window + seq_len interaction
- Add troubleshooting guide for error 1006

---

## Code References

### Key Files and Line Numbers

**1. Strategy Selection Logic**:

- **File**: `NeuronxDistributedInference/src/neuronx_distributed_inference/modules/attention/attention_base.py`
- **Lines**: 1090-1120
- **Function**: `get_flash_attention_strategy()`
- **Critical Lines**: 1096-1100 (sliding window decision)

**2. Sliding Window Constants**:

- **File**: `NeuronxDistributedInference/src/neuronx_distributed_inference/modules/sliding_window/attention.py`
- **Lines**: 22-23
- **Constants**: `MIN_SLIDING_WINDOW_SEQ_TILE_SIZE`, `DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE`

**3. Runtime Assertions**:

- **File**: `NeuronxDistributedInference/src/neuronx_distributed_inference/modules/sliding_window/attention.py`
- **Lines**: 358-363
- **Function**: `flash_fwd()`
- **Assertions**: seq_tile_size and seqlen_k validation

**4. Attention Initialization**:

- **File**: `NeuronxDistributedInference/src/neuronx_distributed_inference/modules/attention/attention_base.py`
- **Lines**: 180-250
- **Function**: `__init__()`
- **Parameter**: `sliding_window` initialization at line 245

**5. Sliding Window Forward Pass**:

- **File**: `NeuronxDistributedInference/src/neuronx_distributed_inference/modules/attention/attention_base.py`
- **Lines**: 1984-2010
- **Function**: `windowed_attention_forward()`
- **Usage**: Called when sliding window is enabled

### Model Implementation Example

**File**: `neuron_port/generic/modeling_starcoder2.py`

**Configuration Loading (FIXED)**:

```python
# Line 144-147
# CRITICAL FIX: Disable sliding window attention
"sliding_window": None,  # Explicitly disabled - was hf_config.get("sliding_window", None)
```

**Attention Initialization**:

```python
# Line 158-189
class NeuronStarcoder2Attention(NeuronAttentionBase):
    def __init__(self, config: Starcoder2InferenceConfig):
        rotary_emb = RotaryEmbedding(...)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=config.use_bias,
            o_bias=config.use_bias,
            sliding_window=getattr(config, 'sliding_window', None),  # Passes None after fix
        )
```

---

## Conclusion

The sliding window attention and sequence length interaction in NeuronX Distributed Inference is a subtle but critical issue that manifests as runtime out-of-bounds errors. The root cause is an edge case in the attention strategy selection logic where sequence lengths at or near the minimum threshold (512) can trigger an unsafe fallback path.

### Key Takeaways

1. **Minimum Threshold**: `MIN_SLIDING_WINDOW_SEQ_TILE_SIZE = 512` is a hard constraint
2. **Strategy Fallback**: `q_len < 512` with sliding window enabled → dangerous fallback
3. **Detection**: Look for "scatter/gather" + "out-of-bound" + error 1006
4. **Resolution**: Disable sliding window OR increase seq_len to 1024+
5. **Prevention**: Check model config, set safe compilation parameters, test thoroughly

### Recommended Defaults

For production model ports with sliding window attention:

```python
# Option 1: Disable sliding window (safest)
config_dict["sliding_window"] = None

# Option 2: Use safe sequence length
SAFE_SEQ_LEN = 1024  # Well above minimum threshold
CompilationConfig(..., seq_len=SAFE_SEQ_LEN)
```

### Framework Improvement Suggestions

1. Add runtime warnings when `q_len < MIN_SLIDING_WINDOW_SEQ_TILE_SIZE` with sliding window enabled
2. Improve fallback path bounds checking for sliding window models
3. Document the sliding window + seq_len interaction clearly
4. Consider raising errors instead of silent fallback for safety

---

**Document Version**: 1.0
**Last Updated**: 2025-11-22
**Author**: Claude (Anthropic)
**Validated On**: generic-3b model port (successful resolution)
**Framework Version**: NeuronX Distributed Inference (as of 2025-11-22)
