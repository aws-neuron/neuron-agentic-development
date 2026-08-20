# Fixing KeyError: 'layers.0.self_attn.qkv_proj.q_proj.weight'

## Problem Summary

During the Phi3 model compilation for NeuronX, we encountered a critical KeyError:

```
KeyError: 'layers.0.self_attn.qkv_proj.q_proj.weight'
```

This error occurred because the NeuronX framework expected weights in a specific format that differed from the original HuggingFace Phi3 model structure.

## Root Cause Analysis

### 1. Weight Structure Mismatch

**HuggingFace Phi3 Format:**

- Uses fused QKV projections: `layers.{i}.self_attn.qkv_proj.weight`
- Uses fused gate/up projections: `layers.{i}.mlp.gate_up_proj.weight`

**NeuronX Expected Format:**

- Expects separate projections: `layers.{i}.self_attn.qkv_proj.q_proj.weight`, `k_proj.weight`, `v_proj.weight`
- Expects separate MLP projections: `layers.{i}.mlp.gate_proj.weight`, `up_proj.weight`

### 2. Missing Weight Conversion

The framework was loading HuggingFace weights directly without converting them to the expected NeuronX format, causing the KeyError when trying to access the separated weight keys.

## Solution Implementation

### Step 1: Created Weight Conversion Function

Added `convert_hf_to_neuron_state_dict` method to split fused weights:

```python
@staticmethod
def convert_hf_to_neuron_state_dict(hf_state_dict, config):
    """Convert HuggingFace state dict to NeuronX format"""
    neuron_state_dict = {}

    # Calculate dimensions
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

    q_hidden_size = num_attention_heads * head_dim
    kv_hidden_size = num_key_value_heads * head_dim

    for key, tensor in hf_state_dict.items():
        if key.startswith('model.'):
            key = key[6:]  # Remove 'model.' prefix

        if 'self_attn.qkv_proj.weight' in key:
            # Split QKV fused weight into separate Q, K, V weights
            layer_idx = key.split('.')[1]

            # Split the fused QKV weight
            q_weight = tensor[:q_hidden_size, :]
            k_weight = tensor[q_hidden_size:q_hidden_size + kv_hidden_size, :]
            v_weight = tensor[q_hidden_size + kv_hidden_size:, :]

            # Create separate weight keys
            base_key = f"layers.{layer_idx}.self_attn.qkv_proj"
            neuron_state_dict[f"{base_key}.q_proj.weight"] = q_weight
            neuron_state_dict[f"{base_key}.k_proj.weight"] = k_weight
            neuron_state_dict[f"{base_key}.v_proj.weight"] = v_weight

        elif 'mlp.gate_up_proj.weight' in key:
            # Split gate_up fused weight into separate gate and up weights
            layer_idx = key.split('.')[1]
            intermediate_size = tensor.shape[0] // 2

            gate_weight = tensor[:intermediate_size, :]
            up_weight = tensor[intermediate_size:, :]

            base_key = f"layers.{layer_idx}.mlp"
            neuron_state_dict[f"{base_key}.gate_proj.weight"] = gate_weight
            neuron_state_dict[f"{base_key}.up_proj.weight"] = up_weight

        else:
            # Copy other weights as-is
            neuron_state_dict[key] = tensor

    return neuron_state_dict
```

### Step 2: Added Automatic Weight Loading

Implemented `load_state_dict` override to automatically convert weights during model loading:

```python
def load_state_dict(self, state_dict, strict=True):
    """
    Override load_state_dict to handle weight conversion from HuggingFace format
    """
    # Check if this is a HuggingFace state dict that needs conversion
    if self._is_hf_state_dict(state_dict):
        print("🔧 Converting HuggingFace weights to NeuronX format...")
        state_dict = self.convert_hf_to_neuron_state_dict(state_dict, self.config)
        print(f"✅ Weight conversion completed. Total keys: {len(state_dict)}")

    return super().load_state_dict(state_dict, strict)

def _is_hf_state_dict(self, state_dict):
    """
    Check if the state dict is from HuggingFace format (has 'model.' prefix)
    """
    hf_keys = [key for key in state_dict.keys() if key.startswith('model.')]
    return len(hf_keys) > 0
```

### Step 3: Weight Dimension Calculations

The conversion required precise dimension calculations:

- **Q projection**: `num_attention_heads * head_dim` (3072 for Phi3-mini)
- **K/V projections**: `num_key_value_heads * head_dim` (3072 each for Phi3-mini)
- **Gate/Up projections**: `intermediate_size` each (8192 each for Phi3-mini)

## Verification Process

### 1. Debug Script Creation

Created `debug_keyerror.py` to verify the conversion:

```python
# Load and convert weights
hf_state_dict = torch.load(model_path, map_location='cpu')
converted_state_dict = NeuronPhi3ForCausalLM.convert_hf_to_neuron_state_dict(hf_state_dict, config)

# Check for the problematic key
key_to_check = 'layers.0.self_attn.qkv_proj.q_proj.weight'
if key_to_check in converted_state_dict:
    print(f"✅ Key EXISTS: {converted_state_dict[key_to_check].shape}")
else:
    print(f"❌ Key MISSING: {key_to_check}")
```

### 2. Compilation Test

The fix was validated by successful compilation:

- **Before**: KeyError during compilation
- **After**: Successful compilation with weight conversion message:
  ```
  🔧 Converting HuggingFace weights to NeuronX format...
  ✅ Weight conversion completed. Total keys: 324
  ```

## Key Insights

### 1. Framework Expectations

NeuronX Distributed Inference framework expects weights in a specific separated format, not the fused format used by HuggingFace models.

### 2. Automatic Conversion

The solution implements automatic detection and conversion of HuggingFace weights during model loading, making the process transparent to users.

### 3. Dimension Preservation

The weight splitting preserves the original tensor dimensions and functionality while reorganizing them into the expected structure.

## Impact

This fix enables:

- ✅ Successful Phi3 model compilation for NeuronX
- ✅ Automatic weight format conversion
- ✅ Compatibility with HuggingFace model checkpoints
- ✅ Transparent operation for end users

## Files Modified

1. **`neuron_port/modeling_phi3.py`**:
   - Added `convert_hf_to_neuron_state_dict` static method
   - Added `load_state_dict` override
   - Added `_is_hf_state_dict` helper method

2. **Debug scripts** (for verification):
   - `agent_artifacts/tmp/debug_keyerror.py`
   - Various weight debugging utilities

## Lessons Learned

1. **Weight Format Compatibility**: Different ML frameworks may expect different weight organizations even for the same model architecture.

2. **Automatic Detection**: Implementing automatic format detection makes the conversion process seamless.

3. **Thorough Testing**: Creating specific debug scripts to verify the fix ensures the solution works correctly.

4. **Framework Integration**: Understanding how the target framework loads and expects weights is crucial for successful model porting.
