# NeuronX Model Porting Guide

## Complete Reference for Porting Transformer Models to AWS Trainium/Inferentia

**Purpose**: This comprehensive guide combines systematic porting procedures with real-world learnings from successful and failed ports. It provides both a quick-reference playbook and deep technical knowledge for porting transformer models to AWS NeuronX.

**Success Rate**: Following this guide, models compile successfully on first try with both context encoding and token generation working perfectly.

---

## Table of Contents

1. [Overview and Philosophy](#overview-and-philosophy)
2. [Critical Framework Patterns (Quick Reference)](#critical-framework-patterns-quick-reference)
3. [Pre-Implementation Research](#pre-implementation-research)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Common Issues and Solutions](#common-issues-and-solutions)
6. [Debugging Guide](#debugging-guide)
7. [Verification and Testing](#verification-and-testing)
8. [Complete Checklists](#complete-checklists)

---

## Overview and Philosophy

### The Golden Rule

**Framework patterns are REQUIRED, not optional.** When ALL working models follow a pattern, you must follow it exactly. The framework expects specific classes, methods, and parameters.

### The Successful Approach vs The Failed Approach

#### ✅ What Works (4-6 hours total)

1. **Start with research, not coding** - Analyze 3-4 working models (1-2 hours)
2. **Follow patterns exactly** - No deviations without strong evidence (2-3 hours)
3. **Test incrementally** - Compile 1 layer first, then full model (1 hour)
4. **Success on first compilation**

#### ❌ What Doesn't Work (10+ hours with failures)

1. Jump into implementation after casually looking at 1-2 models
2. Assume patterns are optional or style choices
3. Add complexity not present in working models (like unnecessary layer_idx)
4. Return base NeuronConfig instead of custom subclass
5. Omit critical framework attributes or parameters
6. Context encoding works ✅, but token generation fails ❌
7. Spend hours trying various fixes
8. Assume framework limitation
9. Eventually discover it was missing configuration

### Key Insights

1. **Patterns are REQUIRED, not optional** - When ALL working models follow a pattern, it's required by the framework
2. **Configuration issues look like framework bugs** - Token generation failures are almost always missing custom NeuronConfig
3. **Don't assume framework limitations** - If other models with similar architectures work, yours can too
4. **The custom NeuronConfig is CRITICAL** - #1 cause of token generation failure
5. **Trust the pattern** - If all working models do it, do it too; if none do it, don't add it

---

## Critical Framework Patterns (Quick Reference)

### Pattern 1: Custom NeuronConfig (REQUIRED)

**Why Critical**: Without this, token generation HLO tracing fails. The framework uses `is_token_gen = False` during token generation, causing wrong code paths and tensor shape mismatches in `attention_base.py:736`.

```python
class YourModelNeuronConfig(NeuronConfig):
    """Custom Neuron configuration - REQUIRED for token generation"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # CRITICAL: Framework uses this to determine attention class
        from .modeling_module import YourModelAttention
        self.attn_cls = YourModelAttention
```

**Common Mistake:**

```python
# ❌ WRONG - Causes token generation failure
class ModelNeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Missing: self.attn_cls = ...
```

---

### Pattern 2: Return Custom Class (REQUIRED)

```python
@classmethod
def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
    # ✅ CORRECT - Return your custom class
    return YourModelNeuronConfig

    # ❌ WRONG - Returns base class, causes token gen failure
    # return NeuronConfig
```

---

### Pattern 3: Complete add_derived_config (REQUIRED)

```python
def add_derived_config(self):
    """Framework expects all these attributes"""
    # REQUIRED: For attention computation distribution
    self.num_cores_per_group = 1

    # Calculate head_dim if missing
    if not hasattr(self, 'head_dim'):
        self.head_dim = self.hidden_size // self.num_attention_heads

    # REQUIRED: All 4 framework attributes
    if not hasattr(self, 'output_attentions'):
        self.output_attentions = False
    if not hasattr(self, 'output_hidden_states'):
        self.output_hidden_states = False
    if not hasattr(self, 'use_return_dict'):
        self.use_return_dict = True
    if not hasattr(self, 'use_cache'):
        self.use_cache = True

    # Set bias flags for attention layers
    if not hasattr(self, 'qkv_bias'):
        self.qkv_bias = getattr(self, 'use_bias', False)
    if not hasattr(self, 'o_bias'):
        self.o_bias = getattr(self, 'use_bias', False)
```

**Common Mistake:**

```python
# ❌ INCOMPLETE - Will cause undefined behavior
def add_derived_config(self):
    self.num_cores_per_group = 1
    # Missing: framework attributes, head_dim
```

---

### Pattern 4: Pass num_cores_per_group (REQUIRED)

```python
class YourModelAttention(NeuronAttentionBase):
    def __init__(self, config):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            # ✅ CRITICAL - Required for distributed computation
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            sliding_window=getattr(config, 'sliding_window', None),
        )
```

**Common Mistake:**

```python
# ❌ MISSING - Causes incorrect tensor shapes
super().__init__(
    config=config,
    # ... other params
    # Missing: num_cores_per_group=config.num_cores_per_group,
)
```

---

### Pattern 5: No layer_idx (USUALLY)

```python
# ✅ CORRECT - Most models don't use layer_idx
class Attention(NeuronAttentionBase):
    def __init__(self, config):  # No layer_idx
        super().__init__(...)

class DecoderLayer(nn.Module):
    def __init__(self, config):  # No layer_idx
        self.self_attn = Attention(config)  # No layer_idx passed

# In model:
self.layers = nn.ModuleList(
    [DecoderLayer(config) for _ in range(n)]  # No layer_idx
)
```

**When layer_idx IS needed**: Check your reference models. If none use it, don't add it.

---

### Pattern 6: Attention Output Unpacking (CRITICAL) ⭐

**Issue**: Token generation fails with dimension mismatch errors like:

```
RuntimeError: Check failed: t->size(dim) == expected_size (1 vs. 128)
Expected tensor to have size 128 at dimension 1, but got size 1
```

**Root Cause**: NeuronAttentionBase returns `NeuronAttentionBaseOutput` which supports BOTH attribute access AND tuple unpacking. However, **only tuple unpacking works correctly**.

**❌ WRONG Pattern** (causes dimension mismatch in token generation):

```python
# In decoder layer forward():
attn_output = self.self_attn(...)
hidden_states = attn_output.hidden_states  # ❌ Causes shape issues
present_key_value = attn_output.present_key_value
```

**✅ CORRECT Pattern** (required for token generation):

```python
# In decoder layer forward():
hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
    hidden_states=hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_value=past_key_value,
    **kwargs,
)
```

**Why This Matters**: The attribute access pattern works during context encoding but causes subtle tensor shape mismatches during token generation HLO tracing. **ALL working models** (GPT-2, Llama3, Mistral, etc.) use tuple unpacking.

**Debugging Tip**: If you see dimension mismatch errors during token generation compilation (not context encoding), check your decoder layer's attention output handling first.

---

### Pattern 7: MLP Return Type

**Standard MLP (most models):**

```python
def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.fc2(x)
    return x, None  # ✅ Tuple for framework compatibility

# In decoder:
hidden_states = self.mlp(hidden_states)[0]  # Extract with [0]
```

**SwiGLU MLP (Llama, Mistral):**

```python
def forward(self, x):
    gate_up = self.gate_up_proj(x)
    gate, up = gate_up.chunk(2, dim=-1)
    x = F.silu(gate) * up
    return self.down_proj(x)  # ✅ Single tensor for SwiGLU

# In decoder:
hidden_states = self.mlp(hidden_states)  # Use directly
```

---

### Pattern 8: Use Correct Base Classes

```python
# ✅ CORRECT: Outer wrapper
class ModelForCausalLM(NeuronBaseForCausalLM):
    _model_cls = ModelBase

# ✅ CORRECT: Inner model
class ModelBase(NeuronBaseModel):
    # Implement setup_attr_for_model() and init_model()
    # Do NOT override forward()

# ❌ WRONG: Using NeuronBaseModel directly without wrapper
class Model(NeuronBaseModel):
    pass
model = Model(config)  # Will fail with parallel group error
```

---

## Pre-Implementation Research

### DO NOT WRITE CODE YET

**Spend 1-2 hours understanding the pattern first. This saves 6-8 hours of debugging later.**

### Step 1: Identify Similar Models

Find 3-4 working models with similar architectures:

```bash
# Search for reference models
find . -name "modeling_*.py" -path "*/neuronx_distributed_inference/*"
find . -name "modeling_*.py" -path "*/NeuroborosFoundations/*"
```

Look for models with:

- Similar attention mechanism (MHA, GQA, MQA)
- Similar activation function (GELU, SiLU, SwiGLU)
- Similar normalization (LayerNorm, RMSNorm)

**Example Reference Models:**

- **For GQA models**: Llama, Mistral, GPT-OSS, Phi3
- **For MHA models**: GPT-2, BERT-style models
- **For MoE models**: PhiMoE, Mixtral

### Step 2: Component Analysis Checklist

Copy and fill this template for **each working model** (3-4 models):

```markdown
## Model: [Name]

### Configuration

- [ ] Has custom NeuronConfig class? (Yes/No)
  - What does **init** set? [list attributes]
- [ ] InferenceConfig.add_derived_config() sets: [list attributes]
- [ ] InferenceConfig.get_required_attributes() includes: [list attributes]
- [ ] InferenceConfig.get_neuron_config_cls() returns: [class name]

### Attention

- [ ] Attention.**init**() signature:
  - Takes layer_idx? (Yes/No)
  - Parameters passed to super().**init**: [list all]

### MLP

- [ ] MLP.forward() return type: Single tensor or tuple?
- [ ] MLP architecture: Standard FFN or SwiGLU?

### Decoder Layer

- [ ] DecoderLayer.**init**() takes layer_idx? (Yes/No)
- [ ] DecoderLayer.forward() MLP call: [exact pattern]

### Model

- [ ] Model.init_model() layer creation: [exact code]
- [ ] lm_head bias: True or False?
- [ ] Tied embeddings: How handled?
```

### Step 3: Identify Common Pattern

After analyzing 3-4 models, categorize findings:

- ✅ Things ALL models do → **REQUIRED pattern**
- ⚠️ Things SOME models do → **Architecture-specific**
- ❌ Things NO models do → **Don't add it!**

**Critical patterns to confirm:**

```
[ ] ALL models have custom NeuronConfig? → REQUIRED
[ ] ALL models set attn_cls in NeuronConfig.__init__? → REQUIRED
[ ] ALL models pass num_cores_per_group to attention? → REQUIRED
[ ] ALL models return custom class from get_neuron_config_cls()? → REQUIRED
[ ] ALL models set framework attrs in add_derived_config()? → REQUIRED
[ ] ANY models use layer_idx? → Check if needed for your architecture
```

---

## Step-by-Step Implementation

### Step 1: Create Custom NeuronConfig Class

```python
class YourModelNeuronConfig(NeuronConfig):
    """
    Neuron-specific configuration for YourModel

    CRITICAL: This class is REQUIRED for token generation to work.
    Without it, token generation HLO tracing fails with tensor shape mismatches.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Import here to avoid circular dependency
        from .modeling_module import NeuronYourModelAttention

        # CRITICAL: Framework uses this during token generation tracing
        self.attn_cls = NeuronYourModelAttention
```

**Why This Matters:**

- **Root Cause of Most Failures**: Token generation HLO tracing fails because the framework doesn't know which attention class to use
- **Symptom**: `is_token_gen = False` during token generation, causing wrong code path
- **Error**: Tensor shape mismatch in `attention_base.py:736` during `perform_prefill()`
- **Impact**: Without this, nothing else matters - context encoding may work but token generation will fail

**Checklist:**

- [ ] Class inherits from `NeuronConfig`
- [ ] `__init__` calls `super().__init__(**kwargs)`
- [ ] `__init__` sets `self.attn_cls` to your attention class
- [ ] Class is defined BEFORE `InferenceConfig` (order matters for imports)

---

### Step 2: Create InferenceConfig Class

```python
from typing import List, Type
import json
import os
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig


class YourModelInferenceConfig(InferenceConfig):
    """Configuration class for YourModel inference on Neuron"""

    def add_derived_config(self):
        """
        Add derived configuration parameters required by the framework

        CRITICAL: This method is called during initialization and MUST set
        all framework-required attributes
        """
        # REQUIRED: Framework uses this for attention computation distribution
        self.num_cores_per_group = 1

        # Calculate head_dim if not present in HF config
        if not hasattr(self, 'head_dim'):
            self.head_dim = self.hidden_size // self.num_attention_heads

        # REQUIRED: Framework expects all 4 of these attributes
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True
        if not hasattr(self, 'use_cache'):
            self.use_cache = True

        # Set bias flags for attention layers
        if not hasattr(self, 'qkv_bias'):
            self.qkv_bias = getattr(self, 'use_bias', False)
        if not hasattr(self, 'o_bias'):
            self.o_bias = getattr(self, 'use_bias', False)

        # Disable sliding window for initial testing (can cause runtime errors)
        # Re-enable for production if needed
        if hasattr(self, 'sliding_window') and self.sliding_window is not None:
            pass  # Leave as-is or set to None for testing

    def get_required_attributes(self) -> List[str]:
        """
        List of required attributes from HuggingFace config.json

        These attributes MUST be present in the HF config or provided during initialization
        """
        return [
            "hidden_size",              # Model hidden dimension
            "num_attention_heads",      # Number of attention heads
            "num_hidden_layers",        # Number of transformer layers
            "num_key_value_heads",      # Number of KV heads (for GQA/MQA)
            "vocab_size",               # Vocabulary size
            "max_position_embeddings",  # Maximum sequence length
            "intermediate_size",        # MLP intermediate dimension
            "hidden_act",               # Activation function name
            "norm_epsilon",             # Layer normalization epsilon
            "use_bias",                 # Whether to use bias in linear layers
            # Add other architecture-specific params
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """
        Return the NeuronConfig class to use

        CRITICAL: MUST return your custom NeuronConfig class, NOT base NeuronConfig
        Returning base NeuronConfig will cause token generation to fail
        """
        return YourModelNeuronConfig  # ✅ Return custom class, NOT NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from HuggingFace model directory

        Args:
            model_path: Path to HuggingFace model directory
            **kwargs: Additional config overrides
        """
        neuron_config = kwargs.pop("neuron_config", None)
        model_path = os.path.expanduser(model_path)
        config_path = os.path.join(model_path, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        def load_config_fn(config_instance):
            """Callback to load config attributes"""
            for key, value in config_dict.items():
                if not key.startswith("_"):
                    setattr(config_instance, key, value)
            for key, value in kwargs.items():
                setattr(config_instance, key, value)

        # CRITICAL: Create default NeuronConfig if none provided
        # This must happen BEFORE calling __init__ to ensure proper initialization order
        if neuron_config is None:
            neuron_config = cls.get_neuron_config_cls()()

        return cls(neuron_config=neuron_config, load_config=load_config_fn)
```

**Critical Points:**

1. **Initialization Order**: `neuron_config` → `load_config()` → `add_derived_config()`
2. **All Framework Attributes Must Be Set**: Missing any will cause undefined behavior
3. **num_cores_per_group**: Required for attention computation distribution across cores
4. **get_neuron_config_cls() Return Value**: Must return YOUR custom class, not base `NeuronConfig`

**Checklist:**

- [ ] `add_derived_config()` sets `num_cores_per_group = 1`
- [ ] `add_derived_config()` calculates `head_dim` if missing
- [ ] `add_derived_config()` sets ALL 4 framework attributes
- [ ] `get_required_attributes()` includes all architecture-specific params
- [ ] `get_neuron_config_cls()` returns YOUR custom class (not `NeuronConfig`)
- [ ] `from_pretrained()` creates `neuron_config` BEFORE calling `__init__`

---

### Step 3: Create Attention Class

```python
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding


class NeuronYourModelAttention(NeuronAttentionBase):
    """YourModel attention implementation for NeuronX"""

    def __init__(self, config: YourModelInferenceConfig):
        """
        Initialize attention layer

        IMPORTANT: NO layer_idx parameter unless the pattern explicitly requires it
        (most models don't use it)
        """
        # Create rotary position embedding
        rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, 'rope_theta', 10000.0),
        )

        # Initialize base attention with ALL required parameters
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,

            # ✅ CRITICAL: Must pass num_cores_per_group
            # Missing this can cause incorrect tensor shapes during distributed execution
            num_cores_per_group=config.num_cores_per_group,

            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            sliding_window=getattr(config, 'sliding_window', None),
        )
```

**Critical Points:**

1. **num_cores_per_group**: Framework needs this to partition attention computation across cores
2. **No layer_idx**: Most models don't need this (check your pattern analysis)
3. **No custom forward()**: Use the base class implementation unless absolutely necessary

**Warning About GQA:**

- **Ignorable Warning**: `TP degree (X) and KV heads (Y) are not divisible. Overriding attention sharding strategy to GQA.CONVERT_TO_MHA!`
- **What it means**: When tensor parallelism degree doesn't divide evenly into KV heads, framework converts GQA to MHA
- **Action**: Ignore - this is expected behavior, not an error

**Checklist:**

- [ ] Inherits from `NeuronAttentionBase`
- [ ] `__init__` takes only `config` (no `layer_idx` unless pattern requires)
- [ ] Creates `RotaryEmbedding` with correct parameters (rope_theta)
- [ ] Passes ALL required parameters to `super().__init__`:
  - [ ] `config`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`
  - [ ] `head_dim`, `rotary_emb`
  - [ ] **`num_cores_per_group`** ✅ CRITICAL
  - [ ] `qkv_bias`, `o_bias`
  - [ ] `sliding_window` (if applicable)
- [ ] No custom `forward()` method (uses base class)

---

### Step 4: Create MLP Class

**Choose the correct architecture for your model by checking HuggingFace implementation:**

#### Option A: Standard MLP (GPT-2, GPT-Neo, some code models)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear
from typing import Tuple


class NeuronYourModelMLP(nn.Module):
    """Standard MLP with single activation"""

    def __init__(self, config: YourModelInferenceConfig):
        super().__init__()
        self.config = config

        # Input projection (hidden_size -> intermediate_size)
        self.c_fc = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        # Activation function (verify against HuggingFace implementation)
        if config.hidden_act == "gelu":
            self.act = F.gelu
        elif config.hidden_act == "gelu_pytorch_tanh":
            self.act = lambda x: F.gelu(x, approximate="tanh")
        elif config.hidden_act == "silu":
            self.act = F.silu
        elif config.hidden_act == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

        # Output projection (intermediate_size -> hidden_size)
        self.c_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for standard FFN

        Returns:
            Tuple of (output_tensor, None) - None for framework compatibility
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)

        # ✅ CRITICAL: Return tuple for framework compatibility
        # Standard MLPs return (output, None)
        return hidden_states, None
```

#### Option B: SwiGLU MLP (Llama, Mistral, Mixtral)

```python
class NeuronYourModelMLP(nn.Module):
    """SwiGLU MLP architecture"""

    def __init__(self, config: YourModelInferenceConfig):
        super().__init__()

        # Combined gate and up projection
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            2 * config.intermediate_size,  # Note: 2x for gate and up
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        # Down projection
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwiGLU

        Returns:
            Single tensor (not a tuple) for SwiGLU architecture
        """
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = F.silu(gate) * up
        hidden_states = self.down_proj(hidden_states)

        # ✅ SwiGLU returns single tensor, not tuple
        return hidden_states
```

**Common Mistake to Avoid:**

```python
# ❌ WRONG: Changing return type based on one model without understanding
# Saw Phi3 return a single tensor, so I changed mine
def forward(self, x):
    return output  # Don't change from tuple to single without checking architecture

# ✅ CORRECT: Follow the pattern for YOUR architecture type
# Standard FFN: return tuple (output, None)
# SwiGLU: return single tensor
```

**Checklist:**

- [ ] Structure matches your model's architecture (FFN vs SwiGLU)
- [ ] Uses `ColumnParallelLinear` for input/gate projection
- [ ] Uses `RowParallelLinear` for output projection
- [ ] Activation function matches HuggingFace exactly
- [ ] Return type matches architecture:
  - [ ] Standard MLP: tuple `(output, None)`
  - [ ] SwiGLU MLP: single tensor
- [ ] `dtype` passed to all parallel layers

---

### Step 5: Create Decoder Layer

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple


class NeuronYourModelDecoderLayer(nn.Module):
    """YourModel decoder layer implementation for NeuronX"""

    def __init__(self, config: YourModelInferenceConfig):
        """
        Initialize decoder layer

        IMPORTANT: NO layer_idx parameter unless pattern requires it
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-attention (no layer_idx passed)
        self.self_attn = NeuronYourModelAttention(config)

        # MLP
        self.mlp = NeuronYourModelMLP(config)

        # Layer normalization
        # Check HuggingFace: Some models use RMSNorm, others use LayerNorm
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_epsilon,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_epsilon,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,  # ✅ IMPORTANT: Capture extra framework arguments
    ) -> Tuple:
        """
        Forward pass for decoder layer

        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
        """
        # Self-attention with pre-normalization
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Attention returns 4 values: (output, kv_cache, cos, sin)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        # MLP with pre-normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # ✅ Handle MLP return based on architecture type
        mlp_output = self.mlp(hidden_states)
        if isinstance(mlp_output, tuple):
            # Standard FFN returns (output, None)
            hidden_states = mlp_output[0]
        else:
            # SwiGLU returns single tensor
            hidden_states = mlp_output

        # Residual connection
        hidden_states = residual + hidden_states

        # Return 5-tuple expected by framework
        # (hidden_states, kv_cache, cos, sin, attention_weights)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)
```

**Common Mistake to Avoid:**

```python
# ❌ WRONG: Adding unnecessary complexity
class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):  # Unnecessary layer_idx
        self.layer_idx = layer_idx
        self.self_attn = Attention(config, layer_idx=layer_idx)

# ✅ CORRECT: Keep it simple if pattern doesn't require it
class DecoderLayer(nn.Module):
    def __init__(self, config):  # No layer_idx
        self.self_attn = Attention(config)
```

**Checklist:**

- [ ] `__init__` takes only `config` (no `layer_idx` unless pattern requires)
- [ ] Creates attention without passing `layer_idx`
- [ ] `forward()` has `**kwargs` to capture framework arguments
- [ ] Attention unpacking uses 4-tuple: `h, kv, cos, sin = attn(...)`
- [ ] MLP handling matches return type (flexible with `isinstance()` check)
- [ ] Returns 5-tuple: `(hidden_states, kv, cos, sin, None)`
- [ ] Uses correct normalization (LayerNorm vs RMSNorm)

---

### Step 6: Create Base Model

```python
import torch.nn as nn
from neuronx_distributed.parallel_layers import ParallelEmbedding, ColumnParallelLinear
from neuronx_distributed_inference.models.model_base import NeuronBaseModel


class NeuronYourModel(NeuronBaseModel):
    """
    YourModel base model for NeuronX

    IMPORTANT: Inherits from NeuronBaseModel, NOT NeuronBaseForCausalLM
    The CausalLM wrapper comes later
    """

    def setup_attr_for_model(self, config: YourModelInferenceConfig):
        """
        Setup attributes required by the framework

        Called BEFORE init_model() to set up instance attributes
        """
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.sliding_window = getattr(config, "sliding_window", None)

    def init_model(self, config: YourModelInferenceConfig):
        """
        Initialize model components

        Called AFTER setup_attr_for_model() to create layers
        """
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled
        )

        # Decoder layers
        # ✅ CRITICAL: Create layers WITHOUT layer_idx unless pattern requires it
        self.layers = nn.ModuleList(
            [NeuronYourModelDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer normalization
        # Check HuggingFace: LayerNorm or RMSNorm?
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_epsilon,
        )

        # Language modeling head
        # ✅ CRITICAL: lm_head belongs HERE in base model, not in CausalLM wrapper
        # Check HuggingFace checkpoint to verify if bias exists (usually False)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,  # Most models don't have bias in lm_head - verify!
            dtype=config.neuron_config.torch_dtype,
            pad=True,
            gather_output=not self.on_device_sampling,
        )
```

**Critical Points:**

1. **Two Init Methods Required**:
   - `setup_attr_for_model()`: Called BEFORE layer creation
   - `init_model()`: Called AFTER to create actual layers

2. **No forward() Method (General Rule)**:
   - **DO NOT override `forward()`** in NeuronBaseModel for RoPE models (LLaMA, Mistral, Qwen)
   - **EXCEPTION**: Models with learned positional embeddings (GPT-2, BERT, RoBERTa) MUST override forward()
   - See: **OVERRIDING_FORWARD_GUIDANCE.md** for complete decision tree
   - Attempting to override incorrectly causes: `TypeError: forward() takes X arguments but Y were given`

3. **lm_head Placement**:
   - Must be in base model's `init_model()`, NOT in CausalLM wrapper
   - Framework expects to find it here: `AttributeError: 'Model' object has no attribute 'lm_head'`

4. **Layer Creation**:
   - No `layer_idx` passed unless pattern requires
   - Simple list comprehension: `[Layer(config) for _ in range(n)]`

**Common Mistakes to Avoid:**

```python
# ❌ WRONG: Overriding forward()
def forward(self, input_ids, attention_mask=None, ...):
    # Custom implementation
    pass
# Results in: TypeError: forward() takes X arguments but Y were given

# ❌ WRONG: lm_head in CausalLM wrapper instead of base model
# Results in: AttributeError: 'Model' object has no attribute 'lm_head'

# ❌ WRONG: Using layer_idx when pattern doesn't require it
self.layers = nn.ModuleList(
    [DecoderLayer(config, layer_idx=i) for i in range(n)]
)

# ✅ CORRECT: Let base class handle forward, keep layers simple
# No forward() method
self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(n)])
```

**Checklist:**

- [ ] Inherits from `NeuronBaseModel`
- [ ] Has `setup_attr_for_model()` method
- [ ] Has `init_model()` method
- [ ] NO custom `forward()` method
- [ ] Layers created without `layer_idx` (unless pattern requires)
- [ ] `lm_head` bias matches checkpoint (verify with HuggingFace - usually `False`)
- [ ] `embed_tokens` and `lm_head` in same class (base model)
- [ ] Correct normalization type (LayerNorm vs RMSNorm)

---

### Step 7: Create CausalLM Wrapper

```python
import torch
from typing import Type
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM


class NeuronYourModelForCausalLM(NeuronBaseForCausalLM):
    """
    YourModel Causal Language Model wrapper for NeuronX

    This is the top-level class that wraps the base model
    """

    # Reference to the actual model class
    _model_cls = NeuronYourModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """
        Load HuggingFace model for weight extraction

        CRITICAL: Loading full model with from_pretrained() can cause meta tensor errors
        during tie_weights(). Load state dict directly instead.

        Args:
            model_path: Path to HuggingFace model directory or model ID
            **kwargs: Additional arguments (usually ignored)
        """
        import torch
        import os

        model_path = os.path.expanduser(model_path)

        # Handle HuggingFace model IDs (e.g., "facebook/opt-1.3b")
        # Framework may pass model ID instead of local path
        if not os.path.exists(model_path):
            possible_paths = [
                f"agent_artifacts/data/{model_path.split('/')[-1]}",
                f"/path/to/downloads/{model_path.split('/')[-1]}",
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    model_path = p
                    break

        # Try pytorch_model.bin first (most common)
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")

            # Create dummy model wrapper - framework only needs state_dict() method
            class DummyModel:
                def __init__(self, sd):
                    self._state_dict = sd
                def state_dict(self):
                    return self._state_dict

            return DummyModel(state_dict)

        # Try safetensors if available
        try:
            from safetensors import safe_open
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = {}
                with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)

                class DummyModel:
                    def __init__(self, sd):
                        self._state_dict = sd
                    def state_dict(self):
                        return self._state_dict

                return DummyModel(state_dict)
        except ImportError:
            pass

        # Last resort: load full model (may cause meta tensor errors)
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to Neuron format

        CRITICAL: Must add rank utilities for tensor parallelism
        CRITICAL: Handle multiple key prefix patterns (model.*, decoder.*, etc.)

        Args:
            state_dict: HuggingFace model state dict
            config: InferenceConfig instance

        Returns:
            Modified state dict with rank utilities added
        """
        neuron_config = config.neuron_config

        # Add rank utilities for vocabulary parallelism
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )

        # Add rank utilities for attention layers (tensor parallelism)
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Add rank utilities for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        # Convert dtypes if needed
        target_dtype = neuron_config.torch_dtype
        for key, value in state_dict.items():
            if value.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                if value.dtype != target_dtype:
                    state_dict[key] = value.to(target_dtype)

        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Update state dict for tied embeddings and lm_head weights

        CRITICAL: Many models tie embed_tokens and lm_head weights
        HuggingFace only saves one copy, but Neuron expects both keys

        Args:
            state_dict: State dict to modify in-place
        """
        # Check if model uses tied weights (common in many transformer models)
        # If lm_head.weight is missing but embed_tokens.weight exists, they're tied
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            # Clone (not reference) to avoid safetensors errors
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls) -> Type[InferenceConfig]:
        """
        Return the configuration class for this model

        Returns:
            YourModelInferenceConfig class (not instance)
        """
        return YourModelInferenceConfig
```

**Critical Points:**

1. **Rank Utilities**: Framework needs these for tensor parallelism distribution
2. **Tied Weights**: If embeddings are tied, must manually add both keys to state dict
3. **Weight Conversion**: Handle dtype conversion to target precision

**Why Tied Weights Need Special Handling:**

```python
# Problem: Models with tied weights save only one copy
# HuggingFace checkpoint: {"embed_tokens.weight": tensor(...)}
# Neuron expects: {"embed_tokens.weight": tensor(...), "lm_head.weight": tensor(...)}

# Initial attempt (doesn't work):
def init_model(self, config):
    self.embed_tokens = ParallelEmbedding(...)
    self.lm_head = ColumnParallelLinear(...)
    self.lm_head.weight = self.embed_tokens.weight  # Python reference
    # Problem: PyTorch state_dict still only saves one key

# Solution: Manually add to state dict after loading
@staticmethod
def update_state_dict_for_tied_weights(state_dict):
    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
```

**Checklist:**

- [ ] Inherits from `NeuronBaseForCausalLM`
- [ ] Sets `_model_cls` to your base model class
- [ ] Has `load_hf_model()` static method
- [ ] Has `convert_hf_to_neuron_state_dict()` static method
- [ ] Adds rank utilities for vocabulary parallelism (if applicable)
- [ ] Adds rank utilities for attention layers (tensor parallelism)
- [ ] Has `update_state_dict_for_tied_weights()` if model ties weights
- [ ] Has `get_config_cls()` class method
- [ ] Handles dtype conversion in `convert_hf_to_neuron_state_dict()`

---

## Common Issues and Solutions

This section documents specific issues encountered during real ports and their solutions.

### Issue 0: Token Generation Dimension Mismatch ⭐ CRITICAL

#### Issue 0.1: Attention Output Unpacking Pattern

**Error**:

```
RuntimeError: torch_xla/csrc/tensor_methods.cpp:216 : Check failed: t->size(dim) == expected_size (1 vs. 128)
Expected tensor to have size 128 at dimension 1, but got size 1 for argument #2 'batch2' (while checking arguments for bmm)
```

**Symptoms**:

- Context encoding compiles successfully ✅
- Token generation fails during HLO tracing ❌
- Error occurs in `attention_base.py` in `perform_prefill` or `torch.matmul`
- Dimension mismatch between expected and actual tensor sizes

**Root Cause**: Using attribute access on attention output instead of tuple unpacking.

**Solution**:

```python
# ❌ WRONG - Causes dimension mismatch
attn_output = self.self_attn(...)
hidden_states = attn_output.hidden_states
present_key_value = attn_output.present_key_value

# ✅ CORRECT - Required pattern
hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
    hidden_states=hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_value=past_key_value,
    **kwargs,
)
```

**Learning**: NeuronAttentionBaseOutput supports both patterns, but only tuple unpacking works correctly. This is the #1 cause of "context encoding works, token generation fails" issues.

---

#### Issue 0.2: Meta Tensor Error During Weight Loading

**Error**:

```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

**Symptoms**:

- HLO generation succeeds ✅
- Compilation succeeds ✅
- Weight loading fails ❌
- Error occurs in `tie_weights()` or `register_parameter()`

**Root Cause**: Using `from_pretrained()` in `load_hf_model()` causes meta tensor creation during weight tying, even with `init_on_device(cpu)` context.

**Solution**: Load state dict directly instead of full model:

```python
@staticmethod
def load_hf_model(model_path, **kwargs):
    import torch
    import os

    model_path = os.path.expanduser(model_path)

    # Load state dict directly from pytorch_model.bin
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")

        # Create dummy wrapper - framework only needs state_dict() method
        class DummyModel:
            def __init__(self, sd):
                self._state_dict = sd
            def state_dict(self):
                return self._state_dict

        return DummyModel(state_dict)

    # Fallback to from_pretrained (may fail)
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False)
```

**Learning**: Direct state dict loading avoids model initialization and tie_weights() that create meta tensors. Framework only needs the state_dict() method.

---

### Issue 1: Configuration Class Problems

#### Issue 1.1: Missing from_pretrained() Method

**Error**: `AttributeError: type object 'ModelInferenceConfig' has no attribute 'from_pretrained'`

**Root Cause**: The inference framework expects configuration classes to have a `from_pretrained()` method to load from model directories.

**Solution**: Implement `from_pretrained()` as shown in Step 2.

**Learning**: Always implement `from_pretrained()` following the pattern in reference implementations (Llama, Phi3, etc.)

---

#### Issue 1.2: Custom NeuronConfig Not Being Used

**Error**: Framework used base `NeuronConfig` instead of model-specific config, causing token generation failures.

**Root Cause**: `get_neuron_config_cls()` returned base class instead of custom class.

**User Feedback**: "the compiler issue is due to a missing super init, or an incorrect overload"

**Solution**:

```python
# ❌ WRONG
@classmethod
def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
    return NeuronConfig  # Returns base class

# ✅ CORRECT
class YourModelNeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = YourAttention

@classmethod
def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
    return YourModelNeuronConfig  # Returns custom class
```

**Learning**: Always create a custom `NeuronConfig` subclass with model-specific attention class.

---

#### Issue 1.3: Missing Attributes During Initialization

**Error**: `AttributeError: 'ModelInferenceConfig' object has no attribute 'hidden_size'`

**Root Cause**: The `load_config` callback is executed BEFORE `add_derived_config()`, but `neuron_config` was None, causing initialization order issues.

**Solution**: Ensure `neuron_config` is created BEFORE calling the parent `__init__`:

```python
if neuron_config is None:
    neuron_config = cls.get_neuron_config_cls()()
config = cls(neuron_config=neuron_config, load_config=load_config_fn)
```

**Learning**: The initialization order is critical: `neuron_config` → `load_config()` → `add_derived_config()`

---

### Issue 2: Base Class Selection

#### Issue 2.1: Parallel Group Not Initialized

**Error**: `AssertionError: intra_layer_model parallel group is not initialized`

**Root Cause**: Extended `NeuronBaseModel` directly instead of `NeuronBaseForCausalLM`.

**Solution**: Use the correct base class hierarchy:

```python
# ❌ WRONG: Using NeuronBaseModel as top-level class
class CustomModel(NeuronBaseModel):
    pass
model = CustomModel(config)  # Will fail with parallel group error

# ✅ CORRECT: Use two-level hierarchy
class CustomForCausalLM(NeuronBaseForCausalLM):
    _model_cls = CustomModel  # Reference to the actual model

class CustomModel(NeuronBaseModel):
    pass

model = CustomForCausalLM(config)  # Correct initialization
```

**Learning**: For causal language models, ALWAYS use `NeuronBaseForCausalLM` as the outer wrapper, not `NeuronBaseModel` directly.

---

### Issue 3: Model Initialization

#### Issue 3.1: Missing Required Methods

**Error**: `NotImplementedError: setup_attr_for_model() is not implemented`

**Root Cause**: `NeuronBaseModel` requires two initialization methods that must be implemented.

**Solution**: Implement both required methods as shown in Step 6:

- `setup_attr_for_model()` - Called BEFORE `init_model()`
- `init_model()` - Called AFTER `setup_attr_for_model()`

**Learning**: Always implement both methods following reference implementations.

---

#### Issue 3.2: Incorrect Forward Method Override

**Error**: `TypeError: Model.forward() takes from 2 to 7 positional arguments but 8 were given`

**Root Cause**: Attempted to override `forward()` with incorrect signature.

**Solution**: **DO NOT override forward()** for RoPE models - let `NeuronBaseModel` handle it.

**EXCEPTION**: If your model uses learned positional embeddings (GPT-2, BERT, RoBERTa), you MUST override forward() with complete signature. See: **OVERRIDING_FORWARD_GUIDANCE.md**

```python
# ❌ WRONG
def forward(self, input_ids, attention_mask=None, position_ids=None, ...):
    # Custom implementation
    pass

# ✅ CORRECT
# No forward() method - base class handles everything
```

**Learning**: The base class `NeuronBaseModel` provides the complete forward pass orchestration. Only implement `setup_attr_for_model()` and `init_model()`.

---

#### Issue 3.3: Missing lm_head in Model

**Error**: `AttributeError: 'CustomModel' object has no attribute 'lm_head'`

**Root Cause**: The `lm_head` must be in the base model, not just in the ForCausalLM wrapper.

**Solution**: Add `lm_head` to `init_model()` as shown in Step 6.

**Learning**: The `lm_head` belongs in the base model's `init_model()`, not in the `ForCausalLM` wrapper.

---

### Issue 4: Weight Tying

#### Issue 4.1: Tied Weights Not Saved in Checkpoint

**Error**: `RuntimeError: Missing weight tensor with key lm_head.weight`

**Root Cause**: Models with tied embeddings (where `embed_tokens.weight` and `lm_head.weight` share the same tensor) only save one copy during compilation, but inference expects both keys.

**Initial Attempt (Doesn't Work)**:

```python
self.lm_head.weight = self.embed_tokens.weight  # Creates Python reference
# Problem: PyTorch state_dict only saves one key for tied weights
```

**Solution**: Manually add the tied weight in `update_state_dict_for_tied_weights()`:

```python
@staticmethod
def update_state_dict_for_tied_weights(state_dict):
    if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
        # Clone (not reference) to avoid safetensors error
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
```

**Learning**: For models with tied weights, you must manually add both keys to the state dict. The framework expects both keys even though they reference the same data.

---

### Issue 5: Tokenizer and Files

#### Issue 5.1: Missing Tokenizer Files

**Error**: `OSError: Can't load tokenizer for '/path/to/model'`

**Root Cause**: Downloaded only model weights, not tokenizer files.

**Solution**: Download complete model directory:

```bash
huggingface-cli download model/name --local-dir path/to/model
```

This includes: `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `special_tokens_map.json`, `config.json`

**Learning**: Always download the complete model directory, not just weights.

---

### Issue 6: Attention Implementation

#### Issue 6.1: Grouped Query Attention Warning (IGNORABLE)

**Warning**: `TP degree (1) and KV heads (2) are not divisible. Overriding attention sharding strategy to GQA.CONVERT_TO_MHA!`

**Root Cause**: When tensor parallelism degree doesn't divide evenly into KV heads, framework must convert GQA to MHA.

**Action**: **IGNORE - THIS IS A WARNING, NOT AN ERROR**. The framework handles this automatically and supports various GQA ratios (including 12:1) without issues.

---

#### Issue 6.2: Sliding Window Attention Safety

**Issue**: Sliding window attention can cause runtime errors with certain sequence lengths.

**Solution**: Disable sliding window for initial testing:

```python
def add_derived_config(self):
    # Disable sliding window for safety during initial testing
    if hasattr(self, 'sliding_window') and self.sliding_window is not None:
        self.sliding_window = None  # Can re-enable for production
```

**Learning**: Test without sliding window first, then enable for production if needed.

---

### Issue 7: Layer Normalization

#### Issue 7.1: RMSNorm vs LayerNorm Confusion

**Issue**: Imported `CustomRMSNorm` but model uses standard `LayerNorm`.

**Solution**: Check the original model implementation:

```python
# Some models use RMSNorm (Llama, Mistral)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# Others use standard LayerNorm (GPT-2, BERT)
self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
```

**Learning**: Always verify which normalization the original model uses. Don't assume.

---

### Issue 8: MLP Architecture

#### Issue 8.1: FFN vs SwiGLU Confusion

**Issue**: Different models use different MLP architectures, leading to incorrect implementations.

**Solution**: Verify architecture in HuggingFace source and implement correctly (see Step 4 for both patterns).

**Common Architectures**:

- **Standard FFN**: GPT-2, BERT, some code models → Returns tuple `(output, None)`
- **SwiGLU**: Llama, Mistral, Mixtral → Returns single tensor
- **GeGLU**: Some T5 variants → Similar to SwiGLU

**Common Mistake**:

```python
# ❌ WRONG: Saw one model return single tensor, changed mine
def forward(self, x):
    return output  # Changed from tuple without understanding architecture

# ✅ CORRECT: Match YOUR model's architecture
# Standard FFN: return (output, None)
# SwiGLU: return output
```

**Learning**: Check original implementation - FFN architecture varies significantly between models.

---

### Issue 9: Model Validation

#### Issue 9.1: Testing with Wrong Prompt Type

**Issue**: Tested code model with Q&A prompt ("What is the capital of France?").

**Result**: Model generated code, not answers (correct behavior for code model, but looks wrong).

**Solution**: Test with appropriate prompts for the model type:

```python
# For code models:
prompts = [
    "def fibonacci(n):",
    "class BinaryTree:",
    "# Function to sort an array\ndef"
]

# For Q&A models:
prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
]

# For chat models:
prompts = [
    "<|user|>What is Python?<|assistant|>",
]
```

**Learning**: Always test with prompts appropriate for the model's training objective.

---

## Debugging Guide

### Debugging Decision Tree

```
Port fails?
├─ Context encoding fails?
│  ├─ Check InferenceConfig.get_required_attributes()
│  │  └─ All model attributes listed? → Add missing ones
│  ├─ Check InferenceConfig.add_derived_config()
│  │  └─ Sets num_cores_per_group? → Add it
│  ├─ Check parallel layers creation
│  │  └─ ColumnParallelLinear/RowParallelLinear correct? → Fix parameters
│  └─ Check attention parameters
│     └─ All required params passed to super().__init__? → Add missing ones
│
└─ Token generation fails?
   ├─ ✅ Custom NeuronConfig class exists?
   │  └─ NO → CREATE IT (see Pattern 1)
   │     └─ YES → attn_cls set in __init__?
   │        └─ NO → SET IT: self.attn_cls = YourAttention
   │           └─ YES → get_neuron_config_cls() returns it?
   │              └─ NO → FIX: return YourModelNeuronConfig
   │                 └─ YES → num_cores_per_group passed?
   │                    └─ NO → PASS IT in attention super().__init__
   │                       └─ YES → add_derived_config() complete?
   │                          └─ NO → ADD all framework attributes
   │                             └─ YES → Compare line-by-line with working model
│
└─ Weight loading fails?
   ├─ Check lm_head bias matches checkpoint
   ├─ Check tied weights handled
   └─ Check weight name mappings
```

### Step 1: Context Encoding Fails

**Symptoms**:

- Compilation crashes during context encoding
- HLO conversion errors
- Shape mismatch errors in early compilation

**Debug Checklist**:

1. **Check Required Attributes**:

   ```python
   # Verify all attributes in get_required_attributes() exist in config.json
   config = YourModelInferenceConfig.from_pretrained(model_path)
   for attr in config.get_required_attributes():
       assert hasattr(config, attr), f"Missing attribute: {attr}"
   ```

2. **Check add_derived_config()**:

   ```python
   # Verify num_cores_per_group is set
   assert config.num_cores_per_group == 1
   assert hasattr(config, 'head_dim')
   ```

3. **Check Attention Parameters**:
   ```python
   # All these must be passed to NeuronAttentionBase.__init__
   required_params = [
       'config', 'hidden_size', 'num_attention_heads',
       'num_key_value_heads', 'head_dim', 'rotary_emb',
       'num_cores_per_group'  # CRITICAL
   ]
   ```

**Debug Commands**:

```bash
# Check compiler logs
ls agent_artifacts/data/neff_output/context_encoding_model/*/log-neuron-cc.txt
cat agent_artifacts/data/neff_output/context_encoding_model/*/log-neuron-cc.txt | tail -100

# Clear cache if seeing JSON or FileNotFoundError
rm -rf /var/tmp/neuron-compile-cache/*
```

---

### Step 2: Token Generation Fails

**This is the critical failure mode.** If context encoding works but token generation fails, it's almost always one of the configuration patterns.

**Typical Symptom**:

```
RuntimeError: Expected tensor to have size X at dimension 1, but got size 1
Location: attention_base.py:736 in perform_prefill()
```

**Root Cause**: The framework is taking the wrong code path (context_encode instead of token_gen) because `is_token_gen = False`.

**Debug Checklist** (in order of likelihood):

1. **✅ Do you have a custom NeuronConfig class?**

   ```python
   # Check if this class exists in your code
   class YourModelNeuronConfig(NeuronConfig):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.attn_cls = YourAttention  # MUST SET THIS
   ```

   **If missing**: This is your problem. Create it (see Pattern 1).

2. **✅ Does get_neuron_config_cls() return your custom class?**

   ```python
   # Check what it returns
   cls = YourModelInferenceConfig.get_neuron_config_cls()
   print(cls)  # Should print: YourModelNeuronConfig, NOT NeuronConfig
   ```

   **If wrong**: Fix it to return `YourModelNeuronConfig`.

3. **✅ Does attention pass num_cores_per_group?**

   ```python
   # In your attention __init__, verify this line exists
   super().__init__(
       # ... other params ...
       num_cores_per_group=config.num_cores_per_group,  # MUST PASS THIS
   )
   ```

   **If missing**: Add it.

4. **✅ Does add_derived_config() set all framework attributes?**
   ```python
   def add_derived_config(self):
       self.num_cores_per_group = 1
       if not hasattr(self, 'head_dim'):
           self.head_dim = self.hidden_size // self.num_attention_heads
       # ALL 4 of these must be set:
       if not hasattr(self, 'output_attentions'):
           self.output_attentions = False
       if not hasattr(self, 'output_hidden_states'):
           self.output_hidden_states = False
       if not hasattr(self, 'use_return_dict'):
           self.use_return_dict = True
       if not hasattr(self, 'use_cache'):
           self.use_cache = True
   ```

**Debug Commands**:

```bash
# Check token generation logs
ls agent_artifacts/data/neff_output/token_generation_model/*/log-neuron-cc.txt
cat agent_artifacts/data/neff_output/token_generation_model/*/log-neuron-cc.txt | tail -100
```

---

### Step 3: Weight Loading Fails

**Symptoms**:

- Missing parameter errors
- Shape mismatch during weight loading
- Tied weights errors

**Debug Checklist**:

1. **Check lm_head bias**:

   ```python
   from safetensors import safe_open

   with safe_open("model.safetensors", framework="pt") as f:
       has_lm_head_bias = "lm_head.bias" in f.keys()
       print(f"lm_head has bias: {has_lm_head_bias}")

   # Update your code accordingly
   self.lm_head = ColumnParallelLinear(
       ...,
       bias=has_lm_head_bias,  # Match checkpoint
   )
   ```

2. **Check tied weights**:

   ```python
   with safe_open("model.safetensors", framework="pt") as f:
       has_lm_head = "lm_head.weight" in f.keys()
       has_embed = "embed_tokens.weight" in f.keys()

       if has_embed and not has_lm_head:
           print("Model uses tied weights - implement update_state_dict_for_tied_weights()")
   ```

3. **Check weight names**:
   ```bash
   # List all weight names in checkpoint
   python -c "
   from safetensors import safe_open
   with safe_open('model.safetensors', framework='pt') as f:
       for key in sorted(f.keys()):
           print(key)
   "
   ```

---

### Common Compiler Issues

#### Issue: JSON Parse Error

**Error**: `[NLA001] Unhandled exception with message: [json.exception.parse_error.101] parse error at line 1, column 1: attempting to parse an empty input`

**Solution**: Delete compiler cache and retry

```bash
rm -rf /var/tmp/neuron-compile-cache/*
# Then rerun compilation
```

#### Issue: File Not Found in NEFF Output

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'agent_artifacts/neff_output/token_generation_model/_tp0_bk0'`

**Solution**: Delete compiler cache and retry

```bash
rm -rf /var/tmp/neuron-compile-cache/*
# Then rerun compilation
```

---

## Verification and Testing

### Phase 1: Single Layer Compilation Test

**Purpose**: Quickly verify basic structure is correct before full compilation.

```python
# In compile script
config = YourModelInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
    reduce_layers=1,  # Only compile 1 layer for testing
)

# Expected time: 20-60 seconds
# Expected result: Both context encoding and token generation succeed
```

**Success Criteria**:

- ✅ Context encoding compiles
- ✅ Token generation compiles
- ✅ Total time: Under 2 minutes

**If this fails**, fix the issues before proceeding. No point compiling the full model if 1 layer doesn't work.

---

### Phase 2: Full Model Compilation

```python
# In compile script
config = YourModelInferenceConfig.from_pretrained(
    model_path,
    neuron_config=neuron_config,
    reduce_layers=None,  # All layers
)

# Expected time: 3-10 minutes depending on model size
# Expected result: Both context encoding and token generation succeed
```

**Expected Output**:

```
✅ Context encoding: SUCCESS
✅ Token generation: SUCCESS
✅ Total time: ~3-5 minutes (for small models)
✅ Total time: ~5-10 minutes (for medium models)
```

---

### Phase 3: Inference Testing

```python
# test_inference.py
from neuronx_distributed_inference.utils.run_inference import run_inference_with_classes

# Test with prompts appropriate for model type
prompts = [
    "def fibonacci(n):",  # For code models
    "What is the capital of France?",  # For Q&A models
]

for prompt in prompts:
    result = run_inference_with_classes(
        model_class_path="path.to.YourModelForCausalLM",
        config_class_path="path.to.YourModelInferenceConfig",
        model_path=model_path,
        compiled_path=compiled_path,
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
    )

    print(f"Prompt: {prompt}")
    print(f"Generated: {result['generated_text']}")
    print(f"Tokens/sec: {result['tokens_per_second']}")
    print()
```

**Quality Indicators**:

- ✅ Text is coherent and relevant to prompt
- ✅ Syntax is correct (for code generation)
- ✅ No repetitive patterns (same phrase repeated)
- ✅ Follows the style of the prompt
- ✅ Reasonable performance (varies by model size)

**Red Flags**:

- ❌ Generates gibberish or random characters
- ❌ Repeats the same token endlessly
- ❌ Crashes or throws errors
- ❌ Generates text unrelated to prompt

---

## Complete Checklists

### Pre-Implementation Checklist

**Before writing any code:**

- [ ] Identified 3-4 working models with similar architecture
- [ ] Filled component analysis checklist for each model
- [ ] Identified ALL required patterns (things all models do)
- [ ] Identified architecture-specific patterns (things some models do)
- [ ] Confirmed pattern for layer_idx usage (or non-usage)
- [ ] Confirmed MLP architecture type (Standard vs SwiGLU)
- [ ] Confirmed normalization type (LayerNorm vs RMSNorm)

---

### Implementation Checklist

**Configuration:**

- [ ] Custom NeuronConfig class created
- [ ] Custom NeuronConfig sets `self.attn_cls` in `__init__`
- [ ] InferenceConfig.get_neuron_config_cls() returns custom class (not base)
- [ ] InferenceConfig.add_derived_config() sets `num_cores_per_group = 1`
- [ ] InferenceConfig.add_derived_config() calculates `head_dim` if missing
- [ ] InferenceConfig.add_derived_config() sets all 4 framework attributes
- [ ] InferenceConfig.get_required_attributes() includes all model-specific attributes
- [ ] InferenceConfig.from_pretrained() creates neuron_config before **init**

**Attention:**

- [ ] Attention class inherits from `NeuronAttentionBase`
- [ ] Attention `__init__` takes only `config` (no `layer_idx` unless pattern requires)
- [ ] Attention passes `num_cores_per_group=config.num_cores_per_group` to super()
- [ ] Attention passes all required parameters to `super().__init__`
- [ ] NO custom `forward()` method in attention

**MLP:**

- [ ] MLP architecture matches reference model (Standard vs SwiGLU)
- [ ] MLP return type matches pattern (tuple for standard, single for SwiGLU)
- [ ] Decoder layer handles MLP return correctly

**Model Structure:**

- [ ] Base model inherits from `NeuronBaseModel`
- [ ] Has `setup_attr_for_model()` method
- [ ] Has `init_model()` method
- [ ] NO custom `forward()` method in base model
- [ ] Layers created without `layer_idx`: `[Layer(config) for _ in range(n)]`
- [ ] `lm_head` in base model's `init_model()` (not in wrapper)
- [ ] `lm_head` bias matches checkpoint (verify with safetensors)
- [ ] Correct normalization type (LayerNorm vs RMSNorm)

**Wrapper:**

- [ ] ForCausalLM inherits from `NeuronBaseForCausalLM`
- [ ] Sets `_model_cls` correctly
- [ ] Has `load_hf_model()` static method
- [ ] Has `convert_hf_to_neuron_state_dict()` static method
- [ ] Adds rank utilities in `convert_hf_to_neuron_state_dict()`
- [ ] Has `update_state_dict_for_tied_weights()` if model ties weights
- [ ] Has `get_config_cls()` class method

---

### Compilation and Testing Checklist

- [ ] Single layer (reduce_layers=1) compiles successfully
  - [ ] Context encoding succeeds
  - [ ] Token generation succeeds
  - [ ] Time < 2 minutes
- [ ] Full model compiles successfully
  - [ ] Context encoding succeeds
  - [ ] Token generation succeeds
  - [ ] Time: 3-10 minutes depending on size
- [ ] Weights load without errors
- [ ] Inference runs without crashes
- [ ] Generated text is coherent and high-quality
- [ ] Model responds appropriately to different prompt types
- [ ] Performance is acceptable (varies by model size and TP degree)

---

### Success Checklist

**Your port is complete when ALL of these are true:**

- [ ] All implementation checklist items complete
- [ ] All compilation and testing checklist items complete
- [ ] No compiler errors or warnings (ignore GQA conversion warning)
- [ ] Documentation updated with model-specific notes
- [ ] Test prompts appropriate for model type prepared
- [ ] Validation tool passes: `python validate_port_done.py`

---

## Key Lessons Learned

### 1. Patterns are REQUIRED, not optional

- When ALL working models follow a pattern, it's required by the framework
- Don't deviate without strong evidence from multiple reference implementations
- Framework patterns aren't style choices - they're requirements

### 2. Start with research, not coding

- Spend 1-2 hours understanding the pattern first
- Fill in the component checklist completely for 3-4 models
- Then implement exactly following the pattern
- This saves 6-8 hours of debugging later

### 3. Don't assume framework limitations

- If other models with similar architecture work, yours can too
- Configuration issues look like framework bugs
- Compare with working models before assuming limitation
- The framework supports 12:1 GQA ratio fine - failures are configuration issues

### 4. The custom NeuronConfig is CRITICAL

- This is the #1 cause of token generation failure
- Must exist and must set `attn_cls`
- Must be returned by `get_neuron_config_cls()`
- Without this, context encoding may work but token generation will fail

### 5. Trust the pattern

- If all working models do it, do it too
- If no working models do it, don't add it
- Don't add complexity that doesn't exist in the pattern
- `layer_idx` is often unnecessary - verify before adding

### 6. Test incrementally

- Compile 1 layer first (20-60 seconds)
- If that fails, fix before full compilation
- Full compilation wastes time if basic structure is wrong

### 7. Use appropriate test prompts

- Code models: Test with code snippets
- Q&A models: Test with questions
- Chat models: Test with chat format
- Wrong prompt type makes good models look broken

### 8. Framework orchestration is complete

- Don't override `forward()` in NeuronBaseModel for RoPE models (99% of cases)
- Exception: MUST override for learned positional embeddings (GPT-2, BERT, RoBERTa)
- See: **OVERRIDING_FORWARD_GUIDANCE.md** for decision tree
- The base classes handle the full execution flow
- Only implement the required initialization methods

---

## Estimated Timeline

### Following This Guide (Success Path):

1. **Research Phase** (1-2 hours)
   - Find 3-4 similar working models
   - Fill in component checklist for each
   - Identify common patterns
   - Verify critical requirements

2. **Implementation Phase** (2-3 hours)
   - Create custom NeuronConfig (15 min)
   - Create InferenceConfig (30 min)
   - Create Attention (30 min)
   - Create MLP (20 min)
   - Create DecoderLayer (20 min)
   - Create Base Model (30 min)
   - Create CausalLM Wrapper (30 min)

3. **Verification Phase** (1 hour)
   - Compile 1 layer: 20-60 seconds → Success ✅
   - Compile full model: 3-10 minutes → Success ✅
   - Run inference: Success on first try ✅
   - Test with multiple prompts: High quality ✅

**Total: 4-6 hours with success on first compilation**

---

### Not Following This Guide (Failure Path):

1. Jump into implementation: 2 hours
2. Context encoding compiles: ✅
3. Token generation fails: ❌
4. Try fixing attention parameters: 1 hour → Still fails
5. Try different base classes: 1 hour → Still fails
6. Try adding layer_idx everywhere: 2 hours → Still fails
7. Assume 12:1 GQA is framework limitation: 2 hours reading docs
8. Ask for help / get correct implementation
9. Realize mistake: Missing custom NeuronConfig class
10. Add custom NeuronConfig: 15 minutes
11. Recompile: Success ✅

**Total: 10+ hours with frustration and multiple failures**

---

## Conclusion

This comprehensive guide combines systematic porting procedures with real-world learnings from both successful and failed ports. The key insight is that **framework patterns are required, not optional**.

By following this approach:

1. **Research first** (understand the pattern by analyzing 3-4 working models)
2. **Implement exactly** (follow the pattern without deviations)
3. **Verify incrementally** (test 1 layer, then full model)

You can successfully port models on the first try, avoiding the common pitfalls that lead to hours of debugging.

**Remember**: When in doubt, compare with 3-4 working models. If they ALL do something, you must do it too. If NONE of them do something, don't add it.

**The #1 mistake**: Missing or incorrectly configured custom NeuronConfig class. This single issue causes most token generation failures.

**Good luck with your port!**
