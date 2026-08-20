# Novel Patterns for NeuronX Model Porting

This document captures advanced patterns discovered during model porting that are not covered in existing documentation.

## ⚠️ CRITICAL: Always Debug Print Checkpoint Keys First

### The Most Common Source of Weight Loading Errors

**DO NOT assume what keys your `convert_hf_to_neuron_state_dict()` function receives. The actual keys depend on the checkpoint format and may vary.**

### Best Practice: Debug Print First

```python
@staticmethod
def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    # ALWAYS do this first when porting a new model
    print(f"DEBUG: First 10 keys received: {list(state_dict.keys())[:10]}")

    neuron_state_dict = {}

    # Now write conversion logic based on ACTUAL keys you see
    for key, value in state_dict.items():
        new_key = key
        # Add your transformations here based on debug output
        neuron_state_dict[new_key] = value

    return neuron_state_dict
```

### Common Prefix Patterns

Different models use different checkpoint structures:

- `"model.decoder.layers.0.weight"` - Decoder-only models with decoder prefix
- `"model.layers.0.weight"` - Standard models (LLaMA, Mistral)
- `"decoder.layers.0.weight"` - Some variants
- `"layers.0.weight"` - No prefix

**Handle multiple prefix patterns robustly:**

```python
# Remove prefixes - handle both "model.decoder." and "decoder."
new_key = key.replace("model.decoder.", "").replace("decoder.", "")

# Or more explicitly:
if new_key.startswith("model.decoder."):
    new_key = new_key.replace("model.decoder.", "", 1)
elif new_key.startswith("decoder."):
    new_key = new_key.replace("decoder.", "", 1)
```

### Validation Steps

1. **Print keys** - See what you actually receive
2. **Run compilation** - Verify weight loading succeeds
3. **Check for missing keys** - Framework will report any missing weights
4. **Remove debug prints** - After confirming it works

---

## 1. Non-Fused QKV Weight Structure (fused_qkv=False)

### The Second Most Common Weight Loading Error

When `fused_qkv=False`, the framework creates a **wrapper structure** that preshard hooks expect. This is NOT obvious from the model code.

### What the Framework Creates

```python
# When you set fused_qkv=False, the framework creates:
self.qkv_proj = GroupQueryAttention_QKV(...)  # Wrapper object
    self.qkv_proj.q_proj = ColumnParallelLinear(...)
    self.qkv_proj.k_proj = ColumnParallelLinear(...)
    self.qkv_proj.v_proj = ColumnParallelLinear(...)
```

### Weight Key Structure

```python
# ❌ WRONG - What you might expect:
"layers.0.self_attn.q_proj.weight"
"layers.0.self_attn.k_proj.weight"
"layers.0.self_attn.v_proj.weight"

# ✅ CORRECT - What the framework actually expects:
"layers.0.self_attn.qkv_proj.q_proj.weight"
"layers.0.self_attn.qkv_proj.k_proj.weight"
"layers.0.self_attn.qkv_proj.v_proj.weight"
```

Note the extra `qkv_proj.` level!

### The Error You'll See

```
RuntimeError: Missing weight tensor with key: layers.0.self_attn.qkv_proj.q_proj.weight
```

Even though your checkpoint has `layers.0.self_attn.q_proj.weight`.

### Solution: Explicit Per-Layer Restructuring

**Recommended approach (robust and debuggable):**

```python
@staticmethod
def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    neuron_state_dict = {}

    # First pass: copy all keys with basic transformations
    for key, value in state_dict.items():
        new_key = key
        # Remove prefixes, rename projections, etc.
        neuron_state_dict[new_key] = value

    # Second pass: restructure QKV weights per layer
    num_layers = config.num_hidden_layers
    for i in range(num_layers):
        # Check if this layer has separate Q/K/V projections
        if f"layers.{i}.self_attn.q_proj.weight" in neuron_state_dict:
            # Pop original keys
            q_weight = neuron_state_dict.pop(f"layers.{i}.self_attn.q_proj.weight")
            k_weight = neuron_state_dict.pop(f"layers.{i}.self_attn.k_proj.weight")
            v_weight = neuron_state_dict.pop(f"layers.{i}.self_attn.v_proj.weight")

            # Add with qkv_proj intermediate level
            neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.q_proj.weight"] = q_weight
            neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.k_proj.weight"] = k_weight
            neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.v_proj.weight"] = v_weight

            # Handle biases if present
            if f"layers.{i}.self_attn.q_proj.bias" in neuron_state_dict:
                q_bias = neuron_state_dict.pop(f"layers.{i}.self_attn.q_proj.bias")
                k_bias = neuron_state_dict.pop(f"layers.{i}.self_attn.k_proj.bias")
                v_bias = neuron_state_dict.pop(f"layers.{i}.self_attn.v_proj.bias")
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.q_proj.bias"] = q_bias
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.k_proj.bias"] = k_bias
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.v_proj.bias"] = v_bias

    return neuron_state_dict
```

### Common Additional Renaming

Output projection often needs renaming too:

```python
# Many models use "out_proj" but framework expects "o_proj"
new_key = new_key.replace("out_proj", "o_proj")
```

### When This Applies

- Your model has **separate** Q, K, V projection weights (not fused into one matrix)
- You set `fused_qkv=False` in your NeuronConfig

### Validation

```python
# After conversion, check the keys:
attn_keys = [k for k in neuron_state_dict.keys() if 'self_attn' in k and 'layers.0' in k]
print(sorted(attn_keys))

# Should see:
# layers.0.self_attn.o_proj.bias
# layers.0.self_attn.o_proj.weight
# layers.0.self_attn.qkv_proj.k_proj.bias
# layers.0.self_attn.qkv_proj.k_proj.weight
# layers.0.self_attn.qkv_proj.q_proj.bias
# layers.0.self_attn.qkv_proj.q_proj.weight
# layers.0.self_attn.qkv_proj.v_proj.bias
# layers.0.self_attn.qkv_proj.v_proj.weight
```

---

## 2. Learned Positional Embeddings Pattern

### Issue

Models using learned positional embeddings (not RoPE or relative position bias) need to add positional information in the forward pass.

### Background

- **RoPE models** (LLaMA, Mistral, Qwen): Position info added in attention layers
- **Relative position models** (T5): Position bias computed in attention
- **Learned position models** (GPT-2, BERT, RoBERTa): Position embeddings must be added to token embeddings

### Discovery

The base model forward pass calls `self.embed_tokens(input_ids)` but doesn't add positional embeddings. For models with learned positional embeddings, you must add them yourself.

### ⚠️ CRITICAL: This is an EXCEPTION to "Don't Override Forward" Rule

**General Rule**: Do NOT override `forward()` in NeuronBaseModel (see OVERRIDING_FORWARD_GUIDANCE.md)

**EXCEPTION**: Models with learned positional embeddings (GPT-2, BERT, RoBERTa, ALBERT) MUST override `forward()` to add position embeddings.

### How to Identify If You Need This Pattern

Check the HuggingFace model implementation:

```python
# If you see this pattern, you NEED to override forward():
class HFModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_positions, hidden_size)  # ← Learned!

    def forward(self, input_ids):
        token_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)
        hidden_states = token_embeds + position_embeds  # ← Must replicate this!
```

**Models that need this**: GPT-2, BERT, RoBERTa, ALBERT, and similar architectures with learned positional embeddings

**Models that DON'T need this**: LLaMA, Mistral, Qwen (use RoPE), T5 (uses relative position bias)

### Solution: Separate Embeddings + Forward Override

**DO NOT create wrapper classes.** Use separate embeddings and override forward():

```python
def init_model(self, config: InferenceConfig):
    # Token embeddings
    self.embed_tokens = ParallelEmbedding(
        config.vocab_size,
        config.hidden_size,
        dtype=config.neuron_config.torch_dtype,
    )

    # Positional embeddings (separate, not wrapped)
    self.embed_positions = ParallelEmbedding(
        config.max_position_embeddings + 2,  # +2 if model uses offset
        config.hidden_size,
        None,  # No padding_idx
        dtype=config.neuron_config.torch_dtype,
    )

    # ... rest of model initialization

def forward(self, input_ids, position_ids=None, ...):
    # Base class already called self.embed_tokens(input_ids)
    # We need to add positional embeddings
    if inputs_embeds is None and input_ids is not None:
        batch_size, seq_length = input_ids.shape

        # Get token embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Generate position_ids if not provided
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # Get positional embeddings (add offset during lookup if model uses offset)
        position_embeddings = self.embed_positions(position_ids + offset)  # offset=2 for some models

        # Combine token and positional embeddings
        inputs_embeds = inputs_embeds + position_embeddings

    # Continue with rest of forward pass...
    return super().forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        # ... other args
    )
```

### Complete Forward Signature (NeuronBaseModel)

When overriding forward(), you MUST match the complete base class signature:

```python
def forward(
    self,
    input_ids,
    attention_mask,
    position_ids,
    seq_ids,
    sampling_params,
    prev_hidden=None,
    adapter_ids=None,
    accepted_indices=None,
    current_length=None,
    medusa_mask=None,
    scatter_index=None,
    slot_mapping=None,
    active_block_table=None,
    num_queries=None,
    computed_context_lens=None,
    tile_q_indices=None,
    tile_block_tables=None,
    tile_masks=None,
    inputs_embeds=None,
    kv_cache=None,
    active_mask=None,
    rotary_position_id=None,
    vision_embeddings=None,
    vision_mask=None,
):
    # Your custom embedding logic
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids + 2)
        inputs_embeds = inputs_embeds + position_embeds

    # Pass ALL parameters to parent
    return super().forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        seq_ids=seq_ids,
        sampling_params=sampling_params,
        prev_hidden=prev_hidden,
        adapter_ids=adapter_ids,
        accepted_indices=accepted_indices,
        current_length=current_length,
        medusa_mask=medusa_mask,
        scatter_index=scatter_index,
        slot_mapping=slot_mapping,
        active_block_table=active_block_table,
        num_queries=num_queries,
        computed_context_lens=computed_context_lens,
        tile_q_indices=tile_q_indices,
        tile_block_tables=tile_block_tables,
        tile_masks=tile_masks,
        inputs_embeds=inputs_embeds,
        kv_cache=kv_cache,
        active_mask=active_mask,
        rotary_position_id=rotary_position_id,
        vision_embeddings=vision_embeddings,
        vision_mask=vision_mask,
    )
```

### Common Mistakes to Avoid

#### Mistake 1: Wrong `get_input_embeddings()` Signature

❌ **WRONG** - Custom signature breaks the model:

```python
def get_input_embeddings(self, input_ids, position_ids):
    # This is WRONG - get_input_embeddings() takes NO parameters!
    return self.embed_tokens(input_ids) + self.embed_positions(position_ids)
```

✅ **CORRECT** - Simple getter with no parameters:

```python
def get_input_embeddings(self):
    # Just return the embedding layer itself
    return self.embed_tokens
```

**Why**: Base class expects `get_input_embeddings()` to return the embedding layer object, not compute embeddings. It's a getter method, not a computation method.

#### Mistake 2: Using nn.Embedding Instead of ParallelEmbedding

❌ **WRONG** - Regular PyTorch embedding:

```python
self.embed_positions = nn.Embedding(
    config.max_position_embeddings,
    config.hidden_size,
)
```

✅ **CORRECT** - Use ParallelEmbedding for distributed training:

```python
self.embed_positions = ParallelEmbedding(
    config.max_position_embeddings + 2,
    config.hidden_size,
    None,
    dtype=config.neuron_config.torch_dtype,
    shard_across_embedding=False,
)
```

#### Mistake 3: Incomplete Forward Signature

❌ **WRONG** - Missing parameters:

```python
def forward(self, input_ids, attention_mask, position_ids, **kwargs):
    # Using **kwargs is fragile and can cause issues
```

✅ **CORRECT** - Explicit parameters matching base class:

```python
def forward(
    self,
    input_ids,
    attention_mask,
    position_ids,
    seq_ids,
    sampling_params,
    # ... all 26 parameters explicitly listed
):
```

````

### Weight Conversion for Separate Embeddings

With separate embeddings (no wrapper), weight conversion is straightforward:

```python
@staticmethod
def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    neuron_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Remove prefixes
        new_key = new_key.replace("model.decoder.", "").replace("decoder.", "")

        # Keys remain flat - no nesting needed:
        # "embed_tokens.weight" stays as "embed_tokens.weight"
        # "embed_positions.weight" stays as "embed_positions.weight"

        neuron_state_dict[new_key] = value

    return neuron_state_dict
````

### Tied Weights Handling

If embeddings are tied with lm_head:

```python
@staticmethod
def update_state_dict_for_tied_weights(state_dict):
    if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    return state_dict
```

---

        state_dict["lm_head.weight"] = state_dict["embed_tokens.token_embedding.weight"].clone()
    return state_dict

````

---

## 3. Positional Embedding Offsets

### Issue
Some model architectures use positional embeddings with index offsets.

### Discovery
Certain models reserve the first N embedding indices for special purposes:
- Index 0: Padding token position
- Index 1: Reserved
- Index 2+: Actual positions (for models with offset=2)

### Critical: Where to Apply the Offset

**Apply the offset DURING embedding lookup, not when generating position_ids:**

```python
# ✅ CORRECT - Apply offset during lookup
position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
position_embeddings = self.embed_positions(position_ids + 2)  # Offset here!

# ❌ WRONG - Don't add offset to position_ids before storing
position_ids = torch.arange(2, seq_length + 2, ...)  # Wrong!
position_embeddings = self.embed_positions(position_ids)
````

### Implementation

```python
def init_model(self, config: InferenceConfig):
    # Embedding table size includes offset
    self.embed_positions = ParallelEmbedding(
        config.max_position_embeddings + 2,  # +2 for offset
        config.hidden_size,
        None,
        dtype=config.neuron_config.torch_dtype,
    )

def forward(self, input_ids, position_ids=None, ...):
    if position_ids is None:
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    # Add offset during embedding lookup
    position_embeddings = self.embed_positions(position_ids + 2)
```

### Why This Matters

- The offset is a property of the embedding table, not the position indices
- Adding offset during lookup keeps position_ids clean for other uses
- Matches HuggingFace implementation pattern
- **This pattern is critical** for models with learned positional embeddings to generate coherent text

### Validation

Always verify against reference implementation:

```python
# Compare with HuggingFace
hf_model = AutoModel.from_pretrained(model_path)
neuron_model = NeuronModel(compiled_path)

inputs = tokenizer("Test", return_tensors="pt")
with torch.no_grad():
    hf_out = hf_model(**inputs)
    neuron_out = neuron_model(inputs['input_ids'])

# Outputs should match closely
assert torch.allclose(hf_out.logits, neuron_out.logits, atol=1e-2)
```

---

## 4. Common Weight Renaming Patterns

### Issue

Different models use different naming conventions that must be mapped to framework expectations.

### Common Renamings

```python
@staticmethod
def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    neuron_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # 1. Remove model-specific prefixes
        new_key = new_key.replace("model.decoder.", "").replace("decoder.", "")

        # 2. Rename output projection (CRITICAL - often missed)
        new_key = new_key.replace("out_proj", "o_proj")

        # 3. Rename top-level final norm (but not per-layer norms)
        if new_key.startswith("final_layer_norm"):
            new_key = new_key.replace("final_layer_norm", "norm")
        # Per-layer final_layer_norm stays unchanged

        neuron_state_dict[new_key] = value

    return neuron_state_dict
```

### Selective Renaming Pattern

When you need different rules for top-level vs per-layer components:

```python
# Rename top-level only (no "layers.X." prefix)
if new_key.startswith("final_layer_norm"):
    new_key = new_key.replace("final_layer_norm", "norm")

# Or rename per-layer only
if new_key.startswith("layers.") and ".final_layer_norm." in new_key:
    # Apply transformation
    pass
```

---

## Summary

### Key Takeaways

1. **Framework preprocessing**: Base class automatically calls `convert_hf_to_neuron_state_dict` during compilation
2. **Learned positional embeddings**: Use wrapper pattern to combine token + position embeddings
3. **Positional offsets**: Some models require offset in positional embedding indices
4. **Nested keys**: Wrapper modules create nested state dict keys requiring careful mapping
5. **Model-specific prefixes**: Handle various checkpoint prefix patterns beyond `model.`
6. **MLP submodule structure**: Map flat MLP weights to framework's expected hierarchy
7. **Selective renaming**: Apply different rules for top-level vs per-layer components
8. **Separate projection structure**: Understand `fused_qkv=False` weight organization
9. **Explicit tied weights**: Always create both keys even when weights are shared

### Validation Checklist

- [ ] Debug print actual keys in `convert_hf_to_neuron_state_dict()`
- [ ] Verify positional embedding strategy (RoPE vs learned vs relative)
- [ ] Check if positional embeddings use offset
- [ ] Test logits match HuggingFace reference exactly
- [ ] Validate weight loading with nested module structures
- [ ] Test generation produces coherent output
- [ ] Check preshard_hook receives correctly formatted keys
- [ ] Validate MLP weight hierarchy matches framework expectations
- [ ] Confirm tied weights exist as separate keys in checkpoint

### When to Use These Patterns

| Pattern                        | Use When                                         |
| ------------------------------ | ------------------------------------------------ |
| Checkpoint preprocessing check | Any model port                                   |
| Embedding wrapper              | Model uses learned positional embeddings         |
| Positional offset              | Model reserves embedding indices                 |
| Nested key mapping             | Using wrapper modules                            |
| Prefix handling                | Model uses non-standard checkpoint prefixes      |
| MLP submodule mapping          | Framework expects different MLP hierarchy        |
| Selective renaming             | Different rules for top-level vs per-layer norms |
| Separate projection mapping    | Model has unfused q/k/v projections              |
| Explicit tied weights          | Model ties embeddings with lm_head               |

---

## 5. Model-Specific Checkpoint Prefixes

### Issue

Different model families use different checkpoint key prefixes beyond the standard `model.` prefix.

### Discovery

Common prefix patterns:

- **Most models**: `model.layers.0.weight`
- **Decoder-only variants**: `decoder.layers.0.weight`
- **Encoder-decoder**: `encoder.layers.0.weight`, `decoder.layers.0.weight`
- **Some models**: No prefix at all

The base class automatically removes the `model.` prefix and calls `convert_hf_to_neuron_state_dict()`, leaving any remaining prefixes for you to handle.

### Solution: Prefix Detection and Removal

```python
@staticmethod
def convert_hf_to_neuron_state_dict(hf_state_dict: dict, config: InferenceConfig) -> dict:
    neuron_state_dict = {}

    for key, value in hf_state_dict.items():
        new_key = key

        # Remove model-specific prefix
        # Check what prefix actually exists in the checkpoint
        if new_key.startswith("decoder."):
            new_key = new_key.replace("decoder.", "", 1)
        elif new_key.startswith("encoder."):
            new_key = new_key.replace("encoder.", "", 1)
        # Note: "model." prefix already removed by base class

        # Continue with other transformations...
        neuron_state_dict[new_key] = value

    return neuron_state_dict
```

### Validation Pattern

Always verify the actual prefix structure by checking what your conversion function receives:

```python
@staticmethod
def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    # Debug: Check what keys we receive after base class preprocessing
    print("Keys after base class:", list(state_dict.keys())[:10])
    # Now you know what prefix to remove!

    neuron_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Remove remaining prefixes...
        neuron_state_dict[new_key] = value
    return neuron_state_dict
```

### Common Patterns

```python
# Pattern 1: Single prefix removal
if new_key.startswith("decoder."):
    new_key = new_key.replace("decoder.", "", 1)  # Use count=1 for safety

# Pattern 2: Multiple possible prefixes
for prefix in ["decoder.", "model.decoder.", "transformer."]:
    if new_key.startswith(prefix):
        new_key = new_key.replace(prefix, "", 1)
        break

# Pattern 3: Conditional based on model type
if config.is_encoder_decoder:
    if new_key.startswith("decoder."):
        new_key = new_key.replace("decoder.", "", 1)
else:
    # No prefix removal needed
    pass
```

---

## 5. MLP Weight Hierarchy Mapping

### Issue

Some models have MLP weights directly under the layer, but the framework expects them under an `mlp` submodule.

### Discovery

HuggingFace structure:

```
layers.0.fc1.weight
layers.0.fc1.bias
layers.0.fc2.weight
layers.0.fc2.bias
```

Framework expects:

```
layers.0.mlp.fc1.weight
layers.0.mlp.fc1.bias
layers.0.mlp.fc2.weight
layers.0.mlp.fc2.bias
```

### Solution: Simple String Manipulation

**Recommended approach (clear and debuggable):**

```python
@staticmethod
def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    neuron_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Add mlp prefix to fc1/fc2 weights
        if ".fc1." in new_key or ".fc2." in new_key:
            parts = new_key.split(".")
            layer_idx = parts[1]  # Extract layer number
            fc_part = ".".join(parts[2:])  # Get fc1/fc2 and weight/bias
            new_key = f"layers.{layer_idx}.mlp.{fc_part}"

        neuron_state_dict[new_key] = value

    return neuron_state_dict
```

### Alternative: Direct Replacement

For simpler cases:

```python
# If MLP always uses specific names
if ".fc1." in new_key:
    new_key = new_key.replace(".fc1.", ".mlp.fc1.")
if ".fc2." in new_key:
    new_key = new_key.replace(".fc2.", ".mlp.fc2.")

# Or for gate/up/down projections (SwiGLU models)
if ".gate_proj." in new_key:
    new_key = new_key.replace(".gate_proj.", ".mlp.gate_proj.")
if ".up_proj." in new_key:
    new_key = new_key.replace(".up_proj.", ".mlp.up_proj.")
if ".down_proj." in new_key:
    new_key = new_key.replace(".down_proj.", ".mlp.down_proj.")
```

### Validation

Check that MLP weights load correctly:

```python
# After conversion, verify keys exist
mlp_keys = [k for k in neuron_state_dict.keys() if 'mlp' in k and 'layers.0' in k]
print(f"MLP keys: {sorted(mlp_keys)}")

# Should see: layers.0.mlp.fc1.weight, layers.0.mlp.fc1.bias, etc.
```

---

## 8. Selective Component Renaming

### Issue

Some components need renaming at the top level but not at the per-layer level, or vice versa.

### Discovery

Example: Layer normalization

- Top-level final norm: `final_layer_norm` → `norm` (framework expects `norm`)
- Per-layer norms: `layers.X.final_layer_norm` → keep as-is (framework expects original name)

### Solution: Conditional Renaming

```python
@staticmethod
def convert_hf_to_neuron_state_dict(hf_state_dict: dict, config: InferenceConfig) -> dict:
    neuron_state_dict = {}

    for key, value in hf_state_dict.items():
        new_key = key

        # Rename top-level final norm only
        if new_key == "final_layer_norm.weight" or new_key == "final_layer_norm.bias":
            new_key = new_key.replace("final_layer_norm.", "norm.")
        # Per-layer final_layer_norm stays unchanged

        neuron_state_dict[new_key] = value

    return neuron_state_dict
```

### Pattern: Position-Aware Renaming

```python
# Rename based on position in key hierarchy
if new_key.startswith("final_layer_norm."):  # Top-level
    new_key = new_key.replace("final_layer_norm.", "norm.")
elif ".final_layer_norm." in new_key:  # Nested in layers
    # Keep as-is
    pass

# Or use regex for precise matching
import re
# Only match top-level (no "layers.X." prefix)
if re.match(r'^final_layer_norm\.(weight|bias)$', new_key):
    new_key = new_key.replace("final_layer_norm.", "norm.")
```

### Common Patterns

```python
# Pattern 1: Top-level only
if not new_key.startswith("layers."):
    new_key = new_key.replace("old_name.", "new_name.")

# Pattern 2: Per-layer only
if new_key.startswith("layers."):
    new_key = new_key.replace("old_name.", "new_name.")

# Pattern 3: Different rules for different layers
if ".self_attn_layer_norm." in new_key:
    # Keep as-is
    pass
elif ".post_attention_layernorm." in new_key:
    new_key = new_key.replace("post_attention_layernorm.", "post_attn_norm.")
```

---

## 9. Separate Projection Weight Structure (fused_qkv=False)

### Issue

When `fused_qkv=False`, the framework creates a specific weight hierarchy that differs from both fused QKV and completely separate projections.

### Discovery

Three different structures:

**1. Fused QKV (fused_qkv=True):**

```
layers.0.self_attn.qkv_proj.Wqkv.weight  # Single fused weight
```

**2. Completely Separate (incorrect for framework):**

```
layers.0.self_attn.q_proj.weight
layers.0.self_attn.k_proj.weight
layers.0.self_attn.v_proj.weight
```

**3. Framework's fused_qkv=False (correct):**

```
layers.0.self_attn.qkv_proj.q_proj.weight
layers.0.self_attn.qkv_proj.k_proj.weight
layers.0.self_attn.qkv_proj.v_proj.weight
```

The `qkv_proj` is a `GroupQueryAttention_QKV` instance that has `q_proj`, `k_proj`, `v_proj` as attributes.

### Solution: Add Intermediate Level

```python
@staticmethod
def convert_hf_to_neuron_state_dict(hf_state_dict: dict, config: InferenceConfig) -> dict:
    neuron_state_dict = {}

    for key, value in hf_state_dict.items():
        new_key = key

        # Map separate projections to qkv_proj structure
        # Must handle both .weight and .bias
        new_key = new_key.replace(".self_attn.q_proj.weight", ".self_attn.qkv_proj.q_proj.weight")
        new_key = new_key.replace(".self_attn.q_proj.bias", ".self_attn.qkv_proj.q_proj.bias")
        new_key = new_key.replace(".self_attn.k_proj.weight", ".self_attn.qkv_proj.k_proj.weight")
        new_key = new_key.replace(".self_attn.k_proj.bias", ".self_attn.qkv_proj.k_proj.bias")
        new_key = new_key.replace(".self_attn.v_proj.weight", ".self_attn.qkv_proj.v_proj.weight")
        new_key = new_key.replace(".self_attn.v_proj.bias", ".self_attn.qkv_proj.v_proj.bias")

        # Output projection also needs mapping
        new_key = new_key.replace(".self_attn.out_proj.", ".self_attn.o_proj.")

        neuron_state_dict[new_key] = value

    return neuron_state_dict
```

### Configuration

Ensure your NeuronConfig sets `fused_qkv=False`:

```python
class ModelNeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fused_qkv = False  # Critical for separate projections
        self.attn_cls = ModelAttention
```

### Validation

```python
# Check attention weight structure
attn_keys = [k for k in state_dict.keys() if 'self_attn' in k and 'layers.0' in k]
print("Attention keys:", sorted(attn_keys))

# Should see:
# layers.0.self_attn.qkv_proj.q_proj.weight
# layers.0.self_attn.qkv_proj.k_proj.weight
# layers.0.self_attn.qkv_proj.v_proj.weight
# layers.0.self_attn.o_proj.weight
```

---

## 6. Explicit Tied Weight Creation

### Issue

Even when a model ties weights implicitly (e.g., `lm_head` shares weights with `embed_tokens`), the compiled checkpoint must have BOTH keys explicitly present.

### Discovery

HuggingFace models often tie weights by reference:

```python
# In HuggingFace model
self.lm_head.weight = self.embed_tokens.weight  # Same object
```

But the framework's weight loading expects both keys in the state dict:

```python
# Framework expects
state_dict = {
    "embed_tokens.weight": tensor(...),
    "lm_head.weight": tensor(...),  # Must exist even if tied
}
```

### Solution: Explicit Cloning

```python
@staticmethod
def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    neuron_state_dict = {}

    # ... other conversions ...

    # Handle tied embeddings - must use .clone()
    if "lm_head.weight" not in neuron_state_dict and "embed_tokens.weight" in neuron_state_dict:
        neuron_state_dict["lm_head.weight"] = neuron_state_dict["embed_tokens.weight"].clone()

    return neuron_state_dict
```

### Why .clone() is Required

```python
# ❌ WRONG - Creates reference, causes safetensors error
state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

# ✅ CORRECT - Creates independent copy
state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
```

Using a reference instead of `.clone()` causes safetensors to fail when saving because it detects duplicate tensor storage.

### Alternative: update_state_dict_for_tied_weights

Some models implement this as a separate method:

```python
@staticmethod
def update_state_dict_for_tied_weights(state_dict: dict):
    """Handle tied embeddings between embed_tokens and lm_head."""
    if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    return state_dict
```

### Validation

```python
# Verify both keys exist
assert "embed_tokens.weight" in state_dict
assert "lm_head.weight" in state_dict

# Verify they have the same values (but different storage)
assert torch.allclose(state_dict["embed_tokens.weight"], state_dict["lm_head.weight"])
assert state_dict["embed_tokens.weight"].data_ptr() != state_dict["lm_head.weight"].data_ptr()
```

---

## 7. Framework Config Attributes for Inference

### Issue

The framework expects certain standard HuggingFace config attributes to exist during inference, even if they're not actively used by the model.

### Discovery

During inference, the framework's base classes check for these attributes:

- `output_attentions`
- `output_hidden_states`
- `use_return_dict`
- `use_cache`

Missing these causes `AttributeError` during model execution.

### Error You'll See

```
AttributeError: 'ModelInferenceConfig' object has no attribute 'output_attentions'
```

### Solution: Add to add_derived_config()

```python
class ModelInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        """Add derived configuration parameters required by the framework"""
        # Model-specific config
        self.num_cores_per_group = 1
        if not hasattr(self, 'head_dim'):
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Framework-required attributes for inference
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True
        if not hasattr(self, 'use_cache'):
            self.use_cache = True
```

### Why hasattr() Checks

Use defensive `hasattr()` checks to avoid overwriting values that may have been intentionally set in the config JSON or passed as kwargs.

### When This Applies

- **All model ports** should include these attributes
- Especially critical for models that don't inherit from standard HuggingFace config classes
- Required even if your model doesn't use attention outputs or hidden states

### Validation

```python
# After creating config, verify attributes exist
config = ModelInferenceConfig.from_pretrained(model_path)
assert hasattr(config, 'output_attentions')
assert hasattr(config, 'output_hidden_states')
assert hasattr(config, 'use_return_dict')
assert hasattr(config, 'use_cache')
```

---

## 12. Position IDs Computation for Learned Embeddings

### Issue

Models with learned positional embeddings need proper position_ids computation, especially for autoregressive generation where past_key_value is used.

### Discovery

Unlike RoPE (computed in attention layers), learned positional embeddings require position_ids to be computed in the model's forward pass before embedding lookup.

### Pattern for Context Encoding (Prefill)

```python
def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_value=None):
    batch_size, seq_length = input_ids.shape

    if position_ids is None:
        # Determine starting position based on past context
        past_length = 0
        if past_key_value is not None and len(past_key_value) > 0:
            past_length = past_key_value[0][0].shape[2]  # KV cache sequence length

        # Create position_ids starting from past_length
        device = input_ids.device
        position_ids = torch.arange(
            past_length,
            seq_length + past_length,
            dtype=torch.long,
            device=device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    # Get embeddings with positions
    hidden_states = self.embed_tokens(input_ids, position_ids)

    # Continue with decoder layers...
```

### Pattern for Token Generation

```python
# During autoregressive generation
for step in range(max_new_tokens):
    # position_ids should be [past_length + step]
    current_position = past_length + step
    position_ids = torch.tensor([[current_position]], dtype=torch.long, device=device)

    outputs = model(
        input_ids=next_token_id,
        position_ids=position_ids,
        past_key_value=past_key_value
    )
```

### Common Mistake

```python
# ❌ WRONG - Always starts from 0, ignores past context
position_ids = torch.arange(seq_length).unsqueeze(0)

# ✅ CORRECT - Accounts for past context length
past_length = past_key_value[0][0].shape[2] if past_key_value else 0
position_ids = torch.arange(past_length, seq_length + past_length).unsqueeze(0)
```

### Why This Matters

- **Without past_length**: Model sees position [0, 1, 2] for every generation step
- **With past_length**: Model sees position [0, 1, 2] then [3], [4], [5]... correctly

### Validation

```python
# Test position_ids computation
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
position_ids = compute_position_ids(input_ids, past_key_value=None)
assert position_ids.tolist() == [[0, 1, 2, 3, 4]]

# With past context
past_kv = create_dummy_past_kv(past_length=10)
position_ids = compute_position_ids(input_ids, past_key_value=past_kv)
assert position_ids.tolist() == [[10, 11, 12, 13, 14]]
```

- See "Learned Positional Embeddings Pattern" (Section 2)
- See "Positional Embedding Offsets" (Section 3)

---
