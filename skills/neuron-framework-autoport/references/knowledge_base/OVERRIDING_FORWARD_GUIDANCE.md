# Overriding Forward Methods in NeuronX Models - Complete Guidance

## Purpose of This Document

This document provides **definitive guidance** on when and how to override `forward()` methods in NeuronX model implementations. It consolidates all guidance from the knowledge base and resolves apparent contradictions.

## Component Hierarchy

NeuronX models have multiple components, each with their own `forward()` methods:

```
NeuronBaseForCausalLM (Application wrapper)
└── NeuronBaseModel (Main model class) ← THIS DOCUMENT FOCUSES HERE
    ├── embed_tokens (ParallelEmbedding)
    ├── layers (ModuleList)
    │   └── DecoderLayer
    │       ├── self_attn (Attention) ← Has forward()
    │       └── mlp (MLP) ← Has forward()
    └── lm_head (ColumnParallelLinear)
```

**This document focuses on overriding `forward()` in `NeuronBaseModel` subclasses.**

## Decision Tree: Should You Override forward() in NeuronBaseModel?

```
START: Porting a new model to NeuronX
│
├─ Does model use RoPE (Rotary Position Embeddings)?
│  ├─ YES → ❌ DO NOT override forward()
│  │        Examples: LLaMA, Mistral, Qwen, Gemma
│  │        Reason: RoPE computed in attention layers
│  │
│  └─ NO → Continue to next check
│
├─ Does model use relative position bias (T5-style)?
│  ├─ YES → ❌ DO NOT override forward()
│  │        Examples: T5, BART
│  │        Reason: Position bias computed in attention
│  │
│  └─ NO → Continue to next check
│
├─ Does model use learned positional embeddings?
│  ├─ YES → ✅ MUST override forward()
│  │        Examples: GPT-2, BERT, RoBERTa, ALBERT
│  │        Reason: Must add position embeddings to token embeddings
│  │        See: Section "Learned Positional Embeddings Pattern"
│  │
│  └─ NO → Continue to next check
│
├─ Does model require custom embedding preprocessing?
│  ├─ YES → ✅ MAY override forward()
│  │        Examples: Multimodal models (vision + text)
│  │        Reason: Custom logic to combine modalities
│  │
│  └─ NO → ❌ DO NOT override forward()
│           Default: Let base class handle everything
```

## How to Identify Learned Positional Embeddings

### Check HuggingFace Model Code

Look for this pattern in the HuggingFace model:

```python
# Pattern indicating learned positional embeddings
class HFModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_positions, hidden_size)  # ← Learned!

    def forward(self, input_ids):
        token_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)
        hidden_states = token_embeds + position_embeds  # ← Addition happens here!
        # ... rest of forward pass
```

### Models with Learned Positional Embeddings

**Confirmed models that NEED forward() override:**

- GPT-2 (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- BERT (bert-base-uncased, bert-large-uncased)
- RoBERTa (roberta-base, roberta-large)
- ALBERT (albert-base-v2, albert-large-v2)
- Any model with learned positional embeddings (check HuggingFace implementation)

**Models that DO NOT need forward() override:**

- LLaMA family (meta-llama/Llama-2-_, meta-llama/Llama-3-_)
- Mistral (mistralai/Mistral-7B-\*)
- Qwen (Qwen/Qwen-_, Qwen/Qwen2-_)
- Gemma (google/gemma-\*)
- Mixtral (mistralai/Mixtral-8x7B-\*)
- DBRX (databricks/dbrx-\*)
- T5 (t5-small, t5-base, t5-large)

## Survey of Existing Implementations

### NeuronxDistributedInference Models

All surveyed models in `/NeuronxDistributedInference/src/neuronx_distributed_inference/models/`:

| Model     | Overrides forward()? | Position Encoding Type |
| --------- | -------------------- | ---------------------- |
| llama     | ❌ NO                | RoPE                   |
| mistral   | ❌ NO                | RoPE                   |
| qwen2     | ❌ NO                | RoPE                   |
| qwen3     | ❌ NO                | RoPE                   |
| qwen3_moe | ❌ NO                | RoPE                   |
| mixtral   | ❌ NO                | RoPE                   |
| dbrx      | ❌ NO                | RoPE                   |
| gpt_oss   | ❌ NO                | RoPE                   |
| llama4    | ❌ NO                | RoPE                   |
| mllama    | ❌ NO                | RoPE (multimodal)      |

**Finding**: None of the existing models override `forward()` because they all use RoPE.

### NeuroborosFoundations Models

All surveyed models in `/NeuroborosFoundations/src/amzn/neuron/neuroboros/models/`:

| Model      | Overrides forward()? | Position Encoding Type |
| ---------- | -------------------- | ---------------------- |
| gemma3     | ❌ NO                | RoPE                   |
| gpt2       | ❌ NO                | RoPE (modified)        |
| gptoss     | ❌ NO                | RoPE                   |
| phi3       | ❌ NO                | RoPE                   |
| phimoe     | ❌ NO                | RoPE                   |
| starcoder2 | ❌ NO                | RoPE                   |

**Finding**: None of these models override `forward()` either. Note that `gpt2` in this collection appears to be a modified version using RoPE, not the original GPT-2 with learned positional embeddings.

### Example: Learned Positional Embedding Model

| Model Type                    | Overrides forward()? | Position Encoding Type        |
| ----------------------------- | -------------------- | ----------------------------- |
| Learned Positional Embeddings | ✅ YES               | Learned positional embeddings |

**Models with learned positional embeddings are the first type in the codebase that require forward() override.**

## Knowledge Base Guidance Audit

### Files Saying "DON'T Override Forward"

1. **COMPREHENSIVE_LLAMA3_NEURONX_GUIDE.md:305**
   - Context: LLaMA3 porting guide
   - Quote: "The NeuronxDistributedInference framework expects models to NOT implement their own forward method"
   - **Applies to**: RoPE models only

2. **COMPREHENSIVE_LLAMA3_NEURONX_GUIDE.md:815**
   - Context: Compilation checklist
   - Quote: "No Custom Forward: Don't implement forward method in main model class"
   - **Applies to**: RoPE models only

3. **NEURONX_PORTING_GUIDE.md:230**
   - Context: Base model implementation
   - Quote: "Do NOT override forward()"
   - **Applies to**: General case (RoPE models)

4. **NEURONX_PORTING_GUIDE.md:532**
   - Context: Best practices
   - Quote: "No custom forward(): Use the base class implementation unless absolutely necessary"
   - **Applies to**: General case, but notes "unless absolutely necessary"

5. **NEURONX_PORTING_GUIDE.md:886**
   - Context: Common mistakes
   - Quote: "DO NOT override `forward()` - base class handles everything"
   - **Applies to**: General case (RoPE models)

6. **NEURONX_PORTING_GUIDE.md:1188**
   - Context: Troubleshooting
   - Quote: "DO NOT override forward() at all - let `NeuronBaseModel` handle it"
   - **Applies to**: General case (RoPE models)

7. **NEURONX_PORTING_GUIDE.md:1821**
   - Context: Best practices summary
   - Quote: "Don't override `forward()` methods unless absolutely necessary"
   - **Applies to**: General case, but notes "unless absolutely necessary"

8. **TRACE_PORT.md:293**
   - Context: GenericModel port
   - Quote: "Don't define custom forward()"
   - **Applies to**: RoPE models

9. **compilation_errors_and_fixes.md:342**
   - Context: Common errors
   - Quote: "Don't implement custom forward methods"
   - **Applies to**: General case (RoPE models)

### Files Saying "DO Override Forward" (Exception Case)

1. **NEURONX_NOVEL_PORTS.md:189-193**
   - Context: Learned positional embeddings pattern
   - Quote: "EXCEPTION: Models with learned positional embeddings (GPT-2, BERT, RoBERTa) MUST override `forward()` to add position embeddings"
   - **Applies to**: Learned positional embedding models ONLY

### Resolution of Contradiction

**There is NO contradiction** - the guidance is context-specific:

- **General Rule (99% of models)**: Don't override forward() - applies to RoPE models
- **Exception (1% of models)**: MUST override forward() - applies to learned positional embedding models

The confusion arises because:

1. Most existing models use RoPE, so "don't override" is the common case
2. The exception case is documented but not cross-referenced in general guidance
3. Learned positional embedding models are rare in the current codebase

## Complete Implementation Pattern for Learned Positional Embeddings

### Step 1: Use ParallelEmbedding for Both Embeddings

```python
def init_model(self, config: InferenceConfig):
    # Token embeddings
    self.embed_tokens = ParallelEmbedding(
        config.vocab_size,
        config.hidden_size,
        padding_idx=config.pad_token_id,
        dtype=config.neuron_config.torch_dtype,
        shard_across_embedding=not config.neuron_config.vocab_parallel,
    )

    # Positional embeddings (separate, not wrapped)
    self.embed_positions = ParallelEmbedding(
        config.max_position_embeddings + offset,  # Add offset if model uses it
        config.hidden_size,
        None,  # No padding_idx for positions
        dtype=config.neuron_config.torch_dtype,
        shard_across_embedding=False,  # Don't shard position embeddings
    )

    # ... rest of model initialization (layers, norm, lm_head)
```

### Step 2: Keep get_input_embeddings() Simple

```python
def get_input_embeddings(self):
    """Return the token embedding layer (NOT position embeddings)"""
    return self.embed_tokens
```

**Common Mistake**: Do NOT override with custom signature:

```python
# ❌ WRONG - This breaks the model!
def get_input_embeddings(self, input_ids, position_ids):
    return self.embed_tokens(input_ids) + self.embed_positions(position_ids)
```

### Step 3: Override forward() with Complete Signature

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
    """
    Override forward to add positional embeddings to token embeddings.

    This is required for models with learned positional embeddings (GPT-2, BERT, etc).
    """
    # Only compute embeddings if not already provided
    if inputs_embeds is None:
        # Get token embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Add positional embeddings with offset (if applicable)
        position_embeds = self.embed_positions(position_ids + offset)  # offset varies by model
        inputs_embeds = inputs_embeds + position_embeds

    # Pass ALL parameters to parent forward
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
        inputs_embeds=inputs_embeds,  # Pass combined embeddings
        kv_cache=kv_cache,
        active_mask=active_mask,
        rotary_position_id=rotary_position_id,
        vision_embeddings=vision_embeddings,
        vision_mask=vision_mask,
    )
```

### Step 4: Weight Conversion

```python
@staticmethod
def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    neuron_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Remove model-specific prefixes
        if new_key.startswith('decoder.'):
            new_key = new_key.replace('decoder.', '', 1)

        # Keys remain flat - no nesting needed:
        # "embed_tokens.weight" stays as "embed_tokens.weight"
        # "embed_positions.weight" stays as "embed_positions.weight"

        neuron_state_dict[new_key] = value

    return neuron_state_dict
```

## Common Mistakes and How to Avoid Them

### Mistake 1: Wrong get_input_embeddings() Signature

❌ **WRONG**:

```python
def get_input_embeddings(self, input_ids, position_ids):
    return self.embed_tokens(input_ids) + self.embed_positions(position_ids)
```

✅ **CORRECT**:

```python
def get_input_embeddings(self):
    return self.embed_tokens
```

**Why**: `get_input_embeddings()` is a getter that returns the embedding layer object, not a computation method.

### Mistake 2: Using nn.Embedding Instead of ParallelEmbedding

❌ **WRONG**:

```python
self.embed_positions = nn.Embedding(max_positions, hidden_size)
```

✅ **CORRECT**:

```python
self.embed_positions = ParallelEmbedding(
    max_positions,
    hidden_size,
    None,
    dtype=config.neuron_config.torch_dtype,
    shard_across_embedding=False,
)
```

**Why**: NeuronX requires `ParallelEmbedding` for distributed training compatibility.

### Mistake 3: Incomplete Forward Signature

❌ **WRONG**:

```python
def forward(self, input_ids, attention_mask, position_ids, **kwargs):
    # Using **kwargs is fragile
```

✅ **CORRECT**:

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

**Why**: Explicit parameters prevent signature mismatches and make the code self-documenting.

### Mistake 4: Not Passing inputs_embeds to Parent

❌ **WRONG**:

```python
def forward(self, input_ids, ...):
    inputs_embeds = self.embed_tokens(input_ids) + self.embed_positions(position_ids)
    # Forgot to pass inputs_embeds to parent!
    return super().forward(input_ids=input_ids, ...)
```

✅ **CORRECT**:

```python
def forward(self, input_ids, ...):
    inputs_embeds = self.embed_tokens(input_ids) + self.embed_positions(position_ids)
    return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, ...)
```

**Why**: Parent forward needs `inputs_embeds` to skip its own embedding computation.

## Symptom-Based Troubleshooting

### Symptom: Model Compiles But Generates Gibberish

**Possible Cause**: Learned positional embeddings not being added

**Check**:

1. Does HuggingFace model have `embed_positions`?
2. Is it `nn.Embedding` (learned) or computed (RoPE)?

**Fix**: Override `forward()` to add positional embeddings (see Section "Complete Implementation Pattern")

### Symptom: TypeError: forward() takes X arguments but Y were given

**Possible Cause**: Incomplete forward signature

**Fix**: Use complete signature with all 26 parameters explicitly listed

### Symptom: Model generates empty output

**Possible Cause**: `inputs_embeds` not passed to parent forward

**Fix**: Ensure `inputs_embeds=inputs_embeds` in `super().forward()` call

## Summary

### When to Override forward() in NeuronBaseModel

✅ **DO Override** if:

- Model uses learned positional embeddings (GPT-2, BERT, RoBERTa, ALBERT)
- Model requires custom embedding preprocessing (multimodal)

❌ **DON'T Override** if:

- Model uses RoPE (LLaMA, Mistral, Qwen, Gemma) - 99% of models
- Model uses relative position bias (T5, BART)
- Model is standard transformer architecture

### Key Takeaways

1. **General rule**: Don't override forward() - applies to RoPE models (99% of cases)
2. **Exception**: MUST override for learned positional embeddings (1% of cases)
3. **Learned positional embedding models** are the first type requiring this exception
4. **Always use ParallelEmbedding**, never nn.Embedding
5. **Keep get_input_embeddings() simple** - just return the layer
6. **Use complete forward signature** - all 26 parameters explicitly
7. **Pass inputs_embeds to parent** - critical for correct behavior

### Cross-References

- **Learned Positional Embeddings Pattern**: NEURONX_NOVEL_PORTS.md Section 2
- **Positional Embedding Offsets**: NEURONX_NOVEL_PORTS.md Section 3
- **General Porting Guide**: NEURONX_PORTING_GUIDE.md
- **LLaMA Example (no override)**: COMPREHENSIVE_LLAMA3_NEURONX_GUIDE.md
