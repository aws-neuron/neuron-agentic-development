# LongRoPE Investigation - GenericMoE v13 Analysis

## Date

October 27, 2025

## Problem Statement

Despite adding `use_scaled_rope=True` in v13, GenericMoE model continues producing gibberish output:

- Test 1: Empty response
- Test 2: "Trans Trans" repetition
- Test 3: "Writing Writing..." gibberish

## Investigation Process

### 1. Knowledge Base Review

Reviewed `/NeuroborosFoundations/knowledge_base/` for similar issues:

- **MoE_Port_Master_Summary.md**: Generic MoE achieved 100% accuracy through systematic debugging
- **Category3_Accuracy_Debugging_Analysis.md**: Key lesson - normalization type mismatch (LayerNorm vs RMSNorm) caused similar gibberish output

### 2. HuggingFace GenericMoE RoPE Implementation

Discovered GenericMoE uses custom `GenericmoeRotaryEmbedding`:

```python
class GenericmoeRotaryEmbedding(nn.Module):
    def __init__(self, config: Optional[GenericmoeConfig] = None):
        super().__init__()
        self.config = config
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "longrope")
            self.short_mscale = config.rope_scaling.get("short_mscale")  # 1.243163121016122
            self.long_mscale = config.rope_scaling.get("long_mscale")    # 1.243163121016122
        else:
            self.rope_type = "default"
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

    def forward(self, x, seq_len=None):
        mscale = None
        if self.config.rope_scaling and seq_len:
            mscale = (
                self.long_mscale
                if seq_len > self.config.rope_scaling["original_max_position_embeddings"]
                else self.short_mscale
            )
        inv_freq, attention_scaling = self.rope_init_fn(self.config, x.device, seq_len)
        mscale = attention_scaling if mscale is None else mscale
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * mscale).to(x.dtype), (emb.sin() * mscale).to(x.dtype)
```

**Key Points:**

- Applies `mscale` multiplier to cos/sin embeddings
- Uses `short_mscale` for seq_len <= 4096, `long_mscale` for longer sequences
- Both mscale values = 1.243163121016122
- Has 64 `short_factor` and 64 `long_factor` arrays for frequency scaling

### 3. NeuronX Framework RoPE Implementation

Examined `NeuronAttentionBase` in `attention_base.py`:

```python
# Line 430-437: Default RoPE path
if not use_polar_compatible_rope:
    cos_cache, sin_cache = self.rotary_emb(V, position_ids)
    Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

elif use_polar_compatible_rope:
    rotary_freqs = precompute_freqs_cis(self.head_dim,
                                        self.neuron_config.max_context_length * 2,
                                        self.rope_theta,
                                        self.use_scaled_rope,  # <-- Only used here
                                        device=Q.device)
    rotary_freqs = rotary_freqs[position_ids]
    Q, K = apply_rotary_polar_compatible(Q.transpose(1, 2), K.transpose(1, 2), rotary_freqs)
```

**Critical Finding:**

- `use_scaled_rope` flag **ONLY** affects `use_polar_compatible_rope` code path
- Default path uses `RotaryEmbedding` module which doesn't support LongRoPE mscale
- The simple `RotaryEmbedding(dim, max_position_embeddings, base)` has no scaling logic

### 4. Checked Llama4 Implementation

Verified Llama4 uses identical pattern:

```python
# Line 311 in modeling_llama4_text.py
use_scaled_rope=getattr(config, "rope_scaling", None) is not None,
```

But this works for Llama4 because it may use `use_polar_compatible_rope` mode or has different rope_scaling format.

## Root Cause Analysis

**The `use_scaled_rope=True` flag does NOT enable LongRoPE for GenericMoE because:**

1. **Wrong Code Path**: Flag only works with `use_polar_compatible_rope=True`, but our implementation uses default path
2. **Missing mscale Logic**: `RotaryEmbedding` doesn't implement mscale multipliers (1.243x)
3. **Missing Scaling Factors**: No support for `short_factor`/`long_factor` arrays (64 values each)
4. **Framework Limitation**: NeuronX framework doesn't have GenericMoE-compatible LongRoPE implementation

## Evidence

### GenericMoE config.json rope_scaling:

```json
{
  "rope_scaling": {
    "type": "longrope",
    "original_max_position_embeddings": 4096,
    "short_factor": [1.0, 1.0, 1.0, ...],  // 64 values
    "long_factor": [1.0, 1.0, 1.0, ...],   // 64 values
    "short_mscale": 1.243163121016122,
    "long_mscale": 1.243163121016122
  }
}
```

### v13 Implementation (INCORRECT):

```python
# NeuronGenericmoeAttention.__init__
rotary_emb = RotaryEmbedding(
    config.hidden_size // config.num_attention_heads,
    max_position_embeddings=config.max_position_embeddings,
    base=rope_theta,
)

super().__init__(
    # ...
    rotary_emb=rotary_emb,
    use_scaled_rope=getattr(config, "rope_scaling", None) is not None,  # ❌ Has no effect!
)
```

## Conclusion

The LongRoPE scaling in GenericMoE requires custom implementation that:

1. Reads `short_mscale` and `long_mscale` from config
2. Applies appropriate mscale multiplier based on sequence length
3. Uses `short_factor`/`long_factor` arrays for frequency adjustments

The NeuronX framework doesn't support this out of the box. The `use_scaled_rope` flag is **insufficient** for GenericMoE's LongRoPE requirements.

## Recommended Solution: Simplified RoPE (v14)

Remove LongRoPE scaling and use standard RoPE:

```python
# Remove use_scaled_rope parameter entirely
super().__init__(
    config=config,
    hidden_size=config.hidden_size,
    num_attention_heads=config.num_attention_heads,
    num_key_value_heads=config.num_key_value_heads,
    head_dim=config.hidden_size // config.num_attention_heads,
    rotary_emb=rotary_emb,
    qkv_bias=getattr(config, "attention_bias", False),
    o_bias=getattr(config, "attention_bias", False),
    # ✅ Remove: use_scaled_rope parameter
)
```

**Rationale:**

- Standard RoPE with `rope_theta=10000` and `max_position_embeddings=131072` may work for inference
- LongRoPE's mscale=1.243 is a ~24% adjustment - may not be critical for basic functionality
- Allows us to test if other issues (normalization, weight loading, etc.) are causing gibberish
- Can revisit LongRoPE implementation if basic RoPE works

## Next Steps (v14)

1. Remove `use_scaled_rope` from both modeling files
2. Recompile model
3. Test inference
4. Compare output quality with/without LongRoPE

## Files Modified in v13 (to be reverted in v14)

- `/home/ec2-user/agents/hariseldon/NeuroborosFoundations/src/amzn/neuron/neuroboros/models/genericmoe/modeling_genericmoe.py` (line 341)
- `/home/ec2-user/agents/hariseldon/neuron_port/modeling_genericmoe.py` (line 325)

## Compilation Status

- v13 compiled successfully in 142 seconds
- No compilation errors
- Issue is runtime accuracy, not compilation
