# GenericMoE v14 Analysis - Simplified RoPE Approach

## Date

October 27, 2025

## Summary

v14 successfully compiled with simplified standard RoPE (removing LongRoPE scaling parameters). Compilation was faster due to cached NEFF files.

## Changes in v14

Removed `use_scaled_rope` parameter from `NeuronGenericMoEAttention.__init__()` in both modeling files:

### Before (v13):

```python
super().__init__(
    config=config,
    hidden_size=config.hidden_size,
    num_attention_heads=config.num_attention_heads,
    num_key_value_heads=config.num_key_value_heads,
    head_dim=config.hidden_size // config.num_attention_heads,
    rotary_emb=rotary_emb,
    qkv_bias=getattr(config, "attention_bias", False),
    o_bias=getattr(config, "attention_bias", False),
    use_scaled_rope=getattr(config, "rope_scaling", None) is not None,  # ❌ Ineffective
)
```

### After (v14):

```python
super().__init__(
    config=config,
    hidden_size=config.hidden_size,
    num_attention_heads=config.num_attention_heads,
    num_key_value_heads=config.num_key_value_heads,
    head_dim=config.hidden_size // config.num_attention_heads,
    rotary_emb=rotary_emb,
    qkv_bias=getattr(config, "attention_bias", False),
    o_bias=getattr(config, "attention_bias", False),
    # ✅ Removed: use_scaled_rope parameter
)
```

## Compilation Results

### v14 Compilation (Successful)

- **Time**: 12.8 minutes (using cached NEFFs from v13)
- **HLO Generation**: 11.6 seconds
- **Token Generation Model**: 0.21 seconds (cached)
- **Context Encoding Model**: 0.33 seconds (cached)
- **Weight Sharding**: ~730 seconds
- **Status**: ✅ SUCCESS

### Comparison with v13:

- v13: 142 seconds total compilation time
- v14: Used cached NEFF files, faster overall

## Expected Inference Behavior

Based on v12 and v13 results (which also had vocab masking), we expect v14 to show **SIMILAR gibberish output**:

- Test 1: Empty or very short response
- Test 2: "Trans Trans" or similar repetition
- Test 3: "Writing Writing..." repetitive gibberish

### Why Removing LongRoPE Won't Fix Gibberish

The investigation document (`longrope_investigation_v13.md`) established that:

1. `use_scaled_rope=True` was already ineffective in v13 (only affects different code path)
2. LongRoPE mscale multiplier (1.243x) is a relatively small adjustment (~24%)
3. The gibberish persisted despite the flag being present

Therefore, **removing the ineffective flag should not change behavior**.

## Next Investigation Steps

Since RoPE configuration (v9, v13, v14) hasn't resolved gibberish, need to investigate:

### 1. Normalization Type (HIGH PRIORITY)

**Lesson from knowledge_base**: Generic MoE had identical gibberish symptoms due to LayerNorm vs RMSNorm mismatch.

**Action Required**:

- Verify GenericMoE uses RMSNorm (not LayerNorm)
- Check `get_rmsnorm_cls()` in modeling_genericmoe.py:165-169
- Compare with HuggingFace GenericMoE implementation

**Current Implementation**:

```python
def get_rmsnorm_cls():
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> GenericmoeRMSNorm
    return GenericmoeRMSNorm if cpu_mode() else CustomRMSNorm
```

**Question**: Is `CustomRMSNorm` compatible with GenericMoE's normalization requirements?

### 2. Weight Loading Verification

- Ensure 0 missing keys in weight loading
- Verify state dict conversion is correctly mapping all weights
- Check if redundant keys being removed are actually irrelevant

### 3. MoE Router Configuration

From GenericmoeInferenceConfig.**init**() (lines 188-193):

```python
self.neuron_config.router_config.dtype = torch.float32
self.neuron_config.router_config.act_fn = "softmax"
```

**Questions**:

- Is router correctly routing to experts?
- Are expert weights being applied correctly?
- Check if `glu_mlp=True` configuration is working

### 4. Expert Weight Application

From convert_genericmoe_hf_to_neuron_state_dict():

- Gate projection (w1) and up projection (w3) concatenation
- Down projection (w2) mapping
- Weight transposition (.T operations)

**Potential Issue**: Weight dimension ordering mismatch?

## Files Modified

- `/home/ec2-user/agents/hariseldon/NeuroborosFoundations/src/amzn/neuron/neuroboros/models/genericmoe/modeling_genericmoe.py` (line 325)
- `/home/ec2-user/agents/hariseldon/neuron_port/modeling_genericmoe.py` (line 325)

## Compilation Artifacts

- Compiled model: `/home/ec2-user/agents/hariseldon/agent_artifacts/data/genericmoe_compiled`
- Context encoding model: Compiled (cached)
- Token generation model: Compiled (cached)
- Weight shards: 16 files (tp0-tp15)

## Recommendation

**DO NOT** continue iterating on RoPE configuration. The issue is elsewhere.

**NEXT ACTION**: Investigate normalization type following the lesson from Generic MoE success story in knowledge_base.

Focus investigation on:

1. RMSNorm implementation compatibility
2. Weight loading/mapping correctness
3. MoE router functionality
4. Expert weight application

## Version History

- v9: Fixed rope_theta (1M → 10K)
- v10: Added attention_bias and lm_head_bias
- v11: Attempted weight truncation (reverted per user feedback)
- v12: Added vocabulary masking at inference level
- v13: Added use_scaled_rope=True (ineffective)
- v14: Removed use_scaled_rope parameter (simplified RoPE)
