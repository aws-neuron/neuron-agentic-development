"""
bias_restoration_template.py — TP-aware bias restoration after get_sharded_checkpoint.

get_sharded_checkpoint removes biases it considers "redundant" (those not marked
as tensor_model_parallel). This template shows how to restore them from the
original HF state dict with TP-aware logic.

Three cases:
1. hf_bias == target_size: RowParallelLinear (e.g., o_proj) — use as-is
2. hf_bias > target_size:  ColumnParallelLinear (e.g., q_proj) — simple chunk-shard
3. hf_bias < target_size:  CONVERT_TO_MHA (e.g., k_proj, v_proj) — replicate then shard

Adapt the bias_map keys and attr_paths to your model's attention structure.
"""
import torch


def restore_attention_biases(inner_model, hf_sd, config, dtype, rank, tp_degree):
    """Restore attention biases removed by get_sharded_checkpoint.

    Args:
        inner_model: The Neuron model after load_state_dict
        hf_sd: Original HF state dict (full, unsharded)
        config: Model config with head_dim, num_hidden_layers, etc.
        dtype: Target dtype (torch.float32 or torch.bfloat16)
        rank: Current TP rank
        tp_degree: Total TP degree
    """
    for i in range(config.num_hidden_layers):
        layer = inner_model.layers[i]

        # Map HF bias keys to module attribute paths within the layer
        # Adapt these to your model's attention structure
        bias_map = {
            f"model.layers.{i}.self_attn.q_proj.bias": "self_attn.qkv_proj.q_proj",
            f"model.layers.{i}.self_attn.k_proj.bias": "self_attn.qkv_proj.k_proj",
            f"model.layers.{i}.self_attn.v_proj.bias": "self_attn.qkv_proj.v_proj",
            f"model.layers.{i}.self_attn.o_proj.bias": "self_attn.o_proj.o_proj",
        }

        for hf_key, attr_path in bias_map.items():
            if hf_key not in hf_sd:
                continue

            # Navigate to the target module
            module = layer
            for part in attr_path.split("."):
                module = getattr(module, part)

            hf_bias = hf_sd[hf_key].to(dtype)

            if tp_degree > 1:
                target_size = module.weight.shape[0]
                head_dim_val = config.head_dim

                if hf_bias.shape[0] == target_size:
                    # Case 1: RowParallelLinear (o_proj)
                    # Bias is full-size, not sharded. Use as-is.
                    pass

                elif hf_bias.shape[0] > target_size:
                    # Case 2: ColumnParallelLinear (q_proj)
                    # Simple chunk-shard: divide bias into tp_degree equal parts
                    chunk = hf_bias.shape[0] // tp_degree
                    hf_bias = hf_bias[rank * chunk : (rank + 1) * chunk]

                else:
                    # Case 3: CONVERT_TO_MHA (k_proj, v_proj)
                    # When tp_degree % num_kv_heads != 0, KV heads are replicated
                    # to match Q heads. Bias must follow same replication pattern.
                    #
                    # Example: 8 KV heads, TP=4, CONVERT_TO_MHA replicates to 64 heads
                    # bias [256] → reshape [8, 32] → repeat_interleave 8x → [64, 32]
                    # → reshape [2048] → shard [512] per rank
                    full_target = target_size * tp_degree
                    num_orig_heads = hf_bias.shape[0] // head_dim_val
                    num_target_heads = full_target // head_dim_val
                    repeats = num_target_heads // num_orig_heads
                    hf_bias = hf_bias.reshape(num_orig_heads, head_dim_val)
                    hf_bias = hf_bias.repeat_interleave(repeats, dim=0).reshape(-1)
                    chunk = full_target // tp_degree
                    hf_bias = hf_bias[rank * chunk : (rank + 1) * chunk]

            module.bias = torch.nn.Parameter(hf_bias)


def restore_moe_expert_biases(inner_model, hf_sd, config, dtype):
    """Restore per-expert down biases lost in standard weight copy.

    For MoE models, per-expert biases (e.g., down_proj_bias with shape [E, H])
    are not handled by the standard weight loading pipeline. Store them as a
    custom attribute on the experts module.

    Adapt to your model's MoE structure.
    """
    for i in range(config.num_hidden_layers):
        layer = inner_model.layers[i]
        key = f"model.layers.{i}.mlp.experts.down_proj_bias"
        if key in hf_sd:
            layer.mlp.experts._per_expert_down_bias = hf_sd[key].to(dtype)
