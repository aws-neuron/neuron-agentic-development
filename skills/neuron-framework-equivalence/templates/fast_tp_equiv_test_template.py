"""
fast_tp_equiv_test.py — Quick TP equivalence test with random weights.

Uses a tiny 1-layer model config for 5-10 second iteration cycles instead of
minutes with the full model. Creates synthetic HF-format safetensors and runs
the real get_state_dict → get_sharded_checkpoint pipeline.

Design rules for TINY_CONFIG:
- num_key_value_heads must be divisible by all TP degrees you test
  (e.g., 8 for TP=4 and TP=8)
- num_local_experts must be divisible by all TP degrees
- num_hidden_layers=1 keeps it fast while exercising the full forward path
- Keep hidden_size and intermediate_size small but non-trivial (512 works well)

Adapt placeholders (<model>, <Model>, etc.) to your model.
"""
import gc
import os
import sys
import tempfile
import traceback

import torch
import torch.multiprocessing as mp

# Adjust these paths for your container layout
sys.path.insert(0, "/root/data-for-equiv-check")

# --- Tiny model config (adapt to your architecture) ---
TINY_CONFIG = dict(
    hidden_size=512,
    intermediate_size=512,
    num_hidden_layers=1,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=32,
    num_local_experts=8,
    num_experts_per_tok=2,
    vocab_size=256,
    rms_norm_eps=1e-5,
    rope_theta=150000.0,
    max_position_embeddings=131072,
    # ... other model-specific config fields ...
)
SEQ_LEN = 16
SEED = 42


def apply_all_patches():
    """Import and apply all patches. Adapt to your model's patches."""
    from rmsnorm_patch import apply_rmsnorm_patch
    from yarn_rotary_patch import apply_yarn_rotary_patch
    # ... all patches ...
    apply_rmsnorm_patch()
    apply_yarn_rotary_patch()


def build_config(tp_degree, dtype):
    from neuronx_distributed_inference.models.config import NeuronConfig
    from modeling_model import ModelInferenceConfig  # noqa: adapt

    neuron_config = NeuronConfig(
        tp_degree=tp_degree, world_size=tp_degree, batch_size=1,
        seq_len=SEQ_LEN, torch_dtype=dtype, save_sharded_checkpoint=True,
        enable_cte_modular_flow=True,
    )
    config = ModelInferenceConfig(neuron_config=neuron_config, **TINY_CONFIG)
    config.add_derived_config()
    return config


def create_fake_hf_weights(tmpdir):
    """Generate random HF-format safetensors with correct key names.

    Keys must match what convert_hf_to_neuron_state_dict expects.
    Adapt the key format to your model's HF checkpoint structure.
    """
    from safetensors.torch import save_file

    H = TINY_CONFIG["hidden_size"]
    I = TINY_CONFIG["intermediate_size"]
    E = TINY_CONFIG["num_local_experts"]
    nH = TINY_CONFIG["num_attention_heads"]
    nKV = TINY_CONFIG["num_key_value_heads"]
    hd = TINY_CONFIG["head_dim"]
    V = TINY_CONFIG["vocab_size"]

    torch.manual_seed(SEED)
    sd = {}
    sd["model.embed_tokens.weight"] = torch.randn(V, H)
    sd["model.norm.weight"] = torch.randn(H)
    sd["model.lm_head.weight"] = torch.randn(V, H)

    for i in range(TINY_CONFIG["num_hidden_layers"]):
        pfx = f"model.layers.{i}"
        q_dim, k_dim, v_dim = nH * hd, nKV * hd, nKV * hd

        # Attention weights — adapt key names to your model
        sd[f"{pfx}.self_attn.qkv_proj.weight"] = torch.randn(q_dim + k_dim + v_dim, H)
        sd[f"{pfx}.self_attn.o_proj.weight"] = torch.randn(H, nH * hd)
        sd[f"{pfx}.self_attn.q_proj.bias"] = torch.randn(q_dim)
        sd[f"{pfx}.self_attn.k_proj.bias"] = torch.randn(k_dim)
        sd[f"{pfx}.self_attn.v_proj.bias"] = torch.randn(v_dim)
        sd[f"{pfx}.self_attn.o_proj.bias"] = torch.randn(H)

        # Layer norms
        sd[f"{pfx}.input_layernorm.weight"] = torch.randn(H)
        sd[f"{pfx}.post_attention_layernorm.weight"] = torch.randn(H)

        # MoE weights — adapt if your model uses dense MLP instead
        sd[f"{pfx}.mlp.experts.gate_up_proj"] = torch.randn(E, H, 2 * I)
        sd[f"{pfx}.mlp.experts.gate_up_proj_bias"] = torch.randn(E, 2 * I)
        sd[f"{pfx}.mlp.experts.down_proj"] = torch.randn(E, I, H)
        sd[f"{pfx}.mlp.experts.down_proj_bias"] = torch.randn(E, H)
        sd[f"{pfx}.mlp.router.weight"] = torch.randn(E, H)

    path = os.path.join(tmpdir, "model.safetensors")
    save_file(sd, path)
    return tmpdir, sd


def load_model(config, dtype, rank, tp_degree, model_path, hf_sd):
    """Load and shard model. Same pipeline as run_inference_cpu.py."""
    from modeling_model import NeuronModelModel, NeuronModelForCausalLM  # noqa: adapt
    from neuronx_distributed.trace.trace import get_sharded_checkpoint
    # from moe_patch import fix_down_proj_layout  # if needed

    inner_model = NeuronModelModel(config)
    if dtype == torch.bfloat16:
        inner_model.bfloat16()
    inner_model.eval()

    model_sd = NeuronModelForCausalLM.get_state_dict(model_path, config)
    # fix_down_proj_layout(model_sd, ...)  # if needed
    get_sharded_checkpoint(model_sd, inner_model, rank, tp_degree)
    inner_model.load_state_dict(model_sd, strict=False)

    # Restore biases (see bias_restoration_template.py)
    # ...

    del model_sd
    gc.collect()
    return inner_model


def manual_forward(inner_model, input_ids, dtype):
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    with torch.no_grad():
        hs = inner_model.embed_tokens(input_ids)
        for layer in inner_model.layers:
            out = layer(hs, attention_mask=causal_mask, position_ids=position_ids,
                        past_key_value=None, use_cache=False)
            hs = out[0]
        hs = inner_model.norm(hs)
        logits = inner_model.lm_head(hs)
    return logits.float().cpu()


# ---------------------------------------------------------------------------
# TP=1 runner
# ---------------------------------------------------------------------------

def run_tp1(dtype, input_ids, model_path, hf_sd):
    from neuronx_distributed.parallel_layers import parallel_state

    os.environ.update({"NXD_CPU_MODE": "1", "WORLD_SIZE": "1",
                        "MASTER_ADDR": "localhost", "MASTER_PORT": "8080", "RANK": "0"})
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.distributed.init_process_group(backend="gloo")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

    config = build_config(1, dtype)
    model = load_model(config, dtype, 0, 1, model_path, hf_sd)
    logits = manual_forward(model, input_ids, dtype)

    del model; gc.collect()
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()
    return logits


# ---------------------------------------------------------------------------
# TP>1 runner (via mp.spawn)
# ---------------------------------------------------------------------------

def _tp_worker(rank, tp_degree, dtype, input_ids, result_dict, model_path, hf_sd):
    from neuronx_distributed.parallel_layers import parallel_state

    os.environ.update({"NXD_CPU_MODE": "1", "RANK": str(rank),
                        "WORLD_SIZE": str(tp_degree),
                        "MASTER_ADDR": "localhost", "MASTER_PORT": "29501"})
    apply_all_patches()  # CRITICAL: re-apply in spawned process

    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=tp_degree)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree)

    try:
        config = build_config(tp_degree, dtype)
        model = load_model(config, dtype, rank, tp_degree, model_path, hf_sd)
        logits = manual_forward(model, input_ids, dtype)
        if rank == 0:
            result_dict["logits"] = logits
            result_dict["success"] = True
        del model; gc.collect()
    except Exception as e:
        traceback.print_exc()
        if rank == 0:
            result_dict["success"] = False
            result_dict["error"] = str(e)
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run_tp_multi(tp_degree, dtype, input_ids, model_path, hf_sd):
    from neuronx_distributed.parallel_layers import parallel_state
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    manager = mp.Manager()
    result_dict = manager.dict()
    mp.spawn(_tp_worker, args=(tp_degree, dtype, input_ids, result_dict, model_path, hf_sd),
             nprocs=tp_degree, join=True)
    if not result_dict.get("success", False):
        raise RuntimeError(f"TP={tp_degree} failed: {result_dict.get('error')}")
    return result_dict["logits"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tp_degrees = [4, 8]  # adapt to your needs
    dtype = torch.bfloat16
    apply_all_patches()

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, TINY_CONFIG["vocab_size"], (1, SEQ_LEN))

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, hf_sd = create_fake_hf_weights(tmpdir)

        print("Running TP=1 reference...")
        ref_logits = run_tp1(dtype, input_ids, model_path, hf_sd)

        for tp in tp_degrees:
            print(f"\nRunning TP={tp}...")
            tgt_logits = run_tp_multi(tp, dtype, input_ids, model_path, hf_sd)

            diff = ref_logits - tgt_logits
            rel_fro = torch.norm(diff) / torch.norm(ref_logits)
            max_abs = diff.abs().max().item()
            status = "PASS" if rel_fro < 1e-2 else "FAIL"
            print(f"  TP=1 vs TP={tp}: rel_fro={rel_fro:.6e}  max_abs={max_abs:.6e}  [{status}]")


if __name__ == "__main__":
    main()
