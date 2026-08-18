"""
run_inference_cpu.py — Neuron model CPU inference at TP=1 and TP>1.

Template for the main E2E inference script. Adapt placeholders (<model>, <Model>,
<patch1>, etc.) to your model.

Key elements:
1. apply_all_patches() — call before any model instantiation
2. CPU mode init — NXD_CPU_MODE=1, gloo backend
3. Weight loading pipeline — get_state_dict → layout fixes → get_sharded_checkpoint → load_state_dict
4. manual_forward() — bypass the NeuronBase wrapper, call inner model directly
5. _tp_worker() — mp.spawn worker with apply_all_patches() re-applied
"""
import os
import sys
import gc
import argparse
import traceback

import torch
import torch.multiprocessing as mp

# Adjust these paths for your container layout
sys.path.insert(0, "/root/data-for-equiv-check")

MODEL_PATH = "/root/equiv-check-rst/<model>-weights"
LOGITS_DIR = "e2e_logits"


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------

def apply_all_patches():
    """Apply all monkey patches BEFORE model instantiation.

    CRITICAL: Must be called:
    1. In the main process before TP=1 runs
    2. Inside each mp.spawn worker before model instantiation
       (mp.spawn creates new processes that do NOT inherit patches)
    """
    from rmsnorm_patch import apply_rmsnorm_patch          # noqa: adapt
    from yarn_rotary_patch import apply_yarn_rotary_patch  # noqa: adapt
    # ... all patches for your model ...
    apply_rmsnorm_patch()
    apply_yarn_rotary_patch()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# TODO: populate from your model's config.json
MODEL_CONFIG = dict(
    hidden_size=...,
    intermediate_size=...,
    num_hidden_layers=...,
    num_attention_heads=...,
    num_key_value_heads=...,
    head_dim=...,
    vocab_size=...,
)


def build_config(tp_degree, dtype):
    """Construct model config for given TP degree and dtype."""
    from neuronx_distributed_inference.models.config import NeuronConfig
    from modeling_model import ModelInferenceConfig  # noqa: adapt import

    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        world_size=tp_degree,
        batch_size=1,
        seq_len=128,
        torch_dtype=dtype,
        save_sharded_checkpoint=True,
        enable_cte_modular_flow=True,
    )
    config = ModelInferenceConfig(neuron_config=neuron_config, **MODEL_CONFIG)
    config.add_derived_config()
    return config


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_model_inner(config, dtype, rank, tp_degree):
    """Load and shard the Neuron model for a given rank."""
    from modeling_model import NeuronModelModel, NeuronModelForCausalLM  # noqa: adapt
    from neuronx_distributed.trace.trace import get_sharded_checkpoint
    # Import any model-specific layout fixes
    # from moe_patch import fix_down_proj_layout

    inner_model = NeuronModelModel(config)
    if dtype == torch.bfloat16:
        inner_model.bfloat16()
    inner_model.eval()

    # 1. Load and convert HF state dict to Neuron key naming
    model_sd = NeuronModelForCausalLM.get_state_dict(MODEL_PATH, config)

    # 2. Apply model-specific layout fixes BEFORE sharding
    # fix_down_proj_layout(model_sd, config.num_hidden_layers, ...)

    # 3. Shard weights for this rank
    get_sharded_checkpoint(model_sd, inner_model, rank, tp_degree)

    # 4. Load sharded weights
    inner_model.load_state_dict(model_sd, strict=False)

    # 5. Restore biases removed by get_sharded_checkpoint
    #    See templates/bias_restoration_template.py

    del model_sd
    gc.collect()
    return inner_model


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def manual_forward(inner_model, input_ids, dtype):
    """Run forward pass directly on the inner model, bypassing NeuronBase wrapper."""
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(
        batch_size, 1, seq_len, seq_len
    )

    with torch.no_grad():
        hidden_states = inner_model.embed_tokens(input_ids)
        for layer in inner_model.layers:
            layer_output = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
            )
            hidden_states = layer_output[0]
        hidden_states = inner_model.norm(hidden_states)
        logits = inner_model.lm_head(hidden_states)

    # Check for NaN/Inf
    assert not torch.isnan(logits).any(), "NaN in logits!"
    assert not torch.isinf(logits).any(), "Inf in logits!"
    return logits.float().cpu()


# ---------------------------------------------------------------------------
# TP=1
# ---------------------------------------------------------------------------

def run_tp1(dtype, input_ids):
    from neuronx_distributed.parallel_layers import parallel_state

    os.environ.update(
        {
            "NXD_CPU_MODE": "1",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "8080",
            "RANK": "0",
        }
    )
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.distributed.init_process_group(backend="gloo")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

    config = build_config(1, dtype)
    model = load_model_inner(config, dtype, 0, 1)
    logits = manual_forward(model, input_ids, dtype)

    del model
    gc.collect()
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()
    return logits


# ---------------------------------------------------------------------------
# TP>1
# ---------------------------------------------------------------------------

def _tp_worker(rank, tp_degree, dtype, input_ids_shared, result_dict, num_layers):
    """Worker function for TP>1 via mp.spawn."""
    from neuronx_distributed.parallel_layers import parallel_state

    os.environ.update(
        {
            "NXD_CPU_MODE": "1",
            "RANK": str(rank),
            "WORLD_SIZE": str(tp_degree),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29501",  # Different from TP=1's 8080!
        }
    )

    # CRITICAL: Re-apply all patches in each spawned process.
    # mp.spawn uses start_method="spawn" (new processes), so patches
    # from the parent are NOT inherited.
    apply_all_patches()

    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=tp_degree
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_degree
    )

    try:
        config = build_config(tp_degree, dtype)
        model = load_model_inner(config, dtype, rank, tp_degree)
        logits = manual_forward(model, input_ids_shared, dtype)
        if rank == 0:
            result_dict["logits"] = logits
            result_dict["success"] = True
        del model
        gc.collect()
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


def run_tp_multi(tp_degree, dtype, input_ids):
    """Launch TP>1 inference via mp.spawn."""
    from neuronx_distributed.parallel_layers import parallel_state

    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    manager = mp.Manager()
    result_dict = manager.dict()
    mp.spawn(
        _tp_worker,
        args=(tp_degree, dtype, input_ids, result_dict, MODEL_CONFIG["num_hidden_layers"]),
        nprocs=tp_degree,
        join=True,
    )
    if not result_dict.get("success", False):
        raise RuntimeError(f"TP={tp_degree} failed: {result_dict.get('error')}")
    return result_dict["logits"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--dump-logits", action="store_true")
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    apply_all_patches()

    # TODO: tokenize your prompt and create input_ids
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # input_ids = tokenizer("The future of AI is", return_tensors="pt").input_ids

    if args.tp_degree == 1:
        logits = run_tp1(dtype, input_ids)
    else:
        logits = run_tp_multi(args.tp_degree, dtype, input_ids)

    print(f"Logits shape: {logits.shape}")
    print(f"  min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")

    if args.dump_logits:
        os.makedirs(LOGITS_DIR, exist_ok=True)
        tag = f"neuron_logits_{args.dtype}_tp{args.tp_degree}"
        path = os.path.join(LOGITS_DIR, f"{tag}.pt")
        torch.save({"logits": logits, "input_ids": input_ids.cpu()}, path)
        print(f"Saved to {path}")
