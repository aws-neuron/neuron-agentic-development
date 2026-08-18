"""
NxDI (NeuronX Distributed Inference) adapter.

Handles: NeuronConfig, from_pretrained, and compiled-model loading via
scripts/nxdi_compiled_loader.py (public NxDI API only).
"""
import importlib
import importlib.util
import os
import sys
from typing import Any, Dict, List

import torch

from .base import StackAdapter
from . import register


@register("nxdi")
class NxDIAdapter(StackAdapter):
    name = "nxdi"

    def init_distributed(self, tp_degree: int = 1) -> None:
        os.environ["NXD_CPU_MODE"] = "1"
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")

        from neuronx_distributed.parallel_layers.parallel_state import (
            initialize_model_parallel,
            model_parallel_is_initialized,
        )

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
        if not model_parallel_is_initialized():
            initialize_model_parallel(tensor_model_parallel_size=tp_degree)

    def create_model(self, target_module_file, target_class_name, target_config_name,
                     hf_model_path, **kwargs):
        module = _import_module(target_module_file)
        InnerClass = getattr(module, target_class_name)
        ConfigClass = getattr(module, target_config_name)

        from neuronx_distributed_inference.models.config import NeuronConfig
        NeuronConfigCls = NeuronConfig
        if hasattr(ConfigClass, "get_neuron_config_cls"):
            NeuronConfigCls = ConfigClass.get_neuron_config_cls()

        neuron_config = NeuronConfigCls(
            tp_degree=1, world_size=1, batch_size=1, seq_len=128,
            torch_dtype=torch.bfloat16, save_sharded_checkpoint=True, on_cpu=True,
        )

        try:
            config = ConfigClass.from_pretrained(hf_model_path, neuron_config=neuron_config)
        except (TypeError, AttributeError):
            from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
            config = ConfigClass(neuron_config, load_config=load_pretrained_config(hf_model_path))

        if hasattr(config, "add_derived_config"):
            config.add_derived_config()

        return InnerClass(config)

    def load_weights(self, model, hf_model_path, dtype=torch.bfloat16):
        if hasattr(model, "load"):
            model.load(hf_model_path)
        else:
            from safetensors.torch import load_file
            import glob
            state = {}
            for f in sorted(glob.glob(os.path.join(hf_model_path, "*.safetensors"))):
                state.update(load_file(f))
            state = {k: v.to(dtype) for k, v in state.items()}
            model.load_state_dict(state, strict=False)
        return model

    def forward(self, model, input_ids, **kwargs):
        seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]
        attention_mask = kwargs.get("attention_mask", torch.ones_like(input_ids))
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            try:
                out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
            except TypeError:
                try:
                    out = model(input_ids, attention_mask=attention_mask)
                except TypeError:
                    out = model(input_ids)

            logits = out.logits if hasattr(out, "logits") else out
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

        return torch.nan_to_num(logits.float(), nan=0.0, posinf=1e6, neginf=-1e6)

    def device_inference(self, model_id, tp_size, prompts, max_tokens=32, **kwargs):
        compiled_path = kwargs.get("compiled_model_path")
        model_class = kwargs.get("model_class")
        config_class = kwargs.get("config_class")

        try:
            from nxdi_compiled_loader import (
                create_compiled_model,
                ensure_generation_config_version,
                patch_generation_mixin,
            )
        except ImportError:
            # Adapter imported without scripts/ on sys.path — add it and retry.
            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from nxdi_compiled_loader import (
                create_compiled_model,
                ensure_generation_config_version,
                patch_generation_mixin,
            )
        from transformers import GenerationConfig
        import transformers

        model, tokenizer, _ = create_compiled_model(
            model_path=model_id,
            compiled_model_path=compiled_path,
            model_class=model_class,
            config_class=config_class,
            batch_size=1,
            seq_len=128,
        )
        model.load(compiled_path)
        ensure_generation_config_version(model)
        patch_generation_mixin()

        gen_config = GenerationConfig(
            do_sample=False, top_k=1, pad_token_id=tokenizer.pad_token_id,
        )
        gen_config.transformers_version = transformers.__version__

        results = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, generation_config=gen_config, max_new_tokens=max_tokens,
                )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            results.append({"text": text, "tokens": output_ids[0].tolist()})

        if hasattr(model, "reset"):
            model.reset()
        return results

    def cleanup(self):
        from neuronx_distributed.parallel_layers.parallel_state import (
            model_parallel_is_initialized,
            destroy_model_parallel,
        )
        if model_parallel_is_initialized():
            destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _import_module(filepath):
    """Dynamically import a Python module from a file path."""
    module_file = os.path.abspath(filepath)
    module_dir = os.path.dirname(module_file)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    mod_name = f"_target_{os.path.basename(module_file).replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(mod_name, module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
