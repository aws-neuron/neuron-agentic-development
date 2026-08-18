"""
Load a compiled NxDI model for device inference.

Reconstructs the exact ``NeuronConfig`` a model was compiled with (by reading
``neuron_config.json`` out of the compiled artifact directory), resolves the
port's model/config classes, builds a tokenizer, and instantiates the model.

Everything here is built on public API only — ``neuronx_distributed_inference``,
``transformers``, and ``torch`` — so the equivalence pipeline's device stages
have no dependency outside the Neuron SDK.

Used by ``adapters/nxdi.py::NxDIAdapter.device_inference()``, which is reached
by Stage 1 (``run_stage1.py``). No other stage needs this module.
"""
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoTokenizer, GenerationConfig


# ---------------------------------------------------------------------------
# Compiled-artifact config recovery
# ---------------------------------------------------------------------------

def load_neuron_config_from_compiled(compiled_path: str) -> Dict[str, Any]:
    """Read ``neuron_config.json`` from a compiled model directory.

    Expected at ``compiled_path/neuron_config.json``. Some compilation flows
    nest it in a subdirectory, so fall back to a recursive search and take the
    first match. The JSON may hold the config under a ``"neuron_config"`` key
    or at the top level; both are handled.

    Raises:
        FileNotFoundError: if no ``neuron_config.json`` exists under the path.
    """
    config_path = Path(compiled_path) / "neuron_config.json"
    if not config_path.exists():
        candidates = sorted(Path(compiled_path).rglob("neuron_config.json"))
        if not candidates:
            raise FileNotFoundError(
                f"neuron_config.json not found under {compiled_path}. "
                "Confirm compilation completed successfully."
            )
        config_path = candidates[0]
        print(f"  neuron_config.json not at root, using: {config_path}")

    with open(config_path) as f:
        config_data = json.load(f)

    return config_data.get("neuron_config", config_data)


# ---------------------------------------------------------------------------
# Class resolution
# ---------------------------------------------------------------------------

def import_class_from_file(file_path: str, class_name: str):
    """Import a single class from an arbitrary ``.py`` file.

    Puts the file's parent directory on ``sys.path`` so sibling imports inside
    the port resolve, then loads the module under a synthetic name to avoid
    colliding with anything already imported.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = Path.cwd() / file_path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    module_name = f"_equiv_module_{path.stem}_{class_name}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def import_class(class_path: str):
    """Resolve a class from either ``path/to/file.py:ClassName`` (a standalone
    ported modeling file) or ``package.module.ClassName`` (an installed port).
    """
    if ".py:" in class_path:
        file_path, class_name = class_path.split(":", 1)
        return import_class_from_file(file_path, class_name)
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


# ---------------------------------------------------------------------------
# transformers / NxDI compatibility shims
# ---------------------------------------------------------------------------

def patch_generation_mixin() -> None:
    """Ensure NxDI's ``HuggingFaceGenerationAdapter`` has ``GenerationMixin``
    in its MRO.

    The adapter does not always inherit it, but the device-inference path calls
    ``.generate()``, which lives on ``GenerationMixin``. Append it if absent.
    Idempotent — checks before mutating.
    """
    from transformers import GenerationMixin

    for mod_path in (
        "neuronx_distributed_inference.utils.accuracy",
        "neuronx_distributed_inference.utils.hf_adapter",
    ):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        cls = getattr(mod, "HuggingFaceGenerationAdapter", None)
        if cls is not None and GenerationMixin not in cls.__bases__:
            cls.__bases__ = cls.__bases__ + (GenerationMixin,)


def ensure_generation_config_version(model) -> None:
    """Guarantee every reachable ``generation_config`` has a string
    ``transformers_version``.

    Without this, transformers' internal version comparison raises
    ``TypeError: '<' not supported between instances of 'NoneType' and 'str'``.
    Covers three cases: the attribute is missing, it is a raw dict, or its
    ``transformers_version`` is not a string.
    """
    import transformers

    for obj in (model, getattr(model, "config", None)):
        if obj is None:
            continue
        gen_config = getattr(obj, "generation_config", None)
        if gen_config is None:
            obj.generation_config = GenerationConfig()
            obj.generation_config.transformers_version = transformers.__version__
        elif isinstance(gen_config, dict):
            obj.generation_config = GenerationConfig(**{
                k: v for k, v in gen_config.items() if k != "transformers_version"
            })
            obj.generation_config.transformers_version = transformers.__version__
        elif not isinstance(getattr(gen_config, "transformers_version", None), str):
            gen_config.transformers_version = transformers.__version__


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype_str) -> torch.dtype:
    """Map the ``torch_dtype`` field of neuron_config.json to a torch dtype."""
    if not isinstance(dtype_str, str):
        return dtype_str if dtype_str is not None else torch.bfloat16
    name = dtype_str.split(".")[-1]
    return getattr(torch, name, torch.bfloat16)


def create_compiled_model(
    model_path: str,
    compiled_model_path: str,
    model_class: str,
    config_class: str,
    batch_size: int = 1,
    seq_len: int = 128,
) -> Tuple[Any, Any, GenerationConfig]:
    """Build a compiled NxDI model, its tokenizer, and a greedy GenerationConfig.

    The ``NeuronConfig`` is reconstructed from the compiled directory's
    ``neuron_config.json`` rather than from the caller's arguments, so it always
    matches what the model was actually compiled with. ``batch_size`` and
    ``seq_len`` are only fallbacks for when the file omits them.

    Note: this does **not** call ``model.load()`` — the caller is responsible
    for loading the compiled weights.

    Args:
        model_path:          HF checkpoint directory (source weights + tokenizer).
        compiled_model_path: Directory holding the compiled artifacts.
        model_class:         ``file.py:ClassName`` or dotted path to the ForCausalLM class.
        config_class:        ``file.py:ClassName`` or dotted path to the InferenceConfig class.

    Returns:
        ``(model, tokenizer, generation_config)``
    """
    from neuronx_distributed_inference.models.config import NeuronConfig, MoENeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    ModelClass = import_class(model_class)
    ConfigClass = import_class(config_class)

    compiled_cfg = load_neuron_config_from_compiled(compiled_model_path)

    # MoE models need MoENeuronConfig; detect from the compiled config.
    is_moe = "moe_tp_degree" in compiled_cfg or "router_config" in compiled_cfg
    NeuronConfigCls = MoENeuronConfig if is_moe else NeuronConfig

    kwargs = {
        "tp_degree": compiled_cfg.get("tp_degree", 1),
        "batch_size": compiled_cfg.get("batch_size", batch_size),
        "seq_len": compiled_cfg.get("seq_len", seq_len),
        "torch_dtype": _resolve_dtype(compiled_cfg.get("torch_dtype")),
        "save_sharded_checkpoint": compiled_cfg.get("save_sharded_checkpoint", True),
        "on_cpu": compiled_cfg.get("on_cpu", False),
    }
    for optional in (
        "world_size", "max_context_length", "enable_bucketing",
        "enable_cte_modular_flow", "ep_degree", "moe_ep_degree",
    ):
        if optional in compiled_cfg:
            kwargs[optional] = compiled_cfg[optional]
    kwargs.setdefault("max_context_length", kwargs["seq_len"])

    neuron_config = NeuronConfigCls(**kwargs)

    # Some ports accept from_pretrained(path, neuron_config=...); older ones
    # take (neuron_config, load_config=...).
    try:
        model_config = ConfigClass.from_pretrained(model_path, neuron_config=neuron_config)
    except (TypeError, AttributeError):
        model_config = ConfigClass(
            neuron_config, load_config=load_pretrained_config(model_path),
        )
    if hasattr(model_config, "add_derived_config"):
        model_config.add_derived_config()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="right", trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    try:
        model = ModelClass.from_pretrained(compiled_model_path, config=model_config)
    except (TypeError, AttributeError):
        model = ModelClass(model_path, model_config)

    generation_config = GenerationConfig.from_pretrained(
        model_path, do_sample=False, top_k=1,
    )

    ensure_generation_config_version(model)
    return model, tokenizer, generation_config
