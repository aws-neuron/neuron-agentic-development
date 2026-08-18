"""
Adapter registry and auto-detection.

Usage:
    adapter = get_adapter("vllm_neuron")
    adapter = get_adapter(auto_detect_from="path/to/modeling_xxx.py")
"""
import re
from typing import Optional

from .base import StackAdapter, ThreeWayResult

_REGISTRY = {}


def register(name: str):
    """Decorator to register an adapter class."""
    def wrapper(cls):
        _REGISTRY[name] = cls
        return cls
    return wrapper


# Import adapters to trigger registration
from . import nxdi as _nxdi  # noqa: F401, E402
from . import vllm_neuron as _vllm  # noqa: F401, E402


def get_adapter(
    target_stack: Optional[str] = None,
    auto_detect_from: Optional[str] = None,
    check_environment: bool = True,
) -> StackAdapter:
    """Get the adapter for a serving stack.

    Args:
        target_stack: Explicit stack name ("nxdi", "vllm_neuron").
        auto_detect_from: Path to target modeling file for auto-detection.
        check_environment: If True (default), run the adapter's environment
            check immediately after construction. This validates pinned
            dependency versions (e.g. vLLM for the vLLM-Neuron adapter) and
            raises ``EnvironmentError`` with an actionable message + early
            exit when the environment is incompatible, instead of crashing
            mid-stage. Set False only for introspection (e.g. listing trees)
            where no stack APIs are exercised.

    Returns:
        Instantiated StackAdapter.

    If neither is provided, defaults to "nxdi" for backward compatibility.
    """
    if target_stack:
        if target_stack not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY.keys()))
            raise ValueError(
                f"No adapter for '{target_stack}'. "
                f"Available: {available}. "
                f"See references/adapter-contract.md to add a new one."
            )
        adapter = _REGISTRY[target_stack]()
    else:
        stack = _detect_stack(auto_detect_from) if auto_detect_from else None
        # Default: NxDI (backward compatible)
        adapter = _REGISTRY[stack]() if stack else _REGISTRY["nxdi"]()

    if check_environment:
        # Fail fast and loud on environment/version skew (clear message +
        # early exit) rather than deep inside a stage. No-op for adapters
        # that don't pin dependency versions (e.g. nxdi).
        adapter.check_environment()

    return adapter


def _detect_stack(filepath: str) -> Optional[str]:
    """Detect serving stack from import statements in a modeling file."""
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except (FileNotFoundError, PermissionError):
        return None

    if re.search(r"from\s+vllm_neuron\b|import\s+vllm_neuron\b", content):
        return "vllm_neuron"
    if re.search(r"from\s+neuronx_distributed_inference\b|import\s+neuronx_distributed_inference\b", content):
        return "nxdi"
    return None


def list_adapters():
    """Return available adapter names."""
    return sorted(_REGISTRY.keys())
