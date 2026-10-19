"""Retire NAD-owned AutoBridge files without deleting user artifacts."""

from pathlib import Path


_SKILL = Path("skills/experimental-neuron-framework-optimization")
RETIRED_FILES = tuple(
    _SKILL / relative
    for relative in (
        "scripts/framework_loop.py",
        "scripts/kernel_loop.py",
        "scripts/STEP6_kernel_loop_integration.md",
        "scripts/example_kernel_loop_job.json",
        "scripts/framework_gate/__init__.py",
        "scripts/framework_gate/common.py",
        "scripts/framework_gate/evidence.py",
        "scripts/framework_gate/outcomes.py",
        "scripts/framework_gate/selftest.py",
        "scripts/framework_gate/stages.py",
        "references/native-campaign.md",
        "scripts/server_lifecycle.py",
        "scripts/server_process.py",
        "scripts/server_readiness.py",
    )
)


def remove_retired_artifacts(destination):
    """Remove only known retired files and their compiled Python copies."""
    root = Path(destination).resolve()
    for relative in RETIRED_FILES:
        path = root / relative
        if not path.parent.resolve().is_relative_to(root):
            raise ValueError(f"retired artifact parent escapes deployment root: {relative}")
        if path.is_file() or path.is_symlink():
            path.unlink()
        if path.suffix != ".py":
            continue
        cache = path.parent / "__pycache__"
        if not cache.resolve().is_relative_to(root):
            raise ValueError(f"retired artifact cache escapes deployment root: {relative}")
        for compiled in cache.glob(f"{path.stem}.*.pyc"):
            if compiled.is_file() or compiled.is_symlink():
                compiled.unlink()
        for suffix in (".pyc", ".pyo"):
            compiled = path.with_suffix(suffix)
            if compiled.is_file() or compiled.is_symlink():
                compiled.unlink()
