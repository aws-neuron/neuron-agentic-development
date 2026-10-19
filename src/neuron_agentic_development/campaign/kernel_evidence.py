"""Admit kernel work only for a complete compiler-observed dispatch configuration."""

import json
from pathlib import Path

from .contracts import ContractError
from .evidence import require_valid_artifacts


def check_dispatch_fidelity(kernel_target, fingerprint):
    shape = kernel_target.get("shape") or {}
    configs = fingerprint.get("configs") or []
    if not isinstance(shape, dict) or not isinstance(configs, list) or not configs:
        return "dispatch_fingerprint carries no observed configs or valid target shape"
    if any(
        not isinstance(config, dict) or not isinstance(config.get("kwargs"), dict)
        or not config["kwargs"] for config in configs
    ):
        return "dispatch_fingerprint configs must contain observed launch arguments"
    fields = set().union(*(set(config["kwargs"]) for config in configs))
    if not set(shape) & fields:
        return "config-fidelity: cannot verify; shape declares none of the observed launch args"
    matched, gaps = [], []
    for config in configs:
        kwargs = config["kwargs"]
        missing = [key for key in kwargs if key not in shape]
        wrong = {
            key: {"shape": shape[key], "model": value}
            for key, value in kwargs.items() if key in shape and shape[key] != value
        }
        if not missing and not wrong:
            matched.append(kwargs)
        else:
            gaps.append((len(missing) + len(wrong), missing, wrong))
    if len(matched) == 1:
        return ""
    if len(matched) > 1:
        return "config-fidelity: shape matches multiple distinct dispatch configs"
    _, missing, wrong = min(gaps, key=lambda gap: gap[0])
    return (
        f"config-fidelity: missing launch args {missing}; contradicts on {wrong}. "
        "Derive every launch argument from the captured compiler dispatch."
    )


def validate_kernel_target(target, project_root):
    if not isinstance(target, dict) or not isinstance(target.get("dispatch_fingerprint"), str):
        raise ContractError("kernel target requires a dispatch_fingerprint JSON artifact")
    root = Path(project_root).resolve()
    path = Path(target["dispatch_fingerprint"])
    if not path.is_absolute():
        path = root / path
    try:
        path = path.resolve(strict=True)
        path.relative_to(root)
        fingerprint = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot read kernel dispatch evidence under project_root: {error}") from error
    if not isinstance(fingerprint, dict) or fingerprint.get("source") != "fxgraph.txt":
        raise ContractError("kernel dispatch evidence must come from the compiler fxgraph.txt")
    problem = check_dispatch_fidelity(target, fingerprint)
    if problem:
        raise ContractError(problem)


def validate_profile_target(state: dict, profile: dict, target: dict) -> None:
    """A complete dispatch from another graph is still the wrong kernel target."""
    path = str((Path(state["spec"]["project_root"]) / target["dispatch_fingerprint"]).resolve())
    sources = [profile] + [item for item in state["submissions"] if item["_accepted"]
                          and item["capability"] == "profile_analysis"
                          and item["payload"]["source_attempt_id"] == profile["attempt_id"]]
    artifacts = [artifact for item in sources for artifact in item["artifacts"] if artifact["path"] == path]
    if not artifacts:
        raise ContractError("dispatch_fingerprint must be admitted with the selected profile or its analysis")
    require_valid_artifacts([{"artifacts": artifacts}], state["spec"]["project_root"])
