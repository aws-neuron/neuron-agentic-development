"""Explicit campaign permissions and immutable baseline identity."""

import re

from .configuration import PRECISION, TRANSPORT, axis_name, canonical_config, validate_precision
from .contracts import ContractError, _keys, _object, _string, canonical_json


def validate_scope(spec: dict) -> None:
    baseline = spec.get("baseline_identity")
    optimization = spec.get("optimization")
    if baseline is not None:
        _object(baseline, "baseline_identity")
        _keys(baseline, "baseline_identity", ("model", "config"), ("source_revision",))
        if baseline["model"] != spec["model"]:
            raise ContractError("baseline_identity.model must equal the immutable campaign model")
        canonical_config(baseline["config"])
        validate_precision(baseline["config"])
        if "source_revision" in baseline:
            _string(baseline["source_revision"], "baseline_identity.source_revision")
    if optimization is not None:
        _object(optimization, "optimization")
        _keys(optimization, "optimization", ("mode", "allowed_axes", "source_write_allowed"))
        if baseline is None:
            raise ContractError("explicit optimization permissions require baseline_identity")
        if optimization["mode"] not in ("optimization", "qualification_only"):
            raise ContractError("optimization.mode must be optimization or qualification_only")
        axes = optimization["allowed_axes"]
        if not isinstance(axes, list):
            raise ContractError("optimization.allowed_axes must be a list")
        normalized = [axis_name(axis) for axis in axes]
        if len(set(normalized)) != len(normalized):
            raise ContractError("optimization.allowed_axes contains duplicate axes")
        if any(any(part in ("model", "checkpoint", "revision", "model_name", "model_path")
                   for part in axis.split("."))
               or axis.split(".")[-1] in PRECISION for axis in normalized):
            raise ContractError("model, checkpoint and precision constraints are immutable")
        if not isinstance(optimization["source_write_allowed"], bool):
            raise ContractError("optimization.source_write_allowed must be boolean")
        if optimization["mode"] == "qualification_only" and (
            axes or optimization["source_write_allowed"]
        ):
            raise ContractError("qualification_only cannot authorize optimization changes")
    contract = spec.get("correctness_contract")
    if contract is not None:
        _object(contract, "correctness_contract")
        _keys(contract, "correctness_contract",
              ("criterion", "reference_method", "corpus_digest", "input_tokens", "output_tokens",
               "reference_identity"), ("requires_logits",))
        for field in ("criterion", "reference_method", "corpus_digest"):
            _string(contract[field], f"correctness_contract.{field}")
        for field in ("input_tokens", "output_tokens"):
            value = contract[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractError(f"correctness_contract.{field} must be a positive integer")
        identity = _object(contract["reference_identity"], "reference_identity")
        _keys(identity, "reference_identity",
              ("checkpoint", "tokenized_prompts_sha256", "code_revision", "runtime", "precision"))
        for field in identity:
            _string(identity[field], f"reference_identity.{field}")
        for value, field in ((contract["corpus_digest"], "corpus_digest"),
                             (identity["tokenized_prompts_sha256"], "tokenized_prompts_sha256")):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ContractError(f"correctness_contract.{field} must be a lowercase SHA-256 digest")
        checkpoint = spec["model"].get("checkpoint")
        if checkpoint is not None and identity["checkpoint"] != checkpoint:
            raise ContractError("reference_identity.checkpoint must match the immutable checkpoint")
        if not isinstance(contract.get("requires_logits", False), bool):
            raise ContractError("correctness_contract.requires_logits must be boolean")


def source_write_allowed(spec: dict) -> bool:
    permission = spec.get("optimization")
    # Existing campaigns retain their workspace-grant checks; new permissions are opt-in.
    return permission["source_write_allowed"] if permission is not None else "baseline_identity" not in spec


def validate_config_permission(spec: dict, config: dict) -> None:
    baseline = spec.get("baseline_identity")
    if baseline is None:
        return
    before, after = canonical_config(baseline["config"]), canonical_config(config)
    allowed = {axis_name(axis) for axis in spec.get("optimization", {}).get("allowed_axes", [])}
    for axis in before.keys() | after.keys():
        if axis == "config_id" or axis in TRANSPORT or axis in allowed:
            continue
        if axis not in before or axis not in after or canonical_json(before[axis]) != canonical_json(after[axis]):
            raise ContractError(f"configuration axis {axis!r} is frozen by baseline_identity")


def validate_scope_request(spec: dict, name: str, inputs: dict, writes: bool) -> None:
    if writes and not source_write_allowed(spec):
        raise ContractError("source repair is blocked: optimization.source_write_allowed is false")
    for field in ("baseline_identity", "correctness_contract"):
        if field in inputs and inputs[field] != spec.get(field):
            raise ContractError(f"{field} contradicts the immutable campaign contract")
    if "config" in inputs:
        validate_config_permission(spec, inputs["config"])
    if name == "autoport" and spec.get("baseline_identity"):
        model = inputs["model_config"]
        if not isinstance(model, dict):
            raise ContractError("autoport model_config must preserve the campaign model object")
        if any(model.get(key) != value for key, value in spec["model"].items()):
            raise ContractError("autoport model_config contradicts the immutable model")


def task_scope(spec: dict) -> dict:
    return {
        "baseline_identity": spec.get("baseline_identity"),
        "optimization": spec.get("optimization", {
            "mode": "qualification_only" if "baseline_identity" in spec else "legacy",
            "allowed_axes": [] if "baseline_identity" in spec else None,
            "source_write_allowed": source_write_allowed(spec),
        }),
        "correctness_contract": spec.get("correctness_contract"),
    }
