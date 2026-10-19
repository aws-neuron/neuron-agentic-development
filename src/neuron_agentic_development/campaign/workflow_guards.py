"""Shared planning and sweep prerequisites for the single campaign workflow."""

from .contracts import ContractError, canonical_json
from .configuration import axis_name, canonical_config, effective_config, source_boundary, validate_precision


def validate_plan(payload, spec):
    cores = spec["hardware"].get("cores_per_node")
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise ContractError("planner reconciliation requires hardware.cores_per_node")
    for candidate in payload["candidates"]:
        for field in ("tp", "dp"):
            value = candidate.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractError(f"planner candidate {field} must be a positive integer")
        if candidate["tp"] * candidate["dp"] > cores:
            raise ContractError("planner candidate TP x DP exceeds cores_per_node")


def validate_profile_config(inputs, plan, entitlement):
    config = inputs["config"]
    validate_precision(config)
    probe = inputs.get("profile_capabilities", {})
    if not isinstance(probe, dict):
        raise ContractError("profile_capabilities must be an object")
    if probe.get("supported") is False:
        raise ContractError("profiling interface probe failed; repair the adapter before capture")
    candidates = {candidate["id"]: candidate for candidate in plan["payload"]["candidates"]}
    candidate = candidates.get(config["config_id"])
    if candidate is None or candidate["id"] != entitlement["payload"]["candidate_id"]:
        raise ContractError("profile config_id must match the current entitled candidate")
    for field in ("tp", "dp"):
        if canonical_config(config).get(field) != candidate[field]:
            raise ContractError(f"profile {field} must match the entitled candidate")


def _value(config, axis):
    normalized = canonical_config(config)
    axis = axis_name(axis)
    if axis not in normalized:
        raise ContractError(f"swept axis {axis!r} is absent from a referenced profile")
    return canonical_json(normalized[axis])


def validate_config_sweep(state, payload):
    """A sweep names actual configuration variation in accepted profile attempts."""
    sweep = payload.get("config_sweep")
    if sweep is None:
        return
    if not isinstance(sweep, dict):
        raise ContractError("config_sweep must be an object")
    axes = sweep.get("axes")
    references = sweep.get("profile_attempt_ids")
    if not isinstance(axes, list) or not axes or any(
        not isinstance(axis, str) or not axis for axis in axes
    ):
        raise ContractError("config_sweep.axes must name the configuration fields swept")
    if not isinstance(references, list) or any(
        not isinstance(reference, str) for reference in references
    ) or len(set(references)) < 2:
        raise ContractError("config_sweep requires at least two distinct profile attempts")
    accepted = [item for item in state["submissions"] if item["_accepted"]]
    boundary = source_boundary(state)
    profiles = {
        item["attempt_id"]: item for item in accepted
        if item["capability"] == "profile" and item["_sequence"] > boundary
        and item["payload"]["measurement_kind"] == "full_model"
    }
    if not set(references).issubset(profiles):
        raise ContractError("config_sweep must reference current full-model profile attempts")
    if payload["source_attempt_id"] not in references:
        raise ContractError("config_sweep must include the profile being analyzed")
    configs = [effective_config(profiles[reference]) for reference in references]
    for axis in axes:
        if len({_value(config, axis) for config in configs}) < 2:
            raise ContractError(f"config_sweep did not vary {axis!r}")
