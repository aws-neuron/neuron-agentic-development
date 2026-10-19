"""Versioned JSON contracts shared by the controller and NAD agents."""

import hashlib
import json
import math
import re
from enum import Enum
from pathlib import Path

CONTRACT_VERSION = "1"


class EvidenceClass(str, Enum):
    PRIMARY_MEASURED = "PRIMARY_MEASURED"
    DERIVED_VERIFIED = "DERIVED_VERIFIED"
    ANALYTIC = "ANALYTIC"
    DIAGNOSTIC = "DIAGNOSTIC"
    PROJECTED = "PROJECTED"
    NARRATIVE = "NARRATIVE"


EVIDENCE_CLASSES = frozenset(member.value for member in EvidenceClass)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SUBMISSION_REQUIRED_FIELDS = (
    "contract_version",
    "campaign_id",
    "task_id",
    "attempt_id",
    "grant_id",
    "task_digest",
    "agent_id",
    "capability",
    "client",
    "status",
    "requested_config",
    "observed_config",
    "payload",
    "artifacts",
    "metrics",
    "claims",
    "next_actions",
    "idempotency_key",
)


class CampaignError(Exception):
    """Base error with a stable CLI category."""

    exit_code = 3


class ContractError(CampaignError):
    exit_code = 2


class NotFoundError(CampaignError):
    exit_code = 4


class ConflictError(CampaignError):
    exit_code = 3


class PolicyError(CampaignError):
    exit_code = 3


class IntegrityError(CampaignError):
    exit_code = 4


def canonical_json(value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"value is not strict JSON: {error}")


def json_digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _object(value, name):
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be a JSON object")
    return value


def _keys(value, name, required, optional=()):
    missing = sorted(set(required) - set(value))
    unknown = sorted(set(value) - set(required) - set(optional))
    if missing:
        raise ContractError(f"{name} is missing: {', '.join(missing)}")
    if unknown:
        raise ContractError(f"{name} has unknown fields: {', '.join(unknown)}")


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value


def _identifier(value, name):
    _string(value, name)
    if not _ID.fullmatch(value):
        raise ContractError(
            f"{name} must match {_ID.pattern} and be at most 64 characters"
        )
    return value


def _version(document, name):
    if document.get("contract_version") != CONTRACT_VERSION:
        raise ContractError(
            f"{name}.contract_version must be {CONTRACT_VERSION!r}"
        )


def _copy(value):
    return json.loads(canonical_json(value))


def validate_campaign_spec(document):
    spec = _object(document, "CampaignSpec")
    _keys(
        spec,
        "CampaignSpec",
        (
            "contract_version",
            "campaign_id",
            "project_root",
            "ask_type",
            "model",
            "hardware",
            "objectives",
            "constraints",
        ),
        ("budget", "same_window_required", "baseline_identity", "optimization",
         "correctness_contract"),
    )
    _version(spec, "CampaignSpec")
    _identifier(spec["campaign_id"], "CampaignSpec.campaign_id")

    root = Path(_string(spec["project_root"], "CampaignSpec.project_root"))
    if not root.is_absolute():
        raise ContractError("CampaignSpec.project_root must be absolute")
    if not root.is_dir():
        raise ContractError("CampaignSpec.project_root must be an existing directory")

    ask_type = spec["ask_type"]
    if ask_type not in ("latency", "throughput", "joint"):
        raise ContractError(
            "CampaignSpec.ask_type must be latency, throughput, or joint"
        )

    model = _object(spec["model"], "CampaignSpec.model")
    _string(model.get("name"), "CampaignSpec.model.name")
    hardware = _object(spec["hardware"], "CampaignSpec.hardware")
    _string(hardware.get("name"), "CampaignSpec.hardware.name")
    cores = hardware.get("cores_per_node")
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise ContractError("CampaignSpec.hardware.cores_per_node must be a positive integer")

    constraints = _object(spec["constraints"], "CampaignSpec.constraints")
    if constraints.get("serving") != "colocated":
        raise ContractError("only colocated serving is supported")
    max_nodes = constraints.get("max_nodes")
    if isinstance(max_nodes, bool) or max_nodes != 1:
        raise ContractError("only single-node campaigns are supported")

    objectives = spec["objectives"]
    if not isinstance(objectives, list) or not objectives:
        raise ContractError("CampaignSpec.objectives must be a non-empty list")
    normalized_objectives = []
    for index, objective in enumerate(objectives):
        name = f"CampaignSpec.objectives[{index}]"
        objective = _object(objective, name)
        _keys(
            objective,
            name,
            ("leg", "metric", "operator", "target", "unit"),
            ("scope",),
        )
        if objective["leg"] not in ("latency", "throughput"):
            raise ContractError(f"{name}.leg must be latency or throughput")
        _string(objective["metric"], f"{name}.metric")
        if objective["operator"] not in ("<=", ">="):
            raise ContractError(f"{name}.operator must be <= or >=")
        if isinstance(objective["target"], bool) or not isinstance(
            objective["target"], (int, float)
        ):
            raise ContractError(f"{name}.target must be numeric")
        if not math.isfinite(objective["target"]):
            raise ContractError(f"{name}.target must be finite")
        _string(objective["unit"], f"{name}.unit")
        if "scope" in objective:
            _string(objective["scope"], f"{name}.scope")
        normalized_objectives.append(_copy(objective))

    legs = [objective["leg"] for objective in normalized_objectives]
    expected = {
        "latency": ["latency"],
        "throughput": ["throughput"],
        "joint": ["latency", "throughput"],
    }[ask_type]
    if sorted(legs) != sorted(expected):
        raise ContractError(
            f"{ask_type} campaigns require objective legs: {', '.join(expected)}"
        )

    same_window = spec.get("same_window_required", False)
    if not isinstance(same_window, bool):
        raise ContractError("CampaignSpec.same_window_required must be boolean")
    if same_window and ask_type != "joint":
        raise ContractError("same_window_required is only valid for joint campaigns")
    budget = _object(spec.get("budget", {}), "CampaignSpec.budget")
    _keys(
        budget,
        "CampaignSpec.budget",
        (),
        ("max_attempts", "max_hardware_attempts", "deadline_at", "closeout_reserve_seconds"),
    )
    for field, value in budget.items():
        if field in ("deadline_at", "closeout_reserve_seconds"):
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"CampaignSpec.budget.{field} must be a non-negative integer")

    from .scope import validate_scope
    from .budget import validate_budget

    validate_scope(spec)
    validate_budget(budget)
    normalized = _copy(spec)
    normalized["project_root"] = str(root.resolve())
    normalized["same_window_required"] = same_window
    normalized["objectives"] = normalized_objectives
    normalized["budget"] = _copy(budget)
    if "optimization" in normalized:
        from .configuration import axis_name

        normalized["optimization"]["allowed_axes"] = [
            axis_name(axis) for axis in normalized["optimization"]["allowed_axes"]
        ]
    return normalized


def validate_action_request(document):
    request = _object(document, "ActionRequest")
    _keys(
        request,
        "ActionRequest",
        (
            "contract_version",
            "campaign_id",
            "state_version",
            "action_id",
            "client",
            "inputs",
            "idempotency_key",
        ),
        ("rationale", "resources"),
    )
    _version(request, "ActionRequest")
    _identifier(request["campaign_id"], "ActionRequest.campaign_id")
    if (
        isinstance(request["state_version"], bool)
        or not isinstance(request["state_version"], int)
        or request["state_version"] < 1
    ):
        raise ContractError("ActionRequest.state_version must be a positive integer")
    _string(request["action_id"], "ActionRequest.action_id")
    if request["client"] not in ("kiro", "claude"):
        raise ContractError("ActionRequest.client must be kiro or claude")
    _object(request["inputs"], "ActionRequest.inputs")
    _identifier(request["idempotency_key"], "ActionRequest.idempotency_key")
    if "rationale" in request:
        _string(request["rationale"], "ActionRequest.rationale")
    if "resources" in request:
        _object(request["resources"], "ActionRequest.resources")
    return _copy(request)


def validate_submission(document):
    submission = _object(document, "AgentSubmission")
    _keys(
        submission,
        "AgentSubmission",
        SUBMISSION_REQUIRED_FIELDS,
        ("failure",),
    )
    _version(submission, "AgentSubmission")
    for field in (
        "campaign_id",
        "task_id",
        "attempt_id",
        "grant_id",
        "agent_id",
        "capability",
        "client",
        "idempotency_key",
    ):
        _identifier(submission[field], f"AgentSubmission.{field}")
    _string(submission["task_digest"], "AgentSubmission.task_digest")
    if submission["status"] not in ("succeeded", "partial", "blocked", "failed"):
        raise ContractError(
            "AgentSubmission.status must be succeeded, partial, blocked, or failed"
        )
    for field in ("requested_config", "observed_config", "payload"):
        _object(submission[field], f"AgentSubmission.{field}")
    for field in ("artifacts", "metrics", "claims", "next_actions"):
        if not isinstance(submission[field], list):
            raise ContractError(f"AgentSubmission.{field} must be a list")
    for action in submission["next_actions"]:
        _identifier(action, "AgentSubmission.next_actions[]")
    if submission["client"] not in ("kiro", "claude"):
        raise ContractError("AgentSubmission.client must be kiro or claude")
    if submission["status"] in ("blocked", "failed"):
        failure = _object(submission.get("failure"), "AgentSubmission.failure")
        _string(failure.get("class"), "AgentSubmission.failure.class")
        _string(failure.get("message"), "AgentSubmission.failure.message")
        _string(failure.get("command"), "AgentSubmission.failure.command")
    elif "failure" in submission:
        failure = _object(submission["failure"], "AgentSubmission.failure")
        for field in ("class", "message", "command"):
            _string(failure.get(field), f"AgentSubmission.failure.{field}")
    return _copy(submission)


def validate_idempotency_key(value):
    return _identifier(value, "idempotency_key")
