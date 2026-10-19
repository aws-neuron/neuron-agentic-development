"""Mechanical envelopes and read-only validation through the existing admission path."""

from copy import deepcopy
from pathlib import Path

from .admission import validate_prepared_submission
from .capabilities import output_contract
from .contracts import CampaignError, ContractError, _keys, _object, validate_submission
from .evidence import admit_artifacts, admit_claims, admit_metrics, admit_submission


def envelope(task: dict, grant: dict) -> dict:
    fields = ("campaign_id", "task_id", "attempt_id", "agent_id", "capability", "client")
    return {
        "contract_version": "1", **{field: task[field] for field in fields},
        "grant_id": grant["grant_id"], "task_digest": grant["task_digest"],
        "requested_config": deepcopy(task["inputs"].get("config", {})),
        "idempotency_key": f"submit-{task['attempt_id']}",
    }


def format_facts(task: dict, grant: dict, facts: dict, key: str) -> dict:
    _object(facts, "facts")
    _keys(facts, "facts", ("status", "observed_config", "payload"),
          ("artifacts", "metrics", "claims", "next_actions", "failure"))
    submission = envelope(task, grant)
    submission.update(deepcopy(facts))
    submission["idempotency_key"] = key
    for name in ("artifacts", "metrics", "claims", "next_actions"):
        submission.setdefault(name, [])
    root = Path(task["project_root"])

    def path(value):
        if not isinstance(value, str):
            return value  # Admission reports malformed paths without guessing.
        return str((root / value).resolve())

    for artifact in submission["artifacts"] if isinstance(submission["artifacts"], list) else []:
        if isinstance(artifact, dict) and "path" in artifact:
            artifact["path"] = path(artifact["path"])
    for metric in submission["metrics"] if isinstance(submission["metrics"], list) else []:
        if isinstance(metric, dict) and "artifact_path" in metric:
            metric["artifact_path"] = path(metric["artifact_path"])
    for claim in submission["claims"] if isinstance(submission["claims"], list) else []:
        if isinstance(claim, dict) and isinstance(claim.get("artifact_paths"), list):
            claim["artifact_paths"] = [path(value) for value in claim["artifact_paths"]]
    return submission


def validation_errors(state: dict, document: dict, now) -> list[str]:
    """Collect independent mechanical failures; never mutate state or specialist facts."""
    errors = []

    def check(function, *args):
        try:
            return function(*args)
        except CampaignError as error:
            errors.append(str(error))
            return None

    submission = check(validate_submission, document)
    if not isinstance(document, dict):
        return errors
    try:
        contract = output_contract(document.get("capability"))
    except (ContractError, TypeError):
        return errors + ["unknown submission capability"]
    if document.get("status") not in contract["status"]:
        errors.append("AgentSubmission.status must be succeeded, partial, blocked, or failed")
    actions = document.get("next_actions")
    if isinstance(actions, list) and any(
        not isinstance(action, str) or action not in contract["allowed_next_actions"] for action in actions
    ):
        errors.append("next_actions must contain allowed capability identifier strings")
    root = state["spec"]["project_root"]
    artifacts = []
    for artifact in document.get("artifacts", []) if isinstance(document.get("artifacts"), list) else []:
        admitted = check(admit_artifacts, [artifact], root)
        artifacts.extend(admitted or [])
    metrics = []
    for metric in document.get("metrics", []) if isinstance(document.get("metrics"), list) else []:
        admitted = check(admit_metrics, [metric], artifacts, {**document, "_project_root": root})
        metrics.extend(admitted or [])
    for claim in document.get("claims", []) if isinstance(document.get("claims"), list) else []:
        check(admit_claims, [claim], artifacts, metrics)
    if submission is not None and not errors:
        check(validate_prepared_submission, state, submission, now)
        if not errors:
            check(admit_submission, submission, root, state["attempts"][submission["task_id"]]["task"])
    return list(dict.fromkeys(errors))
