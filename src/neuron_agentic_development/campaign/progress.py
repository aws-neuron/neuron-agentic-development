"""Record native child observations without controlling processes or certifying arms."""

from pathlib import Path

from .contracts import ContractError, PolicyError, _keys, _object, _string, validate_idempotency_key
from .evidence import admit_artifacts, require_valid_artifacts


def validate_progress(state: dict, document: dict) -> dict:
    _object(document, "Progress")
    _keys(document, "Progress", (
        "campaign_id", "task_id", "idempotency_key", "child_id", "expected_output_path", "state"
    ), ("artifacts", "cleanup_error"))
    validate_idempotency_key(document["idempotency_key"])
    for field in ("campaign_id", "task_id", "child_id", "expected_output_path", "state"):
        _string(document[field], f"Progress.{field}")
    if document["campaign_id"] != state["campaign_id"]:
        raise ContractError("progress campaign_id differs from this campaign")
    attempt = state["attempts"].get(document["task_id"])
    if attempt is None:
        raise PolicyError("progress requires an existing attempt")
    if document["state"] not in ("running", "end_turn", "completed", "failed", "unknown"):
        raise ContractError("progress state must be running, end_turn, completed, failed, or unknown")
    if document["expected_output_path"] != attempt["task"].get("expected_output_path"):
        raise ContractError("progress expected_output_path must match the prepared task")
    child = attempt.get("progress")
    if attempt["status"] != "PREPARED" and not child:
        raise PolicyError("a native child must bind before the submission receipt")
    if child and document["child_id"] != child["child_id"]:
        raise ContractError("attempt is already bound to another native child; reconcile it before retrying")
    if "cleanup_error" in document:
        _string(document["cleanup_error"], "Progress.cleanup_error")
    artifacts = document.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ContractError("Progress.artifacts must be a list")
    require_valid_artifacts([{"artifacts": attempt.get("progress_artifacts", [])}],
                            state["spec"]["project_root"])
    return {**document, "artifacts": admit_artifacts(artifacts, state["spec"]["project_root"])}


def validate_cancel(attempt: dict) -> None:
    child = attempt.get("progress")
    if child and (child["state"] in ("running", "end_turn", "unknown")
                  or Path(child["expected_output_path"]).is_file()):
        raise PolicyError("reconcile the native child and admit available output before cancellation")


def reconciliation(attempt: dict) -> dict:
    output = attempt["task"].get("expected_output_path")
    return {
        "task_id": attempt["task"]["task_id"],
        "child": attempt.get("progress"),
        "expected_output_path": output,
        "output_available": Path(output).is_file() if output else False,
        "submission_receipt": attempt.get("submission_sequence"),
        "artifacts": attempt.get("progress_artifacts", []),
        "resume_action": "submit and validate the existing output" if output and Path(output).is_file()
                         else "reconcile the native child before preparing another attempt",
    }


def pending_children(state: dict) -> list:
    return [reconciliation(attempt) for attempt in state["attempts"].values()
            if attempt.get("progress", {}).get("state") in ("running", "end_turn", "unknown")]
