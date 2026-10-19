"""Select recorded failures that can guide the current campaign's recovery."""

from .contracts import ContractError
from .evidence import require_valid_artifacts
from .configuration import source_boundary
from .scope import source_write_allowed

_RECOVERABLE = {"profile", "autoport", "equivalence", "nki_implementation"}
def failure_sources(state):
    """A diagnostic grant needs current, recorded failure evidence, not a win."""
    boundary = source_boundary(state)
    return {
        result["attempt_id"]: result
        for result in state["submissions"]
        if not result["_accepted"]
        and (result["_sequence"] > boundary or result["capability"] in ("autoport", "nki_implementation")
             and result["_sequence"] == boundary)
        and result["capability"] in _RECOVERABLE
        and result["status"] in ("blocked", "failed", "partial")
        and result.get("failure")
        and result.get("artifacts")
        and not any(later["_accepted"] and later["_sequence"] > result["_sequence"]
                    and later["capability"] == result["capability"]
                    and later["requested_config"] == result["requested_config"]
                    and later["payload"].get("source_attempt_id") == result["payload"].get("source_attempt_id")
                    for later in state["submissions"])
    }


def validate_failure_source(state, inputs):
    source = failure_sources(state).get(inputs["source_attempt_id"])
    if source is None:
        raise ContractError(
            "compiler_debug source_attempt_id must identify an accepted attempt "
            "or a current recorded failure with artifacts"
        )
    if inputs["failure"] != source["failure"]:
        raise ContractError("compiler_debug failure must preserve the recorded failure")
    require_valid_artifacts([source], state["spec"]["project_root"])


def equivalence_failures(state: dict) -> list:
    """Keep failures per source, including partial comparisons and missing evidence."""
    by_source = {}
    for result in state["submissions"]:
        if result["capability"] != "equivalence" or result["_sequence"] <= source_boundary(state):
            continue
        task = state["attempts"][result["task_id"]]["task"]
        source = task["inputs"]["source_attempt_id"]
        by_source[source] = result
    return [item for item in by_source.values() if not item["_accepted"]]


def recovery_handoffs(state: dict) -> list:
    handoffs = []
    for result in equivalence_failures(state):
        failure = result.get("failure") or {}
        kind = result["payload"].get("failure_kind", failure.get("class"))
        if kind not in ("mismatch", "missing_evidence", "unsupported_tooling"):
            kind = "mismatch" if result["payload"].get("passed") is False else "missing_evidence"
        repair = source_write_allowed(state["spec"])
        handoffs.append({
            "attempt_id": result["attempt_id"], "kind": kind,
            "defect": failure.get("message", "comparison evidence is incomplete"),
            "action": "autoport" if kind == "mismatch" and repair else
                      "equivalence" if kind == "missing_evidence" else "compiler_debug",
            "blocked": kind == "mismatch" and not repair,
            "required_permission": "optimization.source_write_allowed" if kind == "mismatch" and not repair else None,
        })
    return handoffs


def validate_equivalence_retry(state: dict, inputs: dict) -> None:
    for result in equivalence_failures(state):
        task = state["attempts"][result["task_id"]]["task"]
        if task["inputs"]["source_attempt_id"] != inputs["source_attempt_id"]:
            continue
        diagnosis = inputs.get("diagnostic_attempt_id")
        if not any(item["_accepted"] and item["capability"] == "compiler_debug"
                   and item["attempt_id"] == diagnosis
                   and item["payload"]["source_attempt_id"] == result["attempt_id"]
                   and item["_sequence"] > result["_sequence"]
                   for item in state["submissions"]):
            raise ContractError(
                "unchanged failed equivalence requires a recorded diagnosis before retry; "
                "repair source under an authorized grant or hand off the missing evidence"
            )
