"""Frozen comparison contracts and content-addressed reference reuse."""

from .contracts import ContractError, json_digest
from .configuration import current_profile, effective_config, execution_identity
from .evidence import require_valid_artifacts
from .measurement_sources import validate_reference_outputs


def reference_contract(state: dict) -> dict | None:
    contract = state["spec"].get("correctness_contract")
    if contract is None:
        return None
    key = json_digest(contract)
    reusable = {}
    for submission in state["submissions"]:
        for artifact in submission["artifacts"]:
            if (artifact["type"] == "reference_outputs"
                    and artifact.get("metadata", {}).get("cache_key") == key):
                reusable[artifact["path"]] = artifact
    if reusable:
        require_valid_artifacts([{"artifacts": list(reusable.values())}],
                                state["spec"]["project_root"])
        for artifact in reusable.values():
            validate_reference_outputs(artifact, contract)
    return {**contract, "cache_key": key, "reusable_artifacts": list(reusable.values())}


def source_execution_identity(state: dict, source_attempt_id: str) -> dict:
    source = next(item for item in state["submissions"]
                  if item["_accepted"] and item["attempt_id"] == source_attempt_id)
    require_valid_artifacts([source], state["spec"]["project_root"])
    attempt = state["attempts"][source["task_id"]]
    return {
        "source_attempt_id": source_attempt_id,
        "task_digest": attempt["grant"]["task_digest"],
        "source_identity": attempt["task"].get("source_identity"),
        "effective_config": effective_config(source),
        "artifacts": {item["path"]: item["sha256"] for item in source["artifacts"]},
    }


def validate_request(state: dict, inputs: dict) -> None:
    contract = state["spec"].get("correctness_contract")
    if contract and contract.get("requires_logits"):
        probe = inputs.get("capture_capabilities", {})
        if not isinstance(probe, dict) or probe.get("logits") is not True:
            raise ContractError("the frozen correctness criterion requires a successful logits capture probe")
    source = next(item for item in state["submissions"]
                  if item["attempt_id"] == inputs["source_attempt_id"])
    task = state["attempts"][source["task_id"]]["task"]
    if source["capability"] == "profile":
        current_profile(state, source["attempt_id"])
        if contract:
            workload = task["inputs"]["workload"]
            for name, alias in (("input_tokens", "input"), ("output_tokens", "output")):
                if workload.get(name, workload.get(alias)) != contract[name]:
                    raise ContractError("profile workload differs from the frozen correctness contract")
    if task.get("source_identity") and source["capability"] not in ("autoport", "nki_implementation"):
        if task["source_identity"] != execution_identity(state):
            raise ContractError("equivalence source execution identity is stale")


def validate_submission_contract(state: dict, task: dict, submission: dict) -> None:
    contract = state["spec"].get("correctness_contract")
    if contract and submission["payload"].get("correctness_contract") != contract:
        raise ContractError("equivalence must preserve the frozen correctness_contract")
    if task["capability"] != "equivalence":
        return
    if contract and task.get("source_execution_identity") != source_execution_identity(
        state, task["inputs"]["source_attempt_id"]
    ):
        raise ContractError("equivalence grant does not bind the current source execution identity")
    source = next(item for item in state["submissions"]
                  if item["attempt_id"] == task["inputs"]["source_attempt_id"])
    if source["capability"] == "profile" and "effective_config" in source["payload"]:
        if effective_config(source) != effective_config(submission):
            raise ContractError("equivalence effective_config differs from the source profile")
