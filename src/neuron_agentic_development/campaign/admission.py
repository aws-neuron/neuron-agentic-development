"""Validate a specialist response against its immutable prepared task and grant."""

from datetime import datetime

from .capabilities import validate_submission_payload
from .contracts import ContractError, PolicyError, json_digest
from .evidence import validate_observed_config
from .kernel_evidence import validate_kernel_target
from .workflow_guards import validate_config_sweep
from .correctness import validate_submission_contract
from .experiments import validate_candidates
from .configuration import validate_precision_claim


def validate_prepared_submission(state: dict, submission: dict, now: datetime) -> None:
    if submission["campaign_id"] != state["campaign_id"]:
        raise ContractError("AgentSubmission campaign_id does not match this campaign")
    attempt = state["attempts"].get(submission["task_id"])
    if attempt is None or attempt["status"] != "PREPARED":
        raise PolicyError("submission does not match an active prepared task")
    task = attempt["task"]
    grant = attempt["grant"]
    expected = {
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "grant_id": grant["grant_id"],
        "task_digest": grant["task_digest"],
        "agent_id": task["agent_id"],
        "capability": task["capability"],
        "client": task["client"],
    }
    for field, value in expected.items():
        if submission[field] != value:
            raise ContractError(f"AgentSubmission.{field} does not match the grant")
    if now > datetime.fromisoformat(grant["expires_at"].replace("Z", "+00:00")):
        raise PolicyError("delegation grant has expired; cancel and prepare a new attempt")
    if json_digest(task) != grant["task_digest"]:
        raise PolicyError("stored task no longer matches its immutable digest")

    task_config = task["inputs"].get("config")
    if task_config is not None and submission["requested_config"] != task_config:
        raise ContractError("submission requested_config differs from the prepared task")
    measured = submission["status"] == "succeeded" or any(
        isinstance(metric, dict) and metric.get("supports_sla") is True for metric in submission["metrics"]
    )
    devices = grant["resources"].get("devices")
    if (
        devices is not None
        and measured
        and submission["observed_config"].get("devices") != devices
    ):
        raise ContractError("observed devices do not match the resource grant")
    reservation = grant["resources"].get("external_reservation_id")
    if (
        reservation is not None
        and measured
        and submission["observed_config"].get("external_reservation_id")
        != reservation
    ):
        raise ContractError(
            "observed external_reservation_id does not match the resource grant"
        )

    payload = submission["payload"]
    capability_name = task["capability"]
    if (
        submission["status"] == "succeeded"
        and capability_name == "hardware_entitlement"
        and payload.get("candidate_id") != task["inputs"]["candidate_id"]
    ):
        raise ContractError("entitlement candidate_id differs from the prepared task")
    if (
        measured
        and capability_name == "profile"
        and payload.get("phase") != task["inputs"]["phase"]
    ):
        raise ContractError("profile phase differs from the prepared task")
    reference_fields = {
        "profile_analysis": "profile_attempt_id",
        "equivalence": "source_attempt_id",
        "compiler_debug": "source_attempt_id",
        "nki_triage": "profile_attempt_id",
        "nki_profile_analysis": "profile_attempt_id",
        "nki_cost_analysis": "profile_attempt_id",
        "nki_compute_analysis": "profile_attempt_id",
        "nki_separation_analysis": "profile_attempt_id",
        "nki_roofline_analysis": "profile_attempt_id",
        "nki_dma_order_analysis": "profile_attempt_id",
        "nki_sharding_analysis": "profile_attempt_id",
        "nki_bottleneck_analysis": "profile_attempt_id",
        "nki_gap_analysis": "profile_attempt_id",
        "nki_offload_analysis": "profile_attempt_id",
        "nki_efficiency_analysis": "profile_attempt_id",
        "nki_implementation": "analysis_attempt_id",
    }
    input_reference = reference_fields.get(capability_name)
    if (
        submission["status"] == "succeeded"
        and input_reference
        and payload.get("source_attempt_id") != task["inputs"][input_reference]
    ):
        raise ContractError("payload source_attempt_id differs from the prepared task")

    validate_submission_payload(
        task["capability"], submission, state["spec"]
    )
    validate_precision_claim(task, submission)
    if submission["status"] == "succeeded":
        if capability_name == "profile_analysis":
            validate_config_sweep(state, payload)
            validate_candidates(state, payload)
        if capability_name.startswith("nki_"):
            validate_kernel_target(task["inputs"]["target"], state["spec"]["project_root"])
    if capability_name == "equivalence" and submission["status"] == "succeeded":
        validate_submission_contract(state, task, submission)
        source = next(
            item for item in state["submissions"]
            if item["attempt_id"] == task["inputs"]["source_attempt_id"] and item["_accepted"]
        )
        if source["capability"] == "profile":
            inputs = state["attempts"][source["task_id"]]["task"]["inputs"]
            validate_observed_config(inputs["config"], submission["observed_config"])
            workload = payload.get("workload")
            if not isinstance(workload, dict):
                raise ContractError("profile equivalence requires the observed workload")
            validate_observed_config(inputs["workload"], workload, "equivalence.workload")
