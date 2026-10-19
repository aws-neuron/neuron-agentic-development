"""Registry and capability-specific contract checks for existing NAD agents."""

from .contracts import ContractError, EVIDENCE_CLASSES, SUBMISSION_REQUIRED_FIELDS


from .registry import CAPABILITIES, NKI_ANALYSES
from .workflow_guards import validate_plan


PAYLOAD_RULES = {
    "equivalence": {
        "correctness_contract": (
            "Preserve task.correctness_contract exactly when supplied; cite primary "
            "reference_outputs with metadata.cache_key = task.reference_contract.cache_key "
            "and a primary equivalence_report binding that key, source_attempt_id, "
            "reference_artifact_paths, correctness_contract, and passed."
        ),
        "workload": (
            "For a profile source, report every requested workload field as observed. "
            "observed_config must match the source profile configuration; short-context "
            "checks cannot certify a longer requested context."
        ),
    },
    "parallelism_plan": {
        "artifacts": "A successful plan requires at least one ANALYTIC artifact.",
        "objective": "string; must equal task.ask_type",
        "candidates": (
            "non-empty array of objects; every object must have a unique "
            "non-empty string id and positive integer tp/dp fitting hardware.cores_per_node"
        ),
        "selected_configs": (
            "non-empty array of objects; every id must reference candidates[].id"
        ),
        "buckets": "array of evaluated context-length buckets",
        "exclusions": "array of rejected candidates and reasons",
    },
    "hardware_entitlement": {
        "artifacts": "Successful entitlement requires at least one ANALYTIC artifact.",
        "candidate_id": "Must equal task.inputs.candidate_id.",
        "reconciliation_residual": "Finite numeric fraction, absolute value at most 0.01.",
        "feasible": "Boolean, not a string.",
    },
    "profile": {
        "phase": "Must equal task.inputs.phase.",
        "measurement_kind": "full_model, reduced, or isolated; report what actually ran.",
        "samples": "Positive integer for a successful measurement.",
        "failure": (
            "On compilation or startup failure, return blocked/failed with the exact "
            "command, error and diagnostic artifacts. Do not change model dimensions, "
            "checkpoint precision or source code under a profiling grant. Recommend "
            "compiler_debug or parallelism_plan; source changes require a new write grant."
        ),
    },
    "profile_analysis": {
        "candidate_experiments": (
            "Optional structured queue: id, capability, inputs, axis, evidence_attempt_ids, "
            "exposed_cost, predicted_benefit, evidence_quality, estimated_chain_seconds, "
            "falsifier, change_scope, plan {control,treatment,drift}. Cite current full-model "
            "profiles; vary one authorized axis; use capacity scope for admission/replicas. "
            "Benefit and evidence quality are fractions; budget the full experiment chain."
        ),
        "config_sweep": (
            "Before kernel work, include axes (configuration field names, dotted for nested "
            "fields) and profile_attempt_ids from at least two current full-model profiles. "
            "Each named axis must actually vary; include source_attempt_id in that set."
        ),
    },
}


def capability(name):
    if not isinstance(name, str):
        raise ContractError("capability must be a string")
    try:
        return CAPABILITIES[name]
    except KeyError:
        raise ContractError(f"unknown capability: {name}")


def validate_inputs(name, inputs):
    spec = capability(name)
    missing = [field for field in spec["required_inputs"] if field not in inputs]
    if missing:
        raise ContractError(
            f"{name} inputs are missing: {', '.join(sorted(missing))}"
        )
    if any(inputs[field] is None for field in spec["required_inputs"]):
        raise ContractError(f"{name} required inputs cannot be null")
    if name == "profile":
        config = inputs["config"]
        if (
            not isinstance(config, dict)
            or not isinstance(config.get("config_id"), str)
            or not config["config_id"]
        ):
            raise ContractError("profile config requires a string config_id")
        if not isinstance(inputs["workload"], dict):
            raise ContractError("profile workload must be an object")
        if inputs["phase"] not in ("prefill", "decode", "joint"):
            raise ContractError("profile phase must be prefill, decode, or joint")


def output_contract(name):
    spec = capability(name)
    contract = {
        "contract_version": "1",
        "capability": name,
        "required_submission": list(SUBMISSION_REQUIRED_FIELDS),
        "required_payload": list(spec["required_payload"]),
        "artifact_fields": ["path", "type", "evidence_class", "metadata (optional)"],
        "metric_fields": [
            "name",
            "value",
            "unit",
            "evidence_class",
            "artifact_path",
            "phase",
            "scope",
            "config_id",
        ],
        "optional_metric_fields": [
            "window_id",
            "supports_sla",
            "statistic",
            "samples",
        ],
        "claim_fields": ["type", "statement", "artifact_paths", "metric_names"],
        "failure_fields": ["class", "message", "command"],
        "status": ["succeeded", "partial", "blocked", "failed"],
        "failure_required_for": ["blocked", "failed"],
        "allowed_next_actions": list(CAPABILITIES),
        "evidence_classes": sorted(EVIDENCE_CLASSES),
        "field_rules": {
            "identity": "Copy task/grant identity fields exactly; task_digest comes from the grant.",
            "requested_config": "Copy task.inputs.config when supplied; otherwise use an object.",
            "observed_config": "Report observed facts; never copy requested settings as observations.",
            "next_actions": "List capability identifier strings from allowed_next_actions, not objects.",
            "artifacts": "Objects with exactly artifact_fields; metadata must be an object.",
            "claims": "Objects with exactly claim_fields; cite submitted artifacts or metric names.",
            "failure": "Required for blocked/failed; class, message and command are non-empty strings.",
            "succeeded": "Requires the capability payload and evidence, not just a completed tool call.",
            "supports_sla": (
                "True only for PRIMARY_MEASURED full-model profile metrics backed by "
                "PRIMARY_MEASURED artifacts. Planner/entitlement metrics must not certify an SLA."
            ),
        },
    }
    if name in PAYLOAD_RULES:
        contract["payload_rules"] = PAYLOAD_RULES[name]
    if spec["hardware"]:
        contract["hardware_observation"] = (
            "observed_config.external_reservation_id must match the grant"
        )
    if spec["source_write"]:
        contract["source_boundary"] = "mutations must remain in the granted workspace"
    return contract


def validate_submission_payload(name, submission, campaign_spec):
    spec = capability(name)
    if submission["agent_id"] != spec["agent_id"]:
        raise ContractError(
            f"{name} must be returned by {spec['agent_id']}, "
            f"not {submission['agent_id']}"
        )
    unknown_next = sorted(set(submission["next_actions"]) - set(CAPABILITIES))
    if unknown_next:
        raise ContractError(
            f"submission proposes unknown capabilities: {', '.join(unknown_next)}"
        )
    payload = submission["payload"]
    if "experiment_outcome" in payload and payload["experiment_outcome"] not in (
        "positive", "negative", "inconclusive"
    ):
        raise ContractError("experiment_outcome must be positive, negative, or inconclusive")
    measured_profile = name == "profile" and any(
        isinstance(metric, dict) and metric.get("supports_sla") is True
        for metric in submission.get("metrics", [])
    )
    if submission["status"] != "succeeded" and not measured_profile:
        return
    missing = [field for field in spec["required_payload"] if field not in payload]
    if missing:
        raise ContractError(
            f"{name} payload is missing: {', '.join(sorted(missing))}"
        )
    if not submission["artifacts"]:
        raise ContractError(f"{name} succeeded without an artifact")

    if name == "parallelism_plan":
        if payload["objective"] != campaign_spec["ask_type"]:
            raise ContractError("planner objective does not match the campaign ask_type")
        if not payload["candidates"] or not payload["selected_configs"]:
            raise ContractError("planner must return candidates and selected configs")
        candidate_ids = [
            candidate.get("id") if isinstance(candidate, dict) else None
            for candidate in payload["candidates"]
        ]
        if any(not isinstance(value, str) or not value for value in candidate_ids):
            raise ContractError("every planner candidate must have a non-empty id")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ContractError("planner candidate ids must be unique")
        selected_ids = {
            selected.get("id") if isinstance(selected, dict) else None
            for selected in payload["selected_configs"]
        }
        if None in selected_ids or not selected_ids.issubset(set(candidate_ids)):
            raise ContractError("selected configs must reference planner candidate ids")
        validate_plan(payload, campaign_spec)
        if not any(
            artifact.get("evidence_class") == "ANALYTIC"
            for artifact in submission["artifacts"]
        ):
            raise ContractError("planner output requires an ANALYTIC artifact")
    elif name == "hardware_entitlement":
        residual = payload["reconciliation_residual"]
        if isinstance(residual, bool) or not isinstance(residual, (int, float)):
            raise ContractError("entitlement reconciliation_residual must be numeric")
        if abs(residual) > 0.01:
            raise ContractError("entitlement reconciliation residual exceeds 1%")
        if not isinstance(payload["feasible"], bool):
            raise ContractError("entitlement feasible must be boolean")
        if not any(
            artifact.get("evidence_class") == "ANALYTIC"
            for artifact in submission["artifacts"]
        ):
            raise ContractError("entitlement output requires an ANALYTIC artifact")
    elif name == "profile":
        if payload["phase"] not in ("prefill", "decode", "joint"):
            raise ContractError("profile phase must be prefill, decode, or joint")
        if payload["measurement_kind"] not in ("full_model", "reduced", "isolated"):
            raise ContractError(
                "profile measurement_kind must be full_model, reduced, or isolated"
            )
        if not submission["metrics"]:
            raise ContractError("profile succeeded without typed metrics")
        if isinstance(payload["samples"], bool) or not isinstance(payload["samples"], int) or payload["samples"] < 1:
            raise ContractError("profile samples must be a positive integer")
    elif name == "profile_analysis":
        if not payload["queries"] or not payload["bottlenecks"]:
            raise ContractError("profile analysis requires queries and bottlenecks")
    elif name == "equivalence" and payload["passed"] is not True:
        raise ContractError("equivalence cannot succeed unless passed is true")
    elif name == "autoport":
        if not isinstance(payload["mutation_manifest"], list):
            raise ContractError("autoport mutation_manifest must be a list")
        for field in ("compile", "smoke"):
            result = payload[field]
            if not isinstance(result, dict) or result.get("passed") is not True:
                raise ContractError(f"autoport {field} must pass before success")
    elif name == "nki_triage":
        routes = payload["routes"]
        if not isinstance(routes, list) or not routes:
            raise ContractError("NKI triage must return at least one route")
        if len(routes) != len(set(routes)):
            raise ContractError("NKI triage routes must be unique")
        unknown = sorted(set(routes) - set(NKI_ANALYSES))
        if unknown:
            raise ContractError(f"NKI triage returned unknown routes: {', '.join(unknown)}")
    elif name in NKI_ANALYSES:
        if not payload["findings"]:
            raise ContractError(f"{name} succeeded without findings")
    elif name == "nki_implementation":
        correctness = payload["correctness"]
        if not isinstance(correctness, dict) or correctness.get("passed") is not True:
            raise ContractError("NKI implementation requires passing correctness evidence")
        measurement = payload["measurement"]
        required = {
            "baseline_cycles",
            "candidate_cycles",
            "confirmation_cycles",
            "accept_margin",
        }
        if not isinstance(measurement, dict) or not required.issubset(measurement):
            raise ContractError(
                "NKI measurement requires baseline, candidate, confirmation, and margin"
            )
        values = [measurement[field] for field in required]
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            for value in values
        ):
            raise ContractError("NKI measurement values must be positive numbers")
        margin = measurement["accept_margin"]
        if not 0 < margin < 1:
            raise ContractError("NKI accept_margin must be between 0 and 1")
        threshold = measurement["baseline_cycles"] * (1 - margin)
        if (
            measurement["candidate_cycles"] >= threshold
            or measurement["confirmation_cycles"] >= threshold
        ):
            raise ContractError(
                "NKI candidate and confirmation must both clear the noise margin"
            )
