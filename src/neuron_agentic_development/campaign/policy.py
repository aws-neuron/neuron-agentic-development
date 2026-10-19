"""Pure legal-action, audit, and verdict decisions."""

from .capabilities import NKI_ANALYSES, capability, output_contract, validate_inputs
from .contracts import ContractError, PolicyError, canonical_json
from .failure_recovery import (
    failure_sources, recovery_handoffs, validate_equivalence_retry, validate_failure_source,
)
from .kernel_evidence import validate_kernel_target, validate_profile_target
from .workflow_guards import validate_plan, validate_profile_config
from .budget import remaining_seconds
from .configuration import current_profile, current_profiles, effective_config
from .correctness import validate_request as validate_correctness_request
from .scope import source_write_allowed, validate_scope_request
from .progress import pending_children


def accepted(state, name=None):
    submissions = [
        submission
        for submission in state["submissions"]
        if submission["_accepted"]
        and (name is None or submission["capability"] == name)
    ]
    return sorted(submissions, key=lambda submission: submission["_sequence"])


def latest(state, name):
    matches = accepted(state, name)
    return matches[-1] if matches else None


def _after(submission, reference):
    return submission is not None and (
        reference is None or submission["_sequence"] > reference["_sequence"]
    )


def _action(state, name, reason):
    spec = capability(name)
    return {
        "action_id": f"{name}:{state['state_version']}",
        "capability": name,
        "agent_id": spec["agent_id"],
        "reason": reason,
        "required_inputs": list(spec["required_inputs"]),
        "resources": {
            "hardware": spec["hardware"],
            "source_write": spec["source_write"],
        },
        "output_contract": output_contract(name),
    }


def budget_status(state):
    limits = state["spec"]["budget"]
    attempts = list(state["attempts"].values())
    hardware_attempts = sum(
        attempt["grant"]["resources"]["hardware_enforcement"] != "not_required"
        for attempt in attempts
    )
    used = {"attempts": len(attempts), "hardware_attempts": hardware_attempts}
    exhausted = []
    if used["attempts"] >= limits.get("max_attempts", float("inf")):
        exhausted.append("max_attempts")
    if used["hardware_attempts"] >= limits.get(
        "max_hardware_attempts", float("inf")
    ):
        exhausted.append("max_hardware_attempts")
    remaining = remaining_seconds(limits)
    if remaining is not None and remaining <= 0:
        exhausted.append("deadline_at")
    return {"limits": limits, "used": used, "exhausted": exhausted,
            "remaining_seconds": remaining}


def _legal_actions(state):
    if (
        state["final"]
        or state["paused"]
        or state["pause_after_task"]
        or state["active_attempt"]
        or pending_children(state)
    ):
        return []

    plan = latest(state, "parallelism_plan")
    if plan is None:
        return [_action(state, "parallelism_plan", "no accepted plan exists")]

    entitlement = latest(state, "hardware_entitlement")
    if not _after(entitlement, plan):
        return [
            _action(
                state,
                "hardware_entitlement",
                "the latest plan has not been entitled",
            )
        ]
    if entitlement["payload"]["feasible"] is False:
        return [
            _action(
                state,
                "parallelism_plan",
                "the selected cell misses the objective at its analytical floor",
            )
        ]

    autoport = latest(state, "autoport")
    equivalence = latest(state, "equivalence")
    if _after(autoport, entitlement) and not _after(equivalence, autoport):
        return [
            _action(
                state,
                "equivalence",
                "the latest source port has not passed equivalence",
            )
        ]

    profiles = current_profiles(state)
    if not profiles:
        return [
            _action(state, "profile", "the entitled configuration needs live evidence"),
            _action(
                state,
                "autoport",
                "model readiness may require an implementation before profiling",
            ),
        ]

    profile_ids = {item["attempt_id"] for item in profiles}
    analyses = [item for item in accepted(state, "profile_analysis")
                if item["payload"]["source_attempt_id"] in profile_ids]
    pending = any(not _profile_equivalence(state, ref) for ref in profile_ids)
    actions = [_action(state, "profile", "measure either baseline leg or a bounded treatment"),
               _action(state, "profile_analysis", "analyze a current profile for its config and leg")]
    if pending:
        actions.append(_action(state, "equivalence", "current profiles need matching correctness evidence"))
    if not analyses:
        return actions

    actions.extend([
        _action(state, "parallelism_plan", "evaluate another supported serving cell"),
        _action(state, "autoport", "apply a framework-level source change"),
        _action(state, "compiler_debug", "investigate a compiler or lowering failure"),
    ])
    if not any(item["payload"].get("config_sweep") and
               _profile_equivalence(state, item["payload"]["source_attempt_id"]) for item in analyses):
        return actions
    actions.append(_action(state, "nki_triage", "route the measured kernel bottleneck"))
    triage = latest(state, "nki_triage")
    if triage and triage["payload"]["source_attempt_id"] in profile_ids:
        actions.extend(
            _action(state, name, "analyze the measured kernel target")
            for name in triage["payload"]["routes"]
        )
    if any(item["payload"]["source_attempt_id"] in profile_ids
           for name in NKI_ANALYSES for item in accepted(state, name)):
        actions.append(
            _action(
                state,
                "nki_implementation",
                "implement a finding from an accepted NKI analysis",
            )
        )
    return actions


def legal_actions(state):
    actions = _legal_actions(state)
    if actions and failure_sources(state):
        names = {action["capability"] for action in actions}
        for name, reason in (
            ("compiler_debug", "diagnose a recorded failure before requiring a successful profile"),
            ("parallelism_plan", "reconsider the serving cell after a recorded execution failure"),
        ):
            if name not in names:
                actions.append(_action(state, name, reason))
    if actions and recovery_handoffs(state) and source_write_allowed(state["spec"]):
        if not any(action["capability"] == "autoport" for action in actions):
            actions.append(_action(state, "autoport", "repair the recorded comparison defect"))
    actions = [action for action in actions if not action["resources"]["source_write"]
               or source_write_allowed(state["spec"])]
    budget = budget_status(state)
    if "max_attempts" in budget["exhausted"] or "deadline_at" in budget["exhausted"]:
        return []
    if "max_hardware_attempts" in budget["exhausted"]:
        return [
            action
            for action in actions
            if not capability(action["capability"])["hardware"]
        ]
    return actions


def action_by_id(state, action_id):
    for action in legal_actions(state):
        if action["action_id"] == action_id:
            return action
    raise PolicyError(f"action is not currently legal: {action_id}")


def _accepted_attempt_ids(state, capability_name=None):
    return {
        submission["attempt_id"]
        for submission in accepted(state, capability_name)
    }


def validate_action_request(state, action, request):
    name = action["capability"]
    inputs = request["inputs"]
    validate_inputs(name, inputs)
    validate_scope_request(state["spec"], name, inputs, capability(name)["source_write"])
    if name.startswith("nki_"):
        validate_kernel_target(inputs["target"], state["spec"]["project_root"])

    if name == "hardware_entitlement":
        plan = latest(state, "parallelism_plan")
        candidates = plan["payload"]["candidates"]
        candidate_ids = {
            candidate["id"] if isinstance(candidate, dict) else candidate
            for candidate in candidates
        }
        if inputs["candidate_id"] not in candidate_ids:
            raise ContractError("candidate_id was not returned by the accepted planner")
    elif name == "profile":
        validate_profile_config(
            inputs, latest(state, "parallelism_plan"), latest(state, "hardware_entitlement")
        )
    elif name == "profile_analysis":
        current_profile(state, inputs["profile_attempt_id"])
    elif name in ("nki_triage",) + NKI_ANALYSES:
        profile = current_profile(state, inputs["profile_attempt_id"])
        validate_profile_target(state, profile, inputs["target"])
        if not _profile_equivalence(state, inputs["profile_attempt_id"]):
            raise ContractError("kernel work requires matching correctness")
        if not any(item["payload"].get("config_sweep") and
                   item["payload"]["source_attempt_id"] == inputs["profile_attempt_id"]
                   for item in accepted(state, "profile_analysis")):
            raise ContractError("kernel work requires a measured sweep for the selected profile")
    elif name == "nki_implementation":
        analysis_attempts = set()
        for analysis_name in NKI_ANALYSES:
            analysis_attempts.update(_accepted_attempt_ids(state, analysis_name))
        if inputs["analysis_attempt_id"] not in analysis_attempts:
            raise ContractError(
                "analysis_attempt_id is not an accepted NKI analysis attempt"
            )
        analysis = next(
            item for item in accepted(state)
            if item["attempt_id"] == inputs["analysis_attempt_id"]
        )
        profile = current_profile(state, analysis["payload"]["source_attempt_id"])
        validate_profile_target(state, profile, inputs["target"])
        target = state["attempts"][analysis["task_id"]]["task"]["inputs"]["target"]
        if inputs["target"] != target:
            raise ContractError("NKI implementation must preserve the analyzed kernel target")
    elif name == "compiler_debug":
        if inputs["source_attempt_id"] not in _accepted_attempt_ids(state):
            validate_failure_source(state, inputs)
    elif name == "equivalence":
        if inputs["source_attempt_id"] not in _accepted_attempt_ids(state):
            raise ContractError("source_attempt_id is not an accepted attempt")
        validate_correctness_request(state, inputs)
        validate_equivalence_retry(state, inputs)


def recommendations(state):
    from .experiments import candidate_queue

    actions = legal_actions(state)
    allowed = {action["capability"] for action in actions}
    ranked = [item["capability"] for item in candidate_queue(state, actions) if item["feasible"]]
    recovery = [item["action"] for item in recovery_handoffs(state) if not item["blocked"]]
    suggested = state["submissions"][-1].get("next_actions", []) if state["submissions"] else []
    return list(dict.fromkeys(name for name in ranked + recovery + suggested if name in allowed))


def result_summaries(state, accepted_only=True):
    submissions = accepted(state) if accepted_only else sorted(
        state["submissions"], key=lambda submission: submission["_sequence"]
    )
    summaries = []
    for submission in submissions:
        summaries.append(
            {
                "sequence": submission["_sequence"],
                "accepted": submission["_accepted"],
                "status": submission["status"],
                "capability": submission["capability"],
                "agent_id": submission["agent_id"],
                "task_id": submission["task_id"],
                "attempt_id": submission["attempt_id"],
                "requested_config": submission["requested_config"],
                "observed_config": submission["observed_config"],
                "effective_config": effective_config(submission),
                "inputs": state["attempts"][submission["task_id"]]["task"]["inputs"],
                "payload": submission["payload"],
                "artifacts": [
                    {
                        "path": artifact["path"],
                        "type": artifact["type"],
                        "evidence_class": artifact["evidence_class"],
                        "sha256": artifact["sha256"],
                    }
                    for artifact in submission["artifacts"]
                ],
                "metrics": submission["metrics"],
                "claims": submission["claims"],
                "failure": submission.get("failure"),
                "next_actions": submission["next_actions"],
            }
        )
    return summaries


def _objective_scope(objective):
    if "scope" in objective:
        return objective["scope"]
    return "single_request" if objective["leg"] == "latency" else "node"


def _profile_equivalence(state, attempt_id):
    return any(
        submission["payload"].get("source_attempt_id") == attempt_id
        for submission in accepted(state, "equivalence")
    )


def objective_results(state):
    results = []
    for objective in state["spec"]["objectives"]:
        matches = []
        for submission in current_profiles(state):
            for metric in submission["metrics"]:
                if (
                    metric["supports_sla"]
                    and metric["name"] == objective["metric"]
                    and metric["unit"] == objective["unit"]
                    and metric["scope"] == _objective_scope(objective)
                    and metric["phase"] == ("prefill" if objective["leg"] == "latency" else "decode")
                ):
                    matches.append((submission["_sequence"], metric, submission["attempt_id"]))
        if not matches:
            results.append({"objective": objective, "metric": None, "met": False})
            continue
        _, metric, attempt_id = sorted(matches, key=lambda item: item[0])[-1]
        if objective["operator"] == "<=":
            met = metric["value"] <= objective["target"]
        else:
            met = metric["value"] >= objective["target"]
        results.append({
            "objective": objective, "metric": metric, "met": met,
            "profile_attempt_id": attempt_id,
        })
    return results


def verdict(state):
    objectives = objective_results(state)
    measured = all(result["metric"] is not None for result in objectives)
    if measured:
        if all(result["met"] for result in objectives) and any(
            not _profile_equivalence(state, result["profile_attempt_id"])
            for result in objectives
        ):
            raise PolicyError("MET requires matching equivalence for every selected profile")
        config_ids = {result["metric"]["config_id"] for result in objectives}
        if len(config_ids) != 1:
            raise PolicyError("objective measurements do not use one serving config")
        configs = {canonical_json(effective_config(current_profile(state, result["profile_attempt_id"])))
                   for result in objectives}
        if len(configs) != 1:
            raise PolicyError("objective measurements use different effective serving configurations")
        if state["spec"]["same_window_required"]:
            windows = {result["metric"].get("window_id") for result in objectives}
            if None in windows or len(windows) != 1:
                raise PolicyError(
                    "joint objectives require one measurement window for this campaign"
                )
        return "MET" if all(result["met"] for result in objectives) else "NOT_YET"

    entitlement = latest(state, "hardware_entitlement")
    if entitlement and entitlement["payload"]["feasible"] is False:
        return "ANALYTIC_MISS"
    if state["submissions"] and state["submissions"][-1]["status"] == "blocked":
        return "BLOCKED"
    if budget_status(state)["exhausted"]:
        return "BLOCKED"
    raise PolicyError("campaign has no complete measured or analytical verdict")


def audit(state, artifact_problems):
    problems = list(artifact_problems)
    if state["active_attempt"]:
        problems.append("an attempt is still active")
    plan = latest(state, "parallelism_plan")
    if plan is None:
        problems.append("no accepted parallelism plan")
    else:
        try:
            validate_plan(plan["payload"], state["spec"])
        except ContractError as error:
            problems.append(str(error))
    entitlement = latest(state, "hardware_entitlement")
    if entitlement is None:
        problems.append("no accepted hardware entitlement")

    try:
        current_verdict = verdict(state)
    except PolicyError as error:
        current_verdict = None
        problems.append(str(error))
    return {
        "ok": not problems,
        "problems": problems,
        "verdict": current_verdict,
        "objectives": objective_results(state),
        "accepted_results": len(accepted(state)),
    }
