"""Validate and rank experiments proposed from this campaign's measured profiles."""

from typing import TypedDict

from .budget import remaining_seconds, seconds
from .capabilities import capability, validate_inputs
from .configuration import axis_name, canonical_config, current_profile, effective_config
from .contracts import ContractError, _identifier, _keys, _object, _string, json_digest
from .scope import source_write_allowed, validate_config_permission


class CandidateExperiment(TypedDict):
    id: str
    capability: str
    inputs: dict
    axis: str
    evidence_attempt_ids: list[str]
    exposed_cost: float
    predicted_benefit: float
    evidence_quality: float
    estimated_chain_seconds: float
    falsifier: str
    change_scope: str
    plan: dict


def validate_candidate(state: dict, candidate: CandidateExperiment) -> str:
    _object(candidate, "candidate_experiment")
    _keys(candidate, "candidate_experiment", CandidateExperiment.__required_keys__)
    _identifier(candidate["id"], "candidate_experiment.id")
    name = candidate["capability"]
    spec = capability(name)
    if name not in ("profile", "parallelism_plan", "autoport", "nki_implementation"):
        raise ContractError("candidate capability must execute a configuration or source experiment")
    _object(candidate["inputs"], "candidate_experiment.inputs")
    validate_inputs(name, candidate["inputs"])
    axis = axis_name(candidate["axis"])
    if spec["source_write"] and not source_write_allowed(state["spec"]):
        raise ContractError("candidate requires forbidden source-write permission")
    for field in ("exposed_cost", "predicted_benefit", "evidence_quality", "estimated_chain_seconds"):
        if seconds(candidate[field], f"candidate.{field}") <= 0:
            raise ContractError(f"candidate.{field} must be positive")
    if candidate["evidence_quality"] > 1 or candidate["predicted_benefit"] > 1:
        raise ContractError("candidate evidence_quality and predicted_benefit must be fractions <= 1")
    _string(candidate["falsifier"], "candidate.falsifier")
    if candidate["change_scope"] not in ("configuration", "capacity", "source"):
        raise ContractError("candidate.change_scope must be configuration, capacity, or source")
    refs = candidate["evidence_attempt_ids"]
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) for ref in refs):
        raise ContractError("candidate requires this campaign's evidence_attempt_ids")
    profiles = [current_profile(state, ref) for ref in refs]
    if any(item["payload"]["measurement_kind"] != "full_model" for item in profiles):
        raise ContractError("candidate evidence must be measured full-model profiles")
    plan = _object(candidate["plan"], "candidate.plan")
    _keys(plan, "candidate.plan", ("control", "treatment", "drift"))
    control, treatment, drift = [canonical_config(plan[arm])
                                 for arm in ("control", "treatment", "drift")]
    for config in plan.values():
        validate_config_permission(state["spec"], config)
    if control != drift:
        raise ContractError("candidate drift must return to the control configuration")
    changed = {key for key in control.keys() | treatment.keys()
               if key != "config_id" and control.get(key) != treatment.get(key)}
    if spec["source_write"]:
        if axis != "source" or changed or candidate["change_scope"] != "source":
            raise ContractError("source experiments must preserve the serving configuration")
    elif changed != {axis}:
        raise ContractError("candidate must vary exactly its declared configuration axis")
    if axis in ("dp", "tp", "max_num_seqs", "concurrency") and candidate["change_scope"] != "capacity":
        raise ContractError("replica/admission experiments must declare capacity change_scope")
    if not any(all(effective_config(item).get(key) == value for key, value in control.items())
               for item in profiles):
        raise ContractError("candidate control must match a witnessed current profile")
    if name == "profile" and canonical_config(candidate["inputs"]["config"]) != treatment:
        raise ContractError("candidate profile inputs must execute its treatment")
    evidence = sorted({artifact["sha256"] for item in profiles for artifact in item["artifacts"]})
    normalized_plan = [{key: value for key, value in arm.items() if key != "config_id"}
                       for arm in (control, treatment, drift)]
    return json_digest({"capability": name, "axis": axis, "plan": normalized_plan, "evidence": evidence})


def validate_candidates(state: dict, payload: dict) -> None:
    candidates = payload.get("candidate_experiments", [])
    if not isinstance(candidates, list):
        raise ContractError("candidate_experiments must be a list")
    ids = set()
    for candidate in candidates:
        validate_candidate(state, candidate)
        if candidate["id"] in ids:
            raise ContractError("candidate_experiments ids must be unique")
        ids.add(candidate["id"])
        if payload["source_attempt_id"] not in candidate["evidence_attempt_ids"]:
            raise ContractError("candidate must cite the profile being analyzed")


def candidate_queue(state: dict, actions: list) -> list:
    legal = {action["capability"]: action["action_id"] for action in actions}
    negative = set()
    for item in state["submissions"]:
        task = state["attempts"][item["task_id"]]["task"]
        if item["status"] in ("failed", "blocked") or item["payload"].get("experiment_outcome") == "negative":
            negative.add(task.get("experiment_fingerprint"))
    candidates = {}
    remaining = remaining_seconds(state["spec"]["budget"])
    for item in state["submissions"]:
        if not item["_accepted"] or item["capability"] != "profile_analysis":
            continue
        for candidate in item["payload"].get("candidate_experiments", []):
            try:
                fingerprint = validate_candidate(state, candidate)
            except ContractError:
                continue  # Stale evidence or a superseded serving cell cannot reopen work.
            if fingerprint in negative:
                continue
            cost = candidate["estimated_chain_seconds"]
            fits = remaining is None or cost <= remaining
            candidates[fingerprint] = {
                **candidate, "fingerprint": fingerprint,
                "action_id": legal.get(candidate["capability"]),
                "feasible": fits and candidate["capability"] in legal,
                "reason": None if fits else "complete chain exceeds deadline minus closeout reserve",
                "rank_score": candidate["exposed_cost"] * candidate["predicted_benefit"]
                              * candidate["evidence_quality"] / cost,
            }
    return sorted(candidates.values(), key=lambda item: (-item["feasible"], -item["rank_score"], item["id"]))


def requested_experiment(state: dict, actions: list, name: str, inputs: dict) -> dict | None:
    if "experiment_id" not in inputs:
        return None
    matches = [item for item in candidate_queue(state, actions) if item["id"] == inputs["experiment_id"]]
    if len(matches) != 1 or not matches[0]["feasible"] or matches[0]["capability"] != name:
        raise ContractError("experiment_id must select one current feasible candidate")
    candidate = matches[0]
    if any(inputs.get(key) != value for key, value in candidate["inputs"].items()):
        raise ContractError("experiment inputs differ from the admitted candidate")
    if inputs.get("estimated_chain_seconds") != candidate["estimated_chain_seconds"]:
        raise ContractError("experiment must preserve its complete chain estimate")
    return candidate
