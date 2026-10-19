"""One pure exporter for tracker, pause and completed-run receipts."""

from .configuration import effective_config
from .contracts import canonical_json
from .failure_recovery import recovery_handoffs
from .policy import _profile_equivalence, audit, objective_results


def measured_gains(state: dict) -> list:
    groups = {}
    for item in state["submissions"]:
        if item["capability"] != "profile":
            continue
        task = state["attempts"][item["task_id"]]["task"]
        for metric in item["metrics"]:
            if not metric["supports_sla"]:
                continue
            key = canonical_json({
                "name": metric["name"], "unit": metric["unit"], "scope": metric["scope"],
                "phase": metric["phase"], "workload": task["inputs"]["workload"],
            })
            groups.setdefault(key, []).append((item, metric))
    results = []
    for measurements in groups.values():
        baseline, initial = measurements[0]
        objective = next((item for item in state["spec"]["objectives"]
                          if item["metric"] == initial["name"] and item["unit"] == initial["unit"]), None)
        if objective is None:
            continue
        sign = 1 if objective["operator"] == "<=" else -1
        best, metric = min(measurements, key=lambda pair: sign * pair[1]["value"])
        before, after = effective_config(baseline), effective_config(best)
        changes = {axis: {"baseline": before.get(axis), "best": after.get(axis)}
                   for axis in before.keys() | after.keys() if before.get(axis) != after.get(axis)}
        results.append({
            "metric": initial["name"], "unit": initial["unit"], "scope": initial["scope"],
            "phase": initial["phase"],
            "workload": state["attempts"][baseline["task_id"]]["task"]["inputs"]["workload"],
            "baseline_attempt_id": baseline["attempt_id"], "baseline": initial["value"],
            "best_attempt_id": best["attempt_id"], "best": metric["value"],
            "gain_fraction": sign * (initial["value"] - metric["value"]) / abs(initial["value"])
                             if initial["value"] else None,
            "configuration_changes": changes,
            "baseline_source_identity": state["attempts"][baseline["task_id"]]["task"].get("source_identity"),
            "best_source_identity": state["attempts"][best["task_id"]]["task"].get("source_identity"),
            "correctness": "passed" if _profile_equivalence(state, best["attempt_id"]) else "missing_or_failed",
            "measurement_status": best["status"],
            "change_scope": "capacity" if any(axis in changes for axis in
                                             ("tp", "dp", "max_num_seqs", "concurrency"))
                            else "configuration_or_source",
            "evidence": [initial["artifact_path"], metric["artifact_path"]],
            "comparison": "observed scoped comparison; no causal certification inferred",
        })
    return results


def export_report(state: dict, lifecycle: str, artifact_problems: list,
                  state_version: int | None = None) -> dict:
    result = audit(state, artifact_problems)
    verdict = result["verdict"] or "BLOCKED"
    if verdict == "MET" and not result["ok"]:
        verdict = "BLOCKED"
    objectives = objective_results(state)
    return {
        "contract_version": "1", "campaign_id": state["campaign_id"],
        "state_version": state["state_version"] if state_version is None else state_version,
        "lifecycle": lifecycle, "verdict": verdict, "scientific_verdict": verdict,
        "objectives": objectives, "unresolved_objectives": [
            item for item in objectives if not item["met"] or
            not _profile_equivalence(state, item["profile_attempt_id"])
        ],
        "accepted_results": result["accepted_results"],
        "observed_gains": measured_gains(state), "problems": result["problems"],
        "recovery": recovery_handoffs(state),
        "resumable": verdict != "MET",
        "resume_action": None if verdict == "MET" else
                         "resume this campaign, reconcile existing work, then select a legal action",
        "deadline_at": state["spec"]["budget"].get("deadline_at"),
    }
