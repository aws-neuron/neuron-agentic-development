"""Transactional application operations for optimization campaigns."""

import uuid
from datetime import datetime, timedelta, timezone

from .admission import validate_prepared_submission
from .capabilities import capability, output_contract
from .contracts import (
    CONTRACT_VERSION,
    ConflictError,
    ContractError,
    PolicyError,
    json_digest,
    validate_action_request,
    validate_campaign_spec,
    validate_idempotency_key,
    validate_submission,
)
from .evidence import admit_submission, require_valid_artifacts, verify_artifacts
from .policy import (
    action_by_id,
    audit as audit_policy,
    budget_status,
    legal_actions,
    recommendations,
    result_summaries,
    validate_action_request as validate_policy_request,
)
from .resources import prepare_resources
from .store import CampaignStore
from .budget import validate_chain
from .configuration import current_profiles, execution_identity, precision_contract
from .correctness import reference_contract, source_execution_identity
from .experiments import candidate_queue, requested_experiment
from .failure_recovery import recovery_handoffs
from .progress import pending_children, reconciliation, validate_cancel, validate_progress
from .reporting import export_report
from .scope import source_write_allowed, task_scope
from .submission_format import envelope, format_facts, validation_errors


def _now():
    return datetime.now(timezone.utc)


def _timestamp(value):
    return value.isoformat().replace("+00:00", "Z")


class CampaignController:
    def __init__(self, store: CampaignStore):
        self.store = store

    @classmethod
    def initialize(cls, document: dict) -> "CampaignController":
        spec = validate_campaign_spec(document)
        return cls(CampaignStore.create(spec))

    @classmethod
    def open(cls, project_root, campaign_id):
        return cls(CampaignStore.open(project_root, campaign_id))

    def _state(self):
        self.store.verify()
        return self.store.state()

    def context(self):
        state = self._state()
        active = state["active_attempt"]
        return {
            "contract_version": CONTRACT_VERSION,
            "campaign_id": state["campaign_id"],
            "state_version": state["state_version"],
            "status": state["status"],
            "spec": state["spec"],
            "budget": budget_status(state),
            "active_task": active["task"] if active else None,
            "reconciliation": reconciliation(active) if active else None,
            "pending_children": pending_children(state),
            "current_profiles": [item["attempt_id"] for item in current_profiles(state)],
            "candidate_experiments": candidate_queue(state, legal_actions(state)),
            "recovery": recovery_handoffs(state),
            "accepted_results": result_summaries(state),
            "attempt_results": result_summaries(state, accepted_only=False),
            "recommended_next_actions": recommendations(state),
            "legal_actions": legal_actions(state),
            "final": state["final"],
            "closeout_receipts": state["closeout_receipts"],
        }

    def status(self):
        state = self._state()
        attempts = {}
        for task_id, attempt in state["attempts"].items():
            attempts[task_id] = {
                "capability": attempt["task"]["capability"],
                "agent_id": attempt["task"]["agent_id"],
                "attempt_id": attempt["task"]["attempt_id"],
                "status": attempt["status"],
                "reconciliation": reconciliation(attempt),
            }
        return {
            "contract_version": CONTRACT_VERSION,
            "campaign_id": state["campaign_id"],
            "state_version": state["state_version"],
            "status": state["status"],
            "attempts": attempts,
            "pending_children": pending_children(state),
            "accepted_results": len(
                [
                    submission
                    for submission in state["submissions"]
                    if submission["_accepted"]
                ]
            ),
            "final": state["final"],
        }

    def _idempotent(self, key, request_digest, kinds):
        event = self.store.event_by_idempotency(key)
        if event is None:
            return None
        if event["kind"] not in kinds:
            raise ConflictError(f"idempotency key {key!r} belongs to another command")
        if event["data"]["request_digest"] != request_digest:
            raise ConflictError(f"idempotency key {key!r} was reused")
        return event

    def prepare(self, document: dict) -> dict:
        request = validate_action_request(document)
        request_digest = json_digest(request)
        existing = self._idempotent(
            request["idempotency_key"], request_digest, {"attempt_prepared"}
        )
        if existing:
            return {
                "contract_version": CONTRACT_VERSION,
                "idempotent": True,
                "state_version": self._state()["state_version"],
                "task": existing["data"]["task"],
                "grant": existing["data"]["grant"],
                "submission_envelope": envelope(existing["data"]["task"], existing["data"]["grant"]),
            }

        state = self._state()
        if request["campaign_id"] != state["campaign_id"]:
            raise ContractError("ActionRequest campaign_id does not match this campaign")
        if request["state_version"] != state["state_version"]:
            raise ConflictError(
                f"stale state_version {request['state_version']}; "
                f"current version is {state['state_version']}"
            )
        if pending_children(state):
            raise PolicyError("reconcile bound native children before preparing another attempt")
        action = action_by_id(state, request["action_id"])
        validate_policy_request(state, action, request)
        validate_chain(state["spec"]["budget"], request["inputs"], _now())
        experiment = requested_experiment(state, legal_actions(state), action["capability"], request["inputs"])
        capability_spec = capability(action["capability"])
        resources = prepare_resources(
            capability_spec, request, state["spec"]["project_root"]
        )

        task_id = uuid.uuid4().hex
        attempt_id = uuid.uuid4().hex
        grant_id = uuid.uuid4().hex
        artifact_root = str(self.store.directory / "attempts" / attempt_id)
        task = {
            "contract_version": CONTRACT_VERSION,
            "campaign_id": state["campaign_id"],
            "state_version": state["state_version"],
            "task_id": task_id,
            "attempt_id": attempt_id,
            "capability": action["capability"],
            "agent_id": action["agent_id"],
            "client": request["client"],
            "project_root": state["spec"]["project_root"],
            "artifact_root": artifact_root,
            "expected_output_path": artifact_root + "/submission.json",
            "source_identity": execution_identity(state),
            "effective_precision": precision_contract(state["spec"], request["inputs"]),
            **task_scope(state["spec"]),
            "ask_type": state["spec"]["ask_type"],
            "objectives": state["spec"]["objectives"],
            "constraints": state["spec"]["constraints"],
            "budget": state["spec"]["budget"],
            "inputs": request["inputs"],
            "accepted_context": result_summaries(state),
            "output_contract": output_contract(action["capability"]),
            "restrictions": {
                "artifacts_must_remain_under": state["spec"]["project_root"],
                "submission_must_preserve_observed_facts": True,
                "source_write_allowed": capability_spec["source_write"] and source_write_allowed(state["spec"]),
                "source_workspace": resources.get("workspace"),
            },
        }
        if experiment:
            task["experiment_fingerprint"] = experiment["fingerprint"]
        if action["capability"] == "equivalence":
            task["reference_contract"] = reference_contract(state)
            task["source_execution_identity"] = source_execution_identity(
                state, request["inputs"]["source_attempt_id"])
        source_id = request["inputs"].get("source_attempt_id")
        if source_id:
            task["source_context"] = next(
                (result for result in result_summaries(state, accepted_only=False)
                 if result["attempt_id"] == source_id),
                None,
            )
        task_digest = json_digest(task)
        issued_at = _now()
        grant = {
            "contract_version": CONTRACT_VERSION,
            "grant_id": grant_id,
            "campaign_id": state["campaign_id"],
            "task_id": task_id,
            "attempt_id": attempt_id,
            "task_digest": task_digest,
            "capability": action["capability"],
            "agent_id": action["agent_id"],
            "client": request["client"],
            "issued_at": _timestamp(issued_at),
            "expires_at": _timestamp(issued_at + timedelta(hours=24)),
            "resources": resources,
        }
        event = self.store.append(
            "attempt_prepared",
            {
                "request_digest": request_digest,
                "task": task,
                "grant": grant,
                "rationale": request.get("rationale", ""),
            },
            expected_version=state["state_version"],
            idempotency_key=request["idempotency_key"],
        )
        return {
            "contract_version": CONTRACT_VERSION,
            "idempotent": False,
            "state_version": event["sequence"],
            "task": task,
            "grant": grant,
            "submission_envelope": envelope(task, grant),
        }

    def submit(self, document: dict) -> dict:
        submission = validate_submission(document)
        request_digest = json_digest(submission)
        existing = self._idempotent(
            submission["idempotency_key"],
            request_digest,
            {"submission_accepted", "submission_recorded"},
        )
        if existing:
            self._finish_pause(existing)
            return {
                "contract_version": CONTRACT_VERSION,
                "idempotent": True,
                "accepted": existing["kind"] == "submission_accepted",
                "status": existing["data"]["submission"]["status"],
                "state_version": self._state()["state_version"],
                "context": self.context(),
            }

        state = self._state()
        validate_prepared_submission(state, submission, _now())
        admitted = admit_submission(submission, state["spec"]["project_root"],
                                    state["attempts"][submission["task_id"]]["task"])
        kind = (
            "submission_accepted"
            if submission["status"] == "succeeded"
            else "submission_recorded"
        )
        event = self.store.append(
            kind,
            {"request_digest": request_digest, "submission": admitted,
             "pause_after_task": state["pause_after_task"]},
            expected_version=state["state_version"],
            idempotency_key=submission["idempotency_key"],
        )
        self._finish_pause(event)
        return {
            "contract_version": CONTRACT_VERSION,
            "idempotent": False,
            "accepted": kind == "submission_accepted",
            "status": submission["status"],
            "state_version": self._state()["state_version"],
            "context": self.context(),
        }

    def _finish_pause(self, event: dict) -> None:
        if not event["data"].get("pause_after_task"):
            return
        key = "closeout-" + event["event_id"]
        state = self._state()
        if not state["paused"] or self.store.event_by_idempotency(key):
            return
        receipt = export_report(state, "PAUSED", self._artifact_problems(state), state["state_version"] + 1)
        self.store.append("run_closed", {"closeout": receipt}, expected_version=state["state_version"],
                          idempotency_key=key)

    def format_submission(self, task_id: str, facts: dict, idempotency_key: str) -> dict:
        validate_idempotency_key(idempotency_key)
        state = self._state()
        attempt = state["attempts"].get(task_id)
        if attempt is None:
            raise ContractError("unknown task_id")
        submission = format_facts(attempt["task"], attempt["grant"], facts, idempotency_key)
        return {"submission": submission, **self.validate_submission(submission)}

    def validate_submission(self, document: dict) -> dict:
        state = self._state()
        errors = validation_errors(state, document, _now())
        return {"valid": not errors, "errors": errors, "state_version": state["state_version"]}

    def progress(self, document: dict) -> dict:
        if not isinstance(document, dict):
            raise ContractError("Progress must be an object")
        validate_idempotency_key(document.get("idempotency_key"))
        digest = json_digest(document)
        if self._idempotent(document["idempotency_key"], digest, {"attempt_progress"}):
            return self.status()
        state = self._state()
        progress = validate_progress(state, document)
        self.store.append("attempt_progress", {"request_digest": digest, "progress": progress},
                          expected_version=state["state_version"],
                          idempotency_key=document["idempotency_key"])
        return self.status()

    def report(self):
        state = self._state()
        return export_report(state, state["status"], self._artifact_problems(state))

    def _artifact_problems(self, state):
        saved = [{"artifacts": attempt.get("progress_artifacts", [])}
                 for attempt in state["attempts"].values()]
        return verify_artifacts(state["submissions"] + saved, state["spec"]["project_root"])

    def pause(self, mode, idempotency_key):
        if mode not in ("after-task", "now"):
            raise ContractError("pause mode must be after-task or now")
        validate_idempotency_key(idempotency_key)
        request_digest = json_digest({"mode": mode})
        if self._idempotent(idempotency_key, request_digest, {"campaign_paused"}):
            return self.context()
        state = self._state()
        if state["final"]:
            raise PolicyError("a finalized campaign cannot be paused")
        if state["paused"] or (state["pause_after_task"] and mode == "after-task"):
            return self.context()
        active = state["active_attempt"]
        effective_mode = mode if active else "now"
        data = {
            "request_digest": request_digest,
            "mode": effective_mode,
            "cancelled_task_id": (
                active["task"]["task_id"]
                if active and effective_mode == "now" and not active.get("progress")
                else None
            ),
            "closeout": export_report(state, "PAUSED", self._artifact_problems(state),
                                      state["state_version"] + 1),
        }
        self.store.append(
            "campaign_paused",
            data,
            expected_version=state["state_version"],
            idempotency_key=idempotency_key,
        )
        return self.context()

    def resume(self, idempotency_key):
        validate_idempotency_key(idempotency_key)
        request_digest = json_digest({"command": "resume"})
        if self._idempotent(idempotency_key, request_digest, {"campaign_resumed"}):
            return self.context()
        state = self._state()
        if state["final"] and not state["final"].get("resumable"):
            raise PolicyError("a finalized campaign cannot be resumed")
        if not state["paused"] and not state["pause_after_task"] and not state["final"]:
            return self.context()
        require_valid_artifacts(state["submissions"], state["spec"]["project_root"])
        for attempt in state["attempts"].values():
            require_valid_artifacts([{"artifacts": attempt.get("progress_artifacts", [])}],
                                    state["spec"]["project_root"])
        self.store.append(
            "campaign_resumed",
            {"request_digest": request_digest},
            expected_version=state["state_version"],
            idempotency_key=idempotency_key,
        )
        return self.context()

    def cancel(self, task_id, idempotency_key):
        validate_idempotency_key(idempotency_key)
        request_digest = json_digest({"task_id": task_id})
        if self._idempotent(idempotency_key, request_digest, {"attempt_cancelled"}):
            return self.context()
        state = self._state()
        active = state["active_attempt"]
        if active is None or active["task"]["task_id"] != task_id:
            raise PolicyError("task is not the active attempt")
        validate_cancel(active)
        self.store.append(
            "attempt_cancelled",
            {"request_digest": request_digest, "task_id": task_id},
            expected_version=state["state_version"],
            idempotency_key=idempotency_key,
        )
        return self.context()

    def audit(self):
        state = self._state()
        problems = self._artifact_problems(state)
        result = audit_policy(state, problems)
        return {
            "contract_version": CONTRACT_VERSION,
            "campaign_id": state["campaign_id"],
            "state_version": state["state_version"],
            **result,
        }

    def finalize(self, idempotency_key):
        validate_idempotency_key(idempotency_key)
        state = self._state()
        request_digest = json_digest({"campaign_id": state["campaign_id"]})
        previous = self._idempotent(idempotency_key, request_digest, {"campaign_finalized"})
        if previous:
            return previous["data"]["final"]
        if state["final"]:
            return state["final"]
        if state["active_attempt"] or pending_children(state):
            raise PolicyError("reconcile the active native child and submit or cancel before finalize")
        final = export_report(state, "COMPLETE", self._artifact_problems(state),
                              state["state_version"] + 1)
        self.store.append(
            "campaign_finalized",
            {"request_digest": request_digest, "final": final},
            expected_version=state["state_version"],
            idempotency_key=idempotency_key,
        )
        return final

    def replay(self):
        self.store.verify()
        state = self.store.state()
        export = self.store.export_jsonl()
        return {
            "contract_version": CONTRACT_VERSION,
            "campaign_id": state["campaign_id"],
            "state_version": state["state_version"],
            "events": len(self.store.events()),
            "status": state["status"],
            "export": export,
            "legal_actions": legal_actions(state),
        }
