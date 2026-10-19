"""Reject known contradictions between native reports and submitted measurements."""

import json
import re
from pathlib import Path

from .contracts import ContractError, json_digest


def read_report(artifact: dict) -> dict:
    path = Path(artifact["path"])
    if path.suffix.lower() != ".json":
        return {}
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot read measurement report {path}: {error}") from error
    if not isinstance(report, dict):
        raise ContractError(f"measurement report must be a JSON object: {path}")
    return report


def validate_primary_metric(metric: dict, artifact: dict) -> None:
    """Inter-token latency is not a measurement of aggregate output throughput."""
    if metric["phase"] != "decode":
        return
    report = read_report(artifact)
    decode = report.get("decode_measurement")
    if not isinstance(decode, dict):
        return
    derivation = str(decode.get("decode_throughput_derivation", "")).lower()
    if "itl" in derivation or "inter-token" in derivation:
        raise ContractError(
            "SLA throughput cannot use an inter-token-latency proxy; "
            "cite the concurrent aggregate measurement and its observation window"
        )


def _token_rows(value, length: int, name: str, count: int | None = None) -> list:
    if (not isinstance(value, list) or not value or count is not None and len(value) != count
            or any(not isinstance(row, list) or len(row) != length
                   or any(type(token) is not int or token < 0 for token in row) for row in value)):
        raise ContractError(f"{name} requires non-empty token rows of exactly {length} tokens"
                            + (f" for {count} prompts" if count is not None else ""))
    return value


def validate_reference_outputs(artifact: dict, contract: dict) -> dict:
    """Validate the producer's actual cached output, not its artifact label."""
    reference = read_report(artifact)
    if reference.get("correctness_contract") != contract:
        raise ContractError("reference_outputs contradicts correctness_contract")
    identity = reference.get("reference_execution_identity")
    expected = {
        **contract["reference_identity"], "criterion": contract["criterion"],
        "method": contract["reference_method"], "corpus_digest": contract["corpus_digest"],
        "input_tokens": contract["input_tokens"], "output_tokens": contract["output_tokens"],
    }
    if not isinstance(identity, dict):
        raise ContractError("reference_outputs requires reference_execution_identity")
    for field, value in expected.items():
        actual = identity.get(field)
        if field in ("code_revision", "runtime"):
            actual = actual.get("declared") if isinstance(actual, dict) else None
        if actual != value or field in ("input_tokens", "output_tokens") and type(actual) is not int:
            raise ContractError(f"reference_execution_identity.{field} contradicts the frozen reference")
    for field in ("python", "torch", "transformers"):
        value = identity["runtime"].get(field)
        if not isinstance(value, str) or not value:
            raise ContractError(f"reference runtime requires the observed {field} version")
    for field in ("runner", "cache"):
        value = identity["code_revision"].get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ContractError(f"reference code_revision.{field} requires its actual SHA-256")
    prompts = _token_rows(reference.get("prompt_token_ids"), contract["input_tokens"], "reference prompts")
    if (json_digest(prompts) != contract["reference_identity"]["tokenized_prompts_sha256"]
            or identity.get("prompt_token_ids") != prompts):
        raise ContractError("reference prompt token digest or execution identity differs")
    _token_rows(reference.get("generated_token_ids"), contract["output_tokens"],
                "reference generated_token_ids", len(prompts))
    cache = reference.get("reference_cache")
    if not isinstance(cache, dict) or cache.get("key") != json_digest(identity):
        raise ContractError("reference_cache key does not bind the actual reference execution identity")
    return reference


def _validate_target_report(report: dict, submission: dict, artifacts: list,
                            reference: dict, expected_identity: dict | None) -> None:
    from .configuration import canonical_config

    contract = submission["payload"]["correctness_contract"]
    for field in ("input_tokens", "output_tokens"):
        if type(report.get(field)) is not int or report[field] != contract[field]:
            raise ContractError(f"equivalence report actual {field} differs from the frozen length")
    if not expected_identity or report.get("execution_identity") != expected_identity:
        raise ContractError("equivalence report execution_identity differs from the granted source")
    if canonical_config(report.get("observed_config")) != canonical_config(submission["observed_config"]):
        raise ContractError("equivalence report observed_config differs from the submitted observation")
    if report.get("reference_execution_identity") != reference["reference_execution_identity"]:
        raise ContractError("equivalence report reference_execution_identity differs from the actual reference")
    exact = (contract["criterion"] in ("greedy_token_match", "greedy_token_exact")
             or contract["reference_method"] in ("exact_tokens", "hf_greedy"))
    if exact:
        rows = _token_rows(report.get("target_token_ids"), contract["output_tokens"],
                           "target_token_ids", len(reference["prompt_token_ids"]))
        if rows != reference["generated_token_ids"]:
            raise ContractError("exact-token equivalence report claims success for mismatched target tokens")
    path = report.get("target_results")
    if path is None:
        if "target_results_sha256" in report:
            raise ContractError("target_results_sha256 requires the admitted target_results path")
        return
    if not isinstance(path, str) or not path:
        raise ContractError("target_results must name an admitted artifact")
    path = str((Path(submission.get("_project_root", ".")) / path).resolve())
    artifact = next((item for item in artifacts if item["path"] == path), None)
    if (artifact is None or artifact["evidence_class"] != "PRIMARY_MEASURED"
            or report.get("target_results_sha256") != artifact["sha256"]):
        raise ContractError("target_results must be admitted PRIMARY_MEASURED evidence with a matching SHA-256")
    target = read_report(artifact)
    expected = {
        "prompt_token_ids": reference["prompt_token_ids"],
        "source_attempt_id": submission["payload"]["source_attempt_id"],
        "execution_identity": expected_identity, "observed_config": report["observed_config"],
    }
    if any(target.get(field) != value for field, value in expected.items()):
        raise ContractError("target_results content contradicts the granted execution or observed inputs")
    _token_rows(target.get("prompt_token_ids"), contract["input_tokens"],
                "target_results prompt_token_ids", len(reference["prompt_token_ids"]))
    rows = _token_rows(target.get("generated_token_ids"), contract["output_tokens"],
                       "target_results generated_token_ids", len(reference["prompt_token_ids"]))
    if exact and rows != report.get("target_token_ids"):
        raise ContractError("target_results generated tokens differ from the equivalence report")


def validate_equivalence_reports(submission: dict, artifacts: list,
                                 expected_execution_identity: dict | None = None) -> None:
    contract = submission["payload"].get("correctness_contract")
    if contract is not None:
        key = json_digest(contract)
        references = {item["path"]: validate_reference_outputs(item, contract)
                      for item in artifacts if item["type"] == "reference_outputs"
                      and item.get("metadata", {}).get("cache_key") == key
                      and item["evidence_class"] == "PRIMARY_MEASURED"}
        reports = [item for item in artifacts if item["type"] == "equivalence_report"
                   and item["evidence_class"] == "PRIMARY_MEASURED"]
        if not references or not reports:
            raise ContractError("equivalence requires primary reference_outputs and equivalence_report")
        for artifact in reports:
            report = read_report(artifact)
            expected = {
                "correctness_contract": contract, "reference_cache_key": key,
                "source_attempt_id": submission["payload"]["source_attempt_id"], "passed": True,
            }
            if any(report.get(field) != value for field, value in expected.items()):
                raise ContractError("equivalence report contradicts the frozen criterion or source identity")
            paths = report.get("reference_artifact_paths")
            if not isinstance(paths, list) or not paths or any(
                not isinstance(path, str) or path not in references for path in paths
            ):
                raise ContractError("equivalence report must cite the exact admitted reference outputs")
            for path in paths:
                _validate_target_report(report, submission, artifacts, references[path],
                                        expected_execution_identity)
    workload = submission["payload"].get("workload", {})
    if not isinstance(workload, dict):
        return
    lengths = [
        workload.get("input_tokens", workload.get("input")),
        workload.get("output_tokens", workload.get("output")),
    ]
    if not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in lengths):
        return
    for artifact in artifacts:
        report = read_report(artifact)
        target = report.get("target")
        if not isinstance(target, dict):
            continue
        capacity = target.get("max_model_len")
        if isinstance(capacity, (int, float)) and capacity < sum(lengths):
            raise ContractError(
                f"equivalence target max_model_len={capacity} cannot cover "
                f"the requested {sum(lengths)}-token context"
            )
