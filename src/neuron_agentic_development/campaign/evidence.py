"""Artifact, metric, and claim admission rules."""

import hashlib
import math
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

from .artifacts import Artifact
from .contracts import ContractError, EVIDENCE_CLASSES, IntegrityError
from .measurement_sources import validate_equivalence_reports, validate_primary_metric

def _inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _hash_file(source):
    """Hash a file or seekable binary stream, including on the Python 3.10 build target."""
    context = source.open("rb") if isinstance(source, Path) else nullcontext(source)
    with context as stream:
        size = stream.seek(0, 2)
        stream.seek(0)
        if hasattr(hashlib, "file_digest"):
            digest = hashlib.file_digest(stream, "sha256")
        else:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), size


def _directory_files(path, root, ancestors=frozenset()):
    """Recurse through local symlinks, rejecting escapes and directory cycles."""
    resolved = path.resolve(strict=True)
    if not _inside(resolved, root):
        raise ContractError(f"artifact directory contains an escaped path: {path}")
    if resolved.is_file():
        yield path
    elif resolved.is_dir():
        if resolved in ancestors:
            raise ContractError(f"artifact directory contains a symlink cycle: {path}")
        for item in path.iterdir():
            yield from _directory_files(item, root, ancestors | {resolved})
    else:
        raise ContractError(f"artifact is not a file or directory: {path}")


def _hash_path(path, root):
    if path.is_file():
        return _hash_file(path)
    if not path.is_dir():
        raise ContractError(f"artifact is not a file or directory: {path}")

    files = sorted(_directory_files(path, root))
    if not files:
        raise ContractError(f"artifact directory is empty: {path}")
    manifest = BytesIO()
    total = 0
    for item in files:
        resolved = item.resolve(strict=True)
        if not _inside(resolved, root):
            raise ContractError(f"artifact directory contains an escaped path: {item}")
        file_digest, size = _hash_file(resolved)
        total += size
        # Preserve the manifest bytes used by already-admitted directory artifacts.
        manifest.write(str(item.relative_to(path)).encode("utf-8"))
        manifest.write(file_digest.encode("ascii"))
    digest, _ = _hash_file(manifest)
    return digest, total


def admit_artifacts(artifacts, project_root):
    root = Path(project_root).resolve(strict=True)
    normalized = []
    seen = set()
    for index, artifact in enumerate(artifacts):
        name = f"artifacts[{index}]"
        artifact = Artifact.from_dict(artifact, name)

        raw_path = Path(artifact.path)
        if not raw_path.is_absolute():
            raw_path = root / raw_path
        try:
            path = raw_path.resolve(strict=True)
        except FileNotFoundError:
            raise ContractError(f"artifact does not exist: {raw_path}")
        if not _inside(path, root):
            raise ContractError(f"artifact escapes project_root: {path}")
        canonical = str(path)
        if canonical in seen:
            raise ContractError(f"duplicate artifact path: {canonical}")
        seen.add(canonical)
        digest, size = _hash_path(path, root)
        if size == 0 and artifact.evidence_class in ("PRIMARY_MEASURED", "ANALYTIC"):
            raise ContractError(f"{name} cannot use empty primary or analytical evidence")
        if artifact.type == "device_trace":
            captures = [path] if path.is_file() else _directory_files(path, root)
            if not any(
                capture.is_file() and capture.suffix.lower() in (".ntff", ".parquet")
                and capture.stat().st_size > 0 for capture in captures
            ):
                raise ContractError("device_trace requires a non-empty NTFF or parquet capture")
        normalized.append(artifact.admitted(path, digest, size))
    return normalized


def validate_observed_config(requested, observed, prefix="observed_config"):
    from .configuration import PRECISION, canonical_config

    if prefix == "observed_config":
        requested, observed = canonical_config(requested), canonical_config(observed)
    for key, expected in requested.items():
        if key not in observed:
            raise ContractError(f"{prefix} is missing requested field {key!r}")
        actual = observed[key]
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                raise ContractError(f"{prefix}.{key} has the wrong type")
            validate_observed_config(expected, actual, f"{prefix}.{key}")
        elif actual != expected and not (
            expected == "auto" and key.split(".")[-1] in PRECISION
        ):
            raise ContractError(
                f"{prefix}.{key}={actual!r} does not match requested value {expected!r}"
            )


def admit_metrics(metrics, artifacts, submission):
    artifact_map = {artifact["path"]: artifact for artifact in artifacts}
    normalized = []
    for index, metric in enumerate(metrics):
        name = f"metrics[{index}]"
        if not isinstance(metric, dict):
            raise ContractError(f"{name} must be an object")
        required = {
            "name",
            "value",
            "unit",
            "evidence_class",
            "artifact_path",
            "phase",
            "scope",
            "config_id",
        }
        optional = {"window_id", "supports_sla", "statistic", "samples"}
        missing = required - set(metric)
        unknown = set(metric) - required - optional
        if missing or unknown:
            raise ContractError(
                f"{name} has missing {sorted(missing)} and unknown {sorted(unknown)}"
            )
        value = metric["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"{name}.value must be numeric")
        if not math.isfinite(value):
            raise ContractError(f"{name}.value must be finite")
        for field in ("name", "unit", "phase", "scope", "config_id"):
            if not isinstance(metric[field], str) or not metric[field]:
                raise ContractError(f"{name}.{field} must be a non-empty string")
        evidence_class = metric["evidence_class"]
        if not isinstance(evidence_class, str) or evidence_class not in EVIDENCE_CLASSES:
            raise ContractError(f"{name}.evidence_class is invalid")
        if not isinstance(metric["artifact_path"], str) or not metric["artifact_path"]:
            raise ContractError(f"{name}.artifact_path must be a non-empty string")
        for field in ("window_id", "statistic"):
            if field in metric and (
                not isinstance(metric[field], str) or not metric[field]
            ):
                raise ContractError(f"{name}.{field} must be a non-empty string")
        if "samples" in metric and (
            isinstance(metric["samples"], bool)
            or not isinstance(metric["samples"], int)
            or metric["samples"] < 1
        ):
            raise ContractError(f"{name}.samples must be a positive integer")

        raw_path = Path(metric["artifact_path"])
        if not raw_path.is_absolute():
            raw_path = Path(submission["_project_root"]) / raw_path
        artifact_path = str(raw_path.resolve())
        artifact = artifact_map.get(artifact_path)
        if artifact is None:
            raise ContractError(f"{name} cites an unsubmitted artifact: {artifact_path}")
        supports_sla = metric.get("supports_sla", False)
        if not isinstance(supports_sla, bool):
            raise ContractError(f"{name}.supports_sla must be boolean")
        if supports_sla:
            if evidence_class != "PRIMARY_MEASURED":
                raise ContractError(
                    f"{name} cannot support an SLA with {evidence_class} evidence"
                )
            if artifact["evidence_class"] != "PRIMARY_MEASURED":
                raise ContractError(
                    f"{name} cannot strengthen its source artifact to PRIMARY_MEASURED"
                )
            if submission["capability"] != "profile":
                raise ContractError(f"{name} SLA evidence must come from profiling")
            if not isinstance(submission.get("payload"), dict) or submission["payload"].get("measurement_kind") != "full_model":
                raise ContractError(f"{name} SLA evidence must be a full-model measurement")
            validate_primary_metric(metric, artifact)
        normalized_metric = dict(metric)
        normalized_metric["artifact_path"] = artifact_path
        normalized_metric["supports_sla"] = supports_sla
        normalized.append(normalized_metric)
    return normalized


def admit_claims(claims, artifacts, metrics):
    paths = {artifact["path"] for artifact in artifacts}
    metric_names = {metric["name"] for metric in metrics}
    normalized = []
    for index, claim in enumerate(claims):
        name = f"claims[{index}]"
        if not isinstance(claim, dict):
            raise ContractError(f"{name} must be an object")
        required = {"type", "statement", "artifact_paths", "metric_names"}
        missing = required - set(claim)
        unknown = set(claim) - required
        if missing or unknown:
            raise ContractError(
                f"{name} has missing {sorted(missing)} and unknown {sorted(unknown)}"
            )
        for field in ("type", "statement"):
            if not isinstance(claim[field], str) or not claim[field]:
                raise ContractError(f"{name}.{field} must be a non-empty string")
        for field in ("artifact_paths", "metric_names"):
            if not isinstance(claim[field], list) or any(
                not isinstance(value, str) or not value for value in claim[field]
            ):
                raise ContractError(f"{name}.{field} must be a list of strings")
        cited_paths = set(claim["artifact_paths"])
        cited_metrics = set(claim["metric_names"])
        if not cited_paths.issubset(paths):
            raise ContractError(f"{name} cites an artifact outside this submission")
        if not cited_metrics.issubset(metric_names):
            raise ContractError(f"{name} cites an unknown metric")
        if not cited_paths and not cited_metrics:
            raise ContractError(f"{name} has no evidence")
        normalized.append(dict(claim))
    return normalized


def admit_submission(submission, project_root, task=None):
    from .configuration import validate_effective_config

    admitted = dict(submission)
    admitted["_project_root"] = str(Path(project_root).resolve())
    artifacts = admit_artifacts(submission["artifacts"], project_root)
    metrics = admit_metrics(submission["metrics"], artifacts, admitted)
    claims = admit_claims(submission["claims"], artifacts, metrics)
    if submission["capability"] == "profile" and (
        submission["status"] == "succeeded" or any(metric["supports_sla"] for metric in metrics)
    ):
        validate_observed_config(
            submission["requested_config"], submission["observed_config"]
        )
        config_id = submission["requested_config"]["config_id"]
        if any(metric["config_id"] != config_id for metric in metrics):
            raise ContractError("profile metric config_id differs from requested_config")
        if not any(
            artifact["type"] == "device_trace"
            and artifact["evidence_class"] == "PRIMARY_MEASURED"
            for artifact in artifacts
        ):
            raise ContractError(
                "successful profile requires a PRIMARY_MEASURED device_trace"
            )
        validate_effective_config(submission, artifacts)
    if submission["capability"] == "equivalence" and submission["status"] == "succeeded":
        validate_equivalence_reports(admitted, artifacts, (task or {}).get("source_execution_identity"))
    admitted.pop("_project_root")
    admitted["artifacts"] = artifacts
    admitted["metrics"] = metrics
    admitted["claims"] = claims
    return admitted


def verify_artifacts(submissions, project_root):
    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        return [f"project_root is unavailable: {project_root}: {error}"]
    if not root.is_dir():
        return [f"project_root is not a directory: {project_root}"]
    problems = []
    for submission in submissions:
        for artifact in submission.get("artifacts", []):
            path = Path(artifact["path"])
            try:
                resolved = path.resolve(strict=True)
                if not _inside(resolved, root):
                    problems.append(f"artifact escaped project_root: {path}")
                    continue
                digest, size = _hash_path(resolved, root)
                if digest != artifact["sha256"] or size != artifact["size_bytes"]:
                    problems.append(f"artifact changed after admission: {path}")
            except (OSError, RuntimeError, ContractError) as error:
                problems.append(f"artifact unavailable after admission: {path}: {error}")
    return problems


def require_valid_artifacts(submissions, project_root):
    problems = verify_artifacts(submissions, project_root)
    if problems:
        raise IntegrityError("; ".join(problems))
