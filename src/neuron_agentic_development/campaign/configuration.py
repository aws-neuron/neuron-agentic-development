"""Canonical configuration axes and witnessed runtime defaults."""

from .contracts import ContractError, canonical_json
from .measurement_sources import read_report

ALIASES = {
    "kv_block_size": "block_size",
    "tensor_parallel_size": "tp",
    "data_parallel_size": "dp",
    "context_length_buckets": "context_buckets",
    "ctx_len_buckets": "context_buckets",
    "seq_len_buckets": "sequence_buckets",
    "torch_dtype": "weight_dtype",
    "weight_precision": "weight_dtype",
    "compute_precision": "compute_dtype",
    "kv_precision": "kv_cache_dtype",
}
PRECISION = {"dtype", "weight_dtype", "compute_dtype", "kv_cache_dtype"}
TRANSPORT = {"devices", "external_reservation_id"}


def axis_name(axis: str) -> str:
    if not isinstance(axis, str) or not axis or any(not p for p in axis.split(".")):
        raise ContractError("configuration axes must be non-empty dotted field names")
    return ".".join(ALIASES.get(part, part) for part in axis.split("."))


def canonical_config(config: dict, prefix="") -> dict:
    if not isinstance(config, dict):
        raise ContractError("configuration must be an object")
    result = {}
    for key, value in config.items():
        if not isinstance(key, str):
            raise ContractError("configuration field names must be strings")
        axis = axis_name(prefix + key)
        if isinstance(value, dict) and value:
            entries = canonical_config(value, axis + ".")
        else:
            if axis.split(".")[-1] in PRECISION and isinstance(value, str):
                value = {"bfloat16": "bf16", "float16": "fp16",
                         "float32": "fp32"}.get(value.lower(), value.lower())
            entries = {axis: value}
        for field, observed in entries.items():
            if field in result and canonical_json(result[field]) != canonical_json(observed):
                raise ContractError(f"contradictory aliases for configuration axis {field}")
            result[field] = observed
    return result


def effective_config(submission: dict) -> dict:
    raw = submission["payload"].get("effective_config", submission["observed_config"])
    return {key: value for key, value in canonical_config(raw).items()
            if key not in TRANSPORT}


def validate_effective_config(submission: dict, artifacts: list) -> None:
    raw = submission["payload"].get("effective_config")
    requested = canonical_config(submission["requested_config"])
    observed = canonical_config(submission["observed_config"])
    effective = canonical_config(raw) if raw is not None else observed
    validate_precision(effective)
    for field, value in requested.items():
        if value == "auto" and field.split(".")[-1] in PRECISION:
            continue
        if field in effective and canonical_json(effective[field]) != canonical_json(value):
            raise ContractError(f"effective_config.{field} contradicts requested_config")
    for field, value in observed.items():
        if field in effective and value != "auto" and effective[field] != value:
            raise ContractError(f"effective_config.{field} contradicts observed_config")
    witnessed = {}
    for artifact in artifacts:
        if artifact["type"] != "runtime_config":
            continue
        if artifact["evidence_class"] != "PRIMARY_MEASURED":
            raise ContractError("runtime_config must be PRIMARY_MEASURED")
        report = read_report(artifact)
        if report.get("attempt_id") != submission["attempt_id"]:
            raise ContractError("runtime_config attempt_id differs from this execution")
        for field, value in canonical_config(report.get("effective_config", {})).items():
            if field in witnessed and witnessed[field] != value:
                raise ContractError(f"runtime reports contradict on {field}")
            witnessed[field] = value
    for field, value in effective.items():
        if field in TRANSPORT:
            continue
        if field in witnessed and witnessed[field] != value:
            raise ContractError(f"effective_config.{field} contradicts runtime evidence")
        needs_witness = field not in requested or requested[field] == "auto"
        if needs_witness:
            if value != "auto" and witnessed.get(field) != value:
                raise ContractError(f"effective_config.{field} requires runtime_config evidence")
    if raw is not None and not set(requested).issubset(effective):
        raise ContractError("effective_config must retain all requested configuration axes")


def validate_precision(config: dict) -> None:
    """Crosscheck explicit premises without interpreting a capability as active precision."""
    values = canonical_config(config)
    groups = (
        ("weight_dtype", "checkpoint.weight_dtype", "checkpoint.dtype"),
        ("compute_dtype", "runtime.compute_dtype"),
        ("kv_cache_dtype", "serve.kv_cache_dtype", "runtime.kv_cache_dtype"),
    )
    for fields in groups:
        known = {values[field] for field in fields
                 if field in values and isinstance(values[field], str) and values[field] != "auto"}
        if len(known) > 1:
            raise ContractError(f"contradictory effective precision: {', '.join(fields)}")
    dtype = values.get("dtype")
    compute = values.get("compute_dtype")
    if dtype not in (None, "auto") and compute not in (None, "auto", dtype):
        raise ContractError("effective compute_dtype contradicts the declared dtype")
    for field in ("weight_dtype", "compute_dtype", "kv_cache_dtype"):
        if field in values and (not isinstance(values[field], str) or not values[field]):
            raise ContractError(f"{field} must be a non-empty precision string")


def source_boundary(state: dict) -> int:
    """Even an unsuccessful writer may have changed source; recertify after it."""
    return max(
        (item["_sequence"] for item in state["submissions"]
         if (item["_accepted"] and item["capability"] in
             ("parallelism_plan", "hardware_entitlement"))
         or item["capability"] in ("autoport", "nki_implementation")),
        default=0,
    )


def current_profiles(state: dict) -> list:
    boundary = source_boundary(state)
    profiles = {}
    for item in state["submissions"]:
        if item["_accepted"] and item["capability"] == "profile" and item["_sequence"] > boundary:
            task = state["attempts"][item["task_id"]]["task"]
            key = canonical_json({
                "config": effective_config(item), "phase": item["payload"]["phase"],
                "workload": task["inputs"]["workload"],
                "source_identity": task.get("source_identity"),
            })
            profiles[key] = item
    return list(profiles.values())


def current_profile(state: dict, attempt_id: str) -> dict:
    for item in current_profiles(state):
        if item["attempt_id"] == attempt_id:
            return item
    raise ContractError("profile_attempt_id must identify a current profile for its config and phase")


def execution_identity(state: dict) -> dict:
    changes = [item for item in state["submissions"]
               if item["capability"] in ("autoport", "nki_implementation")]
    return {
        "model": state["spec"]["model"],
        "baseline_revision": state["spec"].get("baseline_identity", {}).get("source_revision"),
        "source_attempt_id": changes[-1]["attempt_id"] if changes else None,
    }


def precision_contract(spec: dict, inputs: dict) -> dict:
    """Unknown/auto remains unknown until a producer supplies runtime evidence."""
    config = canonical_config(inputs.get("config", spec.get("baseline_identity", {}).get("config", {})))
    choices = {
        "weight": ("weight_dtype", "checkpoint.weight_dtype", "checkpoint.dtype"),
        "compute": ("compute_dtype", "runtime.compute_dtype", "dtype"),
        "kv": ("kv_cache_dtype", "serve.kv_cache_dtype", "runtime.kv_cache_dtype"),
    }
    return {kind: next((config[key] for key in keys
                        if isinstance(config.get(key), str) and config[key] != "auto"), None)
            for kind, keys in choices.items()}


def validate_precision_claim(task: dict, submission: dict) -> None:
    claim = submission["payload"].get("effective_precision")
    if claim is None:
        return
    if not isinstance(claim, dict) or set(claim) != {"weight", "compute", "kv"}:
        raise ContractError("effective_precision requires separate weight, compute and kv fields")
    promised = task.get("effective_precision", {})
    observed = precision_contract({}, {"config": submission["payload"].get(
        "effective_config", submission["observed_config"])})
    for key, value in claim.items():
        if not isinstance(value, str) or not value or value == "auto":
            raise ContractError(f"effective_precision.{key} must name a resolved precision")
        known = promised.get(key)
        if known is not None and known != value:
            raise ContractError(f"effective_precision.{key} contradicts the prepared precision")
        if known is None and (task["capability"] != "profile" or observed[key] != value):
            raise ContractError(f"effective_precision.{key} requires runtime profile evidence")
