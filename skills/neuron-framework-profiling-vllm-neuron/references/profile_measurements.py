"""Persist observed results independently of server cleanup."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            name = stream.name
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def precision_record(cfg):
    checkpoint = Path(cfg["MODEL"]) / "config.json"
    metadata = json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
    runtime_path = cfg.get("RUNTIME_CONFIG")
    runtime = json.loads(Path(runtime_path).read_text()) if runtime_path else {}
    if runtime_path and runtime.get("attempt_id") != cfg.get("ATTEMPT_ID"):
        raise ValueError("runtime_config.attempt_id must match this run's attempt_id")
    observed = runtime.get("effective_config", {})
    if not isinstance(observed, dict):
        raise ValueError("runtime_config.effective_config must be an object")
    if runtime_path:
        write_json(Path(cfg["PROFILE_DIR"]) / "runtime_config.json", {
            "attempt_id": cfg.get("ATTEMPT_ID"), "effective_config": observed,
        })
    requested = cfg["SERVE_CONFIG"]
    fields = {"weight": "weight_dtype", "compute": "compute_dtype", "kv": "kv_cache_dtype"}
    precision = {}
    for axis, field in fields.items():
        wanted = requested.get("dtype" if axis == "compute" else field)
        actual = observed.get(field)
        aliases = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
        if actual is not None and not isinstance(actual, str):
            raise ValueError(f"runtime {field} must be a precision name")
        if actual in (None, "auto"):
            actual = None
        if actual and wanted not in (None, "auto") and aliases.get(actual, actual) != aliases.get(wanted, wanted):
            raise ValueError(f"runtime {field} contradicts the requested precision")
        precision[axis] = {"requested": wanted, "observed": aliases.get(actual, actual)}
    return {
        "requested_config": requested,
        "runtime_config": runtime,
        "runtime_evidence": runtime_path,
        "checkpoint_metadata": {
            "path": str(checkpoint) if checkpoint.is_file() else None,
            "dtype": metadata.get("dtype", metadata.get("torch_dtype")),
            "quantization_config": metadata.get("quantization_config"),
        },
        "actual_precision": precision,
    }


def persist_measurement(cfg, result, returncode):
    path = Path(result)
    data = path.read_bytes() if path.is_file() else None
    report = {
        "run_name": cfg["run_name"], "phase": cfg["phase"],
        "attempt_id": cfg.get("ATTEMPT_ID"),
        "returncode": returncode, "result_path": str(path),
        "result_sha256": hashlib.sha256(data).hexdigest() if data else None,
        "result": json.loads(data) if data else None,
        "run_spec": cfg["RUN_SPEC"],
        "runfile": cfg["RUNFILE"],
    }
    write_json(Path(cfg["PROFILE_DIR"]) / "measurement.json", report)
    # Precision evidence is separate so missing runtime data cannot erase the measured arm.
    write_json(Path(cfg["PROFILE_DIR"]) / "precision.json", precision_record(cfg))
    if returncode or not data:
        raise RuntimeError(f"benchmark failed or produced no result: rc={returncode}, path={path}")
