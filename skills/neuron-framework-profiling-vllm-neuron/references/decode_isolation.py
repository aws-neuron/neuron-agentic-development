#!/usr/bin/env python3
"""Recapture decode on a DECODE-ONLY server, so no prefill can reach the trace.

`profile_run.arm_decode` isolates decode by TIMING: it drives the real engine with a
bench and waits for a scheduler line saying the prefill queue drained before it POSTs
/start_profile. That trigger can fail to fire. The engine logs that line on a fixed
stats tick (10 s by default), and `Avg prompt throughput: 0.0` is a property of the
WINDOW rather than of the engine -- it needs a whole tick containing no prefill. At
CONC=1 the bench runs requests back to back, so each tick holds one ISL-token prefill
plus OSL*ITL of decode, and when the decode leg is shorter than the tick no tick is
ever prefill-free. The signature is a prompt throughput of exactly ISL/tick on every
tick, with the `Running: {bs} reqs, Waiting: 0 reqs` half matching every time.

This module is what `profile_run` falls back to when that happens. It serves with
`NeuronDecodeBenchConnector` (`kv_role=kv_consumer`), which reports every prompt's KV
as already present, so vLLM SKIPS PREFILL and sends each request straight to the decode
loop. No prefill NEFF is compiled and no prefill is ever scheduled, so there is nothing
to leak -- and the strict timed trigger then fires immediately, because prompt
throughput really is 0.0 throughout. Pattern taken from the in-tree benchmark
`test/models/gemma4/gemma4_31b/bf16/performance/test_isolated_decode.py`.

Do NOT pick this up front from the threshold above. ITL is not known before the run,
the fallback is automatic, and pre-empting it means no capture is ever attempted on the
configuration that was actually pinned.

THE KV IS NEVER FILLED. `_fill_blocks` is a no-op because host-side KV writes fail on
Neuron, so decode runs against uninitialised KV and the logits are garbage. Irrelevant
for a profile or for throughput -- the fixed-shape decode NEFF does the same compute
regardless of KV contents -- but never use this trace for anything that reads outputs.

Everything else is reused unchanged: the same run_spec, the same safe_serve isolation,
the same `arm_decode`, `validate_capture`, `emit_manifest`, `teardown`, and the same
`decode_isolation` recipe name. The connector is recorded as `decode_mode` /
`decode_isolation` at the top level of run_spec.json and manifest.json, never merged
into `serve_config`: none of the pinned serving axes change, so a drift check still sees
the pinned configuration served unmodified.

REQUIRES `vllm_neuron/vllm/kv_connector/neuron_decode_bench_connector.py` in the served
tree -- present in the shared vLLM-Neuron base, not only in the gemma4 line. Its own
docstring names vllm 0.21.0; the base classes it subclasses were re-verified present on
vllm 0.29.0.
"""
from __future__ import annotations

from pathlib import Path

from profile_measurements import precision_record, write_json
from profile_run import (
    _log,
    arm_decode,
    check_capabilities,
    emit_manifest,
    emit_run_spec,
    launch_server,
    profile_files,
    summarize_results,
    teardown,
    validate_capture,
    wait_ready,
)

# The whole decode-only serving condition, and it is fully determined: three fields, no
# tuning, no per-model values. That is what makes it safe to apply automatically.
# kv_connector_module_path is required because vLLM cannot import an out-of-tree
# connector by class name alone; kv_role=kv_consumer is the field that skips prefill.
ISOLATION_KV_TRANSFER_CONFIG = {
    "kv_connector": "NeuronDecodeBenchConnector",
    "kv_connector_module_path": (
        "vllm_neuron.vllm.kv_connector.neuron_decode_bench_connector"
    ),
    "kv_role": "kv_consumer",
}
# Lines the worker prints only when it is running as a decode-only server. vLLM accepts
# --kv-transfer-config and a misconfigured connector leaves a perfectly normal engine
# running, which yields a plausible decode trace with prefill mixed in and no error
# anywhere -- so their absence must fail the capture.
DECODE_ONLY_MARKERS = ("Skipping prefill warmup", "kv_role=kv_consumer")


def inject_isolation(cfg: dict) -> None:
    """Add the decode-only connector to serve_config, in place."""
    cfg["SERVE_CONFIG"]["kv_transfer_config"] = dict(ISOLATION_KV_TRANSFER_CONFIG)


def confirm_decode_only_server(cfg: dict) -> list:
    """Read back from serve.log that the server actually skipped prefill."""
    log = Path(cfg["SERVE_LOG"])
    text = log.read_text(errors="ignore") if log.is_file() else ""
    found = [marker for marker in DECODE_ONLY_MARKERS if marker in text]
    if not found:
        raise RuntimeError(
            f"decode isolation: serve.log shows none of {list(DECODE_ONLY_MARKERS)}, so "
            "this server is NOT decode-only and the capture would contain prefill. Check "
            "that the served tree has vllm_neuron/vllm/kv_connector/"
            "neuron_decode_bench_connector.py."
        )
    _log(f"decode-only server CONFIRMED by serve.log markers: {found}")
    return found


def capture(cfg: dict) -> int:
    """Capture decode on a decode-only server. cfg must already be resolved and injected."""
    Path(f"{cfg['PROFILE_DIR']}/logs").mkdir(parents=True, exist_ok=True)
    Path(cfg["PROFILE_OUT"]).mkdir(parents=True, exist_ok=True)
    # Written BEFORE launch so it survives a compile or launch failure.
    emit_run_spec(cfg)
    write_json(Path(cfg["PROFILE_DIR"]) / "precision.json", precision_record(cfg))
    write_json(Path(cfg["PROFILE_DIR"]) / "capabilities.json", check_capabilities(cfg))
    cfg["PROFILE_FILES_BEFORE"] = profile_files(cfg)

    evidence = {
        "serving_condition": "decode-only (kv_role=kv_consumer, prefill skipped)",
        "kv_transfer_config": cfg["SERVE_CONFIG"]["kv_transfer_config"],
        "kv_contents": "uninitialised; outputs are not meaningful",
        "reason": cfg.get("DECODE_MODE_REASON", ""),
        "markers_found": [],
    }
    _log(f"booting DECODE-ONLY server, log={cfg['SERVE_LOG']}")
    try:
        launch_server(cfg)
        wait_ready(cfg)
        _log("READY")
        evidence["markers_found"] = confirm_decode_only_server(cfg)
        # Reused unchanged: with no prefill anywhere, the trigger it waits for is
        # trivially true, so arming is immediate instead of racy.
        arm_decode(cfg)
        validate_capture(cfg)
        summarize_results(cfg)
        return 0
    finally:
        write_json(Path(cfg["PROFILE_DIR"]) / "decode_isolation.json", evidence)
        try:
            emit_manifest(cfg)
        finally:
            teardown(cfg)
