#!/usr/bin/env python3
"""Run a prefill/decode capture through the shared server lifecycle."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling import
from profile_config import (  # noqa: E402
    resolve_config, resolve_config_from_spec, apply_serve_env,
    build_configs, build_serve_config, build_serve_argv,
)
from profile_measurements import persist_measurement, precision_record, write_json  # noqa: E402

def _ts() -> str:
    return time.strftime("%H:%M:%S", time.gmtime())


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def build_bench_argv(cfg: dict, *, warmups: int, num_prompts: int, result_filename: str) -> list:
    return [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", cfg["MODEL"],
        "--host", "127.0.0.1", "--port", str(cfg["PORT"]),
        "--dataset-name", "random",
        "--random-input-len", str(cfg["ISL"]),
        "--random-output-len", str(cfg["OSL"]),
        "--random-range-ratio", "0",
        "--seed", "0",
        "--num-warmups", str(warmups),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(cfg["CONC"]),
        "--request-rate", "inf",
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result",
        "--result-filename", result_filename,
    ]


# ------------------------------------------------------------------ server lifecycle
def launch_server(cfg: dict) -> dict:
    """chdir to the repo (plan §6 option A) then launch via safe_serve (isolated setsid group)."""
    import safe_serve

    os.chdir(cfg["REPO"])  # child inherits cwd; == the bash `cd $VLLM_NEURON_DIR`
    Path(cfg["SERVE_LOG"]).parent.mkdir(parents=True, exist_ok=True)
    # argv is derived from the SAME serve_config that run_spec.json records (built from env in
    # fallback mode, loaded from the spec in input mode) → the launch matches the record.
    argv = build_serve_argv(cfg["SERVE_CONFIG"])
    rc = safe_serve.cmd_launch(
        Namespace(runfile=cfg["RUNFILE"], log=cfg["SERVE_LOG"], command=argv)
    )
    if rc != 0:
        raise SystemExit(f"profile_run: safe_serve launch failed (rc={rc})")
    return safe_serve._load_runfile(cfg["RUNFILE"])


def wait_ready(cfg: dict) -> None:
    import safe_serve

    rc = safe_serve.cmd_wait(
        Namespace(
            runfile=cfg["RUNFILE"],
            health=f"http://127.0.0.1:{cfg['PORT']}/v1/models",
            ready_pattern="",
            log=cfg["SERVE_LOG"],
            timeout=3600.0,
            stale=900.0,
            poll=10.0,
            report_every=60.0,
        )
    )
    if rc != 0:  # pid died / wedge / timeout — teardown fires in the caller's finally
        raise SystemExit(f"profile_run: server never became ready (safe_serve rc={rc})")


def teardown(cfg: dict) -> None:
    """Kill-after-capture: SIGKILL only the recorded PGID (self-guarded). No pkill."""
    if Path(cfg["RUNFILE"]).exists():
        import safe_serve

        rc = safe_serve.cmd_teardown(Namespace(runfile=cfg["RUNFILE"], grace=0))
        write_json(Path(cfg["PROFILE_DIR"]) / "cleanup.json", {"returncode": rc})
        if rc:
            raise RuntimeError(f"server cleanup failed: rc={rc}")


def post(cfg: dict, path: str) -> None:
    url = f"http://127.0.0.1:{cfg['PORT']}{path}"
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except Exception as e:
        raise RuntimeError(f"profiling endpoint {path} failed: {e}") from e


def check_capabilities(cfg: dict) -> dict:
    """Run optional diagnostics in the serving interpreter, without changing run artifacts."""
    report = {"supported": None, "versions": {}, "errors": [], "warnings": []}
    try:
        with tempfile.TemporaryDirectory(prefix="neuron-profile-probe-") as directory:
            run_spec = Path(directory) / "run_spec.json"
            output = Path(directory) / "capabilities.json"
            write_json(run_spec, {"serve_config": cfg["SERVE_CONFIG"]})
            result = subprocess.run(
                [f"{cfg['VENV']}/bin/python",
                 str(Path(__file__).with_name("profiler_adapter.py")),
                 str(run_spec), "--output", str(output)],
                cwd=cfg["REPO"], capture_output=True, text=True, timeout=60,
            )
            if result.returncode:
                raise RuntimeError(f"probe exited {result.returncode}: {result.stderr.strip()}")
            loaded = json.loads(output.read_text())
            if not isinstance(loaded, dict) or not isinstance(loaded.get("warnings"), list):
                raise ValueError("invalid capabilities report")
            report = loaded
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["warnings"].append(f"Profiler diagnostic unavailable: {error}")
    for warning in report["warnings"]:
        _log(f"WARNING: {warning}")
    return report


def profile_files(cfg: dict) -> dict:
    """Snapshot file signatures so old artifacts cannot stand in for a new capture."""
    files = {}
    for path in Path(cfg["PROFILE_OUT"]).rglob("*"):
        if path.is_file():
            stat = path.stat()
            if stat.st_size:
                files[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return files


def validate_capture(cfg: dict) -> None:
    """Check the requested Neuron trace files, independently of the diagnostic probe."""
    kind = cfg["SERVE_CONFIG"].get("profiler_config", {}).get("profiler")
    if kind not in (None, "cuda"):
        _log(f"WARNING: inspect artifacts using the installed {kind!r} profiler's tools")
        write_json(Path(cfg["PROFILE_DIR"]) / "capture.json", {
            "profiler": kind, "validation": "unverified",
        })
        return
    options = cfg["SERVE_CONFIG"].get("additional_config", {}).get("neuron_profiler", {})
    activities = options.get("activities", ["device_profile", "system_profile"])
    patterns = {
        "device_profile": "*.ntff",
        "system_profile": "ntrace.pb",
        "host_memory": "host_mem.pb",
        "cpu_util": "cpu_util.pb",
    }
    if "all" in activities:
        activities = list(patterns)
    before = cfg.get("PROFILE_FILES_BEFORE", {})
    files = [Path(path) for path, signature in profile_files(cfg).items()
             if before.get(path) != signature]
    missing = [activity for activity in activities if activity in patterns
               and not any(path.match(patterns[activity]) for path in files)]
    unknown = [activity for activity in activities if activity not in patterns]
    write_json(Path(cfg["PROFILE_DIR"]) / "capture.json", {
        "requested_activities": activities,
        "artifacts": [str(path) for path in files],
        "missing_activities": missing,
        "unverified_activities": unknown,
    })
    if missing:
        raise RuntimeError(f"profile capture produced no new nonempty traces for: {', '.join(missing)}")
    if unknown:
        _log(f"WARNING: inspect trace artifacts for unfamiliar activities: {', '.join(unknown)}")


# ------------------------------------------------------------------ phase arming
class DecodeArmingFailed(RuntimeError):
    """The timed trigger never fired, so /start_profile was never sent.

    Narrow on purpose: the server was healthy and decoding, only the scheduler line is
    missing. That is the one failure a different serving condition can fix.
    """


def arm_decode(cfg: dict) -> None:
    """Log-poll steady-state arming: wait for BS=STEADY_STATE_BS decode-only, then bracket."""
    bench_log = f"{cfg['PROFILE_DIR']}/logs/bench_c{cfg['CONC']}.log"
    result = f"{cfg['PROFILE_DIR']}/bench_c{cfg['CONC']}.json"
    argv = build_bench_argv(cfg, warmups=2, num_prompts=cfg["NPROMPTS"], result_filename=result)

    _log(f"launching CONC={cfg['CONC']} bench (nprompts={cfg['NPROMPTS']}) in background")
    # Plain client subprocess: the bench holds NO Neuron device (it is an HTTP client),
    # so it does not need safe_serve isolation. We wait on it directly to bracket the
    # profile window. The device-holding server is the only thing safe_serve manages.
    with open(bench_log, "ab", buffering=0) as bf:
        bench = subprocess.Popen(argv, stdout=bf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    try:
        steady_re = re.compile(
            rf"Avg prompt throughput: 0\.0.*Running: {cfg['STEADY_STATE_BS']} reqs, Waiting: 0 reqs"
        )
        _log(f"polling serve.log for BS={cfg['STEADY_STATE_BS']} decode steady state...")
        deadline = time.time() + 300
        serve_log = Path(cfg["SERVE_LOG"])
        while True:
            if bench.poll() is not None:
                _log(f"bench died before profile triggered — see {bench_log}")
                if Path(result).is_file():
                    persist_measurement(cfg, result, bench.returncode)
                raise DecodeArmingFailed(
                    f"bench finished before the BS={cfg['STEADY_STATE_BS']} steady-state "
                    f"line appeared (see {bench_log})"
                )
            if serve_log.exists() and steady_re.search(serve_log.read_text(errors="ignore")):
                _log(f"BS={cfg['STEADY_STATE_BS']} decode steady state DETECTED")
                break
            if time.time() > deadline:
                _log(f"TIMEOUT waiting for BS={cfg['STEADY_STATE_BS']} steady state")
                bench.kill()
                bench.wait()
                raise DecodeArmingFailed(
                    f"no BS={cfg['STEADY_STATE_BS']} steady-state line within the deadline"
                )
            time.sleep(1)

        _log("POST /start_profile (delay=0 max=2)")
        post(cfg, "/start_profile")
        bench.wait()
        _log(f"bench done (rc={bench.returncode})")
        persist_measurement(cfg, result, bench.returncode)
        _log("POST /stop_profile (idempotent — auto-stop likely fired first)")
        post(cfg, "/stop_profile")
        time.sleep(15)
    finally:
        if bench.poll() is None:
            bench.terminate()
            try:
                bench.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bench.kill()
                bench.wait()


def arm_prefill(cfg: dict) -> None:
    """Workload isolation: arm on the idle server, run a single one-shot prefill, stop."""
    _log("server idle; sleeping 5s")
    time.sleep(5)
    _log("POST /start_profile")
    post(cfg, "/start_profile")

    bench_log = f"{cfg['PROFILE_DIR']}/logs/bench_prefill.log"
    result = f"{cfg['PROFILE_DIR']}/bench_prefill.json"
    argv = build_bench_argv(cfg, warmups=0, num_prompts=cfg["NPROMPTS"], result_filename=result)
    _log(f"launching one-shot prefill bench (ISL={cfg['ISL']} OSL={cfg['OSL']})")
    with open(bench_log, "ab", buffering=0) as bf:
        rc = subprocess.run(
            argv, stdout=bf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        ).returncode
    _log(f"bench done (rc={rc})")
    persist_measurement(cfg, result, rc)

    _log("POST /stop_profile")
    post(cfg, "/stop_profile")
    time.sleep(20)  # give NRT time to flush the NTFF


# ------------------------------------------------------------------ run_spec (INPUT) + manifest (OUTPUT)
def _workload_block(cfg: dict) -> dict:
    # ONLY the non-serve driving knobs: what `vllm bench serve` sends (isl/osl/conc/nprompts)
    # plus the profiler-arming trigger (steady_state_bs). Everything that defines the server /
    # compiled graph — tp, max_model_len, SEG (max_num_batched_tokens), max_num_seqs,
    # num_seqs_buckets — lives in serve_config (the compile-key source of truth) and is NOT
    # duplicated here. Litmus test: changing a serve_config value recompiles the graph;
    # changing a workload value drives different traffic at the same compiled server.
    wl = {
        "isl": cfg["ISL"], "osl": cfg["OSL"], "conc": cfg["CONC"],
        "nprompts": cfg["NPROMPTS"],
    }
    if cfg["phase"] == "decode":
        wl["steady_state_bs"] = cfg["STEADY_STATE_BS"]
    return wl


def emit_run_spec(cfg: dict) -> None:
    """Write the INPUT contract (run_spec.json) at LAUNCH time — BEFORE the server starts.

    Split from manifest.json (the OUTPUT) on purpose: input and output have different
    lifecycles and owners. The spec is fully known before launch and is the caller's
    contract; writing it up front means it SURVIVES a compile/launch failure — the exact
    case where you most need to see which flags were used (a wrong compile flag usually
    kills the server before any output manifest could be written). It is also the diffable,
    reproducible handoff-IN artifact.

    In INPUT mode the spec was authored elsewhere and consumed — we do not re-serialize it
    (that could drop fields we do not model); we materialize the consumed spec verbatim in the
    run dir (or leave it if it is already there).
    """
    if cfg.get("INPUT_MODE"):
        # Persist the consumed spec into the run dir. Round-trip the authored JSON (preserving
        # every field, including any we do not model) but correct the profiler output_dir to
        # where this run actually wrote — the driver retargets it at consume time, so the
        # persisted record must reflect that, not the master's placeholder/foreign path.
        doc = json.loads(Path(cfg["RUN_SPEC_SRC"]).read_text())
        if cfg.get("ATTEMPT_ID"):
            doc["attempt_id"] = cfg["ATTEMPT_ID"]
        try:
            doc["serve_config"]["additional_config"]["neuron_profiler"]["output_dir"] = cfg["PROFILE_OUT"]
        except (KeyError, TypeError):
            pass
        # Recorded at the TOP LEVEL, never merged into serve_config: the connector is
        # capture apparatus, like the profiler output_dir retargeted just above, and none
        # of the pinned serving axes change. So a drift check still sees the pinned config
        # served unmodified, while the record stays complete enough to reproduce.
        if cfg.get("DECODE_MODE"):
            doc["decode_mode"] = cfg["DECODE_MODE"]
        connector = cfg["SERVE_CONFIG"].get("kv_transfer_config")
        if connector:
            doc["decode_isolation"] = {
                "kv_transfer_config": connector,
                "reason": cfg.get("DECODE_MODE_REASON", ""),
            }
        Path(cfg["RUN_SPEC"]).write_text(json.dumps(doc, indent=2) + "\n")
        _log(f"run_spec (consumed from {cfg['RUN_SPEC_SRC']}) -> {cfg['RUN_SPEC']} "
             "(output_dir retargeted to this run)")
        return

    spec = {
        "schema_version": "1.0",
        "run_name": cfg["run_name"],
        "phase": cfg["phase"],
        "recipe": "prefill_isolation" if cfg["phase"] == "prefill" else "decode_isolation",
        "workload": _workload_block(cfg),
        # The structured serve/compile spec — the single source of truth for the launch, and
        # the master-friendly form to hand over. The exact `vllm serve` argv is derived from
        # it by build_serve_argv() (the same call launch_server uses), so it is not stored
        # redundantly — a consumer regenerates the command from serve_config when needed.
        "serve_config": cfg["SERVE_CONFIG"],
        # Caller-set, compile-affecting model flags that live outside resolve_config and reach
        # vllm serve only by inheritance — captured so the compile key is self-documenting.
        "model_env": cfg.get("MODEL_ENV", {}),
    }
    if cfg.get("ATTEMPT_ID"):
        spec["attempt_id"] = cfg["ATTEMPT_ID"]
    Path(cfg["RUN_SPEC"]).write_text(json.dumps(spec, indent=2) + "\n")
    _log(f"run_spec -> {cfg['RUN_SPEC']}")


def _find_session_dir(profile_out: str):
    # NEWEST session by mtime. A stale session from a prior/aborted attempt can linger in
    # output/ (the driver tears down the old server but does not purge its session dir), and
    # picking the lexically-first dir (an older pid) would make the manifest point at the wrong
    # NTFF. The session this run just captured is the most recently modified one.
    dirs = [d for d in Path(profile_out).glob("i-*_pid_*/*") if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


# trace_info.pb does not embed a NEFF *filename*; it records the module as
# "module.neff_<hash>", so the hash FOLLOWS ".neff_". The old graph_<hash>.neff pattern
# therefore never matched on this stack: 
_NEFF_HASH_PATTERNS = (
    rb"\.neff_([0-9a-f]{8,})",
    rb"graph_([0-9a-f]+)\.neff",
    rb"(?<![0-9A-Za-z_])([0-9a-f]{32})\.neff",
)


def _neff_hashes(trace_info_pb) -> list:
    """Every model-graph NEFF hash named in trace_info.pb, under any convention."""
    if not trace_info_pb or not Path(trace_info_pb).exists():
        return []
    data = Path(trace_info_pb).read_bytes()
    found: set[str] = set()
    for pattern in _NEFF_HASH_PATTERNS:
        found.update(h.decode() for h in re.findall(pattern, data))
    return sorted(found)


def emit_manifest(cfg: dict) -> None:
    """Reproduce the .sh manifest.json — the analysis skill's §1.0 fast-path entry point."""
    profile_out = cfg["PROFILE_OUT"]
    session_dir = _find_session_dir(profile_out)
    neff_dir = f"{profile_out}/neffs"
    neff_dir = neff_dir if Path(neff_dir).is_dir() else None
    trace_info_pb = f"{session_dir}/trace_info.pb" if session_dir else None

    def pnull(p):
        return str(p) if (p and Path(p).exists()) else None

    result_json = (
        f"{cfg['PROFILE_DIR']}/bench_prefill.json"
        if cfg["phase"] == "prefill"
        else f"{cfg['PROFILE_DIR']}/bench_c{cfg['CONC']}.json"
    )
    manifest = {
        # input contract (serve_config / model_env) lives in run_spec.json; this OUTPUT
        # manifest carries a back-reference to it plus the slim (deduplicated) workload.
        "schema_version": "1.1",
        "recipe": "prefill_isolation" if cfg["phase"] == "prefill" else "decode_isolation",
        "run_name": cfg["run_name"],
        "workload": _workload_block(cfg),
        # Handoff-IN artifact (written at launch; survives a failed compile). See emit_run_spec.
        "run_spec": cfg["RUN_SPEC"] if Path(cfg["RUN_SPEC"]).exists() else None,
        "profile": {
            "session_dir": pnull(session_dir),
            "device_ntff": pnull(f"{session_dir}/profile_nc_0_session_0.ntff" if session_dir else None),
            "system_trace_pb": pnull(f"{session_dir}/ntrace.pb" if session_dir else None),
            "trace_info_pb": pnull(trace_info_pb),
            "cpu_util_pb": pnull(f"{session_dir}/cpu_util.pb" if session_dir else None),
            "host_mem_pb": pnull(f"{session_dir}/host_mem.pb" if session_dir else None),
            "neff_dir": pnull(neff_dir),
        },
        "neff_hashes": _neff_hashes(trace_info_pb),
        "bench_result_json": pnull(result_json),
    }
    # `recipe` is decode_isolation either way, so it cannot say HOW decode was isolated.
    # A timed trace may contain prefill; an isolated one cannot.
    if cfg.get("DECODE_MODE"):
        manifest["decode_mode"] = cfg["DECODE_MODE"]
        manifest["decode_mode_reason"] = cfg.get("DECODE_MODE_REASON", "")
    manifest_path = f"{cfg['PROFILE_DIR']}/manifest.json"
    Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
    _log(f"manifest -> {manifest_path}")


def summarize_results(cfg: dict) -> None:
    emit_manifest(cfg)
    out = Path(cfg["PROFILE_OUT"])
    _log("=== RESULTS ===")
    print(f"artifacts under {out}/")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    for p in files[:20]:
        print(f"  {p}")
    if cfg["phase"] == "prefill":
        for ntff in out.glob("i-*_pid_*/*/profile_nc_0_session_0.ntff"):
            print(f"NTFF: {ntff.stat().st_size} bytes  {ntff}")
    print("NEFFs referenced in trace_info.pb:")
    for h in _neff_hashes(f"{_find_session_dir(cfg['PROFILE_OUT'])}/trace_info.pb"
                          if _find_session_dir(cfg["PROFILE_OUT"]) else None):
        print(f"  graph_{h}.neff")


# ------------------------------------------------------------------ dry-run
def dry_run(cfg: dict) -> int:
    env_delta = apply_serve_env(cfg)
    serve_config = cfg["SERVE_CONFIG"]
    mode = "INPUT (consuming run_spec)" if cfg.get("INPUT_MODE") else "env fallback"
    print(f"=== profile_run --dry-run (phase={cfg['phase']} run={cfg['run_name']} | {mode}) ===")
    if cfg.get("RUN_SPEC_SRC"):
        print(f"# run_spec source: {cfg['RUN_SPEC_SRC']}")
    print(f"\n# cwd for child: {cfg['REPO']}")
    print("\n# serve_config (structured source → run_spec.json):")
    print("  " + json.dumps(serve_config))
    print("\n# serve argv (derived from serve_config):")
    print("  " + " ".join(build_serve_argv(serve_config)))
    print("\n# env delta (set only-if-unset; PATH prepends venv/bin):")
    for k in sorted(env_delta):
        print(f"  {k}={env_delta[k]}")
    print("\n# model_env (caller-set compile-affecting flags → manifest.model_env):")
    if cfg["MODEL_ENV"]:
        for k in sorted(cfg["MODEL_ENV"]):
            print(f"  {k}={cfg['MODEL_ENV'][k]}")
    else:
        print('  (none — declare via PROFILE_MODEL_ENV="NAME1 NAME2 ..." or use '
              "GEMMA4_ / VLLM_NEURON_BACKEND_ prefixes)")
    warmups, nprompts, result = (
        (0, cfg["NPROMPTS"], f"{cfg['PROFILE_DIR']}/bench_prefill.json")
        if cfg["phase"] == "prefill"
        else (2, cfg["NPROMPTS"], f"{cfg['PROFILE_DIR']}/bench_c{cfg['CONC']}.json")
    )
    print("\n# bench argv:")
    print("  " + " ".join(build_bench_argv(cfg, warmups=warmups, num_prompts=nprompts,
                                           result_filename=result)))
    print("\n(no device touched)")
    return 0


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["prefill", "decode"], default="",
                    help="required in env-fallback mode; in input mode it is taken from the "
                         "run_spec (if given it must match).")
    ap.add_argument("run_name")
    ap.add_argument("--run-spec", default="",
                    help="path to a master-authored run_spec.json to CONSUME (input mode). "
                         "If omitted, falls back to $PROFILE_RUN_SPEC, then to "
                         "<profiles>/<run_name>/run_spec.json if it exists; if none is found, "
                         "the config is built from env vars (fallback). Set "
                         "PROFILE_IGNORE_RUN_SPEC=1 to force the env path.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print resolved serve argv, env delta, and both config JSONs; no device.")
    ap.add_argument("--check-capabilities", action="store_true",
                    help="print best-effort profiler diagnostics without launching or changing run files")
    ap.add_argument("--runtime-config",
                    help="optional observed effective_config JSON; match attempt_id when supplied")
    ap.add_argument("--attempt-id", help="optional run identity, also used for campaign integration")
    ap.add_argument("--decode-mode", choices=["timed", "isolated"], default="timed",
                    help="'timed' (default) captures on the real engine and falls back to "
                         "isolation by itself if the trigger never fires -- the right choice "
                         "for a graded run, because the pinned configuration is attempted "
                         "first. 'isolated' skips that attempt and captures on a decode-only "
                         "server directly (references/decode_isolation.py); useful for "
                         "debugging the isolated path.")
    ap.add_argument("--decode-fallback", choices=["isolated", "none"], default="isolated",
                    help="when the timed decode trigger never fires: 'isolated' (default) "
                         "recaptures on a decode-only server; 'none' fails instead. Only an "
                         "arming failure triggers it.")
    args = ap.parse_args(argv)

    # Input-vs-env selection. Consuming a run_spec is the DEFAULT whenever one is available;
    # env-var resolution is the fallback. Precedence for locating a spec:
    #   --run-spec arg  >  $PROFILE_RUN_SPEC  >  <profiles>/<run_name>/run_spec.json (if present)
    # PROFILE_IGNORE_RUN_SPEC=1 forces the env path (escape hatch for "re-run with changed env").
    ignore_spec = os.environ.get("PROFILE_IGNORE_RUN_SPEC", "0") not in ("", "0", "false", "False")
    spec_path = "" if ignore_spec else (args.run_spec or os.environ.get("PROFILE_RUN_SPEC", ""))
    if not spec_path and not ignore_spec:
        _pr = os.environ.get("NEURON_PROJECT_ROOT", os.getcwd())
        _default = f"{_pr}/agent_artifacts/profiles/{args.run_name}/run_spec.json"
        if Path(_default).exists():
            spec_path = _default

    if spec_path and not Path(spec_path).is_file():
        ap.error(f"run_spec does not exist: {spec_path}")
    if spec_path:
        _log(f"INPUT mode: consuming run_spec {spec_path} "
             "(serve_config + model_env from the spec; env serve/model vars are ignored)")
        cfg = resolve_config_from_spec(args.phase, args.run_name, spec_path)
    else:
        if not args.phase:
            print("profile_run: --phase is required in env-fallback mode (no run_spec found).",
                  file=sys.stderr)
            return 2
        _log("env-fallback mode: building the config (and run_spec.json) from environment vars")
        cfg = resolve_config(args.phase, args.run_name)
    cfg["RUNTIME_CONFIG"] = args.runtime_config
    if args.attempt_id and cfg.get("ATTEMPT_ID") not in (None, args.attempt_id):
        ap.error("attempt-id contradicts the run specification")
    cfg["ATTEMPT_ID"] = args.attempt_id or cfg.get("ATTEMPT_ID")

    if args.dry_run:
        return dry_run(cfg)

    apply_serve_env(cfg)
    if args.check_capabilities:
        print(json.dumps(check_capabilities(cfg), indent=2))
        return 0

    if Path(cfg["RUNFILE"]).exists():
        raise SystemExit("Existing runfile: reconcile the native child and saved results; "
                         "resume existing work or use a new attempt name.")

    if cfg["phase"] == "decode":
        cfg["DECODE_MODE"] = args.decode_mode
        if args.decode_mode == "isolated":
            # Deferred import: decode_isolation imports this module, so module scope would
            # be circular.
            import decode_isolation

            _log("decode mode: isolated, requested explicitly. The pinned serving "
                 "configuration is NOT attempted; prefer the default for a graded run.")
            cfg["DECODE_MODE_REASON"] = "requested with --decode-mode isolated"
            decode_isolation.inject_isolation(cfg)
            return decode_isolation.capture(cfg)
    if cfg["phase"] != "decode" or args.decode_fallback == "none":
        return _timed_capture(cfg)
    try:
        return _timed_capture(cfg)
    except DecodeArmingFailed as error:
        # Only an arming failure: the server was healthy and decoding, the scheduler line
        # just never appeared, and that is the one failure a different serving condition
        # fixes. Everything else propagates. Without this the pod is lost -- boot, cold
        # compile and warmup are all spent and no trace is written.
        import decode_isolation

        _log(f"FALLBACK: timed decode arming did not fire ({error})")
        _log("FALLBACK: recapturing on a decode-only server (kv_role=kv_consumer, prefill "
             "skipped). Pinned serve_config axes unchanged; see decode_isolation.json.")
        reset_run_dir(cfg)
        cfg["DECODE_MODE"] = "isolated"
        cfg["DECODE_MODE_REASON"] = f"timed arming failed ({error})"
        decode_isolation.inject_isolation(cfg)
        return decode_isolation.capture(cfg)


def reset_run_dir(cfg: dict) -> None:
    """Clear one attempt's state so a retry in the same directory starts clean.

    launch_server refuses to start when a runfile exists, validate_capture would count
    the dead attempt's files as new, and a stale manifest would be read as the result.
    """
    for name in ("manifest.json", "capture.json", "cleanup.json"):
        Path(f"{cfg['PROFILE_DIR']}/{name}").unlink(missing_ok=True)
    Path(cfg["RUNFILE"]).unlink(missing_ok=True)
    shutil.rmtree(cfg["PROFILE_OUT"], ignore_errors=True)
    _log(f"reset run directory for retry: {cfg['PROFILE_DIR']}")


def _timed_capture(cfg: dict) -> int:
    """The historical path: real engine, arm by waiting for a prefill-free window."""
    Path(f"{cfg['PROFILE_DIR']}/logs").mkdir(parents=True, exist_ok=True)
    Path(cfg["PROFILE_OUT"]).mkdir(parents=True, exist_ok=True)
    # Write the INPUT contract BEFORE launch so it survives a compile/launch failure.
    emit_run_spec(cfg)
    write_json(Path(cfg["PROFILE_DIR"]) / "precision.json", precision_record(cfg))
    write_json(Path(cfg["PROFILE_DIR"]) / "capabilities.json", check_capabilities(cfg))
    cfg["PROFILE_FILES_BEFORE"] = profile_files(cfg)

    _log(f"booting server, log={cfg['SERVE_LOG']}")
    try:
        launch_server(cfg)
        wait_ready(cfg)
        _log("READY")
        arm_prefill(cfg) if cfg["phase"] == "prefill" else arm_decode(cfg)
        validate_capture(cfg)
        summarize_results(cfg)
        return 0
    finally:
        # Covers every exit path (bench death, timeout, KeyboardInterrupt, exception):
        # SIGKILL the server's OWN isolated group. No pkill, no bare &, no shared group.
        try:
            emit_manifest(cfg)
        finally:
            teardown(cfg)


if __name__ == "__main__":
    sys.exit(main())
