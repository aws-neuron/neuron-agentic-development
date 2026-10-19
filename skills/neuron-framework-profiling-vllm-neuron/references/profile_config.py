"""Resolve the profiling run specification and derive the serving command."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REQUIRED_ENV = ("VLLM_NEURON_VENV_PATH", "VLLM_NEURON_DIR", "TARGET_MODEL")

# Compile-affecting, model-specific env that the DRIVER does not own (kernel routing,
# backend toggles, dedup, device pinning, platform target). These live OUTSIDE
# resolve_config and reach `vllm serve` only by process inheritance — so we SNAPSHOT them
# into manifest.json ("model_env") to make a run self-describing and two runs' compile keys
# diffable. Two unioned sources:
#   * explicit (authoritative):  export PROFILE_MODEL_ENV="NAME1 NAME2 ..."
#       — use for anything not matched by the prefixes below, e.g.
#         VLLM_NEURON_DISABLE_INPUT_DEDUP, NEURON_VISIBLE_DEVICES, NEURON_PLATFORM_TARGET_OVERRIDE.
#   * default prefix allowlist (best-effort convenience): any env var starting with one of:
_MODEL_ENV_PREFIXES = ("GEMMA4_", "VLLM_NEURON_BACKEND_")


def _capture_model_env() -> dict:
    """Snapshot caller-set, compile-affecting model env. Call BEFORE apply_serve_env so we
    record what the CALLER brought, not the driver's own workload-agnostic defaults."""
    names = set()
    for tok in os.environ.get("PROFILE_MODEL_ENV", "").replace(",", " ").split():
        names.add(tok)
    names.update(k for k in os.environ if k.startswith(_MODEL_ENV_PREFIXES))
    return {k: os.environ[k] for k in sorted(names) if k in os.environ}


# ------------------------------------------------------------------ config resolution
def resolve_config(phase: str, run_name: str) -> dict:
    """Resolve env + workload knobs, mirroring the bash ``:?`` guards and ``:-`` defaults."""
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        hints = {
            "VLLM_NEURON_VENV_PATH": "the Neuron SDK venv",
            "VLLM_NEURON_DIR": "the private-vllm-neuron repo",
            "TARGET_MODEL": "the local model path (passed to vllm serve --model)",
        }
        for k in missing:
            print(f"profile_run: set {k} to {hints[k]}", file=sys.stderr)
        raise SystemExit(1)

    def geti(key: str, default: int) -> int:
        return int(os.environ.get(key, default))

    project_root = os.environ.get("NEURON_PROJECT_ROOT", os.getcwd())
    profile_dir = f"{project_root}/agent_artifacts/profiles/{run_name}"
    profile_out = f"{profile_dir}/output"

    cfg = {
        "phase": phase,
        "run_name": run_name,
        "VENV": os.environ["VLLM_NEURON_VENV_PATH"],
        "REPO": os.environ["VLLM_NEURON_DIR"],
        "MODEL": os.environ["TARGET_MODEL"],
        "PROJECT_ROOT": project_root,
        "PROFILE_DIR": profile_dir,
        "PROFILE_OUT": profile_out,
        "SERVE_LOG": f"{profile_dir}/logs/serve.log",
        "RUNFILE": f"{profile_dir}/logs/serve.runfile.json",
        "RUN_SPEC": f"{profile_dir}/run_spec.json",
        # --- workload knobs (env-overridable, identical defaults to the .sh) ---
        "TP": geti("TP", 16),
        "ISL": geti("ISL", 8192),
        "MAXLEN": geti("MAXLEN", 10240),
        "SEG": geti("SEG", 8192),
        "NUM_SEQS_MAX": geti("NUM_SEQS_MAX", 48),
        "NUM_SEQS_BUCKETS": json.loads(
            os.environ.get("NUM_SEQS_BUCKETS", "[1,2,4,8,16,24,32,48]")
        ),
        "MAX_TOP_K": geti("MAX_TOP_K", 64),
        "PORT": geti("PORT", 8041),
        "NUM_GPU_BLOCKS": os.environ.get("NUM_GPU_BLOCKS", ""),
        "ENABLE_DGE": os.environ.get("ENABLE_DGE", "0"),
    }

    if phase == "prefill":
        # OSL / CONC / NPROMPTS are PINNED — the mode is a single-shot prefill dispatch.
        cfg.update(OSL=1, CONC=1, NPROMPTS=1)
    else:  # decode
        cfg.update(
            OSL=geti("OSL", 1024),
            CONC=geti("CONC", 8),
            NPROMPTS=geti("NPROMPTS", 50),
        )
        cfg["STEADY_STATE_BS"] = geti("STEADY_STATE_BS", cfg["CONC"])

    # Snapshot caller-set compile-affecting flags NOW (before apply_serve_env mutates env).
    cfg["MODEL_ENV"] = _capture_model_env()
    # Structured serve spec = the single source of truth for the launch, built from the env
    # knobs. (Input mode loads this from a run_spec instead; everything downstream reads
    # cfg["SERVE_CONFIG"] uniformly.)
    cfg["SERVE_CONFIG"] = build_serve_config(cfg)
    cfg["INPUT_MODE"] = False
    cfg["RUN_SPEC_SRC"] = None
    return cfg


def resolve_config_from_spec(phase_arg: str, run_name: str, spec_path: str) -> dict:
    """INPUT mode (the default when a run_spec is provided): consume a master-authored
    run_spec.json. The spec is the source of truth for phase / serve_config / workload /
    model_env. Only host-specific infra (venv, repo, project root, output paths) still comes
    from the environment — those are where THIS box keeps things and where outputs land, not
    part of the portable spec. TARGET_MODEL is NOT required here (the model path is in
    serve_config.model). The env-var path (resolve_config, above) is unchanged and remains
    the fallback when no run_spec is provided.
    """
    # Host infra only — the model comes from the spec, so TARGET_MODEL is not required here.
    for k in ("VLLM_NEURON_VENV_PATH", "VLLM_NEURON_DIR"):
        if not os.environ.get(k):
            print(f"profile_run: set {k} (host infra; still required in input mode)", file=sys.stderr)
            raise SystemExit(1)

    spec = json.loads(Path(spec_path).read_text())
    for key in ("phase", "serve_config", "workload"):
        if key not in spec:
            print(f"profile_run: run_spec {spec_path} missing required key '{key}'", file=sys.stderr)
            raise SystemExit(1)
    phase = spec["phase"]
    if phase_arg and phase_arg != phase:
        print(f"profile_run: --phase {phase_arg} conflicts with run_spec phase '{phase}'", file=sys.stderr)
        raise SystemExit(1)

    sc = dict(spec["serve_config"])  # copy: we retarget output_dir below without mutating the file
    wl = spec["workload"]
    is_pf = phase == "prefill"

    project_root = os.environ.get("NEURON_PROJECT_ROOT", os.getcwd())
    profile_dir = f"{project_root}/agent_artifacts/profiles/{run_name}"
    profile_out = f"{profile_dir}/output"

    # Retarget the profiler output_dir to THIS run's location — a run-location detail, not part
    # of the portable spec (a master authoring the spec need not know where it lands).
    try:
        sc["additional_config"]["neuron_profiler"]["output_dir"] = profile_out
    except (KeyError, TypeError):
        pass

    def _max_top_k(sc: dict) -> int:
        try:
            return sc["additional_config"]["neuron_config"]["on_device_sampling_config"]["max_top_k"]
        except (KeyError, TypeError):
            return 64

    cfg = {
        "phase": phase,
        "run_name": run_name,
        "VENV": os.environ["VLLM_NEURON_VENV_PATH"],
        "REPO": os.environ["VLLM_NEURON_DIR"],
        "MODEL": sc["model"],
        "PROJECT_ROOT": project_root,
        "PROFILE_DIR": profile_dir,
        "PROFILE_OUT": profile_out,
        "SERVE_LOG": f"{profile_dir}/logs/serve.log",
        "RUNFILE": f"{profile_dir}/logs/serve.runfile.json",
        "RUN_SPEC": f"{profile_dir}/run_spec.json",
        "ENABLE_DGE": os.environ.get("ENABLE_DGE", "0"),
        # serve-shape / compile knobs: from serve_config — the single source of truth. NOT
        # duplicated in workload (dedup: serve_config == the compile key).
        "TP": sc["tensor_parallel_size"],
        "MAXLEN": sc["max_model_len"],
        "SEG": sc["max_num_batched_tokens"],
        "NUM_SEQS_MAX": sc["max_num_seqs"],
        "NUM_SEQS_BUCKETS": sc.get("additional_config", {}).get("neuron_config", {}).get("num_seqs_buckets"),
        "NUM_GPU_BLOCKS": str(sc["num_gpu_blocks_override"]) if sc.get("num_gpu_blocks_override") else "",
        "MAX_TOP_K": _max_top_k(sc),
        "PORT": sc["port"],
        # driving knobs: from workload (bench/arming; sane defaults for optional fields)
        "ISL": wl.get("isl", 8192),
        "OSL": wl.get("osl", 1 if is_pf else 1024),
        "CONC": wl.get("conc", 1 if is_pf else 8),
        "NPROMPTS": wl.get("nprompts", 1 if is_pf else 50),
        # source of truth (verbatim from the spec) + the inherited compile flags to replay
        "SERVE_CONFIG": sc,
        "MODEL_ENV": spec.get("model_env", {}),
        "INPUT_MODE": True,
        "RUN_SPEC_SRC": spec_path,
        "ATTEMPT_ID": spec.get("attempt_id"),
    }
    if not is_pf:
        cfg["STEADY_STATE_BS"] = wl.get("steady_state_bs", cfg["CONC"])
    return cfg


def apply_serve_env(cfg: dict) -> dict:
    """Set serve/debug/DGE env into THIS process's os.environ (plan §6 option A).

    ``safe_serve.cmd_launch`` sets neither ``env=`` nor ``cwd=`` on the child, so the
    child inherits the driver's environment and cwd. Mutating our own os.environ with
    ``setdefault`` (only-if-unset, matching bash ``${VAR:-default}``) is the documented,
    deliberate equivalent of ``source activate`` — NRT reads the debug/DGE vars once at
    child init, so setting them before cmd_launch is correct. Returns the delta for
    --dry-run reporting.
    """
    delta = {}

    def setdef(key: str, val: str) -> None:
        if key not in os.environ:
            os.environ[key] = val
        delta[key] = os.environ[key]

    # INPUT mode: the compile-affecting flags come from the spec, not the caller's shell —
    # export them so the vllm serve child inherits them (the spec is authoritative here).
    # (Env-fallback mode leaves these alone; the caller already set them in the environment.)
    if cfg.get("INPUT_MODE"):
        for k, v in cfg.get("MODEL_ENV", {}).items():
            os.environ[k] = str(v)
            delta[k] = str(v)

    # workload-agnostic serve env
    setdef("VLLM_PLUGINS", "neuron")
    setdef("TOKENIZERS_PARALLELISM", "false")
    setdef("NEURON_SKIP_EFA_AFFINITY", "1")
    setdef("HF_HUB_ENABLE_HF_TRANSFER", "0")
    setdef("VLLM_LOGGING_LEVEL", "INFO")
    setdef("NEURON_RT_LOG_LEVEL", "ERROR")
    setdef("VLLM_NEURON_BARRIER_TIMEOUT", "43200")
    setdef("VLLM_ENGINE_READY_TIMEOUT_S", "43200")
    setdef("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "43200")
    # VLLM_CACHE_ROOT is exported by the .sh only if already set; it is already in
    # os.environ when set, so the child inherits it — nothing to do here.
    if os.environ.get("VLLM_CACHE_ROOT"):
        delta["VLLM_CACHE_ROOT"] = os.environ["VLLM_CACHE_ROOT"]

    # compile-time debug info baked into the NEFFs (source attribution for analysis)
    setdef("XLA_IR_DEBUG", "1")
    setdef("XLA_HLO_DEBUG", "1")
    setdef("NEURON_FRAMEWORK_DEBUG", "1")

    # DGE notifications: off by default (floods MoE decode capture; can hang tight-KV
    # dense models). Set ENABLE_DGE=1 for dense-model DMA-efficiency analysis.
    if cfg["ENABLE_DGE"] == "1":
        os.environ["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"
        delta["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"

    # "source activate": prepend the venv bin so the child's `vllm` resolves to the venv.
    venv_bin = f"{cfg['VENV']}/bin"
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    os.environ["VIRTUAL_ENV"] = cfg["VENV"]
    delta["PATH"] = f"{venv_bin}:$PATH"
    delta["VIRTUAL_ENV"] = cfg["VENV"]
    return delta


# ------------------------------------------------------------------ argv / config JSON
def build_configs(cfg: dict) -> tuple:
    """The exact ``--additional-config`` / ``--profiler-config`` dicts (identical to the .sh)."""
    add_cfg = {
        "neuron_config": {
            "num_batched_tokens_buckets": [cfg["SEG"]],
            "num_seqs_buckets": cfg["NUM_SEQS_BUCKETS"],
            "on_device_sampling_config": {"max_top_k": cfg["MAX_TOP_K"]},
        },
        "neuron_profiler": {
            "activities": ["device_profile", "system_profile"],
            "output_dir": cfg["PROFILE_OUT"],
            "neuron_cores": [0],
        },
    }
    # delay=0: arm on the very next iteration after /start_profile.
    # max=2:   auto-stop after 2 recorded iters (the CcOp-aggregation lower bound).
    profiler_cfg = {"profiler": "cuda", "delay_iterations": 0, "max_iterations": 2}
    return add_cfg, profiler_cfg


def build_serve_config(cfg: dict) -> dict:
    """The structured serve/compile spec — the SINGLE SOURCE OF TRUTH for the launch.

    This is what runs in run_spec.json (a master agent authors/hands over this structured
    form, not a fragile argv string) AND what ``build_serve_argv`` is derived from. Because
    the argv is a pure projection of this dict (and both the launch and the recorded spec use
    the same dict), the recorded config can never drift from what actually ran.
    """
    add_cfg, profiler_cfg = build_configs(cfg)
    return {
        "model": cfg["MODEL"],
        "tensor_parallel_size": cfg["TP"],
        "data_parallel_size": 1,
        "dtype": os.environ.get("DTYPE", "auto"),
        "kv_cache_dtype": os.environ.get("KV_CACHE_DTYPE", "auto"),
        "max_model_len": cfg["MAXLEN"],
        "max_num_batched_tokens": cfg["SEG"],
        "max_num_seqs": cfg["NUM_SEQS_MAX"],
        "num_gpu_blocks_override": int(cfg["NUM_GPU_BLOCKS"]) if cfg["NUM_GPU_BLOCKS"] else None,
        "block_size": 16,
        "optimization_level": 2,
        "enable_prefix_caching": False,
        "async_scheduling": True,
        "port": cfg["PORT"],
        "additional_config": add_cfg,
        "profiler_config": profiler_cfg,
    }


def build_serve_argv(sc: dict) -> list:
    """Derive the exact ``vllm serve`` argv from a serve_config dict (see build_serve_config).

    Deriving argv from the structured spec — rather than building it independently — is what
    lets run_spec.json store only the structured `serve_config`: the argv is reproducible from
    it, and this is the same function launch_server uses, so the launch matches the record.
    """
    argv = [
        "vllm", "serve", sc["model"],
        "--port", str(sc["port"]),
        "--tensor-parallel-size", str(sc["tensor_parallel_size"]),
        "--data-parallel-size", str(sc["data_parallel_size"]),
        "--dtype", sc["dtype"],
        "--kv-cache-dtype", sc["kv_cache_dtype"],
        "--max-model-len", str(sc["max_model_len"]),
        "--max-num-batched-tokens", str(sc["max_num_batched_tokens"]),
        "--max-num-seqs", str(sc["max_num_seqs"]),
    ]
    if sc.get("num_gpu_blocks_override"):  # None/0 → omit → vllm auto-sizes
        argv += ["--num-gpu-blocks-override", str(sc["num_gpu_blocks_override"])]
    argv += [
        "--additional-config", json.dumps(sc["additional_config"]),
        "--profiler-config", json.dumps(sc["profiler_config"]),
        "--no-enable-log-requests",
    ]
    for field in ("block_size", "optimization_level"):
        if field in sc:
            argv += ["--" + field.replace("_", "-"), str(sc[field])]
    # JSON-valued passthroughs. kv_transfer_config is what makes a DECODE-ONLY server
    # possible (see decode_isolation.py): with kv_role=kv_consumer the engine treats every
    # prompt's KV as already present and skips prefill entirely. It MUST be emitted from
    # serve_config rather than appended by a caller, otherwise run_spec.json records a
    # configuration the launch did not use and a drift check compares the wrong thing.
    for field in ("kv_transfer_config", "limit_mm_per_prompt"):
        if sc.get(field):
            argv += ["--" + field.replace("_", "-"), json.dumps(sc[field])]
    for field in ("enable_prefix_caching", "async_scheduling"):
        if field in sc:
            if not isinstance(sc[field], bool):
                raise ValueError(f"{field} must be boolean")
            argv += ["--" + ("" if sc[field] else "no-") + field.replace("_", "-")]
    return argv
