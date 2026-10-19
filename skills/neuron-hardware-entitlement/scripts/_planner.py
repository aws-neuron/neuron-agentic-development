"""Locate the parallelism-planner skill and load its analytic engine + oracle.

This skill is a thin primitive on top of the engine that the
``neuron-parallelism-planner`` skill OWNS. We never copy that engine
in -- we put its ``scripts/`` dir on ``sys.path`` and import it, so a single parse
of the config and a single analytic model serve both skills and cannot drift.

The sibling skill is found by RELATIVE PATH: both skills install side by side under
``skills/`` (this skill imports the planner's engine rather than copying it), so the
sibling is always ``../neuron-parallelism-planner/scripts``. No
environment variable is involved.

A run-spec is a direct PATH on the COMMAND LINE (``--runspec``). It is the same
run-spec JSON the planner reads: model display name, ``hf_config`` path, ``hardware``
constants path, ``configs`` cell list. The agent writes it (typically into
``agent_artifacts/tmp/`` beside the hf.json it references by bare filename).

THE ONE ENVIRONMENT VARIABLE, and there are no others:

  NEURON_PROJECT_ROOT   where this run's artifacts go (<root>/agent_artifacts/...)
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_HERE_SKILL = os.path.normpath(os.path.join(_HERE, ".."))   # <skill>/scripts -> <skill>

ARTIFACTS_SUBDIR = os.path.join("agent_artifacts", "tmp")
PROJECT_ROOT_ENV = "NEURON_PROJECT_ROOT"


def resolve_runspec(value: str) -> str:
    """A direct PATH to a run-spec. No model-name / environment resolution.

    The agent writes the run-spec (typically into ``agent_artifacts/tmp/`` beside the hf.json it
    references) and passes its path here. There is no run-spec environment variable and no shipped
    model catalog to name -- a floor silently computed for whatever model was last resolved looks
    exactly like a real one.
    """
    if os.path.isfile(value):
        return value
    raise SystemExit("error: run-spec not found: %s" % value)


def planner_scripts_dir() -> str:
    """Absolute path to the parallelism-planner skill's ``scripts/`` dir.

    Found by relative path from this skill: both skills are installed side by side under
    ``skills/`` (this skill imports the planner's engine rather than copying it), so the sibling
    is always at ``../neuron-parallelism-planner/scripts``. No environment variable
    is involved -- the layout is fixed by how the two skills ship.
    """
    return os.path.normpath(os.path.join(
        _HERE, "..", "..", "neuron-parallelism-planner", "scripts"))


def ensure_engine_on_path() -> str:
    """Put the planner engine on ``sys.path`` (idempotent). Returns the dir."""
    d = planner_scripts_dir()
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"parallelism-planner scripts dir not found at {d!r}. This skill imports that skill's "
            f"engine; the two must be installed side by side under skills/.")
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def _inside_skill_package(path: str) -> bool:
    """True when `path` is this skill package or below it."""
    try:
        return os.path.commonpath([os.path.abspath(path), _HERE_SKILL]) == _HERE_SKILL
    except ValueError:      # different drives / unrelated roots
        return False


def project_root(explicit: str = None) -> str:
    """The PROJECT root: where this run's ``agent_artifacts/`` tree lives.

    Same input, same resolution, and the same env var as the planner skill's
    ``planner.emit.project_root`` -- deliberately, because the two skills are used together on one
    analysis and their artifacts belong in one evidence trail. ``explicit`` (a ``--project-root``
    flag) > ``$NEURON_PROJECT_ROOT`` > the cwd. The cwd fallback is the ordinary case, so nothing
    new is required to run these scripts; the input exists for callers whose cwd is NOT the
    project (a test harness, or a run-spec read in place from somewhere else).

    An explicitly named root must already exist -- creating it would turn a typo into a plausible
    but empty evidence trail somewhere nobody will look.
    """
    root = explicit or os.environ.get(PROJECT_ROOT_ENV)
    if not root:
        return os.getcwd()
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise RuntimeError(
            "project root does not exist: %s. --project-root / $%s must name an EXISTING "
            "project directory (artifacts land in its %s)."
            % (root, PROJECT_ROOT_ENV, ARTIFACTS_SUBDIR))
    return root


def artifacts_dir(project_root_arg: str = None) -> str:
    """Directory for GENERATED artifacts (entitlement floors, attribution reports).

    Artifacts ALWAYS land in ``<project root>/agent_artifacts/tmp`` -- the convention shared with
    the other Neuron skills, which is what lets one run's floors, profiles and reports sit in one
    evidence trail. There is exactly ONE knob, the project root:

      1. ``project_root_arg`` -- the ``--project-root`` CLI flag.
      2. ``$NEURON_PROJECT_ROOT`` (the same variable the planner skill reads).
      3. the cwd, as a RELATIVE ``agent_artifacts/tmp`` (the default).

    There is deliberately no separate "put them exactly here" variable: ``$NEURON_ENTITLEMENT_OUTDIR``
    was removed along with the planner's and the profiling skills' equivalents, because four names
    for one idea let a single campaign scatter its evidence across unrelated directories. To send a
    run somewhere specific, point the project root there.

    If the destination is inside the skill package we REFUSE rather than guess (a hand-run from
    ``scripts/``, or a mis-pointed root): writing there would pollute a deployed skill, and quietly
    diverting elsewhere would hide the mistake.
    """
    if project_root_arg or os.environ.get(PROJECT_ROOT_ENV):
        d = os.path.join(project_root(project_root_arg), ARTIFACTS_SUBDIR)
    else:
        d = ARTIFACTS_SUBDIR
    if _inside_skill_package(d):
        raise RuntimeError(
            "refusing to write generated artifacts inside the skill package (destination=%s). "
            "These scripts expect a PROJECT directory, where artifacts land in %s/. Either run "
            "them from the project, pass --project-root, or set $%s."
            % (os.path.abspath(d), ARTIFACTS_SUBDIR, PROJECT_ROOT_ENV))
    os.makedirs(d, exist_ok=True)
    return d


def resolve_outdir(project_root_arg: str = None) -> str:
    """``artifacts_dir()`` for a CLI: the same destination, but reported as a one-line error.

    These scripts run an oracle forward pass before they write anything, so an unusable
    destination must surface up front and as an operator instruction -- not as a traceback after
    the work.
    """
    try:
        return artifacts_dir(project_root_arg)
    except RuntimeError as e:
        raise SystemExit("error: %s" % e)


def load_runspec(path: str):
    """Return (arch, hw, configs, raw, runspec_dir) for a run-spec path.

    Parses like the planner and then applies the planner's own serving knobs via
    ``entitlement_formulas.apply_runspec_knobs``, so an FP8 run-spec
    (``weight_dtype_bytes`` / ``compute_precision``), an explicit untied-lm-head
    precision exception (``lm_head_dtype_bytes``), an fp8 KV cache
    (``kv_dtype_bytes``), a DCP-prefill server (``apply_prefill_dcp``) or a tightened
    memory budget (``usable_hbm_fraction``) yields the SAME floors here as
    ``python -m planner`` on the identical run-spec. Parsing the run-spec without
    applying the knobs would silently report BF16 floors for an FP8 config -- roughly
    2x high on the decode weight-read floor and 2x slow on the prefill compute floor --
    and a cross-check that disagrees with the planner is worse than no cross-check.

    We deliberately do NOT call ``entitlement_formulas.load_runspec``: it requires a
    ``configs`` list and returns a 4-tuple, whereas the entitlement scripts accept a
    minimal run-spec with no cells (``hbm_attribution`` only needs ``arch``) and need the
    absolute run-spec dir for resolving report paths.
    """
    import json

    ensure_engine_on_path()
    from entitlement_formulas import apply_runspec_knobs, arch_from_hf, load_hardware

    with open(path) as f:
        raw = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    with open(os.path.join(base, raw["hf_config"])) as f:
        arch = arch_from_hf(json.load(f))
    hw = load_hardware(os.path.join(base, raw["hardware"]))
    apply_runspec_knobs(arch, hw, raw)
    return arch, hw, raw.get("configs", []), raw, base


# --- recovered `sweep.cell_entitlement` surface --------------------------------
# The planner retired scripts/sweep.py in 878948e ("segment-based cost engine, retire the
# old sweep"), which replaced `sweep.cell_entitlement` with `planner.build.cell` and moved
# `corrected_weight_bytes` / `embed_table_bytes` into entitlement_formulas.py. That commit
# did not touch THIS skill, so config_entitlement.py / operator_entitlement.py / (via the
# latter) hbm_attribution.py have been dead at import ever since -- `import sweep` raises
# ModuleNotFoundError against a checkout at or after that commit.
#
# We do NOT vendor the deleted sweep.py back in: it predates the FP8 knobs and reads
# `a.dtype_bytes` directly everywhere, so it cannot express a `weight_dtype_bytes: 1`
# run-spec at all. Restoring it would reinstate exactly the engine drift the retirement
# removed. Instead this module -- already the designated seam to the planner engine --
# absorbs the API change and re-exposes the one symbol with no modern equivalent.
#
# WHAT CHANGES vs the retired sweep (deliberate, both toward correctness):
#   * step_ms is now the engine's per-component roofline SERIALIZED across components
#     (sum of max(load, flop) + comm), not the old global max(wl, kv, mm, comm). The old
#     form implicitly assumed attention/FFN/lm_head overlap, which the inter-component
#     collectives forbid. TTFT / decode tok/s therefore move.
#   * FP8 / kv-fp8 / DCP / usable-HBM run-spec knobs now apply (load_runspec above).
# The four engine-total diagnostics are reconstructed from the same per-device byte/FLOP totals.
# Latency is composed from the planner's finer dynamic component records, which are returned
# alongside those totals so Hardware Entitlement does not rebuild the roofline policy.

_BUNDLE_CACHE: dict = {}


def bundle_for(arch, hw, raw=None):
    """A planner ``ModelBundle`` for an already-parsed (arch, hw), memoized.

    ``planner.build.build_bundle`` re-reads the run-spec from disk; the entitlement scripts
    already hold arch/hw, so we assemble the bundle directly to keep ONE parse (and one
    application of the run-spec knobs) per run.
    """
    key = (id(arch), id(hw))
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]
    ensure_engine_on_path()
    from planner.build import ModelBundle, require_lnc2
    from planner.segments import (
        build_flop_segments, build_kv_segments, build_weight_segments, _attn_type_label,
    )

    require_lnc2(hw)
    b = ModelBundle(
        arch=arch, hw=hw,
        weight_segs=build_weight_segments(arch),
        kv_segs=build_kv_segments(arch),
        flop_segs=build_flop_segments(arch),
        meta={_attn_type_label(g): (g.n_kv_heads, g.mla is not None)
              for g in arch.attn_groups},
        raw=raw or {},
    )
    _BUNDLE_CACHE[key] = b
    return b


def cell_entitlement(arch, hw, TP, PP, CP, S, bs, prefill, EP=1) -> dict:
    """One cell's entitlement, in the shape the entitlement scripts consume.

    Signature-compatible with the retired ``sweep.cell_entitlement``. Returns the engine
    decomposition in ms (``weight_load`` / ``kv_load`` / ``activation_load`` / ``matmul`` /
    additive ``comm``), the TE/HBM binding component, ``step_ms``, ``ttft_ms``-or-``itl_ms``,
    ``throughput_tps``, and ``peak_hbm_gb`` -- computed by the current planner engine, so a
    run-spec's FP8 / kv-dtype / DCP knobs are honored here exactly as in
    ``python -m planner``.

    PP must be 1: pipeline parallelism is out of scope for the planner engine (not
    implemented by the Neuron inference stack; ``ShardConfig`` has no PP field). We refuse
    rather than silently ignore a PP>1 request, which would report single-stage latency.

    EP is the old expert-parallel DEGREE. The engine shards routed experts across
    ``world = TP*DP``, so degree E is expressed as DP=E with EP enabled: routed experts
    then divide by TP*E while every other class still divides by TP -- the retired
    engine's split (``routed/(TP*PP*EP)``, ``non_expert/(TP*PP)``) at PP=1.
    """
    if PP != 1:
        raise ValueError(
            "PP=%r: pipeline parallelism is out of scope for the planner engine (not "
            "supported by the Neuron inference stack), so no per-stage entitlement can be "
            "produced. Re-run with PP=1." % (PP,))
    ensure_engine_on_path()
    from planner.build import cell as _cell
    from planner.build import ep_default_enabled
    from planner.plan import peak_hbm
    from planner.shard import ShardConfig

    b = bundle_for(arch, hw)
    # EP-enabled is a property of the MODEL, not of the requested degree: a model with routed
    # experts is expert-parallel even at degree 1 (the grid is then just the TP group), which is
    # how `planner.build.build_grid` constructs its own configs. Deriving it from `EP > 1` made
    # a MoE model at degree 1 look dense to the engine.
    cfg = ShardConfig(TP=TP, CP=CP, DP=max(int(EP), 1),
                      ep_enabled=ep_default_enabled(arch))
    c = _cell(b, cfg, S, bs, prefill)

    # Engine-total decomposition, from the SAME per-device totals the step time uses.
    # Weight, KV, and activation are sub-attributions of one HBM stream; the component
    # floor combines their bytes before applying the HBM/TE maximum.
    bw, te = hw.hbm_bw_bytes_per_s, hw.te_peak_flops
    wl = c["per_device_weight_bytes"] / bw * 1e3
    kv = c["per_device_kv_bytes"] / bw * 1e3
    act = c["per_device_activation_hbm_bytes"] / bw * 1e3
    mm = c["per_device_flops"] / te * 1e3
    cm = c["comm_ms"]
    comps = {
        "weight_load": wl,
        "kv_load": kv,
        "activation_load": act,
        "matmul": mm,
        "comm": cm,
    }
    binding_terms = {
        name: comps[name]
        for name in ("weight_load", "kv_load", "activation_load", "matmul")
    }

    step_ms = c["step_ms"]
    n_tok = (S * bs) if prefill else bs
    out = {
        "TP": TP, "PP": 1, "CP": CP, "EP": int(EP), "cores": cfg.world,
        "bucket": S, "bs": bs, "prefill": prefill,
        **comps,
        "step_ms": step_ms,
        # Communication remains on the serialized critical path and in ``step_ms``, but
        # binding identifies the certified TE/HBM optimization constraint.
        "binding_constraint": max(binding_terms, key=binding_terms.get),
        "binding_engine": c["binding"],
        "throughput_tps": n_tok * 1e3 / step_ms if step_ms else 0.0,
        "arith_intensity": c["ai"],
        "activation_io_policy": arch.activation_io_policy,
        "peak_hbm_gb": peak_hbm(b, cfg, S, bs, prefill) / 1024**3,
        "by_type": c["by_type"],
        "components": c["components"],
        "component_composition": c["component_composition"],
    }
    out["ttft_ms" if prefill else "itl_ms"] = step_ms
    return out
