"""§4-output: serialize a plan() result to the planner JSON artifacts.

Pure serializer -- it RESHAPES the dict plan() already returned (plus the model bundle for
the entitlement metadata); it runs no cost math and never re-invokes cell(). Writes the
artifacts into the project's ``agent_artifacts/tmp`` (see ``artifacts_dir``):

  - pareto_frontier.json        -- per-bucket 3-axis Pareto decode cells (the `pareto` flag).
  - target_config.json          -- per-bucket selected colocated target + serving mode.
  - entitlement.json            -- model-level corrected weight bytes + min footprint.
"""

from __future__ import annotations

import json
import os

from entitlement_formulas import (
    corrected_weight_bytes, embed_table_bytes, load_hardware,  # noqa: F401
)
from planner.build import ModelBundle
from planner.plan import usable_hbm, min_kv_heads

HERE = os.path.dirname(os.path.abspath(__file__))
_SKILL_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))   # <skill>/scripts/planner -> <skill>


ARTIFACTS_SUBDIR = os.path.join("agent_artifacts", "tmp")

# The two -- and only two -- environment variables this family reads:
#
#   NEURON_PROJECT_ROOT   where THIS RUN's outputs go: <root>/agent_artifacts/...
#
# That is the ONE and only environment variable. Everything else is a direct argument: the
# run-spec is a path on the command line (the agent writes it, typically into agent_artifacts/tmp
# beside the hf.json it references), never named by a variable, and behavior is set by CLI flags.
PROJECT_ROOT_ENV = "NEURON_PROJECT_ROOT"


def project_root(explicit: str = None) -> str:
    """The PROJECT root: where this run's ``agent_artifacts/`` tree lives.

    The project is wherever the CALLER works -- the directory a user started their agent in, or
    any directory they nominate. It is an INPUT, not something to be discovered: the resolution
    is ``explicit`` (a ``--project-root`` flag) > ``$NEURON_PROJECT_ROOT`` > the cwd. The cwd
    fallback is the ordinary case and keeps the historical behavior, so nothing new is required
    to run the planner; the input exists for callers whose cwd is NOT the project.

    An explicitly named root must already exist. Creating it would turn a typo into a plausible
    but empty evidence trail somewhere nobody will look.
    """
    root = explicit or os.environ.get(PROJECT_ROOT_ENV)
    if not root:
        return os.getcwd()
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise RuntimeError(
            "project root does not exist: %s. --project-root / $%s must name an "
            "EXISTING project directory (artifacts land in its %s)."
            % (root, PROJECT_ROOT_ENV, ARTIFACTS_SUBDIR))
    return root


def artifacts_dir(project_root_arg: str = None) -> str:
    """The generated-artifacts directory, created if needed.

    Artifacts ALWAYS land in ``<project root>/agent_artifacts/tmp`` -- the same convention the
    other Neuron skills and the framework-optimization orchestrator use
    (``agent_artifacts/{tmp,data,traces}/``), which is what keeps one run's evidence trail in one
    place. There is exactly ONE knob, the project root:

      1. ``project_root_arg`` -- the ``--project-root`` CLI flag.
      2. ``$NEURON_PROJECT_ROOT``.
      3. the cwd, as a RELATIVE ``agent_artifacts/tmp`` (the default).

    There is deliberately no separate "put the artifacts exactly here" variable. This skill used to
    honor ``$PARALLELISM_PLANNER_OUTDIR``, and the profiling skills had ``ARTIFACTS_DIR`` /
    ``PROFILE_OUTPUT_DIR``: four names for one idea, each owned by a different skill, so a
    campaign that set some of them scattered its plan, floors, profiles and reports across
    unrelated directories and no reader could tell which run produced what. To point a run somewhere
    specific, point the project root there.

    If the resolved destination is inside the skill package we REFUSE rather than guess: that means
    either the module was hand-run from ``scripts/`` or a root was mis-pointed at the skill. Writing
    there would pollute a deployed skill, and quietly diverting to a temp dir would hide the mistake
    at the moment it is cheapest to correct. Resolved per call, not at import.
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


def _inside_skill_package(path: str) -> bool:
    """True when `path` is the skill package or below it."""
    try:
        return os.path.commonpath([os.path.abspath(path), _SKILL_ROOT]) == _SKILL_ROOT
    except ValueError:      # different drives / unrelated roots
        return False


def _min_footprint(a, hw) -> int:
    """Smallest TP that holds the weights -> functional-minimum core count."""
    import math
    return int(math.ceil(corrected_weight_bytes(a) / usable_hbm(hw)))


def emit(b: ModelBundle, p: dict, outdir: str = None) -> list:
    """Write the four artifacts from the plan dict `p`. Returns the paths written."""
    outdir = artifacts_dir() if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)
    a, hw = b.arch, b.hw
    model = b.raw.get("model")
    hw_name = b.raw.get("hardware_name", "hw")
    serving = p["serving"]

    pareto_out = {"model": model, "hardware": hw_name, "serving": serving,
                  "disabled_axes": p.get("disabled_axes", []),
                  "buckets": {str(S): [r for r in p["decode_grid"][S] if r.get("pareto")]
                              for S in p["buckets"]}}

    targets_out = {"model": model, "hardware": hw_name, "serving": serving,
                   "cores_per_node": hw.cores_per_node,
                   "disabled_axes": p.get("disabled_axes", []),
                   "prefill_itps_floor": p.get("prefill_itps_floor"),
                   "prefill_ttft_ceiling_ms": p.get("prefill_ttft_ceiling_ms"),
                   "decode_itl_ceiling_ms": p.get("decode_itl_ceiling_ms"),
                   "max_cores": p.get("max_cores"),
                   "max_nonreplicating_tp": min_kv_heads(a), "buckets": {}}
    for S in p["buckets"]:
        targets_out["buckets"][str(S)] = {"selected": p["selected"][S]}

    entitlement_out = {
        "source": f"entitlement (analytic, oracle-validated), {model}, {hw_name}",
        "model": model, "hardware": hw_name,
        "corrected_weight_gb": round(corrected_weight_bytes(a) / 1e9, 2),
        "embed_lm_head_gb": round(embed_table_bytes(a) / 1e9, 2),
        "hbm_capacity_gb_per_core": round(hw.hbm_capacity_bytes_per_core / 1024**3, 1),
        "min_footprint_cores": _min_footprint(a, hw),
    }

    written = []
    os.makedirs(outdir, exist_ok=True)
    for name, obj in (("pareto_frontier.json", pareto_out),
                      ("target_config.json", targets_out),
                      ("entitlement.json", entitlement_out)):
        path = os.path.join(outdir, name)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        written.append(path)
    return written
