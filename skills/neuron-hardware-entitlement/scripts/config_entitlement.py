"""End-to-end hardware entitlement for a SINGLE parallelism config.

This is the top-level primitive: one ``{TP, PP, CP, EP}`` choice plus a bucket-regime
(a list of ``(category, bs, bucket, decode/prefill)`` cells) goes in; the per-cell
theoretical bound comes out. For each cell it calls the validated analytic engine
(``_planner.cell_entitlement``) to get the engine-total decomposition
(weight_load / kv_load / activation_load / matmul, plus additive comm), the TE/HBM
binding constraint, the decode-tps or prefill-ms bound, and per-device peak HBM. It then
RECONCILES each cell against the
per-operator breakdown (``operator_entitlement.analyze``): summing the per-op
compute / weight-read / KV-read floors reproduces the cell's matmul / weight_load /
kv_load, proving the operator decomposition is the same bound opened up, not a
re-derivation. Every artifact states its certified component-engine scope explicitly:
the scope is TE and HBM. An engine outside that scope is unmodeled by this floor and
excluded from its binding maximum; it may still bind measured execution and is not zero work.

EP is the expert-parallel degree, taken from the pinned run-spec row (or the regime file)
like TP/PP/CP. It matters: the planner turns EP ON by default for any model with routed
experts and sweeps the DP axis, so a MoE model's recommended config is normally EP>1 --
evaluating it at EP=1 would report a weight floor num_local_experts times too high.

The analytic engine and oracle are OWNED by the ``neuron-parallelism-planner``
skill and imported through ``_planner`` -- never copied -- so a single config parse
and a single analytic model serve both skills and cannot drift.

A run-spec (``--runspec``, a path or a model name) supplies the model
(HF ``config.json``) and hardware constants. The bucket-regime defaults to the
run-spec's ``configs`` list (pinned to one parallelism config); override it with a
regime file of ``{"TP","PP","CP","EP","cells":[{"category","bucket","bs","prefill"},...]}``.

    python3.12 config_entitlement.py                       # run-spec cells
    python3.12 config_entitlement.py --regime my_cells.json
    python3.12 config_entitlement.py --runspec ./agent_artifacts/tmp/run.json
"""

import argparse
import json
import os

import _planner

_planner.ensure_engine_on_path()

import operator_entitlement as oe  # noqa: E402
# The weight-correction helpers moved from the retired sweep.py into entitlement_formulas.py
# (878948e); cell_entitlement is re-exposed by _planner on the current engine.
from entitlement_formulas import corrected_weight_bytes, embed_table_bytes  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Map the engine's component names to human binding-constraint labels.
BINDING_LABEL = {"weight_load": "weight HBM read", "kv_load": "KV HBM read",
                 "activation_load": "activation HBM read/write",
                 "matmul": "matmul"}


def load_regime(path):
    """A regime override file -> (TP, PP, CP, EP, cells). Cells need bucket/bs/prefill."""
    with open(path) as f:
        spec = json.load(f)
    cells = [{"category": c.get("category", f"b{c['bucket']}_"
                                + ("prefill" if c.get("prefill") else "decode")),
              "bucket": c["bucket"], "bs": c["bs"], "prefill": bool(c.get("prefill"))}
             for c in spec["cells"]]
    return spec["TP"], spec.get("PP", 1), spec.get("CP", 1), spec.get("EP", 1), cells


def default_regime(configs):
    """Derive a bucket-regime from the run-spec's ``configs`` list.

    The run-spec varies TP/PP/CP/EP per row; for a single-config artifact we pin the
    parallelism to the first row and reuse every row's (bucket, bs, regime) as a
    cell, so each cell is evaluated under one consistent config."""
    if not configs:
        raise ValueError("run-spec has no 'configs'; pass --regime instead")
    head = configs[0]
    TP, PP, CP = head["TP"], head.get("PP", 1), head.get("CP", 1)
    EP = head.get("EP", 1)
    cells = []
    for r in configs:
        prefill = "prefill" in r.get("regime", "").lower()
        cat = f"{r['bucket'] // 1024}k_" + ("prefill" if prefill else "decode")
        cells.append({"category": cat, "bucket": r["bucket"], "bs": r["bs"], "prefill": prefill})
    return TP, PP, CP, EP, cells


def entitlement_cell_out(arch, hw, TP, PP, CP, EP, cell, model_name, hardware_name):
    """One cell dict, with the operator-level reconciliation attached."""
    c = _planner.cell_entitlement(arch, hw, TP, PP, CP, cell["bucket"], cell["bs"],
                                  cell["prefill"], EP=EP)
    decomp = {"weight_load_ms": round(c["weight_load"], 3),
              "kv_load_ms": round(c["kv_load"], 3),
              "activation_load_ms": round(c["activation_load"], 3),
              "matmul_ms": round(c["matmul"], 3),
              "comm_ms": round(c["comm"], 3)}
    op = oe.analyze(arch, hw, {"TP": TP, "PP": PP, "CP": CP, "EP": EP,
                               "bucket": cell["bucket"],
                               "bs": cell["bs"], "prefill": cell["prefill"]},
                    model_name, hardware_name, analytic_cell=c)
    return {
        "category": cell["category"],
        "bs": cell["bs"], "bucket": cell["bucket"],
        "TP": TP, "PP": PP, "CP": CP, "EP": EP,
        # cores comes from the engine (world = TP*EP): CP is carved from TP and adds no
        # cores, so multiplying it in here would overstate the config's footprint.
        "cores": c["cores"],
        "engine_scope": op["engine_scope"],
        "entitlement_prefill_ms": round(c["ttft_ms"], 3) if cell["prefill"] else None,
        "entitlement_decode_tps": None if cell["prefill"] else round(c["throughput_tps"], 1),
        "decomposition": decomp,
        "binding_constraint": BINDING_LABEL.get(c["binding_constraint"], c["binding_constraint"]),
        "peak_hbm_gb": round(c["peak_hbm_gb"], 2),
        "components": op["components"],
        "component_composition": op["component_composition"],
        "operator_reconciliation": op["reconciliation"]["rel_error"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runspec", default=None, required=True, help="path to a run-spec JSON (the one the agent wrote, typically under agent_artifacts/tmp/)")
    ap.add_argument("--project-root", default=None,
                    help="the PROJECT directory this run belongs to; the report lands in "
                         "<project-root>/agent_artifacts/tmp/. Defaults to $NEURON_PROJECT_ROOT, "
                         "then to the cwd -- pass it when the cwd is not the project (e.g. the "
                         "run-spec is read in place from elsewhere). Must already exist.")
    ap.add_argument("--regime", default=None, help="regime override JSON (see module docstring)")
    ap.add_argument("-o", "--out", default=None,
                    help="output JSON (default: <artifacts dir>/config_entitlement.json)")
    args = ap.parse_args()
    out = args.out or os.path.join(_planner.resolve_outdir(args.project_root),
                                   "config_entitlement.json")

    runspec = _planner.resolve_runspec(args.runspec)
    arch, hw, configs, raw, _ = _planner.load_runspec(runspec)
    model_name = raw.get("model", "model")
    hardware_name = raw.get("hardware_name", "hardware")

    if args.regime:
        TP, PP, CP, EP, cells = load_regime(args.regime)
    else:
        TP, PP, CP, EP, cells = default_regime(configs)

    out_cells = [entitlement_cell_out(arch, hw, TP, PP, CP, EP, c, model_name, hardware_name)
                 for c in cells]
    result = {
        "source": f"entitlement (analytic, oracle-validated), {model_name}, "
                  f"TP={TP} PP={PP} CP={CP} EP={EP}, {hardware_name}",
        "model": model_name,
        "hardware": hardware_name,
        "config": {"TP": TP, "PP": PP, "CP": CP, "EP": EP},
        "engine_scope": list(oe.CERTIFIED_ENGINE_SCOPE),
        "corrected_weight_gb": round(corrected_weight_bytes(arch) / 1e9, 2),
        "embed_lm_head_gb": round(embed_table_bytes(arch) / 1e9, 2),
        "hbm_capacity_gb_per_core": round(hw.hbm_capacity_bytes_per_core / 1024**3, 1),
        "cells": out_cells,
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"=== config entitlement: TP{TP} PP{PP} CP{CP} EP{EP} "
          f"(weights {result['corrected_weight_gb']} GB, {model_name}) ===")
    print("certified component-engine scope: "
          + "/".join(engine.upper() for engine in result["engine_scope"]))
    print(f"{'category':<14} {'bs':>4} {'bucket':>8} "
          f"{'wl':>7} {'kv':>7} {'act':>7} {'mm':>10} {'comm':>8}  "
          f"{'bound':<26} {'mem':>6}")
    for c in out_cells:
        d = c["decomposition"]
        print(f"{c['category']:<14} {c['bs']:>4} {c['bucket']:>8} "
              f"{d['weight_load_ms']:>7.2f} {d['kv_load_ms']:>7.2f} "
              f"{d['activation_load_ms']:>7.2f} {d['matmul_ms']:>10.2f} "
              f"{d['comm_ms']:>8.2f}  {c['binding_constraint']:<26} {c['peak_hbm_gb']:>5.1f}G")
        rec = c["operator_reconciliation"]
        worst = max(rec.values())
        if worst > 0.05:
            print(f"               !! operator reconciliation off by {worst:.2%}: {rec}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
