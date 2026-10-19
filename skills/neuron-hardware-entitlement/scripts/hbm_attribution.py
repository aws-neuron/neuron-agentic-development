"""Attribute ACTUAL per-tensor resident HBM to operators + diff vs the floor.

The companion to ``operator_entitlement.py``. That script gives the THEORETICAL
per-operator HBM floor (oracle FLOPs/weights + analytic KV, per device). This one
takes the ACTUAL per-device resident manifest from the deployed graph and:

  1. parses every resident tensor (size + shape) from the compiled graph,
  2. attributes each tensor to an operator class by its shape signature,
  3. folds onto the oracle's ``by_type`` schema, and
  4. diffs actual_bytes - floor_bytes per operator -> a ranked overage table.

This is a *matched breakdown only* -- it answers "which operators use more HBM
than their floor, and by how much," not "why." The why is the next step.

WHERE THE RESIDENT BYTES ACTUALLY LIVE (important):
DRAM allocation CSVs from the memory-analysis dump only track the
coloring-allocator-managed DRAM scratch (collectives, dots, spills) -- a few tens
of MB. They do NOT track the model weights and the paged KV pool, because those
enter the graph as fixed-address INPUT constants (the hbm_usage pass calls them
"Unallocated Tensors"). The authoritative resident manifest is the compiled
graph's ENTRY signature: every weight + KV tensor with its exact TP-sharded shape.
Summing it reproduces the "Unallocated Tensors" figure from the hbm_usage log. So
this attributes the ENTRY manifest, not the allocation CSVs.

The actual graph may fuse operators the oracle splits (a fused ``qkv_proj`` = q +
k + v, a fused ``gate_up_proj`` = gate + up), and may or may not tie the embedding
to the lm_head. Attribution maps each resident tensor to a small set of
shape-signature classes -- DERIVED FROM THE MODEL ARCH AND THE FLOOR'S TP, not
hardcoded -- and folds the matching oracle floors together so the diff is
apples-to-apples. See ``build_signatures`` and ``floor_for_class``.

The Arch comes from the same run-spec the floor was generated under (default:
``--runspec``, a path or a model name), so the signatures match
the model the floor describes. The TP used for sharding is read from the floor JSON.

Usage:
    python3.12 hbm_attribution.py \
        --entry-hlo captured_entry_manifest.hlo \
        --floor-json operator_entitlement_decode.json \
        --runspec run-spec.json \
        -o actual_vs_floor.json
"""

import argparse
import json
import os
import re
from collections import defaultdict

import _planner

_planner.ensure_engine_on_path()

import operator_entitlement as oe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

_DTYPE_BYTES = {"bf16": 2, "f32": 4, "f16": 2, "s64": 8, "s32": 4,
                "pred": 1, "s8": 1, "f8e4m3": 1, "f8e5m2": 1, "u8": 1}


# --- ENTRY manifest parsing ---------------------------------------------------

def parse_entry_manifest(hlo_path):
    """Parse the compiled graph's ENTRY signature into a list of resident tensors.

    Returns a list of dicts {dtype, dims (tuple), bytes}. These are the per-device
    resident inputs: weights + paged KV pool + small norm/bias/io tensors."""
    with open(hlo_path) as f:
        hlo = f.read()
    m = re.search(r"ENTRY %[\w.]+ \((.*?)\) -> \(", hlo, re.S)
    if not m:
        raise ValueError(f"no ENTRY signature found in {hlo_path}")
    params = m.group(1)
    rows = []
    for mm in re.finditer(r"(\w+): (\w+)\[([^\]]*)\]", params):
        dtype, shp = mm.group(2), mm.group(3)
        dims = tuple(int(x) for x in shp.split(",") if x.strip().lstrip("-").isdigit())
        n = 1
        for d in dims:
            n *= d
        rows.append({"dtype": dtype, "dims": dims,
                     "bytes": n * _DTYPE_BYTES.get(dtype, 2)})
    return rows


# --- shape signatures derived from the model Arch + TP ------------------------

def build_signatures(arch, TP):
    """Derive the per-device weight/KV shape signatures for this model under TP.

    Returns:
      - ``two_d``: {(rows, cols) -> (class_name, fold_keys)} for 2D weight tensors.
        Keys are ORDERED -- orientation distinguishes the hidden-consuming projections
        ``(H, shard)`` (qkv, gate_up) from the hidden-producing ones ``(shard, H)``
        (o_proj, down), which would otherwise collide as the same dim pair. Both fused
        (qkv, gate_up) and split forms are registered so the attribution matches
        whatever the compiler emitted. ``fold_keys`` are the (op, group_label) rows of
        the operator floor this class folds together.
      - ``kv_by_head_dim``: {head_dim -> (class_name, fold_keys)} for the 4D KV pool,
        keyed on head_dim (the discriminator across attention groups).

    All sizes are per-device: column-parallel projections (q/k/v/gate/up/lm_head)
    shard their OUTPUT dim 1/TP; row-parallel projections (o/down) shard their INPUT
    dim 1/TP; KV heads shard to min(TP, n_kv) then replicate. The fused qkv weight
    always carries q + 2*kv columns -- the ``attention_k_eq_v`` alias is a floor-side
    correction, not a weight-storage one, so it never shrinks the signature."""
    H, I = arch.hidden_size, arch.intermediate_size
    two_d = {}

    def reg(a, b, cls, fold):
        two_d.setdefault((a, b), (cls, fold))  # first registration wins on collision

    # MLP (shared shape across groups; MoE replicates the same expert shapes).
    inter = I // TP
    all_glabels = [oe.group_label(arch, gi, g) for gi, g in enumerate(arch.attn_groups)]
    gate_fold = [("gate_proj", L) for L in all_glabels] + [("up_proj", L) for L in all_glabels]
    down_fold = [("down_proj", L) for L in all_glabels]
    reg(H, 2 * inter, "gate_up_proj", gate_fold)   # fused gate+up
    reg(H, inter, "gate_up_proj", gate_fold)       # split gate or up
    reg(inter, H, "down_proj", down_fold)

    # Attention projections, per group.
    kv_by_head_dim = {}
    for gi, g in enumerate(arch.attn_groups):
        L = oe.group_label(arch, gi, g)
        q_dim = g.n_q_heads * g.head_dim // TP
        kv_shard = min(TP, g.n_kv_heads)
        kv_dim = g.n_kv_heads * g.head_dim // kv_shard
        qkv_fold = [("q_proj", L), ("k_proj", L), ("v_proj", L)]
        reg(H, q_dim + 2 * kv_dim, f"qkv_proj_{L}", qkv_fold)  # fused qkv
        reg(H, q_dim, f"qkv_proj_{L}", qkv_fold)               # split q
        reg(H, kv_dim, f"qkv_proj_{L}", qkv_fold)              # split k or v
        reg(q_dim, H, f"o_proj_{L}", [("o_proj", L)])
        kv_by_head_dim.setdefault(g.head_dim, (f"kv_cache_{L}", [("sdpa", L)]))
        # MoE router: (hidden x num_experts), REPLICATED so its shape does not carry /TP.
        # Without this a real graph's gate tensor falls through to `unattributed_`. Both
        # orientations are registered because NeuronxDistributed can additionally hold the
        # TRANSPOSE (`store_transposed_weights` registers a second `weight_T` Parameter) -- in
        # which case the router class legitimately shows ~2x its floor, since holding two
        # layouts of the same weight is an implementation choice, not entitlement.
        if g.num_experts(arch) > 0:
            reg(H, g.num_experts(arch), f"moe_router_{L}", [("moe_router", L)])
            reg(g.num_experts(arch), H, f"moe_router_{L}", [("moe_router", L)])

    # Embedding / lm_head table: vocab/TP x hidden.
    reg(arch.vocab_size // TP, H, "embed_or_lm_head", [("lm_head", "head")])
    return two_d, kv_by_head_dim


def make_classifier(arch, TP):
    """Return ``classify(dims, dtype) -> (class_name, fold_keys)`` for this model."""
    two_d, kv_by_head_dim = build_signatures(arch, TP)

    def classify(dims, dtype):
        if dtype in ("s64", "s32", "pred"):
            return "io_scalar", []
        if len(dims) == 4:
            hd = dims[3]
            if hd in kv_by_head_dim:
                return kv_by_head_dim[hd]
            return f"kv_cache_hd{hd}", []
        if len(dims) == 2:
            hit = two_d.get(dims)
            if hit:
                return hit
            return f"unattributed_{dtype}{dims}", []
        if len(dims) == 1:
            return "norm_or_bias", []
        return f"unattributed_{dtype}{dims}", []

    return classify


def attribute(rows, classify):
    """Roll resident tensors up per operator class, recording each class's fold keys."""
    buckets = defaultdict(lambda: {"bytes": 0, "count": 0, "shapes": set(), "fold": []})
    for r in rows:
        cls, fold = classify(r["dims"], r["dtype"])
        b = buckets[cls]
        b["bytes"] += r["bytes"]
        b["count"] += 1
        b["shapes"].add((r["dtype"], r["dims"]))
        b["fold"] = fold
    return buckets


# --- floor side (from operator_entitlement_*.json) ----------------------------

def load_floor(floor_json_path):
    """Return (by_op_group, config). by_op_group keyed (op, group) ->
    {weight_bytes, kv_bytes, act_bytes}. ``act_bytes`` is interface-volume metadata and does
    not contribute to the resident HBM floor."""
    with open(floor_json_path) as f:
        d = json.load(f)
    out = {}
    for t in d["by_type"]:
        out[(t["op"], t["group"])] = {
            "weight_bytes": t["weight_bytes"],
            "kv_bytes": t["kv_bytes"],
            "act_bytes": t["act_bytes"],
        }
    return out, d["config"]


def floor_for_class(fold_keys, by_op_group):
    """Sum the oracle weight+KV floor for the rows this class folds.

    Activation interface bytes are retained in ``floor_detail`` for diagnosis, but are not
    resident bytes and therefore never contribute to ``resident_floor_bytes``.
    """
    wt = kv = act = 0.0
    for key in fold_keys:
        fl = by_op_group.get(key)
        if fl:
            wt += fl["weight_bytes"]
            kv += fl["kv_bytes"]
            act += fl["act_bytes"]
    return {"weight_bytes": wt, "kv_bytes": kv, "act_bytes": act,
            "resident_floor_bytes": wt + kv}


# --- diff ----------------------------------------------------------------------

def diff(rows, classify, floor_json_path):
    buckets = attribute(rows, classify)
    by_op_group, floor_cfg = load_floor(floor_json_path)
    total_actual = sum(r["bytes"] for r in rows)

    table = []
    for cls in buckets:
        actual = buckets[cls]["bytes"]
        fl = floor_for_class(buckets[cls]["fold"], by_op_group)
        floor_b = fl["resident_floor_bytes"]
        overage = actual - floor_b
        ratio = (actual / floor_b) if floor_b > 0 else None
        is_kv = cls.startswith("kv_cache")
        table.append({
            "op_class": cls,
            "actual_bytes": actual,
            "actual_gb": round(actual / 1024**3, 4),
            "floor_bytes": floor_b,
            "floor_gb": round(floor_b / 1024**3, 4),
            "overage_bytes": overage,
            "overage_gb": round(overage / 1024**3, 4),
            "actual_over_floor": round(ratio, 3) if ratio is not None else None,
            "tensor_count": buckets[cls]["count"],
            "shapes": sorted(str(d) for _, d in buckets[cls]["shapes"]),
            "floor_detail": {k: fl[k] for k in ("weight_bytes", "kv_bytes", "act_bytes")},
            "config_mismatch_note": (
                "actual KV is the paged-pool reservation (block table, governed by "
                "num_blocks / memory utilization, config-independent within the run); "
                "floor KV is the analytic per-request KV for the floor's bucket/bs. "
                "NOT apples-to-apples -- see methodology." if is_kv else None),
        })
    table.sort(key=lambda x: -x["overage_bytes"])

    wt_actual = sum(t["actual_bytes"] for t in table
                    if not t["op_class"].startswith(("kv_cache", "io_")))
    wt_floor = sum(t["floor_detail"]["weight_bytes"] for t in table)
    return {
        "floor_config": floor_cfg,
        "total_actual_resident_bytes": total_actual,
        "total_actual_resident_gb": round(total_actual / 1024**3, 4),
        "weights_actual_gb": round(wt_actual / 1024**3, 4),
        "weights_floor_gb": round(wt_floor / 1024**3, 4),
        "per_op_class": table,
        "note": "resident_floor = oracle weight+KV bytes/device. overage = actual - "
                "floor, ranked descending. Weights are config-independent and match "
                "apples-to-apples; KV crosses a config/sizing boundary (paged pool vs "
                "analytic per-request) -- flagged per row. Matched breakdown only; "
                "root-cause is a separate step.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry-hlo", default=None,
                    help="compiled graph HLO whose ENTRY signature is the resident manifest")
    ap.add_argument("--floor-json", default=None,
                    help="operator floor JSON (default: <artifacts dir>/operator_entitlement_decode.json)")
    ap.add_argument("--runspec", required=True,
                    help="path to a run-spec JSON for the model arch (the one the agent wrote)")
    ap.add_argument("--project-root", default=None,
                    help="the PROJECT directory this run belongs to; the floor JSON is read from "
                         "and the report written to <project-root>/agent_artifacts/tmp/. Defaults "
                         "to $NEURON_PROJECT_ROOT, then to the cwd. Must already exist.")
    ap.add_argument("-o", "--out", default=None,
                    help="output JSON (default: <artifacts dir>/actual_vs_floor.json)")
    args = ap.parse_args()
    outdir = _planner.resolve_outdir(args.project_root)
    entry_hlo = args.entry_hlo or os.path.join(outdir, "captured_entry_manifest.hlo")
    floor_json = args.floor_json or os.path.join(outdir, "operator_entitlement_decode.json")
    out = args.out or os.path.join(outdir, "actual_vs_floor.json")

    runspec = _planner.resolve_runspec(args.runspec)
    arch, _, _, _, _ = _planner.load_runspec(runspec)
    _, floor_cfg = load_floor(floor_json)
    classify = make_classifier(arch, floor_cfg["TP"])

    rows = parse_entry_manifest(entry_hlo)
    total = sum(r["bytes"] for r in rows)
    print(f"parsed {len(rows)} resident ENTRY tensors, total {total/1024**3:.4f} GB")

    result = {"source_hlo": os.path.basename(entry_hlo),
              **diff(rows, classify, floor_json)}
    print(f"\n{'op_class':<20} {'actual_GB':>10} {'floor_GB':>10} {'overage_GB':>11} {'x':>7}")
    for t in result["per_op_class"]:
        x = t["actual_over_floor"]
        print(f"{t['op_class']:<20} {t['actual_gb']:>10.4f} {t['floor_gb']:>10.4f} "
              f"{t['overage_gb']:>11.4f} {('%.2f' % x) if x is not None else '  --':>7}")

    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
