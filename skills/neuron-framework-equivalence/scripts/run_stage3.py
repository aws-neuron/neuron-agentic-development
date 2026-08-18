#!/usr/bin/env python3
"""
Stage 3: Fault Localization.

Reads Stage 2 output (component R-ratios) and performs change-point detection
+ root-cause classification. No external dependencies beyond numpy.

Usage:
    python3 scripts/run_stage3.py \
        --stage2-output ${EXP_DIR}/results/stage2.json \
        --tau-r 1.2 \
        --output ${EXP_DIR}/results/stage3.json
"""

import argparse
import json
import os
import sys
import math


def classify_root_cause(r_ratio, component_name=""):
    """Classify likely root cause based on R magnitude."""
    name = component_name.lower()
    if r_ratio > 10:
        cause = "missing_algorithm"
        desc = f"R={r_ratio:.1f} >> 10: likely missing algorithm or wrong formula."
        if any(kw in name for kw in ("rotary", "rope", "position")):
            desc += " Suspect: missing positional encoding variant (YaRN, NTK, etc.)."
        elif any(kw in name for kw in ("moe", "expert", "router", "mlp")):
            cause = "routing_ignored"
            desc += " Suspect: MoE routing weights ignored."
    elif r_ratio > 3:
        cause = "missing_multiplier"
        desc = f"R={r_ratio:.2f}: possible missing multiplier or partial algorithm gap."
    elif r_ratio > 1.2:
        cause = "precision_ordering"
        desc = f"R={r_ratio:.2f}: precision ordering issue or missing scaling factor."
        if any(kw in name for kw in ("norm", "rmsnorm", "layernorm")):
            desc += " Suspect: normalization computed in wrong precision."
    elif r_ratio < 1.0:
        cause = "over_precision"
        desc = f"R={r_ratio:.4f} < 1.0: target computes at higher precision than source."
    else:
        cause = "unknown"
        desc = f"R={r_ratio:.4f}: within tolerance."
    return cause, desc


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Fault Localization")
    parser.add_argument("--stage2-output", required=True)
    parser.add_argument("--tau-r", type=float, default=1.2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.stage2_output) as f:
        stage2 = json.load(f)

    results = stage2.get("results", [])
    failing = [r for r in results if not r.get("passed", True)]

    if not failing:
        print("Stage 3: No failures to localize. All components passed.")
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({"faults": [], "num_faults": 0}, f, indent=2)
        return 0

    # Change-point detection: spike vs step
    faults = []
    for i, r in enumerate(results):
        if r.get("passed", True):
            continue
        ratio = r.get("r_ratio")
        if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)):
            faults.append({
                "component": r["component"],
                "r_ratio": None,
                "pattern": "error",
                "root_cause": "test_error",
                "description": "Test failed before R-ratio could be computed.",
            })
            continue

        # Check if subsequent components also fail (step) or just this one (spike)
        subsequent_fails = sum(1 for j in range(i+1, min(i+3, len(results)))
                               if not results[j].get("passed", True))
        pattern = "step" if subsequent_fails >= 2 else "spike"

        cause, desc = classify_root_cause(ratio, r["component"])
        faults.append({
            "component": r["component"],
            "r_ratio": ratio,
            "pattern": pattern,
            "root_cause": cause,
            "description": desc,
        })

    # Sort by R-ratio descending (worst first)
    faults.sort(key=lambda f: -(f["r_ratio"] or 0))

    # Find primary fault (first step pattern)
    primary = next((f for f in faults if f["pattern"] == "step"), faults[0] if faults else None)

    print(f"Stage 3: {len(faults)} fault(s) localized")
    print(f"{'=' * 60}")
    if primary:
        print(f"  Primary fault: {primary['component']}")
        print(f"    R-ratio: {primary['r_ratio']}")
        print(f"    Pattern: {primary['pattern']}")
        print(f"    Root cause: {primary['root_cause']}")
        print(f"    {primary['description']}")
    print(f"\n  All faults:")
    for f in faults:
        r_str = f"R={f['r_ratio']:.4f}" if f['r_ratio'] is not None else "R=ERROR"
        print(f"    [{f['root_cause']}] {f['component']}: {r_str} ({f['pattern']})")
        print(f"      {f['description']}")
    print(f"{'=' * 60}")

    output = {"faults": faults, "num_faults": len(faults), "primary_fault": primary}
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return 1  # Always returns 1 since faults were found


if __name__ == "__main__":
    sys.exit(main())
