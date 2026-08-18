#!/usr/bin/env python3
"""
Stage 6: Distributional and Semantic Verification.

Reads Stage 1 output (enhanced_metrics) and checks:
  Condition B: per-position cosine similarity >= theta
  Condition C: per-position KL divergence (top-k) <= delta

Usage:
    python3 scripts/run_stage6.py \
        --stage1-output ${EXP_DIR}/results/stage1.json \
        --theta 0.95 --delta 0.1 --delta-max 1.0 \
        --output ${EXP_DIR}/results/stage6.json
"""

import argparse
import json
import os
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Stage 6: Distributional & Semantic Verification")
    parser.add_argument("--stage1-output", required=True, help="Path to Stage 1 JSON output")
    parser.add_argument("--theta", type=float, default=0.95, help="Cosine similarity floor (Condition B)")
    parser.add_argument("--rho-tail", type=float, default=0.05, help="Max fraction of positions below theta")
    parser.add_argument("--delta", type=float, default=0.1, help="KL p95 ceiling (Condition C)")
    parser.add_argument("--delta-max", type=float, default=1.0, help="KL max ceiling (Condition C)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.stage1_output) as f:
        stage1 = json.load(f)

    em = stage1.get("enhanced_metrics", {})
    if not em.get("enhanced_metrics_available"):
        print("ERROR: Enhanced metrics not available in Stage 1 output.")
        print("Re-run Stage 1 with a model that supports output_scores=True.")
        return 1

    # ── Condition B: Semantic Consistency ──
    per_token_cos = em.get("per_token_cosine_similarity", [])
    cos_mean = em.get("logit_cosine_similarity_mean", 0)
    cos_min = em.get("logit_cosine_similarity_min", 0)

    if per_token_cos:
        cos_arr = np.array(per_token_cos)
        cos_p5 = float(np.percentile(cos_arr, 5))
        tail_fraction = float(np.mean(cos_arr < args.theta))
        condition_b = cos_p5 >= args.theta and tail_fraction <= args.rho_tail
    else:
        cos_p5 = cos_min
        tail_fraction = 0.0
        condition_b = cos_min >= args.theta

    print(f"Condition B (Semantic Consistency): {'PASS' if condition_b else 'FAIL'}")
    print(f"  cosine sim: mean={cos_mean:.6f}, min={cos_min:.6f}, p5={cos_p5:.6f}")
    print(f"  tail fraction below θ={args.theta}: {tail_fraction:.4f} (threshold ρ={args.rho_tail})")
    print(f"  positions: {len(per_token_cos)}")

    # ── Condition C: Distributional Equivalence ──
    # Collect all available KL variants
    kl_by_scope = {}
    for scope in ["top5", "top10", "top50", "top100"]:
        key = f"per_position_kl_{scope}"
        vals = em.get(key, [])
        if vals:
            kl_by_scope[scope] = vals
    full_kl = em.get("per_position_kl_divergence", [])
    if full_kl:
        kl_by_scope["full_vocab"] = full_kl

    # Primary check on top50 (filters tail noise)
    primary_scope = "top50" if "top50" in kl_by_scope else ("full_vocab" if "full_vocab" in kl_by_scope else None)

    if primary_scope and kl_by_scope[primary_scope]:
        kl_arr = np.array(kl_by_scope[primary_scope])
        kl_mean = float(np.mean(kl_arr))
        kl_p95 = float(np.percentile(kl_arr, 95))
        kl_max = float(np.max(kl_arr))
        condition_c = kl_p95 <= args.delta and kl_max <= args.delta_max
    else:
        # Fallback to aggregate KL
        kl_agg = em.get("kl_divergence", 0)
        kl_mean = kl_p95 = kl_max = kl_agg
        condition_c = kl_agg <= args.delta_max
        primary_scope = "aggregate"

    print(f"\nCondition C (Distributional Equivalence): {'PASS' if condition_c else 'FAIL'}")
    print(f"  Primary check on: {primary_scope}")

    if kl_by_scope:
        n_pos = len(next(iter(kl_by_scope.values())))
        print(f"  Per-position KL by top-k ({n_pos} positions):")
        print(f"    {'scope':<12s} {'mean':>10s} {'p95':>10s} {'max':>10s}")
        print(f"    {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        for scope in ["top5", "top10", "top50", "top100", "full_vocab"]:
            if scope in kl_by_scope:
                arr = np.array(kl_by_scope[scope])
                print(f"    {scope:<12s} {np.mean(arr):>10.4f} {np.percentile(arr, 95):>10.4f} {np.max(arr):>10.4f}")

    print(f"  Thresholds: δ={args.delta} (p95), δ_max={args.delta_max} (max)")

    # ── Additional metrics ──
    print(f"\n  Additional:")
    print(f"    Top-5 agreement: {em.get('top5_agreement', 0):.2%}")
    print(f"    Relative L2 error: mean={em.get('relative_l2_error_mean', 0):.6f}, max={em.get('relative_l2_error_max', 0):.6f}")
    if em.get("mean_prob_of_hf_token"):
        print(f"    P(HF token) under Neuron: {em['mean_prob_of_hf_token']:.6f}")

    # ── Verdict ──
    both_pass = condition_b and condition_c
    print(f"\n{'=' * 60}")
    print(f"  Stage 6: {'PASS' if both_pass else 'FAIL'}")
    print(f"  Condition B: {'PASS' if condition_b else 'FAIL'}, Condition C: {'PASS' if condition_c else 'FAIL'}")
    print(f"{'=' * 60}")

    result = {
        "condition_b_passed": condition_b,
        "condition_c_passed": condition_c,
        "passed": both_pass,
        "semantic": {
            "cos_mean": cos_mean, "cos_min": cos_min, "cos_p5": cos_p5,
            "tail_fraction": tail_fraction, "theta": args.theta,
        },
        "distributional": {
            "primary_scope": primary_scope,
            "kl_mean": kl_mean, "kl_p95": kl_p95, "kl_max": kl_max,
            "delta": args.delta, "delta_max": args.delta_max,
            "kl_by_scope": {
                scope: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}
                for scope, v in kl_by_scope.items()
            },
        },
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return 0 if both_pass else 1


if __name__ == "__main__":
    sys.exit(main())
