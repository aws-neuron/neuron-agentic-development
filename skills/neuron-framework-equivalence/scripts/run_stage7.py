#!/usr/bin/env python3
"""
Stage 7: Downstream Task Evaluation.

Two modes:
  - accuracy_debugger (vLLM-Neuron): uses LmEvalAnalyzer for per-sample deviation
    tracking, then optionally runs prompt-level diagnosis on failures.
  - neuron_bench (legacy/NxDI): aggregate score comparison via neuron_bench.

Usage:
    # vLLM-Neuron (preferred):
    python3 scripts/run_stage7.py \
        --model-path /path/to/model --tp-size 8 \
        --target-stack vllm_neuron \
        --eval-fn run_accuracy_gsm8k_cot \
        --thresholds '{"exact_match,flexible-extract": 0.435}' \
        --diagnose-failures

    # Legacy/NxDI:
    python3 scripts/run_stage7.py \
        --bench-config ${EXP_DIR}/bench_config.yaml \
        --output-dir ${EXP_DIR}/results/stage7 \
        --tolerance 0.02
"""

import argparse
import json
import sys
import os

# Historical Stage 7 sample cap. vLLM-Neuron 0.24 flipped the run_accuracy_*
# default to limit=None (full dataset); Stage 7 preserves the old 200-sample
# behavior by default so an omitted --limit doesn't silently launch a
# full-dataset eval. Use --full-dataset to run the whole dataset.
DEFAULT_SAMPLE_LIMIT = 200


def _run_via_accuracy_debugger(args):
    """Primary path for vLLM-Neuron: uses accuracy_debugger APIs."""
    if not args.model_path:
        raise SystemExit("Error: --model-path is required when using --target-stack vllm_neuron")

    sys.path.insert(0, os.path.dirname(__file__))
    from adapters import get_adapter

    adapter = get_adapter(target_stack=args.target_stack)
    output_dir = args.output_dir or "results/stage7"
    os.makedirs(output_dir, exist_ok=True)

    thresholds = {}
    if args.thresholds:
        thresholds = json.loads(args.thresholds) if isinstance(args.thresholds, str) else args.thresholds

    # Sample-count policy. The 0.24 run_accuracy_* runners default limit=None
    # (full dataset). To avoid surprise multi-hour full-dataset runs, Stage 7
    # keeps the historical safe default of 200 samples unless the user opts in:
    #   --limit N        -> N samples
    #   --full-dataset   -> limit=None (whole dataset)
    #   (neither)        -> DEFAULT_SAMPLE_LIMIT (200)
    eval_kwargs = {}
    if args.full_dataset:
        eval_kwargs["limit"] = None
    elif args.limit is not None:
        eval_kwargs["limit"] = args.limit
    else:
        eval_kwargs["limit"] = DEFAULT_SAMPLE_LIMIT

    eval_fn = None
    if args.eval_fn:
        # Eval runners moved from test.evaluation.eval_runners into source at
        # vllm_neuron.accuracy.lm_eval when the accuracy debugger was merged.
        from vllm_neuron.accuracy import lm_eval as eval_runners
        if not hasattr(eval_runners, args.eval_fn):
            raise SystemExit(f"Error: '{args.eval_fn}' not found in vllm_neuron.accuracy.lm_eval")
        eval_fn = getattr(eval_runners, args.eval_fn)
        if not callable(eval_fn):
            raise SystemExit(f"Error: '{args.eval_fn}' is not callable")

    analysis_kwargs = {
        "model_id": args.model_path,
        "tp_size": args.tp_size,
        "thresholds": thresholds,
        "output_dir": output_dir,
        "max_model_len": args.max_model_len,
    }

    if eval_fn:
        analysis_kwargs["eval_fn"] = eval_fn
        analysis_kwargs["eval_kwargs"] = eval_kwargs
    elif args.input_task_results:
        analysis_kwargs["input_task_results"] = args.input_task_results
    else:
        raise SystemExit("Error: provide --eval-fn or --input-task-results")

    print("=" * 60)
    print("Phase 1: Task-level accuracy analysis (accuracy_debugger)")
    print("=" * 60)

    result = adapter.run_accuracy_analysis(**analysis_kwargs)
    passed = result["passed"]
    deviated_prompts = result.get("deviated_prompts", [])

    print(f"\n  Scores: {result['scores']}")
    print(f"  Passed: {passed}")
    print(f"  Deviated prompts: {len(deviated_prompts)}")
    print(f"  Report: {result.get('report_path', 'N/A')}")

    all_results = {"accuracy_analysis": result}

    # ── Prompt-level diagnosis on failures ──
    if args.diagnose_failures and deviated_prompts:
        print("\n" + "=" * 60)
        print("Phase 2: Prompt-level diagnosis on deviated prompts")
        print("=" * 60)

        diagnosis_result = adapter.run_prompt_diagnosis(
            model_id=args.model_path,
            tp_size=args.tp_size,
            prompts=deviated_prompts[:5],
            output_length=16,
            output_dir=os.path.join(output_dir, "prompt_diagnosis"),
            max_model_len=args.max_model_len,
        )
        all_results["prompt_diagnosis"] = diagnosis_result
        print(f"  Diagnosis report: {diagnosis_result.get('report_path', 'N/A')}")

    summary_file = os.path.join(output_dir, "stage7_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  Stage 7: {'PASS' if passed else 'FAIL'}")
    print(f"  Results saved to {output_dir}")
    print(f"{'=' * 60}")
    return 0 if passed else 1


def _run_via_neuron_bench(args):
    """Fallback path for NxDI/other stacks: aggregate comparison via neuron_bench."""
    from neuron_bench.config import parse_config
    from neuron_bench.run import (
        run_lm_eval_scenarios,
        run_hf_lm_eval_scenarios,
    )
    from neuron_bench.compare import compare_results, print_comparison_report
    from neuron_bench.model_loader import load_model

    config = parse_config(args.bench_config)
    output_dir = args.output_dir or "results/stage7"
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    hf_results = None
    if args.hf_baseline and config.lm_eval:
        print("=" * 60)
        print("Phase 1: HF baseline (FP32 CPU)")
        print("=" * 60)
        hf_results = run_hf_lm_eval_scenarios(
            config.model["model_path"], config.lm_eval, output_dir,
        )
        all_results["lm_eval_hf"] = hf_results

    neuron_results = None
    if config.lm_eval:
        print("=" * 60)
        print("Phase 2: Neuron model")
        print("=" * 60)
        model, tokenizer, generation_config = load_model(config.model)
        neuron_results = run_lm_eval_scenarios(
            model, tokenizer, generation_config, config.lm_eval, output_dir,
        )
        all_results["lm_eval"] = neuron_results

    passed = True
    if hf_results and neuron_results:
        print("=" * 60)
        print("Phase 3: Comparing HF vs Neuron")
        print("=" * 60)
        comparison = compare_results(hf_results, neuron_results, tolerances=config.tolerance)
        all_results["comparison"] = comparison
        print_comparison_report(comparison)
        passed = comparison["overall_pass"]

    summary_file = os.path.join(output_dir, "stage7_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  Stage 7: {'PASS' if passed else 'FAIL'}")
    print(f"  Results saved to {output_dir}")
    print(f"{'=' * 60}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description="Stage 7: Downstream Task Evaluation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-stack", default=None, help="Serving stack: nxdi, vllm_neuron")

    # accuracy_debugger path (vllm_neuron)
    parser.add_argument("--model-path", default=None, help="HF model path")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--eval-fn", default=None, help="Eval runner function name from vllm_neuron.accuracy.lm_eval")
    parser.add_argument("--thresholds", default=None, help="JSON dict of metric→threshold")
    parser.add_argument("--input-task-results", default=None, help="Path to existing eval results (skip running eval)")
    parser.add_argument("--diagnose-failures", action="store_true", default=False,
                        help="Run prompt-level diagnosis on deviated prompts")
    parser.add_argument("--limit", type=int, default=None,
                        help=f"Samples per task. Omit to use the safe default "
                             f"({DEFAULT_SAMPLE_LIMIT}); pass e.g. --limit 50 for a "
                             f"quick check. Use --full-dataset to run everything.")
    parser.add_argument("--full-dataset", action="store_true", default=False,
                        help="Run the full dataset (limit=None). Overrides --limit. "
                             "Note: this can be a long/expensive run.")

    # neuron_bench path (legacy/nxdi)
    parser.add_argument("--bench-config", default=None, help="Path to neuron_bench YAML config")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--hf-baseline", action="store_true", default=True)
    args = parser.parse_args()

    if args.target_stack == "vllm_neuron":
        return _run_via_accuracy_debugger(args)
    elif args.bench_config:
        return _run_via_neuron_bench(args)
    else:
        parser.error("Provide --target-stack vllm_neuron (with --model-path) or --bench-config for legacy path")


if __name__ == "__main__":
    sys.exit(main())
