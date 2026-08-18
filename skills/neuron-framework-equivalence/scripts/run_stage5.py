#!/usr/bin/env python3
"""
Stage 5: E2E Three-Tensor Comparison.

Compares last-position logits from HF FP32, HF BF16, and compiled Neuron model.
Requires the scripts/ directory on PYTHONPATH.

Usage:
    python3 scripts/run_stage5.py \
        --model-path /path/to/hf_model \
        --compiled-model-path /path/to/compiled_model \
        --model-class path/to/modeling.py:NeuronXxxForCausalLM \
        --config-class path/to/modeling.py:XxxInferenceConfig \
        --prompts "The capital of France is" "Water freezes at"
"""

import argparse
import json
import sys
import os

import torch
import numpy as np


def get_last_logits(model, input_ids, attention_mask=None):
    """Get last-position logits, handling both HF and NxDI models."""
    seq_len = input_ids.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        try:
            out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        except (TypeError, AttributeError) as e:
            if "DynamicCache" in str(e) or "get_usable_length" in str(e):
                out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
            else:
                try:
                    out = model(input_ids, attention_mask=attention_mask)
                except TypeError:
                    out = model(input_ids)

        logits = out.logits if hasattr(out, "logits") else out
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

    last = logits[:, -1, :].float()
    return torch.nan_to_num(last, nan=0.0, posinf=1e6, neginf=-1e6)


def _run_via_logit_validation(adapter, args):
    """Fallback Stage 5 for stacks whose reference-model helpers aren't importable.

    Uses the stack's own per-token logit validation instead of our three-tensor
    comparison. Currently only vLLM-Neuron provides this (its accuracy_debugger builds
    the FP32 and dtype references internally, so no HF reference model is needed here).

    Returns a result dict, or None if the stack has no equivalent.
    """
    if adapter.name != "vllm_neuron":
        return None

    print("\n  Falling back to the adapter's per-token logit validation.")
    print("  NOTE: this is a different measurement from the three-tensor R-ratio —")
    print("        it reports the stack's own pass/fail per token, not an R-ratio.")

    output_dir = os.path.dirname(os.path.abspath(args.output)) if args.output else "./accuracy_report"
    try:
        validation = adapter.run_logit_validation(
            model_id=args.model_path,
            tp_size=args.tp_size,
            prompts=args.prompts,
            output_length=args.num_tokens if hasattr(args, "num_tokens") else 16,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"  Logit-validation fallback also failed: {e}")
        return None

    plugin_results = validation.get("plugin_results", {}) or {}
    logit_val = plugin_results.get("logit_val", plugin_results)
    passed = bool(logit_val) and all(
        v.get("passed", False) if isinstance(v, dict) else False
        for v in logit_val.values()
    )

    print(f"\n{'=' * 70}")
    print(f"  STAGE 5 (logit-validation fallback): {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 70}")

    return {
        "passed": passed,
        "method": "adapter_logit_validation_fallback",
        "reason": "reference-model helpers not importable; used stack logit validation",
        "prompts": validation.get("prompts"),
        "report_path": validation.get("report_path"),
        "plugin_results": {k: str(v) for k, v in logit_val.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 5: E2E Three-Tensor")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-model-path", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--config-class", required=True)
    parser.add_argument("--prompts", nargs="+", default=[
        "The capital of France is", "Water freezes at", "The speed of light is approximately",
    ])
    parser.add_argument("--tau-r", type=float, default=1.2)
    parser.add_argument("--output", default=None)
    parser.add_argument("--target-stack", default=None, help="Serving stack: nxdi, vllm_neuron")
    parser.add_argument("--target-module-file", default=None, help="Path to target modeling file")
    parser.add_argument("--target-config-class", default=None, help="Config class name for adapter")
    parser.add_argument("--target-inner-class", default=None, help="Model class name for adapter")
    parser.add_argument("--kv-analysis-on-fail", action="store_true", default=False,
                        help="Run KV cache analysis via accuracy_debugger when R-ratio fails")
    parser.add_argument("--tp-size", type=int, default=8, help="TP size for device-level diagnostics")
    args = parser.parse_args()

    # Import tensor_compare from same scripts/ directory
    sys.path.insert(0, os.path.dirname(__file__))
    from tensor_compare import compare_3tensors
    from adapters import get_adapter

    from transformers import AutoTokenizer

    # Load adapter first (needed for setup_reference_model)
    adapter = get_adapter(
        target_stack=args.target_stack,
        auto_detect_from=args.target_module_file,
    )

    # Load HF source models via adapter's setup_reference_model.
    #
    # The vLLM-Neuron adapter's implementation lives in the vLLM-Neuron repo's test tree,
    # which is not shipped in the wheel. If it can't be imported, fall back to the plugin's
    # own logit validation instead of failing the stage outright.
    try:
        print("Loading HF FP32...")
        fp32_model = adapter.setup_reference_model(args.model_path, dtype=torch.float32)

        print("Loading HF BF16...")
        bf16_model = adapter.setup_reference_model(args.model_path, dtype=torch.bfloat16)
    except ModuleNotFoundError as e:
        print(f"\n  Reference-model helpers unavailable: {e}")
        result = _run_via_logit_validation(adapter, args)
        if result is None:
            print("  No fallback available for this stack — Stage 5 cannot run.")
            return 1
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
        return 0 if result.get("passed") else 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading target model via {adapter.name} adapter...")

    adapter.init_distributed(tp_degree=1)
    try:
        if not args.target_module_file and ":" not in args.model_class:
            sys.exit("Error: --target-module-file is required when --model-class doesn't use path:Class format")
        target_module_file = args.target_module_file or args.model_class.split(":")[0]
        target_class = args.target_inner_class or args.model_class.split(":")[-1]
        target_config = args.target_config_class or args.config_class.split(":")[-1]

        port_model = adapter.create_model(
            target_module_file=target_module_file,
            target_class_name=target_class,
            target_config_name=target_config,
            hf_model_path=args.model_path,
        )
        port_model = adapter.load_weights(port_model, args.model_path)
        port_model.eval()

        # Compare
        results = []
        all_passed = []

        for prompt in args.prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            fp32_logits = get_last_logits(fp32_model, inputs.input_ids, inputs.attention_mask)
            bf16_logits = get_last_logits(bf16_model, inputs.input_ids, inputs.attention_mask)

            # Use adapter for target model forward
            port_out = adapter.forward(port_model, inputs.input_ids)
            # Take last position logits, handle shape differences
            if port_out.dim() == 3:
                port_logits = port_out[:, -1, :].float()
            elif port_out.dim() == 2:
                port_logits = port_out[-1:, :].float()
            else:
                port_logits = port_out.float()
            port_logits = torch.nan_to_num(port_logits, nan=0.0, posinf=1e6, neginf=-1e6)
            if port_logits.dim() == 1:
                port_logits = port_logits.unsqueeze(0)

            # Align shapes (handles Neuron hidden dim padding)
            fp32_a, bf16_a, port_a, shapes_ok = adapter.align_tensors(fp32_logits, bf16_logits, port_logits)

            # Three-way comparison via adapter (BC/σ-ratio for vLLM, R-ratio for NxDI)
            cmp_result = adapter.compare_three_way(
                fp32_a, bf16_a, port_a,
                name=f"stage5/{prompt[:30]}",
                tau_r=args.tau_r,
                plot_on_failure=True,
                output_dir=os.path.dirname(os.path.abspath(args.output)) if args.output else "./accuracy_report",
            )
            all_passed.append(cmp_result.passed)

            # Top-k agreement
            min_v = fp32_a.shape[-1]
            topk = {}
            for k in [1, 5, 10, 50, 100]:
                if k > min_v:
                    continue
                fp32_topk = set(torch.topk(fp32_a.flatten()[:min_v], k).indices.tolist())
                port_topk = set(torch.topk(port_a.flatten()[:min_v], k).indices.tolist())
                topk[k] = len(fp32_topk & port_topk) / k

            print(f"\n  \"{prompt[:50]}\" → {cmp_result.summary()}")
            print(f"    top-1={'match' if topk.get(1,0)==1 else 'MISMATCH'}")
            results.append({
                "prompt": prompt,
                "passed": cmp_result.passed,
                "r_ratio": cmp_result.r_ratio,
                "bc": cmp_result.bc,
                "sigma_ratio": cmp_result.sigma_ratio,
                "linf_ratio": cmp_result.linf_ratio,
                "l2_ratio": cmp_result.l2_ratio,
                "top_k": topk,
            })

        passed = all(all_passed)

        print(f"\n{'=' * 60}")
        print(f"  Stage 5: {'PASS' if passed else 'FAIL'}")
        print(f"  {sum(all_passed)}/{len(all_passed)} prompts passed")
        print(f"{'=' * 60}")

        # ── KV cache analysis on failure (pinpoints divergent layers) ──
        if not passed and args.kv_analysis_on_fail and adapter.name == "vllm_neuron":
            print("\n" + "=" * 60)
            print("  Comparison FAILED — running KV cache analysis to locate divergence")
            print("=" * 60)
            try:
                output_dir = os.path.dirname(os.path.abspath(args.output)) if args.output else "./accuracy_report"
                kv_result = adapter.run_kv_cache_analysis(
                    model_id=args.model_path,
                    tp_size=args.tp_size,
                    prompts=args.prompts[:2],
                    output_length=16,
                    output_dir=os.path.join(output_dir, "kv_analysis"),
                )
                results.append({"kv_cache_analysis": kv_result})
                print(f"  KV analysis report: {kv_result.get('report_path', 'N/A')}")
            except Exception as e:
                print(f"  KV cache analysis skipped: {e}")

        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({"results": results, "passed": passed}, f, indent=2, default=str)

        return 0 if passed else 1
    finally:
        adapter.cleanup()


if __name__ == "__main__":
    sys.exit(main())
