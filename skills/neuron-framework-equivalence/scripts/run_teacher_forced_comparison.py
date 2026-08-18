#!/usr/bin/env python3
"""
Teacher-Forced Per-Position Logit Comparison.

Used by Stages 5 and 6 for proper per-position comparison under teacher forcing.

Pattern (from NxDI's logit_validation):
1. Generate expected tokens from the source model (greedy)
2. Feed the source tokens to the target model
3. At each position, compare logits (both models see the same prefix)
4. If the target diverges, re-feed the source tokens and continue

This ensures per-position KL and cosine are computed on identical contexts,
not contaminated by trajectory divergence after token flips.

Usage:
    python3 scripts/run_teacher_forced_comparison.py \
        --model-path /path/to/hf_model \
        --compiled-model-path /path/to/compiled_model \
        --model-class path/to/modeling.py:NeuronXxxForCausalLM \
        --config-class path/to/modeling.py:XxxInferenceConfig \
        --num-tokens 32 \
        --output results/teacher_forced.json
"""

import argparse
import json
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F


def get_source_logits_and_tokens(model, tokenizer, prompt, num_tokens):
    """Generate from source model with output_scores=True to get per-position logits."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.no_grad():
        try:
            out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=num_tokens, min_new_tokens=num_tokens,
                do_sample=False, return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        except (AttributeError, TypeError):
            out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=num_tokens, min_new_tokens=num_tokens,
                do_sample=False, return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.pad_token_id, use_cache=False,
            )

    # scores: tuple of [batch, vocab] tensors, one per generated position
    source_logits = torch.stack(out.scores)  # [num_tokens, batch, vocab]
    source_tokens = source_logits.argmax(dim=2).T  # [batch, num_tokens]
    return source_logits, source_tokens, input_ids


def get_target_logits_teacher_forced(adapter, model, tokenizer, input_ids, source_tokens, num_tokens):
    """Get target model logits under teacher forcing via adapter.

    Feeds source tokens one at a time, collecting target logits at each position.
    Uses adapter.forward() which handles stack-specific signature differences.
    """
    target_logits_list = []
    current_input = input_ids.clone()

    for t in range(num_tokens):
        logits = adapter.forward(model, current_input)

        # Take last position logits
        if logits.dim() == 3:
            last_logits = logits[:, -1, :].float()
        elif logits.dim() == 2:
            last_logits = logits[-1:, :].float()
        else:
            last_logits = logits.float()
        last_logits = torch.nan_to_num(last_logits, nan=0.0, posinf=1e6, neginf=-1e6)
        if last_logits.dim() == 1:
            last_logits = last_logits.unsqueeze(0)
        target_logits_list.append(last_logits)

        # Teacher forcing: append the SOURCE token, not the target's argmax
        next_token = source_tokens[:, t:t+1]
        current_input = torch.cat([current_input, next_token], dim=1)

    if hasattr(model, "reset"):
        model.reset()

    return torch.stack(target_logits_list)  # [num_tokens, batch, vocab]


def main():
    parser = argparse.ArgumentParser(description="Teacher-Forced Per-Position Comparison")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-model-path", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--config-class", required=True)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--prompts", nargs="+", default=[
        "The capital of France is", "Water freezes at", "The speed of light is approximately",
    ])
    parser.add_argument("--theta", type=float, default=0.95)
    parser.add_argument("--output", default=None)
    parser.add_argument("--target-stack", default=None, help="Serving stack: nxdi, vllm_neuron")
    parser.add_argument("--target-module-file", default=None, help="Path to target modeling file")
    parser.add_argument("--target-config-class", default=None, help="Config class name for adapter")
    parser.add_argument("--target-inner-class", default=None, help="Model class name for adapter")
    parser.add_argument("--tp-size", type=int, default=8, help="TP size for device inference")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from tensor_compare import compare_3tensors
    from adapters import get_adapter

    from transformers import AutoTokenizer

    # Load adapter first (needed for setup_reference_model)
    stack_adapter = get_adapter(
        target_stack=args.target_stack,
        auto_detect_from=args.target_module_file,
    )

    # NOTE: the HF source models are loaded further down, immediately before first use.
    # The vLLM-Neuron branch below never uses them (accuracy_debugger builds its own
    # references), so loading them here would waste two full-precision model loads and
    # would drag in setup_reference_model's dependencies for a path that doesn't need them.

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading target model via {stack_adapter.name} adapter...")

    try:
        if not args.target_module_file and ":" not in args.model_class:
            sys.exit("Error: --target-module-file is required when --model-class doesn't use path:Class format")
        target_module_file = args.target_module_file or args.model_class.split(":")[0]
        target_class = args.target_inner_class or args.model_class.split(":")[-1]
        target_config = args.target_config_class or args.config_class.split(":")[-1]

        # For vLLM-Neuron, use the accuracy_debugger's run_prompt_analysis
        # which handles all the vLLM engine setup, logprobs extraction, and
        # three-way logit validation correctly.
        if stack_adapter.name == "vllm_neuron":
            print("  [vllm_neuron] Using accuracy_debugger run_prompt_analysis")

            os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.pop("NXD_CPU_MODE", None)

            # Point neuron compile cache to the pre-compiled artifacts
            compile_cache = os.path.join(args.compiled_model_path, "neuron", "compile_cache")
            if os.path.isdir(compile_cache):
                os.environ["NEURON_COMPILE_CACHE_URL"] = compile_cache

            from vllm_neuron.accuracy.accuracy_debugger import run_prompt_analysis
            from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.logit_val import LogitValPlugin

            tp_size = args.tp_size
            max_model_len = 256

            server_config = {
                "model": args.model_path,
                "tp_degree": tp_size,
                "max_model_len": max_model_len,
                "batch_size": 1,
                "additional_config": {
                    "neuron_config": {
                        "on_device_sampling_config": {"all_greedy": True},
                        "num_batched_tokens_buckets": [max_model_len],
                        "num_seqs_buckets": [1],
                    }
                },
            }

            # Equiv framework pass criteria: R-ratio < 1.2 (≈20% relative error).
            # Default accuracy_debugger tol_map is designed for production validation
            # and is stricter (1.1% for top-5). Loosen to align with equiv framework.
            equiv_tol_map = {
                "5": (1e-5, 0.05),
                "50": (1e-5, 0.05),
                "1000": (1e-5, 0.05),
                "all": (1e-5, 0.05),
            }

            output_dir = os.path.dirname(os.path.abspath(args.output)) if args.output else "./accuracy_report"

            result = run_prompt_analysis(
                server_config=server_config,
                prompts=args.prompts,
                plugin_steps=[LogitValPlugin(tol_map=equiv_tol_map)],
                output_dir=output_dir,
                output_length=args.num_tokens,
            )

            plugin_results = result.plugin_results
            logit_val = plugin_results.get("logit_val", {})
            passed = all(
                v.get("passed", False) if isinstance(v, dict) else False
                for v in logit_val.values()
            )

            summary = {
                "passed": passed,
                "method": "accuracy_debugger_logit_validation",
                "num_prompts": len(args.prompts),
                "num_tokens": args.num_tokens,
                "report_path": str(output_dir),
                "plugin_results": {k: str(v) for k, v in logit_val.items()},
            }
            print(f"\n{'=' * 70}")
            print(f"  LOGIT VALIDATION: {'PASS' if passed else 'FAIL'}")
            print(f"{'=' * 70}")

            if args.output:
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                with open(args.output, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                print(f"  Results saved to {args.output}")

            stack_adapter.cleanup()
            sys.exit(0 if passed else 1)

        stack_adapter.init_distributed(tp_degree=1)
        target_model = stack_adapter.create_model(
            target_module_file=target_module_file,
            target_class_name=target_class,
            target_config_name=target_config,
            hf_model_path=args.model_path,
        )
        target_model = stack_adapter.load_weights(target_model, args.model_path)
        target_model.eval()

        # Load the HF source models now — first point they are actually needed.
        # (Deliberately after the vLLM-Neuron branch above, which returns before this.)
        print("Loading source model (FP32)...")
        source_fp32 = stack_adapter.setup_reference_model(args.model_path, dtype=torch.float32)

        print("Loading source model (BF16)...")
        source_bf16 = stack_adapter.setup_reference_model(args.model_path, dtype=torch.bfloat16)

        all_results = []

        for prompt in args.prompts:
            print(f"\n{'=' * 70}")
            print(f"  Prompt: \"{prompt}\"")
            print(f"{'=' * 70}")
    
            # Step 1: Get source tokens (FP32, greedy generation)
            print("  Generating source tokens (FP32, greedy)...")
            src_logits_fp32, src_tokens, input_ids = get_source_logits_and_tokens(
                source_fp32, tokenizer, prompt, args.num_tokens,
            )
            num_gen = min(args.num_tokens, src_tokens.shape[1])
    
            # Step 1b: Get source FP32 logits via teacher-forced forward passes (KV-cached)
            print("  Getting source FP32 logits (teacher-forced)...")
            teacher_seq = src_tokens[0, :num_gen]  # [num_gen]
            src_logits_fp32_tf = stack_adapter.generate_teacher_forced_logits(
                source_fp32, input_ids, teacher_seq,
            )  # [num_tokens, 1, vocab] or [num_tokens, vocab]
            if src_logits_fp32_tf.dim() == 2:
                src_logits_fp32_tf = src_logits_fp32_tf.unsqueeze(1)

            # Step 2: Get source BF16 logits (teacher-forced by FP32 tokens, KV-cached)
            print("  Getting source BF16 logits (teacher-forced)...")
            src_logits_bf16_tf = stack_adapter.generate_teacher_forced_logits(
                source_bf16, input_ids, teacher_seq,
            )
            if src_logits_bf16_tf.dim() == 2:
                src_logits_bf16_tf = src_logits_bf16_tf.unsqueeze(1)
    
            # Step 3: Get target logits (teacher-forced by FP32 tokens)
            print("  Getting target logits (teacher-forced)...")
            tgt_logits = get_target_logits_teacher_forced(
                stack_adapter, target_model, tokenizer, input_ids, src_tokens, args.num_tokens,
            )
    
            # Align vocab sizes
            min_vocab = min(src_logits_fp32_tf.shape[-1], src_logits_bf16_tf.shape[-1], tgt_logits.shape[-1])
            min_tokens = min(src_logits_fp32_tf.shape[0], src_logits_bf16_tf.shape[0], tgt_logits.shape[0])
    
            fp32 = src_logits_fp32_tf[:min_tokens, 0, :min_vocab].float()
            bf16 = src_logits_bf16_tf[:min_tokens, 0, :min_vocab].float()
            tgt = tgt_logits[:min_tokens, 0, :min_vocab].float()
            tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1e6, neginf=-1e6)
            bf16 = torch.nan_to_num(bf16, nan=0.0, posinf=1e6, neginf=-1e6)
    
            # Per-position metrics
            per_pos_cos = []
            per_pos_kl = []
            per_pos_r = []
            per_pos_topk = {1: [], 5: [], 10: []}
    
            for t in range(min_tokens):
                # Cosine similarity
                cos = F.cosine_similarity(fp32[t].unsqueeze(0), tgt[t].unsqueeze(0)).item()
                per_pos_cos.append(cos)
    
                # KL divergence (on full vocab)
                ref_probs = F.softmax(fp32[t], dim=-1)
                tgt_log_probs = F.log_softmax(tgt[t], dim=-1)
                kl = max(0.0, F.kl_div(tgt_log_probs.unsqueeze(0), ref_probs.unsqueeze(0), reduction="sum").item())
                per_pos_kl.append(kl)
    
                # R-ratio (three-tensor)
                baseline_err = torch.norm(bf16[t] - fp32[t], p=2).item()
                target_err = torch.norm(tgt[t] - fp32[t], p=2).item()
                # Use a meaningful epsilon: scale by the FP32 norm to avoid
                # division by near-zero when FP32 and BF16 happen to agree exactly
                fp32_norm = torch.norm(fp32[t], p=2).item()
                eps = max(1e-12, fp32_norm * 1e-7)
                r = target_err / (baseline_err + eps)
                per_pos_r.append(r)
    
                # Top-k agreement
                for k in per_pos_topk:
                    if k > min_vocab:
                        continue
                    fp32_topk = set(torch.topk(fp32[t], k).indices.tolist())
                    tgt_topk = set(torch.topk(tgt[t], k).indices.tolist())
                    per_pos_topk[k].append(len(fp32_topk & tgt_topk) / k)
    
            # Print per-position results
            cos_arr = np.array(per_pos_cos)
            kl_arr = np.array(per_pos_kl)
            r_arr = np.array(per_pos_r)
    
            print(f"\n  Per-position results ({min_tokens} positions, teacher-forced):")
            print(f"  {'Pos':>4s} {'R-ratio':>8s} {'Cosine':>8s} {'KL':>10s} {'Top-1':>6s}")
            print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")
            for t in range(min(min_tokens, 10)):  # Show first 10
                top1 = "✓" if per_pos_topk[1][t] == 1.0 else "✗"
                print(f"  {t:>4d} {per_pos_r[t]:>8.4f} {per_pos_cos[t]:>8.4f} {per_pos_kl[t]:>10.4f} {top1:>6s}")
            if min_tokens > 10:
                print(f"  ... ({min_tokens - 10} more positions)")
    
            print(f"\n  Summary:")
            print(f"    R-ratio:  mean={np.mean(r_arr):.4f}, p95={np.percentile(r_arr, 95):.4f}, max={np.max(r_arr):.4f}")
            print(f"    Cosine:   mean={np.mean(cos_arr):.6f}, min={np.min(cos_arr):.6f}, p5={np.percentile(cos_arr, 5):.6f}")
            print(f"    KL:       mean={np.mean(kl_arr):.6f}, p95={np.percentile(kl_arr, 95):.6f}, max={np.max(kl_arr):.6f}")
            print(f"    Top-1:    {np.mean(per_pos_topk[1]):.2%}")
    
            all_results.append({
                "prompt": prompt,
                "num_positions": min_tokens,
                "r_ratio": {"mean": float(np.mean(r_arr)), "p95": float(np.percentile(r_arr, 95)), "max": float(np.max(r_arr))},
                "cosine": {"mean": float(np.mean(cos_arr)), "min": float(np.min(cos_arr)), "p5": float(np.percentile(cos_arr, 5))},
                "kl": {"mean": float(np.mean(kl_arr)), "p95": float(np.percentile(kl_arr, 95)), "max": float(np.max(kl_arr))},
                "top1_agreement": float(np.mean(per_pos_topk[1])),
                "per_position_cosine": per_pos_cos,
                "per_position_kl": per_pos_kl,
                "per_position_r_ratio": per_pos_r,
            })
    
        # Aggregate across prompts
        all_cos = [c for r in all_results for c in r["per_position_cosine"]]
        all_kl = [k for r in all_results for k in r["per_position_kl"]]
        all_r = [r for res in all_results for r in res["per_position_r_ratio"]]
    
        cos_arr = np.array(all_cos)
        kl_arr = np.array(all_kl)
        r_arr = np.array(all_r)
    
        condition_b = float(np.percentile(cos_arr, 5)) >= args.theta
        condition_c_kl_p95 = float(np.percentile(kl_arr, 95))
    
        print(f"\n{'=' * 70}")
        print(f"  TEACHER-FORCED COMPARISON SUMMARY ({len(all_cos)} total positions)")
        print(f"{'=' * 70}")
        print(f"  Condition B (Semantic): {'PASS' if condition_b else 'FAIL'}")
        print(f"    cosine: mean={np.mean(cos_arr):.6f}, min={np.min(cos_arr):.6f}, p5={np.percentile(cos_arr, 5):.6f} (θ={args.theta})")
        print(f"  Condition C (Distributional):")
        print(f"    KL: mean={np.mean(kl_arr):.6f}, p95={condition_c_kl_p95:.6f}, max={np.max(kl_arr):.6f}")
        print(f"  E2E R-ratio:")
        print(f"    mean={np.mean(r_arr):.4f}, p95={np.percentile(r_arr, 95):.4f}, max={np.max(r_arr):.4f}")
        print(f"  Top-1 agreement: {np.mean([r['top1_agreement'] for r in all_results]):.2%}")
        print(f"{'=' * 70}")
    
        output = {
            "per_prompt": all_results,
            "aggregate": {
                "condition_b_passed": condition_b,
                "cosine": {"mean": float(np.mean(cos_arr)), "min": float(np.min(cos_arr)), "p5": float(np.percentile(cos_arr, 5))},
                "kl": {"mean": float(np.mean(kl_arr)), "p95": float(np.percentile(kl_arr, 95)), "max": float(np.max(kl_arr))},
                "r_ratio": {"mean": float(np.mean(r_arr)), "p95": float(np.percentile(r_arr, 95)), "max": float(np.max(r_arr))},
                "num_positions": len(all_cos),
            },
        }

        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")

        return 0
    finally:
        stack_adapter.cleanup()


if __name__ == "__main__":
    sys.exit(main())
