#!/usr/bin/env python3
"""
Stage 1: Smoke Test Runner.

Runs device inference via the adapter then validates accuracy via
greedy token matching against HF FP32 reference.

Usage:
    python3 scripts/run_stage1.py \
        --model-path /path/to/hf_model \
        --compiled-model-path /path/to/compiled_model \
        --model-class path/to/modeling.py:NeuronXxxForCausalLM \
        --config-class path/to/modeling.py:XxxInferenceConfig \
        --target-stack vllm_neuron
"""

import argparse
import json
import sys
import os

import torch




def _run_token_matching(args, prompts, device_results):
    """Fallback path for non-vllm_neuron stacks: HF greedy token comparison."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n  Loading HF reference (FP32)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32, trust_remote_code=True,
    )
    hf_model.eval()

    matching = 0
    total = 0
    per_prompt = []

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        with torch.no_grad():
            hf_out = hf_model.generate(
                **inputs, max_new_tokens=args.num_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        hf_tokens = hf_out[0][inputs.input_ids.shape[1]:].tolist()
        device_tokens = device_results[i]["tokens"][-args.num_tokens:] if len(device_results[i]["tokens"]) > args.num_tokens else device_results[i]["tokens"]

        n = min(len(hf_tokens), len(device_tokens))
        matches = sum(1 for a, b in zip(hf_tokens[:n], device_tokens[:n]) if a == b)
        rate = matches / n if n > 0 else 0
        matching += matches
        total += n
        per_prompt.append({"prompt": prompt[:55], "match_rate": rate})
        print(f"  {prompt[:55]:<55s} {rate:>5.0%}")

    overall_rate = matching / total if total > 0 else 0
    passed = overall_rate > args.pass_threshold
    print(f"\n  Overall: {overall_rate:.2%} ({matching}/{total} tokens)")

    return {
        "token_matching": {"match_rate": overall_rate, "per_prompt_results": per_prompt},
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Smoke Test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-model-path", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--config-class", required=True)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--target-stack", default=None, help="Serving stack: nxdi, vllm_neuron")
    parser.add_argument("--target-module-file", default=None, help="Path to target modeling file (for auto-detect)")
    parser.add_argument("--tp-size", type=int, default=8, help="TP size for device inference")
    parser.add_argument("--pass-threshold", type=float, default=0.30,
                        help="Minimum token match rate to pass (fallback path only)")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from adapters import get_adapter

    # Set compile cache env var early so all LLM instances find pre-compiled NEFFs
    compile_cache = os.path.join(args.compiled_model_path, "neuron", "compile_cache")
    if os.path.isdir(compile_cache):
        os.environ["NEURON_COMPILE_CACHE_URL"] = compile_cache

    adapter = get_adapter(
        target_stack=args.target_stack,
        auto_detect_from=args.target_module_file,
    )
    print(f"  Adapter: {adapter.name}")

    prompts = [
        "The capital of France is",
        "Water freezes at a temperature of",
        "The speed of light is approximately",
        "In machine learning, gradient descent",
        "The largest planet in our solar system",
        "def fibonacci(n):\n    if n <= 1:",
        "The theory of relativity states that",
        "Photosynthesis is the process by which",
        "The human genome contains approximately",
        "In quantum mechanics, the uncertainty principle",
    ]

    print("\n" + "=" * 70)
    print("  Stage 1: Smoke Test (Device Inference)")
    print("=" * 70)

    # ── First: confirm device inference works at all ──
    try:
        device_results = adapter.device_inference(
            model_id=args.model_path,
            tp_size=args.tp_size,
            prompts=prompts,
            max_tokens=args.num_tokens,
            compiled_model_path=args.compiled_model_path,
            model_class=args.model_class,
            config_class=args.config_class,
        )
    except Exception as e:
        print(f"\n  Device inference FAILED: {e}")
        result = {"passed": False, "error": str(e)}
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
        return 1

    # ── Validate accuracy: greedy token matching against HF FP32 ──
    result = _run_token_matching(args, prompts, device_results)

    passed = result["passed"]

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    print(f"\n{'=' * 70}")
    print(f"  STAGE 1 VERDICT: {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 70}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
