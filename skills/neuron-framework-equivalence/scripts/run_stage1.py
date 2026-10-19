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
from pathlib import Path
import hashlib


class UnsupportedComparison(ValueError):
    pass


def _run_token_matching(args, prompts, device_results):
    """Fallback path for non-vllm_neuron stacks: HF greedy token comparison."""
    import torch
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


def run_exact_comparison(args):
    """Compare a producer's current target tokens against a pinned HF reference."""
    from reference_cache import ReferenceCache, checkpoint_identity, digest, validate_tokens, write_json

    contract = json.loads(Path(args.correctness_contract).read_text())
    prepared = json.loads(Path(args.prepared_attempt).read_text())
    prepared = prepared.get("result", prepared)
    task, grant = prepared["task"], prepared["grant"]
    if digest(task) != grant["task_digest"] or task["capability"] != "equivalence":
        raise ValueError("prepared task does not match the equivalence grant")
    if (task["correctness_contract"] != contract or
            task["inputs"]["source_attempt_id"] != args.source_attempt_id):
        raise ValueError("correctness contract or source attempt differs from the grant")
    if (contract["criterion"] not in ("greedy_token_match", "greedy_token_exact") or
            contract["reference_method"] not in ("exact_tokens", "hf_greedy")):
        raise UnsupportedComparison("this path requires an exact greedy-token criterion and reference method")
    if contract.get("requires_logits", False):
        raise UnsupportedComparison("the requested criterion requires a logit-capable comparison")
    reference = contract["reference_identity"]
    actual_checkpoint = checkpoint_identity(args.model_path, args.model_revision)
    if reference["checkpoint"] != actual_checkpoint:
        raise ValueError("actual checkpoint identity differs from the frozen reference")
    precision = reference["precision"]
    if precision not in ("fp32", "bf16", "fp16"):
        raise UnsupportedComparison("HF reference precision must be explicit: fp32, bf16 or fp16")
    prompts = json.loads(Path(args.prompt_token_ids).read_text())
    validate_tokens(prompts, len(prompts), contract["input_tokens"])
    if not prompts or digest(prompts) != reference["tokenized_prompts_sha256"]:
        raise ValueError("prompt token IDs do not match the frozen reference identity")
    target_path = Path(args.target_results).resolve()
    target_bytes = target_path.read_bytes()
    target = json.loads(target_bytes)
    if target["prompt_token_ids"] != prompts:
        raise ValueError("target used different prompt token IDs")
    if not target.get("observed_config") or not target.get("execution_identity"):
        raise ValueError("target results require observed_config and execution_identity")
    if target.get("source_attempt_id") != args.source_attempt_id:
        raise ValueError("target source_attempt_id differs from the granted profile")
    if (target["execution_identity"] != task["source_execution_identity"] or
            target["observed_config"] != task["source_context"]["observed_config"]):
        raise ValueError("target execution identity or configuration differs from the granted source")
    target_tokens = target["generated_token_ids"]
    validate_tokens(target_tokens, len(prompts), contract["output_tokens"])

    import platform
    import torch
    import transformers

    identity = {
        **reference, "checkpoint": actual_checkpoint, "prompt_token_ids": prompts,
        "input_tokens": contract["input_tokens"], "output_tokens": contract["output_tokens"],
        "method": contract["reference_method"], "criterion": contract["criterion"],
        "corpus_digest": contract["corpus_digest"],
        "runtime": {"declared": reference["runtime"], "python": platform.python_version(),
                    "torch": torch.__version__, "transformers": transformers.__version__},
        "code_revision": {
            "declared": reference["code_revision"],
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cache": hashlib.sha256(Path(__file__).with_name("reference_cache.py").read_bytes()).hexdigest(),
        },
    }

    def generate():
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype={"fp32": torch.float32, "bf16": torch.bfloat16,
                                         "fp16": torch.float16}[precision],
            trust_remote_code=True, revision=args.model_revision,
        ).eval()
        outputs = []
        with torch.no_grad():
            for tokens in prompts:
                ids = torch.tensor([tokens])
                result = model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    min_new_tokens=contract["output_tokens"],
                    max_new_tokens=contract["output_tokens"], do_sample=False,
                    output_scores=False, output_logits=False, return_dict_in_generate=False,
                )
                outputs.append(result[0, len(tokens):].tolist())
        if checkpoint_identity(args.model_path, args.model_revision) != actual_checkpoint:
            raise ValueError("checkpoint changed during reference generation")
        return outputs

    expected, cache = ReferenceCache(args.reference_cache).get_or_create(identity, generate)
    reference_path = Path(args.output).resolve().with_name(Path(args.output).stem + "_reference.json")
    write_json(reference_path, {
        "correctness_contract": contract, "reference_execution_identity": identity,
        "prompt_token_ids": prompts, "generated_token_ids": expected,
        "reference_cache": cache,
    })
    return {
        "passed": expected == target_tokens, "criterion": contract["criterion"],
        "comparison_state": "matched" if expected == target_tokens else "mismatch",
        "method": contract["reference_method"], "correctness_contract": contract,
        "input_tokens": contract["input_tokens"], "output_tokens": contract["output_tokens"],
        "reference_identity": reference, "reference_cache": cache,
        "reference_cache_key": digest(contract),
        "reference_artifact_paths": [str(reference_path)],
        "source_attempt_id": args.source_attempt_id,
        "task_digest": grant["task_digest"],
        "reference_execution_identity": identity,
        "target_token_ids": target_tokens,
        "observed_config": target["observed_config"],
        "execution_identity": target["execution_identity"],
        "target_results": str(target_path),
        "target_results_sha256": hashlib.sha256(target_bytes).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Smoke Test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-model-path")
    parser.add_argument("--model-class")
    parser.add_argument("--config-class")
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--target-stack", default=None, help="Serving stack: nxdi, vllm_neuron")
    parser.add_argument("--target-module-file", default=None, help="Path to target modeling file (for auto-detect)")
    parser.add_argument("--tp-size", type=int, default=8, help="TP size for device inference")
    parser.add_argument("--pass-threshold", type=float, default=0.30,
                        help="Minimum token match rate to pass (fallback path only)")
    parser.add_argument("--correctness-contract", help="frozen exact-token correctness JSON")
    parser.add_argument("--prompt-token-ids", help="JSON array of tokenized prompts")
    parser.add_argument("--target-results", help="current target token and execution evidence JSON")
    parser.add_argument("--reference-cache", help="bounded reusable HF token reference directory")
    parser.add_argument("--source-attempt-id", help="profile attempt selected by the equivalence grant")
    parser.add_argument("--prepared-attempt", help="JSON returned by neuron-campaign prepare")
    parser.add_argument("--model-revision", help="immutable HF commit for a remote model-path")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    if args.correctness_contract:
        if not all((args.prompt_token_ids, args.target_results, args.reference_cache,
                    args.source_attempt_id, args.prepared_attempt, args.output)):
            parser.error("exact comparison requires prompt-token-ids, target-results, reference-cache, source-attempt-id, prepared-attempt and output")
        try:
            result = run_exact_comparison(args)
        except (OSError, KeyError, ValueError, RuntimeError, ImportError) as error:
            if isinstance(error, UnsupportedComparison):
                state = "unsupported_tooling"
            elif isinstance(error, (OSError, KeyError, ValueError)):
                state = "missing_evidence"
            else:
                state = "execution_failed"
            result = {"passed": False, "error": str(error), "comparison_state": state}
        from reference_cache import write_json
        write_json(args.output, result)
        return 0 if result["passed"] else 1
    if any((args.prompt_token_ids, args.target_results, args.reference_cache,
            args.source_attempt_id, args.prepared_attempt, args.model_revision)):
        parser.error("reference reuse requires an explicit correctness-contract")
    if not all((args.compiled_model_path, args.model_class, args.config_class)):
        parser.error("smoke test requires compiled-model-path, model-class and config-class")
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
