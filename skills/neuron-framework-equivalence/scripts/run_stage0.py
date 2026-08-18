#!/usr/bin/env python3
"""
Stage 0: Build Model Trees for source and target.

Generates compressed trees, pretty-printed ASCII, and flat module paths
for both models. The target model is instantiated in CPU mode (TP=1).

Usage:
    python3 scripts/run_stage0.py \
        --source-model-path /path/to/hf_model \
        --target-model-path /path/to/hf_model \
        --target-module-file /path/to/modeling_xxx.py \
        --target-inner-class NeuronXxxModel \
        --target-config-class XxxInferenceConfig \
        --output-dir experiments/model_tree
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Stage 0: Build Model Trees")
    parser.add_argument("--source-model-path", required=True, help="Path to source (HF) model weights")
    parser.add_argument("--target-model-path", required=True, help="Path to HF-compatible weights for target config")
    parser.add_argument("--target-module-file", required=True, help="Path to target modeling .py file")
    parser.add_argument("--target-inner-class", required=True, help="Inner model class name (NeuronXxxModel)")
    parser.add_argument("--target-config-class", required=True, help="Config class name (XxxInferenceConfig)")
    parser.add_argument("--output-dir", required=True, help="Output directory for tree artifacts")
    parser.add_argument("--target-stack", default=None, help="Serving stack: nxdi, vllm_neuron (auto-detected if omitted)")
    args = parser.parse_args()

    # Import tree utilities from same directory
    sys.path.insert(0, os.path.dirname(__file__))
    from stage0_scaffolding import build_model_tree, save_model_tree, compressed_tree_to_pretty_string

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Source Model Tree ──
    print("=" * 70)
    print("  Building Source Model Tree")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM

    print(f"  Loading source model from: {args.source_model_path}")
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source_model_path, torch_dtype="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    source_model.eval()

    tree, full, paths = build_model_tree(source_model, "model")
    pretty = save_model_tree(tree, full, paths, args.output_dir, "model_tree_source")
    print(f"\n  Source tree ({len(paths)} module paths):")
    print(pretty)

    del source_model
    import gc
    gc.collect()

    # ── Target Model Tree (CPU mode, inner model only) ──
    print("\n" + "=" * 70)
    print("  Building Target Model Tree (CPU mode)")
    print("=" * 70)

    from adapters import get_adapter
    adapter = get_adapter(
        target_stack=args.target_stack,
        auto_detect_from=args.target_module_file,
    )
    print(f"  Adapter: {adapter.name}")

    adapter.init_distributed(tp_degree=1)

    print(f"  Instantiating target model (CPU mode, structure only)...")
    target_model = adapter.create_model(
        target_module_file=args.target_module_file,
        target_class_name=args.target_inner_class,
        target_config_name=args.target_config_class,
        hf_model_path=args.target_model_path,
    )

    tree, full, paths = build_model_tree(target_model, "model")
    pretty = save_model_tree(tree, full, paths, args.output_dir, "model_tree_target")
    print(f"\n  Target tree ({len(paths)} module paths):")
    print(pretty)

    print("\n" + "=" * 70)
    print("  Stage 0 complete. Trees saved to:", args.output_dir)
    print("  Next: Compare the trees and build component_mapping.json")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
