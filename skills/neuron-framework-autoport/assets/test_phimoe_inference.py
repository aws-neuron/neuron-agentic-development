#!/usr/bin/env python3
"""
Test script for the generic inference utility with PhiMoE.
"""

import sys
import os
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from run_inference import run_inference_with_classes
from neuron_port.modeling_phimoe import NeuronPhiMoEForCausalLM, PhiMoeInferenceConfig


def main():
    # Configuration
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "agent_artifacts/data/Phi-3.5-MoE-instruct")
    hf_model_id = "microsoft/Phi-3.5-MoE-instruct"
    compiled_output_path = os.path.join(current_dir, "agent_artifacts/data/phi35_moe_tp_ep_compiled_fixed")

    success, result, metrics = run_inference_with_classes(
        model_class=NeuronPhiMoEForCausalLM,
        config_class=PhiMoeInferenceConfig,
        model_path=model_path,
        compiled_path=compiled_output_path,
        prompt="What is the capital of France?",
        max_new_tokens=30,
        temperature=0.7,
        top_p=0.9
    )

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
