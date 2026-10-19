#!/usr/bin/env python3
"""
Test script for the generic inference utility with Phi3.
"""

import sys
import os
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from run_inference import run_inference_with_classes
from neuron_port.modeling_phi3 import NeuronPhi3ForCausalLM, Phi3InferenceConfig


def main():
    # Configuration
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "agent_artifacts/data/Phi-3-mini-4k-instruct")
    hf_model_id = "microsoft/Phi-3-mini-4k-instruct"
    compiled_output_path = os.path.join(current_dir, "agent_artifacts/data/phi3-mini-4k-compiled")
    
    success, result, metrics = run_inference_with_classes(
        model_class=NeuronPhi3ForCausalLM,
        config_class=Phi3InferenceConfig,
        model_path=model_path,
        compiled_path=compiled_output_path,
        prompt="The meaning of life is",
        max_new_tokens=30,
        temperature=0.7,
        top_p=0.9
    )

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
