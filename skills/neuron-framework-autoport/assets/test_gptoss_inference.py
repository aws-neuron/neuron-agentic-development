#!/usr/bin/env python3
"""
Test script for the generic inference utility with GPT-OSS.
"""

import sys
import os
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from run_inference import run_inference_with_classes
from neuron_port.modeling_gptoss import NeuronGptOssForCausalLM, GptOssInferenceConfig


def main():
    # Configuration
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "gpt_oss_hf_official_dequantized")
    hf_model_id = "gpt-oss"
    compiled_output_path = os.path.join(current_dir, "gptoss_compiled_full_workflow")
    
    success, result, metrics = run_inference_with_classes(
        model_class=NeuronGptOssForCausalLM,
        config_class=GptOssInferenceConfig,
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
