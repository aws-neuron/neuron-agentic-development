#!/usr/bin/env python3
"""
Example showing how to use the model compiler for PhiMoE
Includes the complete workflow for prepared weights and compilation.
"""

import sys
import os
import warnings
import time
from pathlib import Path
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import the port produced by this skill and the skill's compilation utility.
SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

# Import required classes
from neuron_port.modeling_phimoe import NeuronPhiMoEForCausalLM, PhiMoeInferenceConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig
from model_compiler import DirectModelCompiler, CompilationConfig


def require_model_weights(model_path: str) -> str:
    """Return an existing model directory prepared by the setup flow."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Download them into agent_artifacts/data before compiling."
        )
    return model_path


def compile_phimoe_full():
    """Complete PhiMoE workflow using prepared model weights."""
    print("=" * 80)
    print("PhiMoE Model Compilation - Full Workflow (Prepared Weights + Compile)")
    print("=" * 80)
    print()

    # Configuration
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "agent_artifacts/data/Phi-3.5-MoE-instruct")
    compiled_output_path = os.path.join(current_dir, "agent_artifacts/data/phi35_moe_tp_ep_compiled_fixed")

    print(f"📝 Paths:")
    print(f"   Model Path: {model_path}")
    print(f"   Output Path: {compiled_output_path}")
    print()

    # Step 1: Validate model weights prepared by the setup flow
    print("📦 Step 1: Validate Prepared Model Weights")
    print("-" * 40)
    prepared_model_path = require_model_weights(model_path)
    print(f"✅ Model weights found at: {prepared_model_path}")
    print()

    # Step 2: Create compilation configuration
    print("⚙️  Step 2: Configure Compilation")
    print("-" * 40)
    config = CompilationConfig(
        model_class=NeuronPhiMoEForCausalLM,
        config_class=PhiMoeInferenceConfig,
        neuron_config_class=MoENeuronConfig,
        model_path=prepared_model_path,
        output_path=compiled_output_path,
        batch_size=1,
        seq_len=2048,
        tp_degree=16,
        use_fp16=True,
    )

    print("Configuration:")
    print(f"  Model Class: {config.model_class.__name__}")
    print(f"  Config Class: {config.config_class.__name__}")
    print(f"  Neuron Config Class: {config.neuron_config_class.__name__}")
    print(f"  Model Path: {config.model_path}")
    print(f"  Output Path: {config.output_path}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  TP degree: {config.tp_degree}")
    print(f"  Use FP16 (bfloat16): {config.use_fp16}")
    print()

    # Step 3: Compile model
    print("🚀 Step 3: Compile Model")
    print("-" * 40)
    print("   Expected Duration: 30-60 minutes")
    print()

    start_time = time.time()

    compiler = DirectModelCompiler(config)
    success = compiler.compile()

    elapsed_time = time.time() - start_time

    print()
    print("=" * 80)
    if success:
        print("✅ COMPILATION SUCCESSFUL!")
        print("=" * 80)
        print()
        print(f"⏱️  Compilation Time: {elapsed_time/60:.1f} minutes")
        print(f"📁 Compiled model saved to: {compiled_output_path}")
        print()
        print("🎯 Next: Run inference test:")
        print("   python3 skills/neuron-framework-autoport/assets/test_phimoe_inference.py")
        print()
        return True
    else:
        print("❌ COMPILATION FAILED!")
        print("=" * 80)
        print()
        print(f"⏱️  Time elapsed: {elapsed_time/60:.1f} minutes")
        print("⚠️  Check the error messages above for details")
        print()
        return False


def compile_phimoe_existing():
    """Compile PhiMoE using existing weights (no download)."""
    print("=" * 80)
    print("PhiMoE Model Compilation - Using Existing Weights")
    print("=" * 80)
    print()

    # Configuration
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "agent_artifacts/data/Phi-3.5-MoE-instruct")
    compiled_output_path = os.path.join(current_dir, "agent_artifacts/data/phi35_moe_tp_ep_compiled_fixed")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("   Prepare the weights before running this example")
        return False

    print(f"📝 Paths:")
    print(f"   Model Path: {model_path}")
    print(f"   Output Path: {compiled_output_path}")
    print()

    # Create compilation configuration
    print("⚙️  Step 1: Configure Compilation")
    print("-" * 40)
    config = CompilationConfig(
        model_class=NeuronPhiMoEForCausalLM,
        config_class=PhiMoeInferenceConfig,
        neuron_config_class=MoENeuronConfig,
        model_path=model_path,
        output_path=compiled_output_path,
        batch_size=1,
        seq_len=2048,
        tp_degree=16,
        use_fp16=True,
    )

    print("Configuration:")
    print(f"  Model Class: {config.model_class.__name__}")
    print(f"  Config Class: {config.config_class.__name__}")
    print(f"  Neuron Config Class: {config.neuron_config_class.__name__}")
    print(f"  Model Path: {config.model_path}")
    print(f"  Output Path: {config.output_path}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  TP degree: {config.tp_degree}")
    print(f"  Use FP16 (bfloat16): {config.use_fp16}")
    print()

    # Compile model
    print("🚀 Step 2: Compile Model")
    print("-" * 40)
    print("   Expected Duration: 30-60 minutes")
    print()

    start_time = time.time()

    compiler = DirectModelCompiler(config)
    success = compiler.compile()

    elapsed_time = time.time() - start_time

    print()
    print("=" * 80)
    if success:
        print("✅ COMPILATION SUCCESSFUL!")
        print("=" * 80)
        print()
        print(f"⏱️  Compilation Time: {elapsed_time/60:.1f} minutes")
        print(f"📁 Compiled model saved to: {compiled_output_path}")
        print()
        print("🎯 Next: Run inference test:")
        print("   python3 skills/neuron-framework-autoport/assets/test_phimoe_inference.py")
        print()
        return True
    else:
        print("❌ COMPILATION FAILED!")
        print("=" * 80)
        print()
        print(f"⏱️  Time elapsed: {elapsed_time/60:.1f} minutes")
        print("⚠️  Check the error messages above for details")
        print()
        return False


def main():
    """Main function with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="PhiMoE Model Compiler")
    parser.add_argument("--example", choices=["full", "existing"],
                       default="full", help="Which example to run (full=validate+compile, existing=compile only)")

    args = parser.parse_args()

    if args.example == "full":
        success = compile_phimoe_full()
    else:  # existing
        success = compile_phimoe_existing()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
