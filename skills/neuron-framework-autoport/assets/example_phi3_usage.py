#!/usr/bin/env python3
"""
Example showing how to use the refactored model compiler for Phi3
Includes the complete workflow for prepared weights: compile and test.
"""

import sys
import os
from pathlib import Path

# Import the port produced by this skill and the skill's compilation utility.
SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(SKILL_ROOT / "scripts"))


def require_model_weights(model_path: str) -> str:
    """Return an existing model directory prepared by the setup flow."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Download them into agent_artifacts/data before compiling."
        )
    return model_path

def example_phi3_full_workflow():
    """Example: Complete Phi3 workflow using prepared model weights."""
    print("📋 Example: Complete Phi3 Workflow (Prepared Weights + Compile)")
    print("=" * 70)

    # Import the required classes and utilities
    
    from neuron_port.modeling_phi3 import NeuronPhi3ForCausalLM, Phi3InferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig
    from model_compiler import DirectModelCompiler, CompilationConfig
    
    # Configuration
    current_dir = os.getcwd()
    hf_model_id = "microsoft/Phi-3-mini-4k-instruct"
    model_path = os.path.join(current_dir, "agent_artifacts/data/Phi-3-mini-4k-instruct")
    compiled_output_path = os.path.join(current_dir, "agent_artifacts/data/phi3-mini-4k-compiled")
    
    print("🎯 Workflow Configuration:")
    print(f"  HuggingFace Model: {hf_model_id}")
    print(f"  Model Path: {model_path}")
    print(f"  Compiled Output: {compiled_output_path}")
    print()
    
    try:
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
            model_class=NeuronPhi3ForCausalLM,
            config_class=Phi3InferenceConfig,
            neuron_config_class=NeuronConfig,
            model_path=prepared_model_path,
            output_path=compiled_output_path,
            batch_size=1,
            seq_len=128,
            tp_degree=1,  # Start with TP=1 for Phi3
            use_fp16=True,
        )
        
        print("Configuration:")
        print(f"  Model Class: {config.model_class.__name__}")
        print(f"  Config Class: {config.config_class.__name__}")
        print(f"  Model Path: {config.model_path}")
        print(f"  Output Path: {config.output_path}")
        print(f"  TP Degree: {config.tp_degree}")
        print(f"  Sequence Length: {config.seq_len}")
        print(f"  Use FP16: {config.use_fp16}")
        print()
        
        # Step 3: Compile model
        print("🚀 Step 3: Compile Model")
        print("-" * 40)
        compiler = DirectModelCompiler(config)
        success = compiler.compile()
        
        if success:
            print("\n🎉 COMPLETE WORKFLOW SUCCESS!")
            print("=" * 70)
            print("✅ All steps completed successfully:")
            print("  1. ✅ Prepared model weights validated")
            print("  2. ✅ Model compiled for NeuronX")
            print()
            print(f"📁 Compiled model ready at: {compiled_output_path}")
            print()
            print("🚀 Ready for inference! Use:")
            print(f"python {SKILL_ROOT / 'scripts' / 'run_inference.py'} \\")
            print(f"  --model_class NeuronPhi3ForCausalLM \\")
            print(f"  --config_class Phi3InferenceConfig \\")
            print(f"  --model_path {prepared_model_path} \\")
            print(f"  --compiled_path {compiled_output_path} \\")
            print(f"  --prompt 'The meaning of life is'")
            
            return True
        else:
            print("\n❌ Compilation failed!")
            return False
            
    except Exception as e:
        print(f"\n💥 Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_phi3_prepare_and_compile():
    """Example: Validate prepared Phi3 weights and compile using direct imports."""
    print("📋 Example: Compile Prepared Phi3 Model (Direct Classes)")
    print("=" * 60)
    
    from neuron_port.modeling_phi3 import NeuronPhi3ForCausalLM, Phi3InferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig
    from model_compiler import DirectModelCompiler, CompilationConfig
    
    # Step 1: Validate model weights prepared by the setup flow
    current_dir = os.getcwd()
    hf_model_id = "microsoft/Phi-3-mini-4k-instruct"
    model_path = os.path.join(current_dir, "agent_artifacts/data/Phi-3-mini-4k-instruct")

    print(f"📦 Validating prepared weights for: {hf_model_id}")
    prepared_model_path = require_model_weights(model_path)
    print(f"✅ Model weights found at: {prepared_model_path}")

    # Step 2: Create configuration with direct class references
    config = CompilationConfig(
        model_class=NeuronPhi3ForCausalLM,       # Direct model class
        config_class=Phi3InferenceConfig,        # Direct config class
        neuron_config_class=NeuronConfig,        # Direct neuron config class
        model_path=prepared_model_path,          # Use prepared model path
        output_path=os.path.join(current_dir, "agent_artifacts/data/phi3-mini-4k-compiled"),  # Where to save compiled artifacts
        batch_size=1,
        seq_len=128,
        tp_degree=1,                             # Start with TP=1 for simplicity
        use_fp16=True,                           # Use bfloat16 for memory efficiency
    )
    
    print("Configuration:")
    print(f"  Model class: {config.model_class.__name__}")
    print(f"  Config class: {config.config_class.__name__}")
    print(f"  NeuronConfig class: {config.neuron_config_class.__name__}")
    print(f"  Model path: {config.model_path}")
    print(f"  Save model to: {config.model_path}")
    print(f"  Save compiled to: {config.output_path}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  TP degree: {config.tp_degree}")
    print(f"  Use FP16: {config.use_fp16}")
    
    # Step 2: Create compiler
    print("\n🔧 Creating compiler...")
    compiler = DirectModelCompiler(config)
    
    # Step 3: Run compilation
    print("\n🚀 Starting compilation...")
    print("This will:")
    print("  1. Validate the prepared Phi3 model weights")
    print("  2. Load model using NeuronPhi3ForCausalLM")
    print("  3. Configure using Phi3InferenceConfig")
    print("  4. Create NeuronConfig for compilation")
    print("  5. Compile for NeuronX hardware")
    print("  6. Save compiled artifacts")
    
    success = compiler.compile()
    
    if success:
        print("\n🎉 SUCCESS!")
        print(f"Model compiled and ready at: {config.output_path}")
        
        # Show how to run inference
        print("\n📝 To run inference:")
        print(f"python3 {SKILL_ROOT / 'scripts' / 'run_inference.py'} \\")
        print(f"  --model_path {config.output_path} \\")
        print(f"  --hf_model_path {config.model_path} \\")
        print(f"  --prompt 'The meaning of life is' \\")
        print(f"  --max_new_tokens 50")
    else:
        print("\n❌ Compilation failed!")
    
    return success

def example_phi3_compile_existing():
    """Example: Compile existing Phi3 model (no download) using direct classes."""
    print("📋 Example: Compile Existing Phi3 Model (Direct Classes)")
    print("=" * 60)
    
    from neuron_port.modeling_phi3 import NeuronPhi3ForCausalLM, Phi3InferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig
    from model_compiler import DirectModelCompiler, CompilationConfig
    
    # Check if model exists
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "agent_artifacts/data/Phi-3-mini-4k-instruct")
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Prepare model weights in agent_artifacts/data first")
        return False

    # Configuration for existing model with direct classes
    config = CompilationConfig(
        model_class=NeuronPhi3ForCausalLM,       # Direct model class
        config_class=Phi3InferenceConfig,        # Direct config class
        neuron_config_class=NeuronConfig,        # Direct neuron config class
        model_path=model_path,
        output_path=os.path.join(current_dir, "agent_artifacts/data/phi3-mini-4k-compiled-256"),
        batch_size=1,
        seq_len=256,                    # Longer sequence
        tp_degree=1,
        use_fp16=True,
    )
    
    print("Configuration:")
    print(f"  Model class: {config.model_class.__name__}")
    print(f"  Config class: {config.config_class.__name__}")
    print(f"  Model path: {config.model_path}")
    print(f"  Output path: {config.output_path}")
    print(f"  Sequence length: {config.seq_len}")
    
    # Create and run compiler
    compiler = DirectModelCompiler(config)
    success = compiler.compile()
    
    return success

def show_python_api_usage():
    """Show how to use the Python API with direct classes."""
    print("📋 Python API Usage Examples (Direct Classes)")
    print("=" * 60)
    
    print("1. Complete workflow - validate and compile Phi3:")
    print("""
from neuron_port.modeling_phi3 import NeuronPhi3ForCausalLM, Phi3InferenceConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from model_compiler import DirectModelCompiler, CompilationConfig

# Step 1: Use model weights prepared by the setup flow
model_path = require_model_weights("./agent_artifacts/data/Phi-3-mini-4k-instruct")

# Step 2: Configure compilation
config = CompilationConfig(
    model_class=NeuronPhi3ForCausalLM,
    config_class=Phi3InferenceConfig,
    neuron_config_class=NeuronConfig,
    model_path=model_path,
    output_path="./phi3_compiled",
    batch_size=1,
    seq_len=128,
    tp_degree=1,
    use_fp16=True
)

# Step 3: Compile
compiler = DirectModelCompiler(config)
success = compiler.compile()
""")
    
    print("\n2. Compile existing model:")
    print("""
config = CompilationConfig(
    model_class=NeuronPhi3ForCausalLM,
    config_class=Phi3InferenceConfig,
    neuron_config_class=NeuronConfig,
    model_path="./Phi-3-mini-4k-instruct",
    output_path="./phi3_compiled_existing",
    batch_size=1,
    seq_len=256,
    tp_degree=1,
    use_fp16=True
)
""")
    
    print("\n3. Run inference:")
    print("""
# Use the generic inference utility
python skills/neuron-framework-autoport/scripts/run_inference.py \\
  --model_class NeuronPhi3ForCausalLM \\
  --config_class Phi3InferenceConfig \\
  --model_path ./Phi-3-mini-4k-instruct \\
  --compiled_path ./phi3_compiled \\
  --prompt 'The meaning of life is' \\
  --max_new_tokens 50
""")
    
    print("\n📝 Note: Uses refactored architecture with separated concerns")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phi3 Model Compiler Examples")
    parser.add_argument("--example", choices=["full", "prepared", "download", "existing", "api"],
                       default="api", help="Which example to run")
    
    args = parser.parse_args()
    
    if args.example == "full":
        print("🚀 Running complete workflow example...")
        success = example_phi3_full_workflow()
    elif args.example in ("prepared", "download"):
        print("🚀 Running prepared-weights compile example...")
        success = example_phi3_prepare_and_compile()
    elif args.example == "existing":
        print("�  Running compile existing model example...")
        success = example_phi3_compile_existing()
    else:
        print("📋 Showing Python API usage examples...")
        show_python_api_usage()
        success = True
    
    if success:
        print("\n✅ Example completed successfully!")
    else:
        print("\n❌ Example failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
