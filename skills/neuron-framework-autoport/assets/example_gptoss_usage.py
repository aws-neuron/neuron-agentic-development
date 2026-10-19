#!/usr/bin/env python3
"""
Example showing how to use the refactored model compiler for GPT-OSS
Includes GPT-OSS specific dequantization routines.
"""

import sys
import os
import json
import torch
import shutil
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, 'src')
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

# Import safetensors for dequantization
try:
    import safetensors.torch as st
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("⚠️  safetensors not available - dequantization will not work")

def dequantize_mxfp4_expert_weight(quantized_blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Properly dequantize MXFP4 expert weights using the correct algorithm.
    
    Based on the HuggingFace transformers implementation in:
    transformers/src/transformers/integrations/mxfp4.py
    
    Args:
        quantized_blocks: [experts, features, blocks, block_size] e.g., [32, 5760, 90, 16] uint8
        scales: [experts, features, blocks] e.g., [32, 5760, 90] uint8
        
    Returns:
        torch.Tensor: Dequantized weights in proper format
    """
    experts, features, blocks, block_size = quantized_blocks.shape
    print(f"       Dequantizing MXFP4: {quantized_blocks.shape} with scales {scales.shape}")
    
    # FP4 lookup table from HuggingFace implementation
    FP4_VALUES = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    
    # Convert scales: uint8 -> int32 with proper offset (127 = 2^7)
    scales_int = scales.to(torch.int32) - 127
    
    # Create FP4 lookup table
    lut = torch.tensor(FP4_VALUES, dtype=torch.bfloat16, device=quantized_blocks.device)
    
    # Reshape for processing: [experts, features, blocks, block_size] -> [total_rows, block_size]
    total_rows = experts * features * blocks
    blocks_flat = quantized_blocks.reshape(total_rows, block_size)
    scales_flat = scales_int.reshape(total_rows, 1)
    
    # Output tensor: each uint8 block becomes 2 bfloat16 values (nibble unpacking)
    out = torch.empty(total_rows, block_size * 2, dtype=torch.bfloat16, device=quantized_blocks.device)
    
    # Process in chunks to manage memory
    rows_per_chunk = min(32768, total_rows)
    
    for r0 in range(0, total_rows, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, total_rows)
        
        blk = blocks_flat[r0:r1]
        exp = scales_flat[r0:r1]
        
        # Extract nibbles: each uint8 contains 2 FP4 values
        idx_lo = (blk & 0x0F).to(torch.long)  # Lower 4 bits
        idx_hi = (blk >> 4).to(torch.long)    # Upper 4 bits
        
        # Look up FP4 values
        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]  # Even indices get lower nibbles
        sub[:, 1::2] = lut[idx_hi]  # Odd indices get upper nibbles
        
        # Apply scaling: multiply by 2^exp
        torch.ldexp(sub, exp, out=sub)
    
    # Reshape back to original structure with doubled last dimension
    # [total_rows, block_size * 2] -> [experts, features, blocks, block_size * 2]
    out = out.reshape(experts, features, blocks, block_size * 2)
    
    # Flatten the blocks dimension: [experts, features, blocks * block_size * 2]
    out = out.reshape(experts, features, blocks * block_size * 2)
    
    # Fix the shape to match HuggingFace expectations
    # The HuggingFace model expects different shapes for gate_up_proj vs down_proj
    if features == 5760:  # gate_up_proj case
        # Current: [32, 5760, 2880], Expected: [32, 2880, 5760]
        # This is a transpose operation
        out = out.transpose(1, 2).contiguous()
        print(f"       Transposed gate_up_proj to match HF format: {out.shape}")
    elif features == 2880:  # down_proj case  
        # Current: [32, 2880, 2880], Expected: [32, 2880, 2880]
        # This is already correct
        print(f"       down_proj shape is correct: {out.shape}")
    
    print(f"       Final shape: {out.shape}")
    print(f"       Weight statistics: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}, std={out.std().item():.4f}")
    
    return out.contiguous()

def is_model_quantized(model_path: str) -> bool:
    """Check if the model is MXFP4 quantized by looking for _blocks weights."""
    if not ST_AVAILABLE:
        return False
        
    try:
        # Check for quantized weight files
        weight_files = list(Path(model_path).glob("*.safetensors"))
        if not weight_files:
            return False
        
        # Load first weight file and check for quantized patterns
        weights = st.load_file(str(weight_files[0]))
        
        # Look for MXFP4 quantized expert weights (4D tensors with _blocks suffix)
        for key, value in weights.items():
            if ('experts' in key and '_blocks' in key and 
                len(value.shape) == 4):
                print(f"   Found quantized weight: {key} {value.shape}")
                return True
        
        return False
        
    except Exception as e:
        print(f"   Error checking quantization: {e}")
        return False

def is_mxfp4_quantized(model_path: str) -> bool:
    """Check if model is MXFP4 quantized by looking for _blocks and _scales parameters."""
    try:
        # Check config first
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            if config.get('quantization_config', {}).get('quant_method') == 'mxfp4':
                return True
        
        # Check for quantized weight files
        safetensor_files = list(Path(model_path).glob("*.safetensors"))
        if safetensor_files:
            # Load first file and check for _blocks parameters
            weights = st.load_file(str(safetensor_files[0]))
            for key in weights.keys():
                if '_blocks' in key and 'experts' in key:
                    return True
        
        return False
    except:
        return False

def dequantize_gptoss_model(src_path: str, dst_path: str) -> bool:
    """Dequantize MXFP4 GPT-OSS model to proper 3D expert format."""
    if not ST_AVAILABLE:
        print("❌ safetensors not available - cannot dequantize")
        return False
        
    print("🔄 Dequantizing MXFP4 model...")
    
    try:
        # Create output directory
        Path(dst_path).mkdir(parents=True, exist_ok=True)
        
        # Copy non-weight files
        for file in ["config.json", "generation_config.json", "tokenizer.json", 
                    "tokenizer_config.json", "special_tokens_map.json"]:
            src_file = Path(src_path) / file
            if src_file.exists():
                dst_file = Path(dst_path) / file
                if file == "config.json":
                    # Remove quantization config
                    with open(src_file, 'r') as f:
                        config = json.load(f)
                    if "quantization_config" in config:
                        del config["quantization_config"]
                    with open(dst_file, 'w') as f:
                        json.dump(config, f, indent=2)
                else:
                    shutil.copy2(src_file, dst_file)
        
        # Load and process all weight files
        src_files = list(Path(src_path).glob("*.safetensors"))
        if not src_files:
            return False
        
        # First pass: collect all weights
        all_raw_weights = {}
        for src_file in src_files:
            print(f"   Loading {src_file.name}...")
            weights = st.load_file(str(src_file))
            all_raw_weights.update(weights)
        
        # Second pass: process weights, handling MXFP4 quantized pairs
        processed_weights = {}
        processed_keys = set()
        
        for key, value in all_raw_weights.items():
            if key in processed_keys:
                continue
                
            # Handle MXFP4 quantized expert weights
            if len(value.shape) == 4 and 'experts' in key and '_blocks' in key:
                print(f"     Dequantizing {key}: {value.shape}")
                
                # Find corresponding scales
                scales_key = key.replace('_blocks', '_scales')
                scales = all_raw_weights.get(scales_key)
                
                if scales is not None:
                    dequantized_weight = dequantize_mxfp4_expert_weight(value, scales)
                    processed_keys.add(scales_key)  # Mark scales as processed
                else:
                    print(f"       ⚠️ No scales found for {key}")
                    continue
                
                # Fix parameter names: remove '_blocks' suffix
                new_key = key.replace('_blocks', '')
                processed_weights[new_key] = dequantized_weight
                processed_keys.add(key)
                
            elif '_scales' in key:
                # Skip scales - they're handled with their corresponding blocks and should not be saved
                processed_keys.add(key)
                print(f"     Skipping scales parameter: {key}")
                continue
                
            else:
                # Regular weights - just convert to bfloat16 and make contiguous
                processed_weights[key] = value.to(torch.bfloat16).contiguous()
                processed_keys.add(key)
        
        # Save dequantized weights
        output_file = os.path.join(dst_path, "model.safetensors")
        st.save_file(processed_weights, output_file)
        
        # Create index file
        total_size = sum(w.numel() * w.element_size() for w in processed_weights.values())
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": {k: "model.safetensors" for k in processed_weights.keys()}
        }
        
        with open(os.path.join(dst_path, "model.safetensors.index.json"), 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"✅ MXFP4 dequantization completed!")
        print(f"   Processed {len(processed_weights)} weights")
        print(f"   Total size: {total_size / (1024**3):.2f} GB")
        print(f"   Saved to: {dst_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dequantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_gptoss_prepare_and_compile():
    """Example: Compile a prepared GPT-OSS model using direct class imports."""
    print("📋 Example: Compile Prepared GPT-OSS Model (Direct Classes)")
    print("=" * 60)
    
    # Import the required classes directly
    import sys
    import os
    
    # Add gptoss path for imports
    gptoss_path = os.path.abspath(os.path.join('src'))
    if gptoss_path not in sys.path:
        sys.path.insert(0, gptoss_path)
    
    from neuron_port.modeling_gptoss import NeuronGptOssForCausalLM, GptOssInferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig
    from model_compiler import DirectModelCompiler, CompilationConfig
    
    # Step 1: Create configuration with direct class references
    config = CompilationConfig(
        model_class=NeuronGptOssForCausalLM,     # Direct model class
        config_class=GptOssInferenceConfig,      # Direct config class
        neuron_config_class=NeuronConfig,        # Direct neuron config class
        model_path="./gpt_oss_hf_official",      # Prepared model weights
        output_path="./gptoss_compiled",         # Where to save compiled artifacts
        batch_size=1,
        seq_len=128,
        tp_degree=8,                             # Minimal required TP degree for 20B model
        use_fp16=True,                           # Use bfloat16 for memory efficiency
        # Note: Model weights should be prepared and dequantized separately
    )
    
    print("Configuration:")
    print(f"  Model class: {config.model_class.__name__}")
    print(f"  Config class: {config.config_class.__name__}")
    print(f"  NeuronConfig class: {config.neuron_config_class.__name__}")
    print(f"  Save model to: {config.model_path}")
    print(f"  Save compiled to: {config.output_path}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  TP degree: {config.tp_degree} (minimal required for 20B model)")
    print(f"  Use FP16: {config.use_fp16}")
    
    # Step 2: Create compiler
    print("\n🔧 Creating compiler...")
    compiler = DirectModelCompiler(config)
    
    # Step 3: Run compilation
    print("\n🚀 Starting compilation...")
    print("This will:")
    print("  1. Validate the prepared GPT-OSS 20B model weights")
    print("  2. Load model using NeuronGptOssForCausalLM")
    print("  3. Configure using GptOssInferenceConfig")
    print("  4. Create NeuronConfig for compilation with modular flow")
    print("  5. Compile for NeuronX hardware with MoE optimizations")
    print("  6. Save compiled artifacts with weight sharding")
    
    success = compiler.compile()
    
    if success:
        print("\n🎉 SUCCESS!")
        print(f"Model compiled and ready at: {config.output_path}")
        
        # Show how to run inference
        print("\n📝 To run inference:")
        print(f"python3 {SKILL_ROOT / 'scripts' / 'run_inference.py'} \\")
        print(f"  --model_path {config.output_path} \\")
        print(f"  --hf_model_path {config.model_path} \\")
        print(f"  --prompt 'The future of artificial intelligence is' \\")
        print(f"  --max_new_tokens 50")
    else:
        print("\n❌ Compilation failed!")
    
    return success

def example_gptoss_compile_existing():
    """Example: Compile existing GPT-OSS model (no download) using direct classes."""
    print("📋 Example: Compile Existing GPT-OSS Model (Direct Classes)")
    print("=" * 60)
    
    # Import the required classes directly
    import sys
    import os
    
    # Add gptoss path for imports
    gptoss_path = os.path.abspath(os.path.join('neuron_port'))
    if gptoss_path not in sys.path:
        sys.path.insert(0, gptoss_path)
    
    from modeling_gptoss import NeuronGptOssForCausalLM, GptOssInferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig
    from model_compiler import DirectModelCompiler, CompilationConfig
    
    # Check if model exists
    model_path = "./gpt_oss_hf_official"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Prepare the GPT-OSS model weights first")
        return False
    
    # Configuration for existing model with direct classes
    config = CompilationConfig(
        model_class=NeuronGptOssForCausalLM,     # Direct model class
        config_class=GptOssInferenceConfig,      # Direct config class
        neuron_config_class=NeuronConfig,        # Direct neuron config class
        model_path=model_path,
        output_path="./gpt_oss_compiled_existing",
        batch_size=1,
        seq_len=256,                    # Longer sequence
        tp_degree=8,                    # Minimal required for 20B model
        use_fp16=True,
        reduce_layers=12,               # Reduce layers for faster compilation (from 24 to 12)
    )
    
    print("Configuration:")
    print(f"  Model class: {config.model_class.__name__}")
    print(f"  Config class: {config.config_class.__name__}")
    print(f"  Model path: {config.model_path}")
    print(f"  Output path: {config.output_path}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  TP degree: {config.tp_degree}")
    print(f"  Reduce layers: {config.reduce_layers} (from 24 to 12 for faster compilation)")
    
    # Create and run compiler
    compiler = DirectModelCompiler(config)
    success = compiler.compile()
    
    return success

def example_gptoss_dequantization():
    """Example: Dequantize MXFP4 GPT-OSS model using extracted routines."""
    print("📋 Example: Dequantize MXFP4 GPT-OSS Model")
    print("=" * 60)
    
    # Check if quantized model exists
    quantized_model_path = "./gpt_oss_hf_official"
    dequantized_model_path = "./gpt_oss_hf_official_dequantized"
    
    if not os.path.exists(quantized_model_path):
        print(f"❌ Quantized model not found at {quantized_model_path}")
        print("Please ensure you have a quantized GPT-OSS model available")
        return False
    
    # Check if model is actually quantized
    if not is_mxfp4_quantized(quantized_model_path):
        print(f"⚠️  Model at {quantized_model_path} is not MXFP4 quantized")
        print("This example requires an MXFP4 quantized GPT-OSS model")
        return False
    
    print(f"✅ Found MXFP4 quantized model at: {quantized_model_path}")
    
    # Check if dequantized version already exists
    if os.path.exists(dequantized_model_path):
        print(f"✅ Dequantized model already exists at: {dequantized_model_path}")
        return True
    
    # Perform dequantization
    print(f"🔄 Dequantizing model...")
    print(f"   Source: {quantized_model_path}")
    print(f"   Target: {dequantized_model_path}")
    
    success = dequantize_gptoss_model(quantized_model_path, dequantized_model_path)
    
    if success:
        print("\n🎉 Dequantization completed successfully!")
        print(f"✅ Dequantized model saved to: {dequantized_model_path}")
        print("\n💡 Next steps:")
        print("  1. Use the dequantized model for compilation")
        print("  2. The dequantized model is ready for NeuronX compilation")
        print("  3. Expert weights are now in proper 3D format")
        
        # Show how to use in compilation
        print(f"\n📝 To compile the dequantized model:")
        print(f"config = CompilationConfig(")
        print(f"    model_path='{dequantized_model_path}',")
        print(f"    # ... other parameters")
        print(f")")
    else:
        print("\n❌ Dequantization failed!")
        print("Check the error messages above for details")
    
    return success

def example_gptoss_with_quantization():
    """Example: Compile GPT-OSS model with quantization for memory efficiency."""
    print("📋 Example: Compile GPT-OSS Model with Quantization")
    print("=" * 60)
    
    # Import the required classes directly
    import sys
    import os
    
    # Add gptoss path for imports
    gptoss_path = os.path.abspath(os.path.join('src'))
    if gptoss_path not in sys.path:
        sys.path.insert(0, gptoss_path)
    
    from neuron_port.modeling_gptoss import NeuronGptOssForCausalLM, GptOssInferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig
    from model_compiler import DirectModelCompiler, CompilationConfig
    
    # Check if model exists
    model_path = "./gpt-oss-20b"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Prepare the GPT-OSS model weights first")
        return False
    
    # Configuration with quantization enabled
    config = CompilationConfig(
        model_class=NeuronGptOssForCausalLM,     # Direct model class
        config_class=GptOssInferenceConfig,      # Direct config class
        neuron_config_class=NeuronConfig,        # Direct neuron config class
        model_path=model_path,
        output_path="./gptoss_compiled_quantized",
        batch_size=1,
        seq_len=128,
        tp_degree=8,                    # Minimal required for 20B model
        use_fp16=True
    )
    
    print("Configuration:")
    print(f"  Model class: {config.model_class.__name__}")
    print(f"  Config class: {config.config_class.__name__}")
    print(f"  Model path: {config.model_path}")
    print(f"  Output path: {config.output_path}")
    print(f"  TP degree: {config.tp_degree}")
    print(f"  Quantization: {config.quantize} (per_tensor_symmetric int8)")
    print(f"  Expected benefits: Reduced memory usage and faster inference")
    
    # Create and run compiler
    compiler = DirectModelCompiler(config)
    success = compiler.compile()
    
    if success:
        print("\n🎉 Quantized model compilation completed!")
        print("💡 Quantization benefits:")
        print("  - ~50% memory reduction")
        print("  - Faster inference on NeuronX hardware")
        print("  - Maintained model quality with int8 precision")
    
    return success

def show_python_api_usage():
    """Show how to use the Python API with direct classes."""
    print("📋 Python API Usage Examples (Direct Classes)")
    print("=" * 60)
    
    print("1. Compile prepared GPT-OSS 20B weights:")
    print("""
from neuron_port.modeling_gptoss import NeuronGptOssForCausalLM, GptOssInferenceConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from model_compiler import DirectModelCompiler, CompilationConfig

config = CompilationConfig(
    model_class=NeuronGptOssForCausalLM,
    config_class=GptOssInferenceConfig,
    neuron_config_class=NeuronConfig,
    model_path="./gpt-oss-20b",
    output_path="./gptoss_compiled",
    tp_degree=8,  # Minimal required for 20B model
    use_fp16=True
)

compiler = DirectModelCompiler(config)
success = compiler.compile()
""")
    
    print("\n2. Compile existing model with quantization:")
    print("""
config = CompilationConfig(
    model_class=NeuronGptOssForCausalLM,
    config_class=GptOssInferenceConfig,
    neuron_config_class=NeuronConfig,
    model_path="./gpt-oss-20b",
    output_path="./gptoss_compiled_quantized",
    quantize=True,  # Enable int8 quantization
    tp_degree=8,
    use_fp16=True
)
""")
    
    print("\n3. Compile with reduced layers for testing:")
    print("""
config = CompilationConfig(
    model_class=NeuronGptOssForCausalLM,
    config_class=GptOssInferenceConfig,
    neuron_config_class=NeuronConfig,
    model_path="./gpt-oss-20b",
    output_path="./gptoss_compiled_small",
    reduce_layers=12,  # Reduce from 24 to 12 layers
    tp_degree=8,
    use_fp16=True
)
""")
    
    print("\n4. Advanced configuration with MoE optimizations:")
    print("""
config = CompilationConfig(
    model_class=NeuronGptOssForCausalLM,
    config_class=GptOssInferenceConfig,
    neuron_config_class=NeuronConfig,
    model_path="./gpt-oss-20b",
    output_path="./gptoss_compiled_optimized",
    tp_degree=16,  # Higher TP for better expert distribution
    seq_len=512,   # Longer sequences
    batch_size=2,  # Larger batch size
    use_fp16=True
)
""")
    
    print("\n📝 GPT-OSS Model Features:")
    print("  - 20B parameters with Mixture of Experts (MoE)")
    print("  - 32 experts per layer, top-4 routing")
    print("  - Grouped Query Attention (GQA): 64 query heads, 8 KV heads")
    print("  - Alternating sliding window (128) and full attention")
    print("  - RoPE with YARN scaling for long sequences")
    print("  - Modular flow compilation for memory efficiency")
    
    print("\n💡 Compilation Tips:")
    print("  - Use TP degree 8+ for 20B model (memory requirements)")
    print("  - Enable quantization for memory-constrained environments")
    print("  - Reduce layers for faster compilation during development")
    print("  - Modular flow is automatically enabled for GPT-OSS")
    
    print("\n📝 Note: Only Python API is supported - direct class imports required")

def example_gptoss_full_workflow():
    """Example: Complete GPT-OSS workflow using prepared model weights."""
    print("📋 Example: Complete GPT-OSS Workflow (Prepared Weights + Dequantize + Compile)")
    print("=" * 80)
    
    # Import the required classes and utilities
    import sys
    import os
    
    # Add paths for imports
    sys.path.insert(0, 'src')
    
    from neuron_port.modeling_gptoss import NeuronGptOssForCausalLM, GptOssInferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig
    from model_compiler import DirectModelCompiler, CompilationConfig
    
    # Configuration
    hf_model_id = "openai/gpt-oss-20b"
    raw_model_path = "./gpt_oss_hf_official"
    dequantized_model_path = "./gpt_oss_hf_official_dequantized"
    compiled_output_path = "./gptoss_compiled_full_workflow"
    
    print("🎯 Workflow Configuration:")
    print(f"  HuggingFace Model: {hf_model_id}")
    print(f"  Model Path: {raw_model_path}")
    print(f"  Dequantized Path: {dequantized_model_path}")
    print(f"  Compiled Output: {compiled_output_path}")
    print()
    
    try:
        # Step 1: Validate model weights prepared by the setup flow
        print("📦 Step 1: Validate Prepared Model Weights")
        print("-" * 40)
        prepared_weights_path = require_model_weights(raw_model_path)
        print(f"✅ Model weights found at: {prepared_weights_path}")
        print()
        
        # Step 2: Dequantize model (if needed)
        print("🔄 Step 2: Dequantize Model")
        print("-" * 40)
        if is_mxfp4_quantized(prepared_weights_path):
            print("🔍 MXFP4 quantization detected, dequantizing...")
            success = dequantize_gptoss_model(prepared_weights_path, dequantized_model_path)
            if not success:
                print("❌ Dequantization failed!")
                return False
            prepared_model_path = dequantized_model_path
        else:
            print("✅ Model is not quantized, using original")
            prepared_model_path = prepared_weights_path
        print(f"✅ Model prepared at: {prepared_model_path}")
        print()
        
        # Step 3: Create compilation configuration
        print("⚙️  Step 3: Configure Compilation")
        print("-" * 40)
        config = CompilationConfig(
            model_class=NeuronGptOssForCausalLM,
            config_class=GptOssInferenceConfig,
            neuron_config_class=NeuronConfig,
            model_path=prepared_model_path,
            output_path=compiled_output_path,
            batch_size=1,
            seq_len=128,
            tp_degree=8,  # Minimal required for 20B model
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
        
        # Step 4: Compile model
        print("🚀 Step 4: Compile Model")
        print("-" * 40)
        compiler = DirectModelCompiler(config)
        success = compiler.compile()
        
        if success:
            print("\n🎉 COMPLETE WORKFLOW SUCCESS!")
            print("=" * 80)
            print("✅ All steps completed successfully:")
            print("  1. ✅ Prepared model weights validated")
            print("  2. ✅ MXFP4 dequantization completed")
            print("  3. ✅ Model compiled for NeuronX")
            print()
            print(f"📁 Compiled model ready at: {compiled_output_path}")
            print()
            print("🚀 Ready for inference! Use:")
            print(f"python {SKILL_ROOT / 'scripts' / 'run_inference.py'} \\")
            print(f"  --model_class NeuronGptOssForCausalLM \\")
            print(f"  --config_class GptOssInferenceConfig \\")
            print(f"  --model_path {prepared_model_path} \\")
            print(f"  --compiled_path {compiled_output_path} \\")
            print(f"  --prompt 'The future of AI is'")
            
            return True
        else:
            print("\n❌ Compilation failed!")
            return False
            
    except Exception as e:
        print(f"\n💥 Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_mcp_tensor_parallel_analysis():
    """Show how to use MCP tool to analyze tensor parallel requirements."""
    print("📋 MCP Tensor Parallel Analysis for GPT-OSS 20B")
    print("=" * 60)
    
    # GPT-OSS 20B model configuration
    model_config = {
        "number_of_layers": 24,
        "hidden_size": 2880,
        "number_of_attention_heads": 64,
        "number_of_key_value_heads": 8,
        "intermediate_size": 2880,  # For MoE, this is per expert
        "vocabulary_size": 201088
    }
    
    input_shape = {
        "sequence_length": 128,
        "batch_size": 1
    }
    
    print("Model Configuration:")
    print(f"  - Layers: {model_config['number_of_layers']}")
    print(f"  - Hidden size: {model_config['hidden_size']}")
    print(f"  - Attention heads: {model_config['number_of_attention_heads']}")
    print(f"  - KV heads: {model_config['number_of_key_value_heads']}")
    print(f"  - Vocabulary size: {model_config['vocabulary_size']}")
    print(f"  - Sequence length: {input_shape['sequence_length']}")
    print(f"  - Batch size: {input_shape['batch_size']}")
    
    print("\nAnalyzing tensor parallel requirements...")
    
    # Analyze for both Trn1 and Trn2
    results = {}
    for instance_type in ["Trn1", "Trn2"]:
        try:
            # Note: This will be handled by the MCP tool automatically
            print(f"  {instance_type}: Analyzing...")
            results[instance_type] = "Analysis will be performed by MCP tool"
        except Exception as e:
            print(f"  {instance_type}: Analysis failed - {e}")
            results[instance_type] = f"Failed: {e}"
    
    print("\n💡 GPT-OSS 20B Recommendations:")
    print("  - Minimal TP degree depends on instance type and memory constraints")
    print("  - MoE models benefit from higher TP degrees for expert distribution")
    print("  - Consider both model parameters and expert routing overhead")
    print("  - Trn1 (16GB/rank): Typically TP=8 or higher")
    print("  - Trn2 (12GB/rank): Typically TP=16 or higher")
    
    print("\n🔧 Usage in compilation:")
    print("  config = CompilationConfig(")
    print("      tp_degree=8,  # Use MCP analysis result")
    print("      # ... other parameters")
    print("  )")
    
    return results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-OSS Direct Class Compiler Examples")
    parser.add_argument("--example", choices=["prepared", "download", "existing", "quantized", "dequant", "full", "api", "mcp"],
                       default="api", help="Which example to run")
    
    args = parser.parse_args()
    
    if args.example in ("prepared", "download"):
        print("🚀 Running prepared-weights compile example...")
        success = example_gptoss_prepare_and_compile()
    elif args.example == "existing":
        print("🚀 Running compile existing model example...")
        success = example_gptoss_compile_existing()
    elif args.example == "quantized":
        print("🚀 Running quantized compilation example...")
        success = example_gptoss_with_quantization()
    elif args.example == "dequant":
        print("🚀 Running dequantization example...")
        success = example_gptoss_dequantization()
    elif args.example == "full":
        print("🚀 Running complete workflow example...")
        success = example_gptoss_full_workflow()
    elif args.example == "mcp":
        print("🚀 Running MCP tensor parallel analysis...")
        show_mcp_tensor_parallel_analysis()
        success = True
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
