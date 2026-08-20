# Model Implementation Guide for NeuronX Distributed Inference

This guide provides a comprehensive approach to implementing language models for the NeuronX Distributed Inference framework. It covers common issues, solutions, and best practices based on successful implementations of models like Mistral, Qwen2, and DBRX.

## Table of Contents

1. [Module Structure](#module-structure)
2. [Configuration Class Implementation](#configuration-class-implementation)
3. [Model Initialization](#model-initialization)
4. [Weight Conversion](#weight-conversion)
5. [Attention Implementation](#attention-implementation)
6. [MLP Implementation](#mlp-implementation)
7. [Model Compilation and Inference](#model-compilation-and-inference)
8. [Debugging and Error Handling](#debugging-and-error-handling)
9. [Model-Specific Considerations](#model-specific-considerations)
10. [Validation with Existing Implementations](#validation-with-existing-implementations)

## Module Structure

### Issue

The module structure needs to be properly set up for imports to work correctly.

### Solution

- Create proper directory structure (`models/your_model/`)
- Ensure `__init__.py` files export the necessary classes
- Install the package in development mode with `pip install -e .`

### Example

```
src/neuronx_distributed_inference/models/your_model/
├── __init__.py
└── modeling_your_model.py
```

In `__init__.py`:

```python
from neuronx_distributed_inference.models.your_model.modeling_your_model import (
    YourModelInferenceConfig,
    NeuronYourModelForCausalLM,
    NeuronYourModelModel,
)

__all__ = [
    "YourModelInferenceConfig",
    "NeuronYourModelForCausalLM",
    "NeuronYourModelModel",
]
```

## Configuration Class Implementation

### Issue

The model configuration class needs proper implementation of `from_pretrained` method.

### Solution

- Implement `from_pretrained` to read from model configuration files
- Handle `neuron_config` parameter correctly (extract from kwargs)
- Set proper default values for all required parameters

### Example

```python
class YourModelInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        # Add model-specific parameters

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration"""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            # Add model-specific required attributes
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use"""
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "YourModelInferenceConfig":
        """
        Load configuration from a pretrained model directory

        Args:
            model_path: Path to the model directory
            **kwargs: Additional arguments to override configuration

        Returns:
            YourModelInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)

        # Read config file and create config dict
        config_path = os.path.join(model_path, "config.json")  # or params.json, etc.
        with open(config_path, "r") as f:
            params = json.load(f)

        # Create config dict with defaults from config file
        config_dict = {
            "hidden_size": params.get("hidden_size", 1024),
            "num_attention_heads": params.get("num_attention_heads", 16),
            # Add other parameters...
        }

        # Override with remaining kwargs
        config_dict.update(kwargs)

        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)
        return config
```

## Model Initialization

### Issue

The model initialization process requires proper handling of distributed environment.

### Solution

- Initialize distributed environment before creating the model
- Initialize parallel groups with proper tensor parallelism degree
- Create a `from_config` class method for model initialization

### Example

```python
# Initialize distributed environment
import torch.distributed as dist
if not dist.is_initialized():
    dist.init_process_group(backend="xla", init_method="pjrt://", world_size=1, rank=0)

# Initialize parallel groups
import neuronx_distributed as nxd
nxd.parallel_layers.parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=args.tp_degree,
)

# Model class with from_config method
class NeuronYourModelForCausalLM(NeuronBaseForCausalLM):
    """
    Your model causal language model for inference
    """
    _model_cls = NeuronYourModelModel

    @classmethod
    def from_config(cls, config):
        """
        Create a model from a configuration

        Args:
            config: Model configuration

        Returns:
            NeuronYourModelForCausalLM: Model instance
        """
        return cls(config=config)
```

## Weight Conversion

### Issue

The weights from original format need to be properly converted to NeuronX format.

### Solution

- Implement `load_checkpoint` to load weights from original checkpoint
- Implement `convert_to_neuron_state_dict` to convert weights to NeuronX format
- Map weight names correctly (e.g., `output.weight` → `lm_head.weight`)

### Example

```python
@staticmethod
def convert_to_neuron_state_dict(state_dict, config):
    """
    Convert weights from original format to NeuronX format

    Args:
        state_dict: Original state dictionary
        config: Model configuration

    Returns:
        Dict[str, torch.Tensor]: NeuronX format state dictionary
    """
    neuron_state_dict = {}

    # Token embeddings
    if "embeddings.weight" in state_dict:
        neuron_state_dict["embed_tokens.weight"] = state_dict["embeddings.weight"].clone()

    # Final normalization
    if "norm.weight" in state_dict:
        neuron_state_dict["norm.weight"] = state_dict["norm.weight"].clone()

    # Output projection
    if "output.weight" in state_dict:
        neuron_state_dict["lm_head.weight"] = state_dict["output.weight"].clone()

    # Decoder layers
    for i in range(config.num_hidden_layers):
        # Attention weights
        if f"layers.{i}.attention.query.weight" in state_dict:
            neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.q_proj.weight"] = state_dict[f"layers.{i}.attention.query.weight"].clone()
        if f"layers.{i}.attention.key.weight" in state_dict:
            neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.k_proj.weight"] = state_dict[f"layers.{i}.attention.key.weight"].clone()
        if f"layers.{i}.attention.value.weight" in state_dict:
            neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.v_proj.weight"] = state_dict[f"layers.{i}.attention.value.weight"].clone()
        if f"layers.{i}.attention.output.weight" in state_dict:
            neuron_state_dict[f"layers.{i}.self_attn.o_proj.weight"] = state_dict[f"layers.{i}.attention.output.weight"].clone()

        # MLP weights
        # ... (model-specific MLP weight conversion)

        # Layer norms
        # ... (model-specific layer norm weight conversion)

    # Add rank information for tensor parallelism
    neuron_config = config.neuron_config
    tp_degree = neuron_config.tp_degree

    # Add rank information for attention
    for i in range(config.num_hidden_layers):
        neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

    # Add rank information for base model
    neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

    return neuron_state_dict
```

## Attention Implementation

### Issue

The attention mechanism needs to handle different attention types correctly.

### Solution

- Implement proper handling of tensor parallelism with attention
- Handle cases where TP degree and KV heads are not divisible
- Use `NeuronAttentionBase` with proper configuration

### Example

```python
class NeuronYourModelAttention(NeuronAttentionBase):
    """
    Your model attention implementation for NeuronX
    """
    def __init__(self, config):
        rotary_emb = None
        if getattr(config, "position_embedding_type", "rotary") == "rotary":
            rotary_emb = RotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                max_position_embeddings=getattr(config, "max_position_embeddings", 4096),
                base=getattr(config, "rotary_base", 10000.0),
            )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=rotary_emb,
            rope_theta=getattr(config, "rope_theta", 10000.0),
            use_scaled_rope=getattr(config, "use_scaled_rope", False),
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            sliding_window=getattr(config, "sliding_window", None),
        )
```

## MLP Implementation

### Issue

The MLP implementation needs to match the model's architecture.

### Solution

- Implement MLP with appropriate activation function
- Use appropriate layer structure (separate or combined projections)
- Calculate intermediate size based on model parameters

### Example

```python
class NeuronYourModelMLP(nn.Module):
    """
    Your model MLP implementation for NeuronX
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Calculate intermediate size
        intermediate_size = getattr(config, "intermediate_size", 4 * config.hidden_size)

        # Create MLP layers based on architecture
        if getattr(config, "mlp_type", "swiglu") == "swiglu":
            # Separate gate and up projections for SwiGLU
            self.gate_proj = ColumnParallelLinear(
                config.hidden_size,
                intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
            )

            self.up_proj = ColumnParallelLinear(
                config.hidden_size,
                intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
            )

            self.act_fn = nn.SiLU()
        else:
            # Combined projection for GELU
            self.fc_in = ColumnParallelLinear(
                config.hidden_size,
                intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
            )

            self.act_fn = nn.GELU()

        # Down projection
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, x):
        if hasattr(self, "gate_proj"):
            # SwiGLU activation
            gate_output = self.act_fn(self.gate_proj(x))
            up_output = self.up_proj(x)

            # Multiply gate and up outputs
            intermediate_output = gate_output * up_output
        else:
            # GELU activation
            intermediate_output = self.act_fn(self.fc_in(x))

        # Apply down projection
        output = self.down_proj(intermediate_output)

        return output, None  # Return None as second output for compatibility
```

## Model Compilation and Inference

### Issue

The model compilation and inference process requires proper setup.

### Solution

- Create separate scripts for compilation and inference
- Use minimal stable settings as recommended in the guide
- Implement proper error handling and debugging

### Example

```python
# Create NeuronConfig with minimal stable settings
neuron_config = NeuronConfig(
    tp_degree=1,           # No tensor parallelism initially
    batch_size=1,          # Single batch
    seq_len=512,           # Smaller sequence length
    enable_bucketing=False, # No bucketing
    save_sharded_checkpoint=False,  # No sharding
    # NO on_device_sampling_config
)

# Compilation script
def compile_model():
    # Load configuration
    config = YourModelInferenceConfig.from_pretrained(
        args.checkpoint_path,
        neuron_config=neuron_config,
    )

    # Initialize model
    model = NeuronYourModelForCausalLM.from_config(config)

    # Load weights
    state_dict = load_checkpoint(args.checkpoint_path)
    neuron_state_dict = convert_to_neuron_state_dict(state_dict, config)
    model.load_state_dict(neuron_state_dict)

    # Compile model
    model.compile_model()

    # Save compiled model
    model.save_pretrained(args.output_path)

# Inference script
def run_inference():
    # Load compiled model
    model = NeuronYourModelForCausalLM.from_pretrained(args.model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Tokenize input
    inputs = tokenizer(args.prompt, return_tensors="pt")

    # Generate text
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
```

## Debugging and Error Handling

### Issue

Debugging distributed models can be challenging.

### Solution

- Create a debug script to check checkpoint files, configuration, etc.
- Add verbose logging to track the compilation process
- Follow a systematic debugging approach

### Example

```python
def check_model_config(checkpoint_path):
    """Check if configuration file exists and has the expected format"""
    config_path = os.path.join(checkpoint_path, "config.json")  # or params.json, etc.
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        return False

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Check required fields
        required_fields = [
            "hidden_size", "num_attention_heads", "num_hidden_layers",
            "vocab_size", "max_position_embeddings"
        ]
        for field in required_fields:
            if field not in config:
                logger.error(f"Required field '{field}' not found in configuration")
                return False

        logger.info(f"Configuration looks good: {config}")
        return True

    except Exception as e:
        logger.error(f"Error reading configuration: {e}")
        return False

def check_checkpoint_files(checkpoint_path):
    """Check if checkpoint files exist"""
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".bin") or f.endswith(".pth")]
    if not checkpoint_files:
        logger.error(f"No checkpoint files found in {checkpoint_path}")
        return False

    logger.info(f"Found checkpoint files: {checkpoint_files}")
    return True

def check_weight_loading(checkpoint_path):
    """Check if weights can be loaded"""
    try:
        # Load a small part of the weights to verify
        checkpoint_file = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
        checkpoint = torch.load(checkpoint_file, map_location="cpu")

        # Check if expected keys are present
        expected_keys = ["embeddings.weight", "norm.weight", "output.weight"]
        for key in expected_keys:
            if key not in checkpoint:
                logger.error(f"Expected key '{key}' not found in checkpoint")
                return False

        logger.info("Weight loading check passed")
        return True

    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        return False
```

## Model-Specific Considerations

### 1. Mixture of Experts Models (e.g., DBRX, Mixtral)

- **Router Implementation**:

  ```python
  class NeuronMoERouter(nn.Module):
      def __init__(self, config):
          super().__init__()
          self.linear_router = ColumnParallelLinear(
              config.hidden_size,
              config.num_experts,
              bias=False,
              gather_output=True,
              dtype=config.neuron_config.torch_dtype,
          )
          self.top_k = config.num_experts_per_token

      def forward(self, hidden_states):
          router_logits = self.linear_router(hidden_states)
          routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
          routing_weights = F.softmax(routing_weights, dim=-1)
          return routing_weights, selected_experts
  ```

- **Expert Weight Sharding**:

  ```python
  # Convert expert weights
  for e in range(config.num_experts):
      # Copy gate_proj and up_proj after concatenation
      gate_proj_weights = state_dict[f"experts.{e}.gate_proj.weight"].clone()
      up_proj_weights = state_dict[f"experts.{e}.up_proj.weight"].clone()

      gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
      gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
      gate_proj_slice.copy_(gate_proj_weights)
      up_proj_slice = torch.narrow(gate_up_proj_slice, 2, intermediate_size, intermediate_size)
      up_proj_slice.copy_(up_proj_weights)
  ```

### 2. Sliding Window Attention (e.g., Mistral)

- **Configuration**:

  ```python
  class MistralInferenceConfig(InferenceConfig):
      def add_derived_config(self):
          self.sliding_window = 4096  # Set sliding window size
  ```

- **Attention Implementation**:
  ```python
  super().__init__(
      config=config,
      hidden_size=config.hidden_size,
      num_attention_heads=config.num_attention_heads,
      num_key_value_heads=config.num_key_value_heads,
      head_dim=config.hidden_size // config.num_attention_heads,
      rotary_emb=rotary_emb,
      sliding_window=getattr(config, "sliding_window", None)
  )
  ```

### 3. Different Normalization Types

- **RMSNorm Implementation**:

  ```python
  def get_rmsnorm_cls():
      # Initialize to the appropriate implementation of RMSNorm
      # If infer on NXD -> CustomRMSNorm
      # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
      return MistralRMSNorm if cpu_mode() else CustomRMSNorm

  # Usage
  self.input_layernorm = get_rmsnorm_cls()(
      config.hidden_size,
      eps=config.rms_norm_eps,
  )
  ```

### 4. Different Position Embedding Types

- **Rotary Position Embeddings**:

  ```python
  rotary_emb = RotaryEmbedding(
      config.hidden_size // config.num_attention_heads,
      max_position_embeddings=config.max_position_embeddings,
      base=config.rope_theta,
  )
  ```

- **Absolute Position Embeddings**:
  ```python
  self.position_embeddings = nn.Embedding(
      config.max_position_embeddings,
      config.hidden_size,
  )
  ```

## Validation with Existing Implementations

This implementation guide has been validated against existing model implementations in the NeuronX Distributed Inference framework:

### Mistral Implementation

- **Module Structure**: ✅ Follows proper directory structure with `models/mistral/`
- **Configuration Class**: ✅ Properly implements `MistralInferenceConfig` with required attributes
- **Attention Implementation**: ✅ Uses `NeuronAttentionBase` with rotary embeddings and sliding window
- **Weight Conversion**: ✅ Properly converts weights with tensor parallelism handling
- **MLP Implementation**: ✅ Uses SwiGLU activation with separate gate and up projections
- **Model Compilation**: ✅ Uses minimal stable settings for initial compilation

### Qwen2 Implementation

- **Module Structure**: ✅ Follows proper directory structure with `models/qwen2/`
- **Configuration Class**: ✅ Properly implements `Qwen2InferenceConfig` with model-specific parameters
- **Attention Implementation**: ✅ Uses `NeuronAttentionBase` with bias terms
- **Weight Conversion**: ✅ Properly converts weights with tensor parallelism handling
- **Compiler Arguments**: ✅ Uses model-specific optimizations for compilation

### DBRX (Mixture of Experts) Implementation

- **Module Structure**: ✅ Follows proper directory structure with `models/dbrx/`
- **Configuration Class**: ✅ Properly implements `DbrxInferenceConfig` with model-specific parameter names
- **Weight Conversion**: ✅ Handles Mixture of Experts architecture with router weights and expert weights
- **MLP Implementation**: ✅ Implements MoE with router network and expert MLPs

## Conclusion

This implementation guide provides a comprehensive approach to implementing language models for the NeuronX Distributed Inference framework. By following this guide, you can successfully implement a wide range of model architectures, including standard decoder-only models, models with sliding window attention, and Mixture of Experts models.

Start with minimal settings and gradually add optimizations after basic functionality is verified. Use the debugging tools provided to diagnose and fix issues that may arise during implementation.
