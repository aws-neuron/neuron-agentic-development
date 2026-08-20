# Existing Model Architectures in NeuronSDK

This document provides a comprehensive architectural analysis of all models supported in the NeuronxDistributed and NeuronxDistributedInference frameworks. These frameworks implement efficient distributed training and inference of large language models on AWS Neuron hardware.

## Table of Contents

1. [Common Architecture Components](#common-architecture-components)
2. [Decoder-Only Language Models](#decoder-only-language-models)
3. [Mixture-of-Experts Models](#mixture-of-experts-models)
4. [Multimodal Models](#multimodal-models)
5. [Encoder-Decoder Models](#encoder-decoder-models)
6. [Vision Models](#vision-models)
7. [Architectural Relationships](#architectural-relationships)
8. [Parallelization Strategies](#parallelization-strategies)
9. [Optimization Techniques](#optimization-techniques)

## Common Architecture Components

All models in the NeuronxDistributed framework share several foundational architectural components that enable efficient distributed computation on Neuron hardware.

### Base Model Structure

Every model inherits from `NeuronBaseModel` which provides:

- **Initialization**: Model setup with configuration validation
- **Parallelization**: Tensor, sequence, and pipeline parallelism support
- **Optimization**: KV cache management, sampling, and memory optimization
- **Tracing**: XLA compilation and graph optimization

### Attention Mechanisms

All attention implementations inherit from `NeuronAttentionBase` which provides:

- **Parallel Linear Layers**: QKV projections using `ColumnParallelLinear`, output projection using `RowParallelLinear`
- **Flash Attention**: Optimized attention computation with multiple strategies
- **KV Cache Management**: Efficient key-value pair storage and retrieval
- **Rotary Position Embeddings**: RoPE implementation for positional encoding

### Normalization

- **RMSNorm**: Root Mean Square normalization used across most models
- **CustomRMSNorm**: Neuron-optimized implementation for distributed inference
- **LayerNorm**: Standard layer normalization for specific models (T5, DBRX, CLIP)

### Parallelization Infrastructure

- **Tensor Parallelism (TP)**: Splits model parameters across devices
- **Sequence Parallelism (SP)**: Distributes sequence processing
- **Context Parallelism (CP)**: Optimizes long context attention
- **Expert Parallelism (EP)**: Distributes MoE experts across devices
- **Data Parallelism (DP)**: Processes different batches on different devices

## Decoder-Only Language Models

### 1. LLaMA Architecture

**Base Implementation**: `NeuronLlamaModel` in `modeling_llama.py`

**Core Components**:

- **Embedding**: `ParallelEmbedding` with vocabulary sharding
- **Decoder Layers**: Stack of `NeuronLlamaDecoderLayer`
- **Attention**: `NeuronLlamaAttention` with GQA support
- **MLP**: `NeuronLlamaMLP` with SwiGLU activation
- **Normalization**: RMSNorm for input and post-attention
- **Output**: `ColumnParallelLinear` language modeling head

**Key Features**:

- **Grouped Query Attention (GQA)**: Reduces KV cache memory usage
- **RoPE**: Rotary position embeddings for positional encoding
- **SwiGLU Activation**: Gated linear unit in MLP layers
- **Fused QKV**: Optional fusion of query, key, value projections
- **Quantization Support**: FP8 and INT8 quantization with custom kernels

**Architectural Details**:

```python
# Attention mechanism
class NeuronLlamaAttention(NeuronAttentionBase):
    - Multi-head attention with GQA optimization
    - RoPE for positional encoding
    - Flash attention for efficient computation
    - KV cache for autoregressive generation

# MLP structure
class NeuronLlamaMLP:
    - gate_proj: Linear(hidden_size, intermediate_size)
    - up_proj: Linear(hidden_size, intermediate_size)
    - down_proj: Linear(intermediate_size, hidden_size)
    - activation: SwiGLU (gate_proj(x) * silu(up_proj(x)))
```

### 2. Mistral Architecture

**Base Implementation**: `NeuronMistralModel` in `modeling_mistral.py`

**Inheritance**: Extends LLaMA architecture with specific modifications

**Key Differences from LLaMA**:

- **Sliding Window Attention**: Limits attention to recent tokens for efficiency
- **Configuration**: `MistralInferenceConfig` with sliding window parameters
- **Attention**: `NeuronMistralAttention` with sliding window support

**Architectural Details**:

```python
class NeuronMistralAttention(NeuronAttentionBase):
    - Inherits GQA from base attention
    - Adds sliding_window parameter
    - Optimized for inference with limited context
```

### 3. Qwen2 Architecture

**Base Implementation**: `NeuronQwen2Model` in `modeling_qwen2.py`

**Key Features**:

- **QKV Bias**: Supports bias in QKV projections (configurable)
- **Output Bias**: Configurable bias in output projection
- **RoPE**: Standard rotary position embeddings
- **MLP Reuse**: Reuses `NeuronLlamaMLP` implementation

**Configuration Differences**:

```python
class Qwen2InferenceConfig(InferenceConfig):
    - qkv_bias: True (default)
    - o_bias: False (default)
    - Reuses LLaMA MLP architecture
```

### 4. Qwen3 Architecture

**Base Implementation**: `NeuronQwen3Model` in `modeling_qwen3.py`

**Key Features**:

- **Q-K Normalization**: Applies RMSNorm to query and key vectors
- **Enhanced RoPE**: Improved rotary position embeddings
- **Long Context**: Optimized for extended sequence lengths

**Architectural Innovation**:

```python
class NeuronQwen3Attention:
    - Standard attention computation
    - Post-projection Q-K normalization
    - Enhanced stability for long sequences
```

### 5. DeepSeek Architecture

**Base Implementation**: `NeuronDeepSeekModel` in `modeling_deepseek.py`

**Key Features**:

- **Custom RoPE**: Specialized rope utilities in `rope_util.py`
- **Optimized Kernels**: Neuron-specific optimizations
- **Extended Context**: Support for very long sequences

## Mixture-of-Experts Models

### 1. Mixtral (MoE) Architecture

**Base Implementation**: `NeuronMixtralModel` in `modeling_mixtral.py`

**MoE Structure**:

- **Base Architecture**: Built on Mistral foundation
- **Expert Count**: 8 experts per layer
- **Top-K Routing**: k=2 (each token routed to 2 experts)
- **Router**: `RouterTopK` for expert selection
- **Expert MLPs**: `ExpertMLPs` with GLU activation

**Key Components**:

```python
class NeuronMixtralDecoderLayer:
    - self_attn: NeuronMixtralAttention (same as Mistral)
    - mlp: MoE module with 8 experts
    - Router network for expert selection
    - Expert weight combination

# MoE Module Structure
MoE(
    router=RouterTopK(num_experts=8, top_k=2),
    expert_mlps=ExpertMLPs(8 experts with GLU),
    load_balancing=True
)
```

**State Dict Conversion**:

- Converts HuggingFace checkpoint format to Neuron MoE format
- Concatenates gate_proj and up_proj weights
- Reshapes expert weights for efficient computation

### 2. Qwen3-MoE Architecture

**Base Implementation**: `NeuronQwen3MoeModel` in `modeling_qwen3_moe.py`

**Enhanced MoE Features**:

- **Q-K Normalization**: Inherits from Qwen3 with expert routing
- **Configurable Experts**: Variable number of experts per layer
- **Normalized Routing**: Probability normalization for expert selection
- **Advanced Load Balancing**: Sophisticated token distribution

**Key Innovations**:

```python
class NeuronQwen3MoEAttention:
    - Q-K normalization with RMSNorm
    - Custom q_layernorm and k_layernorm
    - Enhanced stability in MoE context

# MoE Configuration
- num_experts: Configurable (typically 8-64)
- num_experts_per_tok: Top-k routing
- norm_topk_prob: Normalized routing probabilities
```

### 3. DBRX (MoE) Architecture

**Base Implementation**: `NeuronDbrxModel` in `modeling_dbrx.py`

**Unique Features**:

- **Fused QKV**: Built-in QKV fusion for efficiency
- **LayerNorm**: Uses standard LayerNorm instead of RMSNorm
- **Custom Expert Layout**: Specialized expert organization
- **Clip QKV**: Optional QKV value clipping

**Architectural Details**:

```python
class NeuronDbrxBlock:
    - self_attn: NeuronDbrxAttention with fused QKV
    - ffn: MoE module with custom expert layout
    - LayerNorm for normalization layers

# Configuration
- fused_qkv: True (built-in)
- clip_qkv: Optional value clipping
- Custom state dict conversion from DBRX format
```

## Multimodal Models

### 1. LLaMA Multimodal (MLLaMA) Architecture

**Base Implementation**: `NeuronMllamaModel` in `modeling_mllama.py`

**Multimodal Components**:

- **Text Model**: `NeuronMllamaTextModel` (LLaMA-based)
- **Vision Model**: `NeuronMllamaVisionModel` (separate implementation)
- **Cross-Attention**: Vision-text interaction layers
- **Image Processing**: Tile-based image encoding

**Key Features**:

```python
class NeuronMllamaModel:
    - text_model: Language model component
    - vision_model: Vision encoder component
    - cross_attention_layers: Vision-text fusion
    - multimodal_kv_cache: Specialized cache management

# Configuration
class MllamaInferenceConfig:
    - text_config: Text model configuration
    - vision_config: Vision model configuration
    - cross_attention_layers: Layer indices for cross-attention
```

**Vision Architecture**:

- **Patch Embedding**: Image to patch conversion
- **Transformer Layers**: Vision transformer blocks
- **Global Layers**: Cross-attention with text
- **Tile Processing**: Multi-resolution image handling

### 2. Pixtral Architecture

**Base Implementation**: `NeuronPixtralTextModel` in `modeling_pixtral.py`

**Inheritance**: Extends `NeuronLlamaModel` with vision capabilities

**Key Components**:

- **Text Model**: Inherits from LLaMA
- **Vision Model**: `NeuronPixtralVisionModel`
- **Vision Wrapper**: `PixtralVisionModelWrapper`
- **Image Processing**: Custom image transformation pipeline

### 3. LLaMA 4 (Multimodal) Architecture

**Base Implementation**: `NeuronLlama4TextModel` in `modeling_llama4_text.py`

**Advanced Multimodal Features**:

- **Vision Integration**: `modeling_llama4_vision.py`
- **Enhanced Cross-Attention**: Improved vision-text fusion
- **Flexible Input**: Support for various input modalities

## Encoder-Decoder Models

### 1. T5 Architecture

**Base Implementation**: `NeuronT5EncoderModel` in `modeling_t5.py`

**Encoder-Decoder Structure**:

- **Encoder**: Stack of T5 encoder layers
- **Decoder**: Separate decoder implementation (if needed)
- **Attention**: Bidirectional for encoder, causal for decoder
- **Relative Position**: T5-style relative position embeddings

**Key Components**:

```python
class NeuronT5EncoderModel:
    - embed_tokens: Shared embedding layer
    - encoder_layers: Stack of T5 encoder blocks
    - final_layer_norm: T5LayerNorm
    - Relative position bias handling

# T5 Layer Structure
class NeuronT5LayerFF:
    - DenseReluDense: MLP with configurable activation
    - Gated activation support (T5v1.1+)
    - LayerNorm for normalization
```

**Attention Mechanism**:

- **Relative Position Bias**: T5-style position encoding
- **Bidirectional**: Full attention in encoder
- **Cross-Attention**: Encoder-decoder attention (when applicable)

## Vision Models

### 1. CLIP Architecture

**Base Implementation**: `NeuronCLIPTextModel` in `modeling_clip.py`

**Dual Encoder Structure**:

- **Text Encoder**: Transformer-based text processing
- **Vision Encoder**: Vision transformer (separate)
- **Contrastive Learning**: Image-text alignment

**Text Encoder Details**:

```python
class NeuronCLIPTextModel:
    - embeddings: Token and position embeddings
    - encoder_layers: Stack of CLIP encoder layers
    - final_layer_norm: LayerNorm
    - Causal attention masking

class NeuronCLIPAttention:
    - Multi-head attention with parallel projections
    - Causal masking for text generation
    - Standard softmax attention
```

### 2. Flux (Diffusion) Architecture

**Base Implementation**: `NeuronFluxTransformer2DModel` in `modeling_flux.py`

**Diffusion Transformer**:

- **2D Transformer**: Specialized for image generation
- **Attention Blocks**: Modified attention for diffusion
- **Conditioning**: Text and image conditioning support

### 3. VAE Decoder Architecture

**Base Implementation**: `ModelWrapperVAEDecoder` in `modeling_vae.py`

**Variational Autoencoder**:

- **Decoder Network**: Upsampling and convolution layers
- **Latent Processing**: Latent space to image conversion
- **Integration**: Works with diffusion models

## Architectural Relationships

### Inheritance Hierarchy

```
NeuronBaseModel
├── NeuronLlamaModel
│   ├── NeuronMistralModel
│   ├── NeuronQwen2Model
│   ├── NeuronQwen3Model
│   ├── NeuronDeepSeekModel
│   └── NeuronPixtralTextModel
├── NeuronMixtralModel (MoE)
├── NeuronQwen3MoeModel (MoE)
├── NeuronDbrxModel (MoE)
├── NeuronMllamaModel (Multimodal)
├── NeuronT5EncoderModel (Encoder-Decoder)
└── NeuronCLIPTextModel (Vision-Language)
```

### Attention Inheritance

```
NeuronAttentionBase
├── NeuronLlamaAttention
│   ├── NeuronMistralAttention (+ sliding window)
│   ├── NeuronQwen2Attention (+ bias options)
│   └── NeuronQwen3Attention (+ Q-K norm)
├── NeuronMixtralAttention (MoE)
├── NeuronQwen3MoEAttention (MoE + Q-K norm)
├── NeuronDbrxAttention (MoE + fused QKV)
├── NeuronT5Attention (relative position)
└── NeuronCLIPAttention (causal masking)
```

### MLP Architecture Patterns

1. **Standard MLP** (T5, CLIP):

   ```python
   Linear(hidden_size, intermediate_size) -> Activation -> Linear(intermediate_size, hidden_size)
   ```

2. **GLU MLP** (LLaMA family):

   ```python
   gate_proj = Linear(hidden_size, intermediate_size)
   up_proj = Linear(hidden_size, intermediate_size)
   down_proj = Linear(intermediate_size, hidden_size)
   output = down_proj(silu(gate_proj(x)) * up_proj(x))
   ```

3. **MoE MLP** (Mixtral, Qwen3-MoE, DBRX):
   ```python
   router = RouterTopK(num_experts, top_k)
   experts = [GLU_MLP for _ in range(num_experts)]
   output = combine_expert_outputs(route_tokens(x))
   ```

## Parallelization Strategies

### Tensor Parallelism Patterns

1. **QKV Projection Sharding**:
   - `ColumnParallelLinear` for Q, K, V projections
   - Sharding across attention heads
   - Efficient all-gather for attention computation

2. **Output Projection Sharding**:
   - `RowParallelLinear` for output projection
   - Reduce-scatter for gradient computation
   - Memory-efficient communication

3. **MLP Sharding**:
   - Column-parallel for input projections
   - Row-parallel for output projections
   - Expert-parallel for MoE models

### Sequence Parallelism

- **Activation Sharding**: Distribute activations across sequence dimension
- **Communication Optimization**: Reduce memory usage for long sequences
- **Gradient Synchronization**: Efficient backward pass communication

### Context Parallelism

- **Long Context Support**: Handle sequences beyond single device memory
- **Attention Sharding**: Distribute attention computation
- **KV Cache Distribution**: Efficient cache management across devices

### Expert Parallelism (MoE)

- **Expert Distribution**: Assign experts to different devices
- **Token Routing**: Efficient token-to-expert assignment
- **Load Balancing**: Ensure even expert utilization
- **Communication**: Minimize inter-device token movement

## Optimization Techniques

### Memory Optimizations

1. **KV Cache Management**:
   - **Standard Cache**: `KVCacheManager` for basic caching
   - **Block Cache**: `BlockKVCacheManager` for memory efficiency
   - **Multimodal Cache**: `MultimodalKVCacheManager` for vision-language models
   - **Data Parallel Cache**: `DataParallelKVCacheManager` for distributed inference

2. **Gradient Checkpointing**:
   - Selective activation recomputation
   - Memory-compute trade-off optimization
   - Layer-wise checkpointing strategies

3. **Quantization Support**:
   - **FP8 Quantization**: Reduced precision computation
   - **INT8 Quantization**: Integer-based computation
   - **Custom Kernels**: Optimized quantized operations

### Computation Optimizations

1. **Flash Attention Strategies**:
   - `UNSHARDED_KERNEL`: Single device attention
   - `SHARDED_KERNEL`: Multi-device attention
   - `CONTEXT_PARALLEL_KERNEL`: Long context optimization
   - `SLIDING_WINDOW_KERNEL`: Limited context attention

2. **Fused Operations**:
   - **Fused QKV**: Combined query, key, value computation
   - **Fused MLP**: Combined gate and up projections
   - **Custom Kernels**: NKI-optimized operations

3. **Sampling Optimizations**:
   - **On-Device Sampling**: Avoid host-device transfers
   - **Batched Sampling**: Efficient multi-request handling
   - **Speculative Decoding**: Accelerated generation

### Hardware-Specific Optimizations

1. **Neuron Compiler Integration**:
   - **XLA Compilation**: Graph-level optimization
   - **NKI Kernels**: Custom kernel implementations
   - **Memory Layout**: Optimized tensor layouts

2. **Communication Optimization**:
   - **All-Reduce**: Efficient gradient synchronization
   - **All-Gather**: Efficient activation gathering
   - **Reduce-Scatter**: Memory-efficient reduction

3. **Pipeline Optimization**:
   - **Async Execution**: Overlapped computation and communication
   - **Bucketing**: Dynamic batching for efficiency
   - **Prefetching**: Optimized data loading

## Model-Specific Optimizations

### LLaMA Family Optimizations

- **Quantized MLP Kernels**: Custom INT8/FP8 MLP computation
- **RMSNorm Kernels**: Optimized normalization
- **RoPE Kernels**: Efficient rotary embedding computation

### MoE Model Optimizations

- **Expert Routing**: Optimized token-to-expert assignment
- **Load Balancing**: Dynamic expert utilization
- **Communication**: Minimized inter-device transfers

### Multimodal Optimizations

- **Vision Processing**: Optimized image encoding
- **Cross-Attention**: Efficient vision-text fusion
- **Memory Management**: Specialized cache for multimodal data

### Long Context Optimizations

- **Context Parallelism**: Distributed long sequence processing
- **Sliding Window**: Limited attention for efficiency
- **Chunked Processing**: Memory-efficient long context handling

This comprehensive architectural overview demonstrates the sophisticated design patterns and optimizations implemented in the NeuronSDK for efficient large language model inference on AWS Neuron hardware. Each model builds upon common foundations while implementing specific optimizations for their unique architectural requirements.
