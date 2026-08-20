# NeuronSDK Model Architectures

This document provides an architectural overview of the models supported in the NeuronxDistributed and NeuronxDistributedInference frameworks. These frameworks implement efficient distributed training and inference of large language models on AWS Neuron hardware.

## Common Architecture Components

All models in the NeuronxDistributed framework share several key architectural components:

### Parallelization Strategies

- **Tensor Parallelism (TP)**: Splits model parameters across multiple devices
- **Sequence Parallelism (SP)**: Distributes sequence processing across devices
- **Context Parallelism (CP)**: Optimizes attention computation for long sequences
- **Data Parallelism (DP)**: Processes different batches on different devices

### Attention Mechanisms

- **Multi-head Attention**: All models use some form of multi-head attention
- **Group Query Attention (GQA)**: Optimizes KV cache usage by sharing key-value heads
- **Rotary Position Embeddings (RoPE)**: Used for positional encoding in most models

### Normalization

- **RMSNorm**: Most models use RMSNorm instead of LayerNorm for better performance
- **CustomRMSNorm**: Optimized implementation for Neuron hardware

### Optimization Techniques

- **Flash Attention**: Optimized attention computation
- **KV Cache Management**: Efficient handling of key-value pairs for autoregressive generation
- **Sharded Kernels**: Distributed computation across devices

## Model-Specific Architectures

### 1. Mistral

Mistral is a decoder-only transformer model with the following architecture:

- **Attention**: Uses Grouped-Query Attention (GQA) where multiple query heads share the same key-value heads
- **Normalization**: RMSNorm for input and post-attention normalization
- **Position Encoding**: Rotary Position Embeddings (RoPE)
- **MLP**: Uses SwiGLU activation function (similar to LLaMA)
- **Special Features**:
  - Sliding window attention mechanism to limit context for efficiency
  - Optimized for inference with efficient KV cache handling

### 2. Mixtral (MoE)

Mixtral is a Mixture-of-Experts (MoE) model based on the Mistral architecture:

- **Attention**: Same GQA mechanism as Mistral
- **MLP Layer**: Replaced with a Mixture-of-Experts layer
- **MoE Architecture**:
  - 8 experts per layer
  - Top-k routing (k=2) - each token is processed by the 2 most relevant experts
  - Router network determines which experts to use for each token
  - Expert outputs are weighted and combined
- **Parallelization**:
  - Expert parallelism for distributing experts across devices
  - Token shuffling for load balancing

### 3. Qwen3

Qwen3 is a decoder-only transformer model with:

- **Attention**: Uses GQA with Q-K normalization
- **Normalization**: RMSNorm for all normalization layers
- **Position Encoding**: Rotary Position Embeddings (RoPE)
- **MLP**: Uses SwiGLU activation (similar to LLaMA)
- **Special Features**:
  - Q-K normalization: Applies RMSNorm to query and key vectors after projection
  - Optimized for long context lengths

### 4. Qwen3-MoE

Qwen3-MoE is the Mixture-of-Experts variant of Qwen3:

- **Attention**: Same as Qwen3 with Q-K normalization
- **MLP Layer**: Replaced with a Mixture-of-Experts layer
- **MoE Architecture**:
  - Multiple experts per layer
  - Top-k routing with normalized probabilities
  - GLU-based MLP experts
- **Special Features**:
  - Optimized expert routing and load balancing
  - Efficient expert weight sharing

### 5. DBRX

DBRX is a transformer-based model with:

- Likely based on a decoder-only architecture
- Specialized for distributed inference on Neuron hardware

### 6. DeepSeek

DeepSeek model support includes:

- Transformer-based architecture
- Optimized for Neuron hardware

### 7. T5

T5 is an encoder-decoder model, different from the decoder-only models above:

- **Architecture**: Encoder-decoder transformer
- **Special Features**: Supports text-to-text tasks

### 8. CLIP

CLIP is a multi-modal model:

- **Architecture**: Dual encoder (vision and text)
- **Special Features**: Supports image-text tasks

## Key Architectural Differences

### Attention Mechanisms

1. **Standard Multi-head Attention**: Used in basic transformer models
2. **Grouped-Query Attention (GQA)**: Used in Mistral, Mixtral, Qwen3 - reduces KV cache size
3. **Q-K Normalization**: Used in Qwen3 and Qwen3-MoE - applies normalization to query and key vectors

### MLP Architectures

1. **Standard MLP**: Used in basic transformer models
2. **SwiGLU Activation**: Used in LLaMA-style models including Mistral
3. **Mixture-of-Experts**: Used in Mixtral and Qwen3-MoE - replaces standard MLP with router and multiple expert networks

### Parallelization Strategies

1. **Tensor Parallelism**: All models support sharding parameters across devices
2. **Sequence Parallelism**: Optimizes memory usage for long sequences
3. **Expert Parallelism**: Specific to MoE models for distributing experts
4. **Context Parallelism**: Optimizes attention computation for very long contexts

## Inference Optimizations

1. **Flash Attention**: Optimized attention computation for faster inference
2. **KV Cache Management**: Efficient handling of key-value pairs for autoregressive generation
3. **On-device Sampling**: Optimized token generation directly on device
4. **Chunked Prefill**: Processing long contexts in chunks for memory efficiency
5. **Sliding Window Attention**: Limiting context window for efficiency in models like Mistral

## Mixture-of-Experts (MoE) Implementation

The MoE architecture in NeuronxDistributed consists of:

1. **Router**: Determines which experts to use for each token
   - Uses a learned weight matrix to compute routing probabilities
   - Implements top-k routing to select the most relevant experts

2. **Expert MLPs**: Multiple feed-forward networks that process tokens
   - Each expert specializes in different aspects of the language
   - Typically uses GLU activation functions

3. **Token Routing**: Process of directing tokens to experts
   - Computes affinity scores between tokens and experts
   - Selects top-k experts for each token
   - Combines expert outputs weighted by routing probabilities

4. **Load Balancing**: Ensures even distribution of tokens across experts
   - Token shuffling to improve load distribution
   - Optional capacity factors to control expert utilization

5. **Parallelization**: Strategies for distributing MoE computation
   - Expert parallelism: distributing experts across devices
   - Tensor parallelism: sharding expert parameters

## Conclusion

The NeuronxDistributed framework provides a comprehensive implementation of modern transformer architectures optimized for AWS Neuron hardware. It supports both standard transformer models (Mistral, Qwen3) and Mixture-of-Experts models (Mixtral, Qwen3-MoE), with various parallelization strategies for efficient distributed training and inference.

The architecture emphasizes flexibility, allowing different parallelization strategies to be combined based on the specific requirements of the model and hardware configuration. The implementation includes specialized kernels and optimizations for the Neuron hardware, enabling efficient inference of large language models.
