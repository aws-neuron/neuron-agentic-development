# Text-to-Video Diffusion Model Porting Guide for AWS Neuron (v3.0)

A complete, self-contained guide for porting text-to-video diffusion models to AWS Neuron hardware (Trainium/Inferentia).

**Scope:** Any text-to-video diffusion pipeline — HunyuanVideo, WAN 2.1, CogVideoX, Mochi, etc. The guide is model-agnostic. Specific models are used only as examples.

---

## Table of Contents

1. [How to Use This Guide](#1-how-to-use-this-guide)
2. [Anatomy of Text-to-Video Pipelines](#2-anatomy)
3. [Phase 1: Discovery — Extract the Component Graph and All Constants](#3-discovery)
4. [Phase 2: Component Implementation](#4-components)
5. [Phase 3: Pipeline Assembly](#5-pipeline-assembly)
6. [Neuron-Specific Patterns](#6-neuron-patterns)
7. [XLA Compatibility Reference](#7-xla-reference)
8. [Validation](#8-validation)
9. [Pitfalls — Silent Failures That Produce Wrong Output](#9-pitfalls)
10. [Completion Checklist](#10-checklist)
11. [Field-Tested Debugging Lessons (v2.0)](#11-field-tested-lessons)
12. [Field-Tested Lessons from Causal-VAE T2V Port (v3.0)](#12-causal-vae-lessons)

---

## 1. How to Use This Guide

This guide has three phases. **Execute them in order. Do not skip ahead.**

| Phase               | What                                                                                      | Why                                                                                                          |
| ------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Discovery** (§3)  | Run the original pipeline on CPU. Map every component, extract every constant and shape.  | Constants you don't extract here will be guessed wrong later.                                                |
| **Components** (§4) | Implement and validate each component independently on Neuron.                            | Each component must pass cosine similarity >0.998 vs CPU before moving on.                                   |
| **Assembly** (§5)   | Wire components into the final pipeline, matching the original's execution order exactly. | The assembly must reproduce which components run once vs per-step, what inputs they take, and in what order. |

**The most common failure mode is skipping Discovery.** An agent that jumps to implementation will guess constants (crop offsets, sequence lengths, conditioning timestep values) and get them wrong. These errors produce plausible but incorrect output with no error messages.

**The second most common failure mode is "incremental shortcuts."** An agent that runs text encoding on CPU "temporarily" to unblock backbone testing will never go back and compile it for Neuron. The completion checklist (§10) prevents this.

### Mindset Rules

These rules override everything else in this document.

**R1: No shortcuts within a component.** You MAY divide the pipeline into components (e.g., T5, backbone, VAE) and port each independently — this is the correct divide-and-conquer approach. You MUST NOT compile a partial component. If the backbone has 54 layers, compile all 54. Partial components hide HBM limits and produce misleading validation results.

**R2: The image is your only ground truth.** You can read images. Use this ability. After every end-to-end pipeline run, read the output image and verify it matches the prompt. Numerical metrics (cosine, std) are necessary but NOT sufficient. A frame with std=108 can be random colorful noise.

**R3: Never guess. Never trial-and-error.** Every constant, shape, and key mapping must come from reading source code or running the original pipeline. Guessing produces plausible but wrong output.

**R4: Be skeptical of your own work.** When debugging, assume YOUR code is wrong, not the framework. In both ports that informed this guide, every single bug was in the porting code, never in NxDI or the Neuron compiler.

**R5: Compare the same model against itself.** When validating, compare your model on Neuron vs your model on CPU — not against the HF model. Two different implementations compound tiny numerical differences over many layers, producing misleading cosine values (e.g., 0.60) that look like bugs but aren't.

**R6: Evaluate all options before coding.** When hitting a constraint, spend 10 minutes listing ALL possible solutions with tradeoffs before writing any code. This saves hours of debugging dead ends.

---

## 2. Anatomy of Text-to-Video Pipelines

Every text-to-video diffusion pipeline has the same high-level structure, but the specific components vary:

```
Text Prompt
    │
    ├─→ [Text Encoders]          ──→ text embeddings       (called ONCE)
    ├─→ [Conditioning Modules]   ──→ refined/projected      (called ONCE)
    │                                 embeddings
    │   ┌──────────────────────────────────────────────────┐
    └─→ │ Denoising Loop (N steps × K for CFG)             │
        │   [Denoising Backbone]                           │  (called N×K)
        └──────────────────────────────────────────────────┘
            │
            └─→ [VAE Decoder]    ──→ Video frames           (called ONCE)
```

**What varies across models:**

| Aspect                  | Examples                                                                        |
| ----------------------- | ------------------------------------------------------------------------------- |
| Number of text encoders | 1 (e.g., T5-XXL), 2 (e.g., multimodal LLM + glyph encoder), 2 (e.g., T5 + CLIP) |
| Conditioning modules    | Token refiner, none, pooled projection — varies by model                        |
| Token preprocessing     | Reorder + zero-mask, simple concat, chat-template encoding — varies by model    |
| Backbone architecture   | Dual-stream MMDiT, single-stream DiT, cross-attention DiT                       |
| Attention type          | Joint self-attention (MMDiT), cross-attention (DiT), both                       |
| CFG multiplier K        | 2 (standard), 1 (no CFG / distilled)                                            |
| VAE type                | Non-causal 3D Conv VAE, causal 3D Conv VAE                                      |

**What is always the same:**

- Text encoders run once, backbone runs N×K times, VAE runs once
- The backbone is the optimization target (dominates runtime)
- The denoising loop uses a scheduler on CPU
- Latent dimensions are determined by VAE config

**The guide does NOT assume any specific component exists.** Phase 1 (Discovery) determines what YOUR pipeline has. Phase 2 gives implementation patterns for each component TYPE. Phase 3 wires them together.

### Compilation Framework Selection

The framework depends on model size:

| Model Size              | Framework                            | TP       | When to use                                          |
| ----------------------- | ------------------------------------ | -------- | ---------------------------------------------------- |
| >2B params              | NxDI `NeuronApplicationBase`         | TP≥2     | Backbone, large text encoders                        |
| 200M-2B params          | Either NxDI or `torch_neuronx.trace` | Optional | Medium text encoders, large conditioning modules     |
| <200M params            | `torch_neuronx.trace`                | No       | Small encoders, conditioning modules, utility models |
| Conv3d-heavy (any size) | `torch_neuronx.trace` × N shards     | No       | VAE decoders (Conv3d exceeds instruction limits)     |
| <1ms compute            | CPU                                  | N/A      | Scheduler, simple pre/post-processing                |

### HBM Budget Rule

**Before choosing single-core vs TP, do this math:**

```
neff_size ≈ (params × 2 bytes) + graph_overhead(0.5-2GB) + scratchpad(1GB)
```

On trn2 with LNC=2: **24GB per logical core**. If `neff_size > 22GB`, you MUST use TP≥2. Do not attempt single-core compilation — it will compile successfully but fail at load time with `Allocation Failure`, wasting the compilation time.

| Params (bf16) | Estimated NEFF | Fits single core?          |
| ------------- | -------------- | -------------------------- |
| <5B           | <12GB          | ✅ Yes                     |
| 5-10B         | 12-22GB        | ⚠️ Maybe — check carefully |
| >10B          | >22GB          | ❌ No — use TP≥2           |

---

## 3. Phase 1: Discovery — Extract the Component Graph and All Constants

**Every item below must be completed before you write a single line of porting code.** Record all values in a constants file.

### 3.0 Know Your Hardware

Before anything else, understand the target hardware constraints. These determine every decision you make — TP degree, CP degree, which components fit on one core, and whether the port is feasible at all.

```bash
# What instance type?
curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "Check instance type manually"

# How many NeuronCores?
neuron-ls 2>/dev/null | grep -c "NeuronCore" || ls /dev/neuron* 2>/dev/null | wc -l

# Neuron SDK version
pip show torch-neuronx 2>/dev/null | grep Version
neuronx-cc --version 2>/dev/null
```

Record these values:

| Constant             | How to find it                                                         | Example (trn2.48xlarge)       |
| -------------------- | ---------------------------------------------------------------------- | ----------------------------- |
| `INSTANCE_TYPE`      | Instance metadata or `ec2 describe-instances`                          | trn2.48xlarge                 |
| `TOTAL_NEURON_CORES` | `neuron-ls`                                                            | 64 (32 physical)              |
| `HBM_PER_CORE`       | Instance spec (24GB for trn2, 16GB for trn1)                           | 24GB                          |
| `LNC_DEFAULT`        | `NEURON_RT_VIRTUAL_CORE_SIZE` env var (2 = LNC2, 1 = LNC1)             | 2 (LNC2)                      |
| `EFFECTIVE_HBM`      | HBM_PER_CORE / LNC_DEFAULT (with LNC2: 24GB shared by 2 logical cores) | 24GB per logical core pair    |
| `INSTRUCTION_LIMIT`  | ~5M per NEFF (hard compiler limit)                                     | 5,000,000                     |
| `SDK_VERSION`        | `pip show torch-neuronx`                                               | Check for known bugs/features |

**Why this matters:**

- A 4.7B-parameter text encoder needs ~9.4GB in bf16. At LNC=2 (24GB per core pair), it fits. At LNC=1 (24GB per single core), it also fits but uses a full physical core.
- A 30-block backbone at seq_len=20,280 generates 37M instructions — 7× over the 5M limit. You MUST know this before attempting compilation.
- A trn2.48xlarge has 64 cores — you can run backbone on cores 0-7 and VAE on cores 8-15 simultaneously. A trn2.3xlarge has only 16 cores — you may need to run components sequentially.
- Different SDK versions have different compiler capabilities and bugs. Check release notes for your version.

### 3.1 Run the Original Pipeline on CPU

Run the complete original pipeline end-to-end on CPU with a test prompt:

```python
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
output = pipe(prompt="a cat walking on a beach at sunset", num_frames=33, ...)
```

Save the output frames — these are your ground truth for end-to-end validation.

### 3.2 Map the Component Graph

Read the pipeline's `__call__` method source code. For every neural network invocation, record:

| Field            | What to record                                                            |
| ---------------- | ------------------------------------------------------------------------- |
| **Name**         | What the component is (e.g., "T5 text encoder", "token refiner")          |
| **Call site**    | Where in `__call__` it's invoked — before loop, inside loop, after loop   |
| **Frequency**    | Once, or per-step                                                         |
| **Inputs**       | Tensor names, shapes, dtypes                                              |
| **Outputs**      | Tensor names, shapes, dtypes                                              |
| **Conditioning** | Does it take a timestep? If so, is it the loop timestep or a fixed value? |
| **Size**         | Parameter count                                                           |
| **Dependencies** | Which other components' outputs does it consume?                          |

**This is the most important step in the entire guide.** The component graph IS the port specification. Every component you find here must be compiled for Neuron. Every constant you extract here prevents a wrong guess later.

**Pay special attention to:**

- Components that take a timestep — is it the denoising loop timestep, or a fixed value (e.g., 1000.0)?
- Components that run ONCE before the loop vs PER-STEP inside the loop
- Any cropping, slicing, or reshaping of intermediate tensors (record exact offsets)

### 3.3 Extract Text Encoder Constants

For each text encoder, trace the tokenization and embedding extraction:

```python
# What is the tokenizer's max_length?
# Does the pipeline crop the output? At what offset? How many tokens?
# What hidden layer is extracted (last? second-to-last? third-from-last)?
```

Record:

| Constant         | How to find it                                                          |
| ---------------- | ----------------------------------------------------------------------- |
| `SEQ_LEN`        | `max_length` in the tokenizer call                                      |
| `CROP_START`     | Offset where pipeline slices hidden states (0 if no crop)               |
| `CROP_LEN`       | Number of tokens extracted                                              |
| `EXTRACT_LAYER`  | Which hidden state layer (e.g., -1, -2, -3)                             |
| `SYSTEM_MESSAGE` | Exact system prompt text, if any (affects tokenization and crop offset) |

**Do NOT guess these values.** Read the pipeline source code. A wrong `CROP_START` silently shifts the token window and degrades output.

### 3.3a 🆕 Text Encoder Preprocessing Traps

These traps were discovered during prior T2V ports. Each one silently produces wrong output.

**Trap 1: System message whitespace.** Extract the exact system message using `repr()`:

```python
import inspect
sig = inspect.signature(Pipeline._get_mllm_prompt_embeds)
system_message = sig.parameters['system_message'].default
print(repr(system_message))  # Shows \n, \t, multiple spaces
```

A system message with `\n        1.` (newline + 8 spaces) vs `1.` (single space) changes tokenization by 5 tokens. This 40-character whitespace difference was the difference between generating a cat and generating a horse.

**Trap 2: Tokenizer class mismatch.** The pipeline may use a fast tokenizer while `AutoTokenizer` loads the slow version. They can produce different token counts. Always verify:

```python
assert my_tokenizer_output['attention_mask'].sum() == pipe_tokenizer_output['attention_mask'].sum()
```

**Trap 3: `apply_chat_template` input format.** `format_text_input()` returns `[[{system}, {user}]]` (list of conversations). The pipeline passes `formatted[0]` (single conversation) or `formatted` (batch). Passing the wrong nesting level changes tokenization silently.

**Trap 4: Decoder models used as encoders.** Some text encoders are decoder-architecture models with `is_decoder=True`. They use **causal** (lower-triangular) attention masking even when used for encoding. Check:

```python
print(model.config.is_decoder)  # True = MUST use causal mask
```

Using bidirectional masking on a causal model produces cosine ~0.60 vs the reference — close enough to look like a numerical issue, far enough to produce wrong images.

**Trap 5: Hidden layer extraction index.** `output_hidden_states=True` returns `[embedding, layer_0, ..., layer_N]` — that's `N+1` entries. `hidden_states[-3]` is layer `N-2`, not layer `N-3`. Verify the index:

```python
out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
print(f"Total hidden states: {len(out.hidden_states)}")  # N+1
print(f"hidden_states[-3] is layer index: {len(out.hidden_states) - 3}")
```

### 3.4 Extract Conditioning Module Constants

For each conditioning module (refiner, projector, etc.):

| Constant         | How to find it                                                            |
| ---------------- | ------------------------------------------------------------------------- |
| `RUNS_ONCE`      | Is it called before the loop (True) or inside the loop (False)?           |
| `FIXED_TIMESTEP` | If it takes a timestep, is it fixed (e.g., 1000.0) or from the scheduler? |
| `INPUT_SHAPES`   | Exact shapes of all inputs                                                |
| `OUTPUT_SHAPES`  | Exact shapes of all outputs                                               |

### 3.5 Extract VAE Constants

```python
import json
with open("path/to/vae/config.json") as f:
    cfg = json.load(f)
```

| Constant          | How to find it                                                   |
| ----------------- | ---------------------------------------------------------------- |
| `SCALING_FACTOR`  | `cfg["scaling_factor"]`                                          |
| `SPATIAL_FACTOR`  | `cfg["spatial_compression_ratio"]` or `cfg["ffactor_spatial"]`   |
| `TEMPORAL_FACTOR` | `cfg["temporal_compression_ratio"]` or `cfg["ffactor_temporal"]` |
| `LATENT_CHANNELS` | `cfg["latent_channels"]`                                         |

### 3.6 Compute Latent Dimensions

```python
latent_h = pixel_height // SPATIAL_FACTOR
latent_w = pixel_width // SPATIAL_FACTOR
latent_f = (num_frames - 1) // TEMPORAL_FACTOR + 1  # check model-specific formula
```

### 3.7 Trace VAE Decoder Shapes on CPU

Run the full VAE decoder on CPU. Print the tensor shape after every block:

```python
x = torch.randn(1, LATENT_CHANNELS, latent_f, latent_h, latent_w)
with torch.no_grad():
    x = decoder.conv_in(x);              print(f"conv_in: {x.shape}")
    x = decoder.mid_block.resnets[0](x); print(f"mid_resnet0: {x.shape}")
    # ... trace EVERY block, EVERY upsample ...
```

**Save these shapes. Use them exactly for compilation. No guessing.**

### 3.8 Estimate HBM Memory Per Core

Each NeuronCore has 24GB HBM (trn2). Estimate:

```
weights_per_core = total_params × 2 bytes / tp_degree
attention_per_core = heads_per_core × seq_len² × 2 bytes
overhead = ~3GB (compiler scratchpad)
total must be < 24GB
```

### 3.9 Inspect Source Weight Keys

```python
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    for key in sorted(f.keys())[:30]:
        print(f"{key}: {f.get_tensor(key).shape}")
```

### 3.10 Write the Constants File

Create a single file with every value discovered above:

```python
# pipeline_constants.py — ALL values from Discovery phase

# Component graph (example — yours will differ)
COMPONENTS = {
    "text_encoder_1": {"framework": "nxdi", "tp": 2, "frequency": "once"},
    "text_encoder_2": {"framework": "trace", "tp": None, "frequency": "once"},
    "conditioning_module": {"framework": "trace", "tp": None, "frequency": "once"},
    "backbone": {"framework": "nxdi", "tp": 2, "frequency": "per_step"},
    "vae_decoder": {"framework": "trace_sharded", "tp": None, "frequency": "once"},
}

# Text encoder constants (example)
TEXT_ENC_1_SEQ_LEN = ...
TEXT_ENC_1_CROP_START = ...
TEXT_ENC_1_CROP_LEN = ...
TEXT_ENC_1_EXTRACT_LAYER = ...

# Conditioning constants (example)
REFINER_FIXED_TIMESTEP = ...  # or None if no refiner
REFINER_RUNS_ONCE = True

# VAE constants
SCALING_FACTOR = ...
SPATIAL_FACTOR = ...
TEMPORAL_FACTOR = ...

# Latent dimensions
T_LAT, H_LAT, W_LAT = ...
LATENT_CHANNELS = ...

# Backbone constants
HIDDEN_SIZE = ...
NUM_HEADS = ...
HEAD_DIM = ...
NUM_LAYERS = ...
```

---

## 4. Phase 2: Component Implementation

For each component in your component graph (§3.2), implement and validate it independently. The patterns below cover the component TYPES you may encounter. **Not all pipelines have all types.** Only implement what your Discovery phase found.

### Type A: Large Model (>2B params) — NxDI with TP

**Applies to:** Large text encoders (e.g., T5-XXL, multimodal LLMs), transformer backbones.

**Pattern:**

1. Define a `nn.Module` subclass with TP layers (see §6)
2. Define an `InferenceConfig` subclass with model constants
3. Define a `ModelWrapper` subclass with `input_generator()` and `get_model_instance()`
4. Define a `NeuronApplicationBase` subclass with `convert_hf_to_neuron_state_dict()`
5. Compile, load, validate

**TP layer replacement:**

- Attention Q/K/V: `ColumnParallelLinear(gather_output=False)`
- Attention O: `RowParallelLinear(input_is_parallel=True)`
- FFN up: `ColumnParallelLinear(gather_output=False)` → activation → FFN down: `RowParallelLinear(input_is_parallel=True)`
- Modulation linear (e.g., AdaLayerNorm): `ColumnParallelLinear(gather_output=True)` (output is chunked element-wise)
- Large vocabulary embedding: `ParallelEmbedding(shard_across_embedding=True)`
- QK norm (if present): `CustomRMSNorm` from NxDI (fused hardware call)

**What goes inside the NEFF vs outside:**

- Inside: everything that runs on every forward call — embeddings, projections, attention, FFN, norms, output projection
- Inside: time/timestep embedding (sinusoidal → linear → activation → linear) — this is cheap and avoids a CPU→device transfer per step
- Inside: RoPE as `register_buffer` (pre-computed in `__init__`, NOT in `forward()`)
- Outside: scheduler step (CPU, trivial)

**RoPE:** Pre-compute in `__init__` and store as `register_buffer`. Do NOT use `torch.arange` or `torch.meshgrid` in `forward()` — they create CPU tensors during XLA tracing. For 3D video RoPE, compute per-axis frequencies and concatenate:

```python
def _init_rope(self, config):
    grid = torch.meshgrid(
        *[torch.arange(s, dtype=torch.float32) for s in spatial_dims],
        indexing="ij",
    )
    # ... compute per-axis frequencies, store as buffer
    self.register_buffer("rope_cos", ...)
    self.register_buffer("rope_sin", ...)
```

**Attention mask:** Use `-1e9` for masked positions, NOT `float('-inf')` (causes NaN in softmax). Pre-compute on CPU if the mask depends on variable-length inputs.

**Weight conversion:** Write a `convert_hf_to_neuron_state_dict` that maps HF weight keys to your Neuron model's keys. Always verify parameter count matches:

```python
orig = sum(v.numel() for v in hf_sd.values() if key_is_relevant(v))
neuron = sum(v.numel() for v in neuron_sd.values())
assert orig == neuron, f"Weight count mismatch: {orig} vs {neuron}"
```

**Compiler args:** `--model-type=transformer -O1 --auto-cast=none`

### Type A-1: 🆕 Decoder Model Used as Encoder (e.g., autoregressive LLM text encoders)

**Applies to:** Decoder-architecture models used as text encoders in diffusion pipelines (e.g., multimodal LLMs, GPT-family models).

**Critical differences from standard encoder implementation:**

1. **Causal attention mask required.** Pre-compute as `register_buffer`:

   ```python
   causal = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN))
   self.register_buffer("causal_mask", causal)
   ```

   In forward: `mask = self.causal_mask[:S, :S] * attention_mask[:, None, None, :]`

2. **M-RoPE (multimodal RoPE).** Some multimodal models use section-based RoPE with `mrope_section` config. For text-only input, all position dimensions are identical, but the cos/sin interleaving must match exactly:

   ```python
   mrope_section_2 = [s * 2 for s in config.rope_scaling['mrope_section']]
   cos_chunks = cos_full.split(mrope_section_2, dim=-1)
   cos = torch.cat([cos_chunks[i][i % 3] for i in range(len(cos_chunks))], dim=-1)
   ```

3. **GQA (Grouped Query Attention).** With `num_kv_heads < num_attention_heads`, use `repeat_interleave` to expand KV heads before attention:

   ```python
   k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
   v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
   ```

4. **Hidden state extraction.** The pipeline extracts a specific intermediate layer, not the final output. Your model must return the correct layer:

   ```python
   self.extract_idx = config.num_hidden_layers + EXTRACT_LAYER  # e.g., 28 + (-3) = 25
   for i, layer in enumerate(self.layers):
       x = layer(x, ...)
       if i == self.extract_idx:
           extracted = x.clone()
   return extracted  # NOT self.norm(x)
   ```

5. **NxDI compile vs inference process model:**
   - Compile: `torchrun --nproc_per_node=<tp_degree> compile.py`
   - Inference: `python3 run.py` (single process — NxDI handles multi-core via `app.load()`)

### Type B: Small Model (<2B params) — torch_neuronx.trace

**Applies to:** Small text encoders (e.g., CLIP, ByT5), token refiners, conditioning modules, utility models.

**Pattern:**

1. Wrap the model in a thin `nn.Module` with explicit `forward` signature
2. Trace with `torch_neuronx.trace`
3. Save with `torch.jit.save`

```python
class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, *args):
        return self.model(*args)

traced = torch_neuronx.trace(wrapper, example_inputs,
    compiler_args="--model-type=transformer -O1 --auto-cast=none")
torch.jit.save(traced, save_path)
```

**Fixed input shapes.** All shapes are baked in at trace time. Use the exact shapes from Discovery.

**`torch.sort` does not work on trn2.** Use `torch.topk` instead. `torch.topk` works on both trn1 and trn2.

### Type C: Conv3d-Heavy Model (VAE Decoder) — Sharded Tracing

**Applies to:** Any VAE decoder with 3D convolutions on video-resolution tensors.

**Why sharded:** A single Conv3d on a full-resolution grid generates millions of instructions, exceeding the compiler's 5-10M instruction limit. The decoder must be split into individual blocks.

**⚠️ Causal vs Non-Causal:** If the VAE uses `CausalConv3d`, see §12.1 for the cache-aware tracing procedure. The pattern below applies to non-causal VAEs. For causal VAEs, each block needs multiple variants for different cache states.

**Pattern:**

1. Use the shapes from Discovery (§3.7) — no guessing
2. Split into individual blocks (conv_in, mid_resnets, mid_attn, up_resnets, up_upsamples, norm_conv_out)
3. Trace each block independently
4. Chain at runtime

**Three common rewrites for Neuron compatibility:**

1. **CausalConv3d: replicate → constant padding.** `F.pad(mode='replicate')` on 5D tensors generates XLA concat ops that exceed SBUF. Use `mode='constant'`. Quality impact: cosine ~0.97 (visually negligible).

2. **High-dimensional rearrange (>6D): decompose to sequential 6D ops.** XLA segfaults on 8D view+permute. Decompose pixel-shuffle patterns into sequential 6D steps. Quality impact: exact (cosine 1.0).

3. **Large self-attention: spatial pooling.** If the VAE mid-block has self-attention over >5K tokens, it will exceed instruction limits. Downsample 2× with `avg_pool2d` before attention, `interpolate` back after. Use `kernel_size=2`, NOT 4. Quality impact: cosine ~1.0.

**Compiler args:** `--model-type=unet-inference -O1 --auto-cast=none`

---

## 5. Phase 3: Pipeline Assembly

Wire all compiled components into the final pipeline. **The assembly must exactly reproduce the original pipeline's execution order from Discovery (§3.2).**

### The General Pattern

```python
# ── STAGE 1: Components that run ONCE, before the loop ──
# Execute in dependency order from your component graph.
# This typically includes: text encoding, conditioning, token preprocessing.

for component in once_before_loop_components:
    outputs[component] = compiled_models[component](*inputs[component])
    # Free component if not needed later: del compiled_models[component]; gc.collect()

# ── STAGE 2: Pre-compute static inputs (CPU) ──
# Attention masks, initial noise, any constant tensors.

# ── STAGE 3: Denoising loop ──
scheduler.set_timesteps(num_steps, ...)
for i, t in enumerate(scheduler.timesteps):
    # Prepare per-step inputs (e.g., concatenate latents with conditioning)
    # Call backbone K times for CFG
    for cfg_variant in cfg_variants:
        pred = backbone(*backbone_inputs_for_variant)
    # Combine predictions (e.g., uncond + scale * (cond - uncond))
    # Scheduler step (CPU, float32 for numerical stability)

# ── STAGE 4: Components that run ONCE, after the loop ──
# Typically just VAE decode.
video = vae_decoder.decode(latents)
```

### Rules for Assembly

1. **Match the original's execution order exactly.** If the original calls the refiner before the loop with a fixed timestep, your port must do the same. If the original has no refiner, don't add one.

2. **Every component from your component graph must run on Neuron.** No CPU fallbacks unless explicitly flagged (e.g., `--vae_on_cpu` for environments where VAE compilation isn't feasible).

3. **Components that take a timestep:** verify whether they use the loop timestep or a fixed value. This is the single most common source of bugs. A conditioning module called per-step with the loop timestep when it should be called once with a fixed timestep will produce subtly wrong output.

4. **CFG requires K forward passes through the backbone per step.** Typically K=2 (conditional + unconditional). Each pass uses different text embeddings but the same noisy latent.

5. **Scheduler step should use float32** for numerical stability, then cast back to bfloat16.

6. **Free components after use** (`del model; gc.collect()`) to reclaim HBM for the next component.

### Incremental Integration Testing

Run the full end-to-end pipeline after each component migration, reading the image each time:

1. All on CPU → read image → baseline ✓
2. Backbone on Neuron, rest on CPU → read image → still correct? ✓
3. Backbone + text encoder on Neuron → read image → still correct? ✓
4. All on Neuron → read image → still correct? ✓ Done.

This isolates which component introduction breaks the output. If step 3 produces a horse instead of a cat, you know the text encoder port is wrong.

### Compile Script

Provide a single `compile_all.py` that compiles every component:

```python
def main():
    for component_name, component_config in COMPONENTS.items():
        compile_component(component_name, component_config, output_dir)
```

### Inference Script

Provide a single `run_inference.py` that runs the full pipeline:

```python
def generate(prompt, output_dir, compiled_dir, num_steps, guidance_scale, seed):
    # Load all compiled models
    # Execute Stage 1-4 above
    # Save frames
```

---

## 6. Neuron-Specific Patterns

### NxDI Application Structure

Every NxDI component follows this pattern:

```python
class MyConfig(InferenceConfig):
    def get_required_attributes(self): return [...]

    @classmethod
    def from_pretrained(cls, model_path, tp_degree=2, **kwargs):
        neuron_config = NeuronConfig(tp_degree=tp_degree, torch_dtype=torch.bfloat16, ...)
        return cls(neuron_config=neuron_config, **kwargs)

class MyModel(nn.Module):
    def __init__(self, config): ...
    def forward(self, *args): ...

class MyWrapper(ModelWrapper):
    def input_generator(self): ...      # example inputs for tracing
    def get_model_instance(self): ...   # creates the model
    def forward(self, *args): ...       # delegates to self._forward

class MyApplication(NeuronApplicationBase):
    _model_cls = MyModel
    def __init__(self, model_path, *args, **kwargs): ...
    def forward(self, *args): ...
    def get_compiler_args(self): return "--model-type=transformer -O1 --auto-cast=none"

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config): ...
```

### Weight Conversion Pattern

```python
@staticmethod
def convert_hf_to_neuron_state_dict(state_dict, config):
    KEY_MAP = {"old_prefix.": "new_prefix.", ...}
    new_sd = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in KEY_MAP.items():
            if new_key.startswith(old):
                new_key = new + new_key[len(old):]
                break
        # Common attention/FFN remapping (diffusers convention)
        new_key = new_key.replace("attn.to_out.0.", "attn.to_out.")
        new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj.")
        new_key = new_key.replace(".ff.net.2.", ".ff.out.")
        new_sd[new_key] = value.clone().detach().contiguous()
    return new_sd
```

---

## 7. XLA Compatibility Reference

### Operations That DON'T Trace (inside NxDI NEFFs)

| Operation                            | Fix                                                |
| ------------------------------------ | -------------------------------------------------- |
| `torch.arange` in forward()          | Pre-compute as `register_buffer`                   |
| `torch.meshgrid` in forward()        | Pre-compute as `register_buffer`                   |
| `torch.where` (dynamic output shape) | Use `torch.index_select` with pre-computed indices |
| `float('-inf')` in masks             | Use `-1e9`                                         |
| `F.pad(mode='replicate')` on 5D      | Use `mode='constant'`                              |
| 8D tensor reshapes                   | Decompose into sequential ≤6D operations           |
| `torch.sort` (trn2 only)             | Use `torch.topk`                                   |
| Unused input tensors                 | Remove — XLA ignores them, then crashes            |

### Operations That Trace Fine

| Operation                                                    | Notes                                        |
| ------------------------------------------------------------ | -------------------------------------------- |
| `F.scaled_dot_product_attention`                             | Preferred for attention                      |
| `F.gelu(approximate="tanh")` / `nn.GELU(approximate="tanh")` | Works                                        |
| `torch.cat`, `torch.index_select`                            | Works                                        |
| `nn.Conv3d`                                                  | Works but generates large instruction graphs |
| `torch.topk`                                                 | Works on both trn1 and trn2                  |

### Input Constraints

- All inputs must be consumed by at least one operation
- All input dtypes must match the compiled dtype (typically bfloat16, except indices which are long)
- All input shapes are fixed at compile time

---

## 8. Validation

### Per-Component Validation

After each component compiles, compare Neuron output vs CPU reference:

```python
cosine = F.cosine_similarity(cpu_out.flatten().float(), neuron_out.flatten().float(), dim=0)
assert cosine > 0.998, f"Component failed: cosine={cosine}"
```

For VAE decoder (due to padding workaround): cosine >0.97 is acceptable.

### End-to-End Validation

Generate video with the same prompt used in Discovery (§3.1). Compare against saved CPU reference frames:

1. Subject is present and recognizable
2. Frame std > 60 (not flat/gray)
3. Spatial structure is correct
4. Temporal coherence across frames

### 🆕 8a. Visual Validation Is Mandatory

**Numerical metrics can pass while the image is garbage.** A frame with std=108 and "correct" color statistics can be random colorful noise. A frame with cosine 0.99 against a reference can show a horse instead of a cat.

**Rule:** After every pipeline change, visually inspect the output image. Compare side-by-side with the CPU reference. If you cannot view images directly, compute per-region statistics:

```python
arr = np.array(img)
top, bot = arr[:H//2].mean(), arr[H//2:].mean()
print(f"top={top:.0f} bot={bot:.0f} R={arr[:,:,0].mean():.0f} G={arr[:,:,1].mean():.0f} B={arr[:,:,2].mean():.0f}")
```

A beach sunset should have: bright top (sky) > dark bottom (ground), R > G > B (warm tones).

### Failure Diagnosis

| Symptom                                 | Likely Cause                                                  |
| --------------------------------------- | ------------------------------------------------------------- |
| Correct scene but missing/wrong subject | Text encoder `CROP_START` or `SEQ_LEN` is wrong               |
| Flat gray output (std < 20)             | Conditioning module called with wrong timestep, or broken CFG |
| NaN in output                           | `float('-inf')` in attention mask                             |
| Wrong output resolution                 | Wrong VAE spatial/temporal factor                             |
| Blurry/wrong spatial detail             | Wrong RoPE convention or axis ordering                        |
| Garbled output, correct shapes          | Token concatenation order wrong                               |
| Degraded quality, subtle                | Dropped weight layer in conditioning module                   |
| Good quality but wrong content          | Text encoder `SEQ_LEN` too short — prompt truncated           |

---

## 9. Pitfalls — Silent Failures That Produce Wrong Output

Each pitfall below was discovered from failed ports. They produce plausible but wrong output with no error message.

### P1: Conditioning module called per-step instead of once (or vice versa)

Read the original pipeline's `__call__` method. If a module is called before the loop, call it before the loop. If it takes a fixed timestep (not the scheduler's), use that fixed value. Getting this wrong changes the conditioning signal.

### P2: Wrong text encoder crop offset or sequence length

These constants must be extracted from the pipeline source code (§3.3), not guessed. A wrong crop offset shifts the entire token window. A too-short sequence length truncates the prompt. Both produce output that looks "almost right" but is wrong.

### P3: Skipping token preprocessing

If the original pipeline reorders, masks, or projects tokens before feeding them to the backbone, your port must do the same. Simply zeroing padding tokens is NOT equivalent to reordering them — padding tokens still occupy attention positions.

### P4: Dropping weight layers to "simplify"

If the original has `linear_1 → activation → linear_2`, your port must have both. Every `nn.Linear` in the original must exist in the port. Verify with parameter count matching.

### P5: Wrong RoPE convention

Multiple conventions exist (real vs complex, interleaved vs split-half, rotate_half vs unbind/stack). Verify your implementation matches the original by comparing Q/K values after RoPE application on CPU.

### P6: Building dynamic tensors inside the NEFF

`torch.arange`, `torch.meshgrid`, dynamic mask construction — all must be pre-computed as `register_buffer` or passed as inputs. They create CPU tensors during XLA tracing.

### P7: Using CPU fallbacks "temporarily"

If a component runs on CPU "to unblock testing," it will stay on CPU forever. Use the completion checklist (§10).

### P8: Inconsistent world_size across NxDI components

All NxDI components must share the same `world_size` (set to the maximum TP degree needed).

### P9: Wrong VAE scaling factor or spatial factor

Read from config, don't hardcode. A wrong spatial factor doubles or halves all dimensions.

### P10: Token concatenation order

If the backbone expects encoder tokens in a specific order (e.g., `[text_1, text_2, image]`), reversing or rearranging silently breaks attention patterns.

### 🆕 P11: System message whitespace

If the pipeline uses a system message for text encoding (e.g., chat-template-based encoders), the exact whitespace matters. A system message with `\n        1.` vs ` 1.` changes tokenization by multiple tokens, shifting the entire embedding window. Extract the system message with `repr()` to see hidden whitespace.

### 🆕 P12: Causal vs bidirectional attention in text encoders

Decoder-architecture models used as text encoders require causal (lower-triangular) attention masking. Using bidirectional masking produces cosine ~0.60 — close enough to look like a numerical issue, far enough to produce completely wrong conditioning. Always check `model.config.is_decoder`.

### 🆕 P13: Cosine similarity comparison target

When debugging a Neuron-compiled model, always compare the **same model** on Neuron vs CPU first (`cosine(neuron_out, cpu_out_same_model)`). Only then compare against the HF reference model. If the first is 1.0 and the second is <0.99, the bug is in your model architecture (attention mask, RoPE, layer extraction), not in Neuron or NxDI weight loading.

### 🆕 P14: Encoder output is all-zeros for certain prompts

Some text encoders (e.g., glyph-specific encoders) only activate for specific input patterns. For normal prompts, the output is all-zeros. This is correct behavior — do not "fix" it by running the encoder on the raw prompt.

### P15: Compiling partial models

Compiling a subset of layers (e.g., 8 of 54) to "test faster" hides critical issues: HBM limits that only appear at full model size, numerical accumulation errors, and semantically wrong output that passes numerical checks. Always compile the full component.

---

## 10. Completion Checklist

**The port is not done until every box is checked.** This checklist is generated from YOUR component graph (§3.2), not from a fixed list.

### For EACH component in your component graph:

- [ ] Compiled for Neuron (NxDI or traced, per framework selection)
- [ ] Validated: cosine >0.998 vs CPU reference (>0.97 for VAE)
- [ ] Weight conversion verified: parameter count matches original
- [ ] Runs on Neuron in the final pipeline (not a CPU fallback)

### Pipeline assembly:

- [ ] Execution order matches original pipeline's `__call__` method
- [ ] Components that run once DO run once (not per-step)
- [ ] Components that run per-step DO run per-step (not once)
- [ ] Any fixed timestep values match the original (not the loop timestep)
- [ ] CFG uses the correct number of backbone calls per step
- [ ] Scheduler step uses float32 for numerical stability

### End-to-end:

- [ ] Generated video matches CPU reference (subject present, correct scene)
- [ ] Frame std > 60 (not flat/gray)
- [ ] No NaN in any intermediate tensor

### Artifacts:

- [ ] `pipeline_constants.py` — all constants from Discovery in one file
- [ ] `compile_all.py` — single script that compiles every component
- [ ] `run_inference.py` — single script that runs the full pipeline

---

## 11. 🆕 Field-Tested Debugging Lessons

These lessons come from actual T2V model ports to trn2. Each cost hours of debugging.

### 11.1 Never Reimplement What You Can Trace Directly

Always try `torch_neuronx.trace` on the HF model first. If it fits on one core, you're done — cosine will be >0.999 with zero architecture work. If it doesn't fit (>11B params on trn2), use NxDI with TP — but reimplement the architecture **exactly**, not approximately. Run a CPU cosine comparison of your reimplementation vs the HF model before ever touching Neuron.

### 11.2 The Debugging Comparison Protocol

When a component produces wrong output on Neuron:

1. **Same model, Neuron vs CPU:** `cosine(my_model_neuron, my_model_cpu)` — should be >0.999
2. **My model vs HF model, both CPU:** `cosine(my_model_cpu, hf_model_cpu)` — should be >0.999
3. **Short input test:** Try 20 tokens instead of 1108. If short works but long doesn't, the bug is in the attention mask or padding handling.
4. **Valid tokens only:** Compare only the non-padding tokens. If valid-only cosine is high but full-sequence cosine is low, the padding behavior differs (acceptable).

If step 1 fails: Neuron compilation issue (rare).
If step 2 fails: Architecture bug in your reimplementation (common).
If both pass but end-to-end output is wrong: Preprocessing bug (tokenization, system message, crop offset).

### 11.3 NxDI Compile vs Inference Process Model

- **Compile:** `torchrun --nproc_per_node=<tp_degree>` — multi-process, needed for HLO generation
- **Inference:** Single `python3` process — NxDI handles multi-core internally via `app.load()`
- **Visible cores:** Set `NEURON_RT_VISIBLE_CORES=0-<2*tp-1>` (2 physical cores per logical core on trn2 LNC=2)

The NxDI Flux example (`generate_flux.py`) demonstrates this pattern — it's a plain `python3` script, not `torchrun`.

### 11.4 Pre-compute ALL Constant Tensors

Dynamic tensor creation in `forward()` creates CPU tensors during XLA tracing. Pre-compute in `__init__` as `register_buffer`:

| Must pre-compute                             | Why                                          |
| -------------------------------------------- | -------------------------------------------- |
| RoPE cos/sin (including M-RoPE interleaving) | `torch.arange` in forward creates CPU tensor |
| Causal attention mask                        | `torch.tril` in forward creates CPU tensor   |
| Position indices                             | Same                                         |
| Any grid or meshgrid                         | Same                                         |

### 11.5 VAE Conv3d Instruction Limits

The VAE decoder (even at 1.26B params) generates 7.9M instructions from Conv3d ops, exceeding the 5M limit. **Do not attempt to trace the full VAE.** Split into individual blocks (conv_in, mid_resnets, mid_attn, up_resnets, up_upsamples, norm_out), trace each separately, chain at runtime. Use `--model-type=unet-inference` for VAE blocks.

### 11.6 Visual Validation Is the Only Ground Truth

| Metric                | What it tells you | What it DOESN'T tell you            |
| --------------------- | ----------------- | ----------------------------------- |
| `std > 60`            | Not flat/gray     | Could be random colorful noise      |
| `cosine > 0.99`       | Numerically close | Could show a horse instead of a cat |
| `R > G > B`           | Warm tones        | Could be any warm-toned scene       |
| **Visual inspection** | **Everything**    | —                                   |

Always look at the actual image. Compare side-by-side with the CPU reference.

### 11.7 Summary of Root Causes from Non-Causal VAE Port

| Symptom                                    | Root Cause                                        | How to Avoid                            |
| ------------------------------------------ | ------------------------------------------------- | --------------------------------------- |
| Colorful noise output                      | Raw encoder output without pipeline preprocessing | Run Discovery phase (§3) completely     |
| Wrong subject (e.g., horse instead of cat) | Bidirectional attention on a causal encoder model | Check `model.config.is_decoder` (§3.3a) |
| Slightly wrong subject                     | System message whitespace difference              | Extract with `repr()` (§3.3a)           |
| Cosine 0.60 "framework broken"             | Comparing different models, not same model        | Use comparison protocol (§11.2)         |
| NEFF won't load (OOM)                      | Model too large for single core                   | Check HBM budget first (§2)             |
| Cosine 0.59 "RoPE wrong"                   | Actually correct — wrong comparison target        | Use comparison protocol (§11.2)         |

---

## 12. 🆕 Field-Tested Lessons from Causal-VAE T2V Port (v3.0)

These lessons come from porting a 1.3B-parameter T2V model with a causal 3D VAE to trn2.48xlarge. They extend the lessons above with new failure modes specific to causal VAEs, context parallelism, and across-heads normalization.

### 12.1 Causal vs Non-Causal VAE — The Critical Distinction

The §4 Type C pattern (split into blocks, trace each, chain at runtime) was written for a **non-causal** VAE. Some models (e.g., those with `CausalConv3d`) use a **causal** VAE. The difference:

|                       | Non-Causal VAE                 | Causal VAE                                             |
| --------------------- | ------------------------------ | ------------------------------------------------------ |
| Temporal padding      | Symmetric (sees past + future) | Asymmetric (sees past only)                            |
| Processing            | All frames at once             | One frame at a time with cache                         |
| Cache                 | None                           | 33-entry stateful cache                                |
| Cache lifecycle       | N/A                            | None → "Rep" string → 1-frame tensor → 2-frame tensor  |
| Block variants needed | 1 per block                    | 3 per block (\_0f, \_1f, regular) + upsampler variants |
| Total traced blocks   | ~18                            | ~55                                                    |

**For causal VAEs, follow this procedure:**

1. Run the full decoder on CPU for frames 0, 1, and 2 with hooks to capture per-block input shapes and cache shapes at each frame
2. Identify cache state transitions (None → sentinel → 1-frame → 2-frame)
3. Trace separate block variants for each cache state
4. For temporal upsamplers with sentinel initialization ("Rep" path), trace spatial-only (`_sp`) and first-chunk (`_fc`) variants
5. At runtime, select the correct variant based on frame index and manage cache accumulation (including padding 1-frame cache to 2-frame for the transition)

**The VAE blocks are frame-count-agnostic.** The same 55 blocks work for 13, 49, or any number of output frames — just iterate more times. No recompilation needed for different frame counts.

### 12.2 XLA Silently Drops Unused Inputs

If a `forward()` method takes a tensor input but only uses `tensor.shape` (not the values), XLA tracing drops the input entirely. The NEFF runs without error but produces wrong output.

**Known affected patterns:**

- RoPE computed from `hidden_states.shape` (e.g., a `RotaryPosEmbed` module that only reads `.shape`)
- Dynamic mask construction from `input.shape[1]`
- Any positional encoding derived from input dimensions

**Detection:** Look for this XLA warning during tracing:

```
UserWarning: Received an input tensor that was unused or used in a non-static way
```

**Fix:** Pre-compute the result on CPU and pass it as an explicit tensor input.

**Impact if missed:** Cosine drops to ~0.994 (looks fine for one step, destroys diffusion over 20 steps with CFG amplification).

### 12.3 XLA-Incompatible Operations in VAE Decoders

Two operations commonly found in causal 3D VAEs crash XLA:

| Operation                             | Error                             | Fix                                    |
| ------------------------------------- | --------------------------------- | -------------------------------------- |
| `x[:, :, -N:, :, :]` where dim < N    | "Value out of range"              | `x[:, :, max(0, x.shape[2]-N):, :, :]` |
| `F.interpolate(mode='nearest-exact')` | "Unknown custom-call API version" | `mode='nearest'`                       |

**Apply fixes via monkey-patching** (replace forward methods at runtime), NOT by modifying the installed diffusers source. This keeps the port self-contained.

### 12.4 NxDI ModelBuilder vs torch_neuronx.trace — Decision Framework

| Situation                           | Use                                                     | Why                                                                   |
| ----------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------- |
| Model fits on 1 core, no TP needed  | `torch_neuronx.trace`                                   | Simpler, no framework overhead                                        |
| Model needs TP≥2                    | NxDI ModelBuilder                                       | Handles weight sharding, parallel layers, multi-core compilation      |
| NxDI constant-folds a dynamic input | `torch_neuronx.trace`                                   | NxDI's tracing can bake input values as constants                     |
| Need context parallelism            | `torch_neuronx.trace` with NxD communication primitives | CP splits sequence, not model — use TP infrastructure for all-gathers |

**Always try NxDI first for TP.** Only fall back to `torch_neuronx.trace` if NxDI constant-folds critical dynamic inputs (test by changing an input at inference and checking if output changes).

### 12.5 The 5M Instruction Limit and How to Work Around It

The Neuron compiler has a hard 5M instruction limit per NEFF. This is the most common blocker for video models.

**What generates instructions:**

- Attention: O(seq_len²) — dominates for long sequences
- Conv3d: O(channels × spatial × temporal × kernel) — dominates for VAE
- All-gathers: ~1K instructions each, adds up with many layers

**Instruction count scaling (example: 30-block DiT backbone):**

| seq_len            | TP=1    | CP=4    | CP=8      |
| ------------------ | ------- | ------- | --------- |
| 6,240 (13 frames)  | ~5M ✅  | N/A     | N/A       |
| 7,800 (17 frames)  | 6.9M ❌ | N/A     | N/A       |
| 20,280 (49 frames) | 37M ❌  | 9.5M ❌ | 5.1M ❌\* |

\*CP=8 at 5.1M is 2.5% over — solved by splitting into 2 NEFFs of 15 blocks each (~2.5M each).

**Solutions (in order of preference):**

1. **Context parallelism** — splits sequence across ranks, reduces per-rank compute AND instructions
2. **Split into multiple NEFFs** — trace groups of blocks separately, chain at runtime (1-2ms overhead per NEFF boundary)
3. **Block-by-block tracing** — trace each block individually (higher overhead, last resort)

**CP is preferred over TP for instruction reduction** because TP reduces model width but not sequence length, and the all-gathers add instruction overhead. CP reduces the effective sequence length per rank.

### 12.6 Context Parallelism Implementation

CP for diffusion backbones is straightforward because attention is bidirectional (not causal):

1. Split the patch sequence across ranks: each rank gets `seq_len / cp_degree` tokens
2. In self-attention: Q is local, K/V are all-gathered from all ranks
3. RoPE: pre-gather once before the block loop (saves N_blocks × 2 all-gathers)
4. Cross-attention: text embeddings are replicated on all ranks (small, no splitting needed)
5. Output: all-gather after the last block to reconstruct the full sequence

**Key optimization:** With pure CP (no TP), each rank has all attention heads locally. RMSNorm across heads works without all-gather. Only K/V need all-gathering in self-attention.

**Use NxD's `gather_from_tensor_model_parallel_region` for all-gathers** — it works with the TP infrastructure even when used for CP (just initialize TP with `tensor_model_parallel_size=cp_degree`).

### 12.7 Across-Heads QK Normalization with TP

Some models use `qk_norm="rms_norm_across_heads"` — RMSNorm applied to the full Q/K vector (all heads concatenated) BEFORE splitting into heads. This is incompatible with standard TP because each rank only has `inner_dim/tp` elements after the column-parallel projection.

**Solution for TP:** All-gather Q/K → RMSNorm on full vector → slice back to local shard. Adds 4 all-gathers per attention layer (Q_gather, K_gather for self-attn and cross-attn).

**Solution for CP:** No issue — each rank has all heads locally.

**Per-head RMSNorm approximation does NOT work.** Cosine drops to ~0.97, which destroys diffusion quality over 20 steps.

### 12.8 Fixed Frame Count Per NEFF

Backbone NEFFs are compiled for a specific latent temporal dimension. Different frame counts require recompilation. This is standard for Neuron/XLA (e.g., NxDI's Flux implementation does the same for different image resolutions).

**VAE blocks are frame-count-agnostic** — they process one frame at a time and work for any frame count.

**Mitigation:** Pre-compile a set of standard frame counts (13f, 49f, 81f) and select at runtime.

### 12.9 LNC Mismatch Between Components

T5-XXL (4.7B params, 9.4GB) needs LNC=1 (24GB per core). The backbone needs LNC=2 (12GB per core, more instruction budget). LNC cannot be switched within a process.

**Solution:** Run T5 in a separate subprocess with `NEURON_RT_VIRTUAL_CORE_SIZE=1`, save embeddings to disk, load in the main process.

### 12.10 The Crosshatch Artifact (Low Frame Count)

At 13 output frames (4 latent), the VAE's temporal upsampler (`time_conv`) produces a crosshatch pattern on textured surfaces. This is NOT a Neuron issue — it's present in CPU fp32 inference too.

**Root cause:** Insufficient temporal context. With only 4 latent frames, the temporal convolution cache has 1-2 frames of history, producing aliased interpolation.

**Fix:** Generate ≥49 frames (13 latent). Confirmed clean at 49 frames on both CPU and Neuron.

### 12.11 Core Allocation for Multi-Component Pipelines

trn2.48xlarge has 64 NeuronCores (32 physical). Use different core sets for different components to avoid allocation conflicts:

```bash
# T5: cores 0-3, LNC=1
NEURON_RT_VISIBLE_CORES=0-3 NEURON_RT_VIRTUAL_CORE_SIZE=1 python3 encode_t5.py

# Backbone CP=8: cores 0-7
NEURON_RT_VISIBLE_CORES=0-7 torchrun --nproc_per_node=8 denoise.py

# VAE: cores 8-15 (while backbone NEFFs are still loaded on 0-7)
NEURON_RT_VISIBLE_CORES=8-15 python3 decode_vae.py
```

### 12.12 Meta-Lesson: Evaluate All Options Before Coding

A recurring mistake in this port was jumping into the first viable approach without evaluating alternatives:

- **VAE:** Tried full-decoder tracing → tiled decode → no-cache blocks → cached block-by-block (which the guide prescribed from the start)
- **Higher frame count:** Jumped to TP → discovered instruction limit → tried CP → worked
- **Block-by-block vs CP:** Block-by-block was the obvious "more of the same" approach, but CP was fundamentally better

**Rule:** When hitting a constraint, spend 10 minutes listing ALL possible solutions with tradeoffs before writing any code. This saves hours of debugging dead ends.

### 12.13 Summary of Causal-VAE Port Bugs

| Bug                                | Symptom                             | Detection                       | Fix                                                |
| ---------------------------------- | ----------------------------------- | ------------------------------- | -------------------------------------------------- |
| XLA drops shape-only inputs (RoPE) | Cosine 0.994, muted output          | XLA warning in trace log        | Pre-compute on CPU, pass as input                  |
| NxDI constant-folds attention mask | T5 ignores padding                  | Same output for different masks | Use `torch_neuronx.trace`                          |
| Padding not zeroed after T5        | Colorful noise                      | Cosine 0.05 vs pipeline         | Zero positions beyond seq_len                      |
| XLA negative indexing              | Compile error                       | Error message                   | `max(0, x.shape[2] - N)`                           |
| XLA nearest-exact                  | Compile error                       | Error message                   | `mode='nearest'`                                   |
| CausalConv3d segfaults XLA         | Segfault                            | Crash                           | Block-by-block tracing                             |
| Stale VAE cache                    | Crosshatch on Neuron-decoded frames | Visual inspection               | Return updated cache from blocks                   |
| Low frame count crosshatch         | Crosshatch on ALL platforms         | CPU fp32 reference              | Generate ≥49 frames                                |
| 5M instruction limit               | Compile error (exit 70)             | Compiler message                | CP + split NEFFs                                   |
| LNC mismatch                       | Can't load T5 + backbone            | Runtime error                   | Separate subprocesses                              |
| Old NEFFs cached in memory         | Shape mismatch after retrace        | Runtime error                   | Reload blocks after retracing                      |
| VAE block traced before patches    | Wrong output (cos 0.004)            | Per-block cosine check          | Always apply patches first, trace ALL blocks fresh |
