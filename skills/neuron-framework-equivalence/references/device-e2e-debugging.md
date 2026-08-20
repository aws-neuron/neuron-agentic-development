# Device End-to-End Equivalence Debugging

When CPU E2E passes but device (NeuronX) produces divergent output, the bug lies in the translation from CPU-mode Python to compiled XLA/NEFF execution.

**Core principle:** **1-layer isolation** is the key technique — it cuts compile time from 13+ minutes to ~5 minutes, enabling rapid iteration through the fix-compile-verify cycle.

---

## Phase 1: Quantify Device Divergence

Measure how far device output is from the HF reference:

1. Generate HF reference logits (FP32 + BF16) for a fixed prompt
2. Run device E2E with the same prompt, capture logits
3. Compute error_ratio = `||device - fp32|| / ||bf16 - fp32||`

| error_ratio | Interpretation          | Action                                               |
| ----------- | ----------------------- | ---------------------------------------------------- |
| <= 1.2      | Within BF16 precision   | **PASS** — done                                      |
| 1.2 - 10    | Moderate divergence     | Precision or scaling bug                             |
| 10 - 100    | Significant divergence  | Missing operation or wrong code path                 |
| 100+        | Catastrophic divergence | Fundamental issue (wrong weights, missing algorithm) |

---

## Phase 2: 1-Layer Isolation

Override `config.num_hidden_layers = 1` when creating the model. Generate 1-layer HF reference intermediates (FP32 + BF16) using hooks. Compile 1-layer device model with `TensorCaptureConfig`. Run 3-tensor comparison at each module boundary.

### Choosing Modules to Capture

| #   | Module                              | What It Captures   | Sharded?         |
| --- | ----------------------------------- | ------------------ | ---------------- |
| 1   | `embed_tokens`                      | Token embeddings   | Gathered         |
| 2   | `layers.0.input_layernorm`          | Pre-attention norm | No               |
| 3   | `layers.0.self_attn`                | Attention output   | No (all-reduced) |
| 4   | `layers.0.post_attention_layernorm` | Pre-MLP norm       | No               |
| 5   | `layers.0.mlp`                      | MLP output         | No (all-reduced) |
| 6   | `layers.0`                          | Full layer output  | No               |
| 7   | `norm`                              | Final norm         | No               |
| 8   | `lm_head`                           | Logits             | Gathered         |

### Reading the Comparison Table

```
Module                            | baseline_err | target_err  | error_ratio | Status
embed_tokens                      | 0.00e+00     | 0.00e+00    | inf*        | PASS*
layers.0.input_layernorm          | 1.59e-03     | 1.59e-03    | 1.0000      | PASS
layers.0.post_attention_layernorm | 5.01e-03     | 6.38e+01    | 127.34      | FAIL <<<
layers.0.mlp                      | 3.70e-02     | 4.82e+01    | 1303.78     | FAIL
```

The **first FAIL** is where the bug originates. Everything after may cascade from it.

---

## Phase 3: Root Cause Diagnosis

### Strategy 1: Read the First Failure

| First failing module       | Likely cause                                                       |
| -------------------------- | ------------------------------------------------------------------ |
| `embed_tokens`             | Embedding vocabulary partition issue (SPMDRank)                    |
| `post_attention_layernorm` | Attention computation bug (missing algorithm, wrong dispatch path) |
| `layers.0.mlp` / `experts` | MoE routing, weight layout, or bias issue                          |
| `norm` / `lm_head`         | Cascaded error from earlier layers                                 |

### Strategy 2: Manual Component Reconstruction

Extract weights from the checkpoint, run the component manually on CPU, compare with/without a hypothesized fix:

```python
from safetensors.torch import load_file
hf_sd = load_file("model.safetensors")

output_with_fix = component_forward_with_fix(input_tensor, weights)
output_without_fix = component_forward_without_fix(input_tensor, weights)

cos_with = F.cosine_similarity(output_with_fix.flatten().unsqueeze(0),
                                hf_output.flatten().unsqueeze(0))
cos_without = F.cosine_similarity(output_without_fix.flatten().unsqueeze(0),
                                   hf_output.flatten().unsqueeze(0))
```

### Strategy 3: Code Path Tracing

Device may use a different dispatch path than CPU:

```python
# Example: NeuronAttentionBase.forward dispatches based on sliding_window
if self.sliding_window:
    return self.windowed_attention_forward(...)   # -> perform_prefill_windowed_attn
else:
    return self.standard_causal_attention_forward(...)  # -> perform_prefill
```

If a patch only covers one path, the component works on CPU (which happens to use the patched path) but fails on device (which uses the unpatched path).

---

## Phase 4: Fix-Compile-Verify Cycle

```
Write/update patch
    → Copy patch to Docker
    → Recompile 1-layer model (~5 min)
    → Run device inference with tensor capture
    → Compare intermediates (3-tensor table)
    → First-fail moved downstream? → Fix worked! Repeat for remaining failures
    → Same module still fails? → Revise the patch, try again
    → All modules PASS? → Proceed to full model validation
```

Name each compiled model distinctly: `compiled_1layer_embfix/`, `compiled_1layer_sinkfix/`, etc.

---

## Phase 5: Full Model Validation

Once all 1-layer modules pass:

1. Compile the full N-layer model with all patches applied
2. Run inference with user-provided prompts or default prompt set
3. Compare logits at last position using 3-tensor comparison
4. Verify top-1 token match between HF and device
5. Run multi-token generation (20 tokens) — check for coherent output

| Metric                             | Threshold                       |
| ---------------------------------- | ------------------------------- |
| error_ratio (last position logits) | <= 1.2                          |
| Top-1 token match                  | Yes                             |
| Top-5 token overlap                | >= 4/5                          |
| 20-token generation                | Coherent, no repetition/garbage |

---

## Phase 6: Clean Up

1. Unify patches: merge device-only and CPU-only patches into single files (use Pattern 5 from device-component-debugging)
2. Remove scaffolding: delete debug prints, temporary hooks, intermediate scripts
3. Re-validate: run full model with cleaned-up patches to ensure no regressions

---

## Common Root Causes

| Root Cause                               | Error Magnitude  | Fix Pattern                                                      |
| ---------------------------------------- | ---------------- | ---------------------------------------------------------------- |
| Missing SPMDRank                         | 100x-600x        | Add SPMDRank, inject in pre_shard_weights_hook                   |
| Missing algorithm on alternate code path | 100x+            | Trace code paths, patch the missed path                          |
| Weight layout mismatch                   | 1000x+           | Fix layout in pre_shard_weights_hook before sharding             |
| Missing bias Parameters                  | 10x+             | Create bias Parameter in **init**, pass bias flags to base class |
| Patches not applied in mp.spawn workers  | matches no-patch | Re-apply all patches inside each worker function                 |

---

## XLA-Compatible Patch Patterns

See [device-component-debugging.md](device-component-debugging.md) for the five key patterns:

1. SPMDRank instead of parallel_state
2. torch.index_select instead of torch.narrow
3. Framework \_reduce instead of torch.distributed
4. Parameters must exist before tracing
5. Unified CPU+device patches

---

## Device-Specific Pitfalls

1. **TensorCaptureConfig requires OnDeviceSamplingConfig** — without it, captured tensors are silently not returned. Always pair them when compiling with tensor capture.
2. **self_attn captures cos_cache, not hidden_states** — use `post_attention_layernorm` as an attention quality proxy instead.
3. **TP-gathered logits are N×vocab_size** — after TP gathering, logits are `tp_degree * vocab_size` wide. Take first `vocab_size` entries: `logits[:, :, :vocab_size]`.
4. **"Removing redundant keys" warning is normal** — the framework's preshard hook remaps individual q/k/v into combined qkv. This warning does not indicate an error.

---

Based on: GPT-OSS 20B device E2E debugging (Feb-Apr 2026)
