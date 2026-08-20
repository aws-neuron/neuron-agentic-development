---
name: cpu-component-debugging
description: Debug and fix failing component-level equivalence tests on CPU by systematically comparing reference and target implementations, identifying the root cause of divergence, and producing runtime monkey patches that restore equivalence without modifying original source files. For device-specific component debugging, see device-component-debugging.
---

# CPU Component-Level Debugging

When a component equivalence test (from the `component-testing` skill) fails **on CPU**, use this workflow to diagnose the root cause, understand exactly how the reference implementation differs from the target, and produce a monkey patch that fixes the target at runtime. Once components pass on CPU, use `device-component-debugging` to validate and fix them on device.

**Core principle:** Never modify the original target model file. All fixes are delivered as standalone monkey-patch files that are imported and applied before the test runs.

---

## Prerequisites

- A **failing component test** from the `component-testing` skill (e.g., `test_02_rotary_emb.py` with error ratio >> 1.2)
- A **running Docker container** with both reference and target implementations importable
- Access to both source files: the reference (e.g., HuggingFace `transformers`) and the target (e.g., Neuron model code)

---

## Workflow

### Phase 1: Diagnose — Read the Test and Log

1. **Read the failing test file** to understand what is being compared (which classes, which forward signatures, what shapes are expected).
2. **Read the test log** to extract the error ratio and which specific metric (cos, sin, output, etc.) is failing.
3. **Note the error magnitude.**
   - 100x+ → formula/algorithm mismatch
   - 1.2–3x → precision ordering issue
   - < 1.0 → patch computes at _higher_ precision than the reference (dtype over-precision)

### Phase 2: Compare Implementations Side-by-Side

4. **Locate both implementations.** Find the exact source files for both the reference and target classes. The reference is typically inside the Docker container's Python packages (e.g., `/opt/conda/lib/python3.*/site-packages/transformers/...`). The target is in the mounted model directory (e.g., `/root/data-for-equiv-check/`).

5. **Read both implementations completely.** Don't skim — read the `__init__`, `forward`, and any helper functions. Pay attention to:
   - Config parameters consumed by the reference but ignored by the target
   - Mathematical operations present in one but not the other
   - Differences in output shape or dtype handling

6. **Identify the root cause category:**

   | Category                       | Symptoms                                                                           | Example                                           |
   | ------------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------- |
   | **Missing algorithm**          | Error ratio 50x–1000x+. Target uses a simpler formula.                             | YaRN scaling omitted from RoPE                    |
   | **Missing multiplier/scaling** | Error ratio 1.3x–2x. Values are proportionally off.                                | `attention_scaling` factor not applied            |
   | **Config parameter gap**       | Target config missing a field the algorithm needs.                                 | `original_max_position_embeddings` absent         |
   | **Precision ordering**         | Error ratio 1.2x–15x. One output (e.g., cos) passes but another (e.g., sin) fails. | Scaling applied after bf16 cast instead of before |
   | **Shape mismatch**             | Comparison invalid. Shapes differ between reference and target.                    | `[bs, seq, dim/2]` vs `[bs, seq, dim]`            |
   | **Routing/logic ignored**      | Error ratio 1000x+. Target produces structurally different output.                 | MoE routing weights ignored                       |

7. **Run numerical diagnostics inside Docker** to confirm. Compare intermediate values (e.g., `inv_freq`, `freqs`, `cos`, `sin`) between reference and target:

   ```bash
   docker exec <container> bash -c "
   cd /root/equiv-check-rst/tests && \
   NXD_CPU_MODE=1 python3 -c '
   import sys
   sys.path.insert(0, \"/root/data-for-equiv-check\")
   from conftest import _init_cpu_env_tp1
   _init_cpu_env_tp1()

   # ... instantiate both, compare intermediates ...
   '"
   ```

### Phase 3: Write the Monkey Patch

8. **Create a patch file** at `{EXP_DIR}/patches/<patch_name>.py`. The file must:
   - Be self-contained (no modifications to original files)
   - Provide an `apply_<name>_patch()` function as the public API
   - Be idempotent (safe to call multiple times, using a `_patched` class attribute guard)
   - Preserve the framework's expected output format (shapes, dtypes, tensor layout)

9. **Follow this patch structure:**

   ```python
   """
   Monkey-patch for <TargetClass> to add <missing feature>.

   Usage:
       from <patch_name> import apply_<name>_patch
       apply_<name>_patch()
   """

   import math
   import torch


   def _compute_correct_values(config):
       """Port the reference algorithm, parameterized by config."""
       # ... replicate the reference computation exactly ...
       return result_tensor, scaling_factor


   def apply_<name>_patch():
       """Monkey-patch <TargetClass>. Call BEFORE instantiation."""
       from modeling_<model> import TargetClass

       if getattr(TargetClass, "_patched", False):
           return

       _original_init = TargetClass.__init__

       def _patched_init(self, config):
           _original_init(self, config)
           # Compute and store corrected values
           self._corrected_values = _compute_correct_values(config)

       def _patched_forward(self, *args, **kwargs):
           # Use corrected values in the forward pass
           ...

       TargetClass.__init__ = _patched_init
       TargetClass.forward = _patched_forward
       TargetClass._patched = True
   ```

### Phase 4: Pitfalls and Common Fixes

10. **Apply lessons from known pitfalls** (ordered by frequency of occurrence):

#### Pitfall 1: Config parameter gaps

The target config's dictionary may omit parameters that the reference config includes by default. Always check what the reference config provides vs what the target config provides.

**Fix:** Add fallback resolution in the patch. Prefer deriving missing values from known ones.

```python
# Example: original_max_position_embeddings missing from rope_scaling
original_max_pos = rope_scaling.get("original_max_position_embeddings")
if original_max_pos is None:
    original_max_pos = getattr(config, "original_max_position_embeddings", None)
if original_max_pos is None:
    # Derive from known relationship: original * factor = max
    original_max_pos = int(config.max_position_embeddings / factor)
```

#### Pitfall 2: Precision ordering (fp32 vs bf16 cast timing)

The reference may apply a scaling multiplier in fp32 before casting to the output dtype. If your patch applies the multiplier after the framework already cast to bf16, you get precision loss.

**Symptom:** One metric (e.g., cos) passes but a related one (e.g., sin) fails with error ratio 5x–15x.

**Fix:** Reimplement the forward computation inline in the patch, applying all multipliers in fp32 before the final `.to(dtype=x.dtype)` cast. Do NOT delegate to the framework's forward then multiply afterward.

```python
def _patched_forward(self, x, position_ids):
    # Compute entirely in fp32, apply scaling, THEN cast
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * self.attention_scaling   # fp32 multiply
    sin = emb.sin() * self.attention_scaling   # fp32 multiply
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)  # cast last
```

#### Pitfall 3: PyTorch registered buffer assignment

`nn.Module.register_buffer("name", None)` creates a buffer slot. Direct assignment (`module.name = tensor`) may silently fail to update the buffer value depending on the PyTorch version and buffer state.

**Symptom:** You set `self.child_module.inv_freq = computed_tensor` in `__init__`, but reading it back in `forward` still returns the default value.

**Fix:** Store corrected tensors on the wrapper module as a plain attribute (e.g., `self._corrected_inv_freq`), and use that directly in the patched forward. Don't try to overwrite framework module buffers.

```python
def _patched_init(self, config):
    _original_init(self, config)
    # Store on self, NOT on self.child_module
    self._corrected_inv_freq = compute_correct_inv_freq(config)

def _patched_forward(self, x, position_ids):
    inv_freq = self._corrected_inv_freq.to(x.device)  # use our value
    ...
```

#### Pitfall 4: Output shape conventions

The reference and target may produce outputs with different shapes for the same logical data (e.g., `[bs, seq, dim/2]` vs `[bs, seq, dim]` via concatenation). The patch must preserve the target's output shape so downstream framework code continues to work.

**Fix:** Match the reference's algorithm but emit the target's shape format.

#### Pitfall 5: Dtype must match the reference — no more, no less

The patch must use the **same dtype strategy** as the reference. Read every `.float()`, `.to(torch.float32)`, and `.to(dtype)` call in the reference's forward. Replicate them exactly — don't add extras, don't omit any.

**Two failure modes:**

| Mode                | Symptom            | Cause                                                  |
| ------------------- | ------------------ | ------------------------------------------------------ |
| **Under-precision** | Error ratio 1.2–2x | Target computes in bf16 where reference uses fp32      |
| **Over-precision**  | Error ratio < 1.0  | Patch adds `.float()` calls the reference doesn't have |

Error ratio < 1.0 is a bug: it means the target is closer to the fp32 ground truth than the bf16 reference is, which can only happen if the patch computes at higher precision.

**Case A — Reference upcasts, target doesn't (under-precision):**

RMSNorm: HF does `hidden_states.to(torch.float32)` before `pow(2).mean()`. Neuron did `.to(torch.bfloat16)`. Error ratio 1.85x. Fix: match the fp32 upcast.

```python
# Wrong — computes variance in bf16
hidden_states = hidden_states.to(torch.bfloat16)

# Correct — matches HF's fp32 variance computation
hidden_states = hidden_states.to(torch.float32)
```

**Case B — Reference stays native, patch adds fp32 (over-precision):**

MoE experts: HF has zero `.float()` calls — matmuls, biases, routing weights all stay in native dtype. An initial patch added `.float()` to einsum operands, producing error ratio 0.82. Fix: remove all `.float()` calls.

```python
# Wrong — forces fp32 where HF uses native dtype
output = torch.einsum('nei,eih->neh', x.float(), w.float())

# Correct — stays in native dtype like HF
output = torch.einsum('nei,eih->neh', x, w)
```

**Case C — Reference upcasts for specific operations only:**

YaRN RoPE: HF computes `inv_freq @ position_ids` in fp32, applies `attention_scaling` in fp32, then casts to input dtype. Applying scaling _after_ a bf16 cast caused sin error ratio 13.7x. Fix: replicate the same fp32 → scale → cast sequence.

### Phase 5: Integrate and Verify

11. **Modify the test file** to import and apply the patch at module level:

    ```python
    from <patch_name> import apply_<name>_patch
    apply_<name>_patch()
    ```

12. **Remove expected-failure markers** from the test (the `"EXPECTED FAIL"` messages, the error printouts). Update the docstring to reflect the patched state.

13. **Run the individual test** inside Docker:

    ```bash
    docker exec <container> bash -c "
    cd /root/equiv-check-rst/tests && \
    NXD_CPU_MODE=1 python3 -m pytest <test_file>.py -v"
    ```

14. **Verify error ratio is ~1.0**, not just below 1.2. A perfect patch should yield error ratio 1.0000 (all error comes from bf16 quantization, none from algorithm mismatch).

15. **Run the full test suite** to confirm no regressions:

    ```bash
    docker exec <container> bash -c "
    cd /root/equiv-check-rst/tests && bash run_all.sh"
    ```

---

## Debugging Checklist

When a patch doesn't produce the expected result, verify in order:

1. **Is the patch actually applied?** Print `TargetClass._patched` and check that patched attributes (e.g., `self.attention_scaling`) have the expected values.
2. **Are intermediate values correct?** Print the patched `inv_freq` / weights / intermediates and compare against the reference's values numerically.
3. **Is there a config parameter gap?** Print both configs' `rope_scaling` (or equivalent) dicts side by side. Look for missing keys.
4. **Is there a precision ordering issue?** Compare fp32-to-fp32 (bypassing bf16). If the pure fp32 comparison shows zero diff, the algorithm is correct and the remaining error is from dtype cast timing.
5. **Is there a buffer assignment issue?** Print the value you assigned vs what the forward reads back. If they differ, store on the wrapper instead.
6. **Is the patch using the same dtype strategy as the reference?** Check for `.float()` calls in the patch that don't exist in the reference. Error ratio < 1.0 means over-precision.

---

## Deliverables

For each fixed component, produce:

| File                             | Location                   | Purpose                                                          |
| -------------------------------- | -------------------------- | ---------------------------------------------------------------- |
| `<patch_name>.py`                | `{EXP_DIR}/patches/`       | Standalone monkey-patch file                                     |
| Updated `test_NN_<component>.py` | `experiments/<exp>/tests/` | Test imports and applies patch, expected-failure markers removed |

---

## Reference Example: GPT-OSS YaRN RotaryEmbedding

**Failing test:** `test_02_rotary_emb.py` — error ratio 130x (cos) and 133x (sin).

**Root cause diagnosis:**

- HF `GptOssRotaryEmbedding` uses `ROPE_INIT_FUNCTIONS["yarn"]` which blends interpolated/extrapolated `inv_freq` and applies `attention_scaling = 1.3466` to cos/sin.
- Neuron `NeuronGptOssRotaryEmbedding` wraps framework's basic `RotaryEmbedding(dim, max_pos, base)` which only computes standard RoPE. The `rope_scaling` config is stored but never used.

**Patch:** `yarn_rotary_patch.py` — `apply_yarn_rotary_patch()`:

1. `_patched_init`: computes YaRN `inv_freq` and `attention_scaling`, stores as `self._yarn_inv_freq` and `self.attention_scaling`
2. `_patched_forward`: computes rotary embeddings in fp32 using YaRN inv_freq, applies attention_scaling before dtype cast, emits `cat((freqs, freqs), dim=-1)` format

**Pitfalls encountered during development:**

1. Neuron config's `rope_scaling` dict was missing `original_max_position_embeddings` (4096), so the correction range was computed against 131072, making YaRN inv_freq nearly identical to default. Fixed by deriving: `original = max_position_embeddings / factor`.
2. Framework's `register_buffer("inv_freq", None)` resisted direct assignment. Fixed by storing `_yarn_inv_freq` on the wrapper instead.
3. Applying `attention_scaling` after the framework cast cos/sin to bf16 caused sin error ratio of 13.7x. Fixed by reimplementing forward inline, applying scaling in fp32 before the cast.

**Result:** Error ratio 1.0000 for both cos and sin after patch.

---

## Reference Example: GPT-OSS RMSNorm

**Failing test:** `test_00_rmsnorm.py` — error ratio 1.85x (hidden by relaxed 2.0 tolerance).

**Root cause:** Precision ordering (Pitfall 5, Case A). Neuron `StandardRMSNorm` casts to `torch.bfloat16` before `pow(2).mean()`. HF `GptOssRMSNorm` casts to `torch.float32`. Computing variance in bf16 loses precision due to 7-bit mantissa.

**Patch:** `rmsnorm_patch.py` — patches the `get_rmsnorm_cls()` factory function (not the class directly, since it creates a new class per call). The patched forward casts to `float32` matching HF.

**Pitfall encountered:** Patching the class returned by one `get_rmsnorm_cls()` call didn't affect subsequent calls because the function defines a new class each time. Fixed by patching the factory function itself.

**Result:** Error ratio 1.0000. Default 1.2 tolerance restored.

---

## Reference Example: GPT-OSS MLP/MoE

**Failing test:** `test_06_mlp_moe.py` — error ratio 1809x. `test_05_experts.py` — error ratio 1774x.

**Root cause:** Three bugs:

1. `NeuronGptOssExperts.forward()` ignores `router_indices` and `routing_weights` — all tokens go through all 32 experts instead of top-4.
2. `NeuronGptOssMLP.forward()` uses `router_scores.sum(dim=-1)` as multiplier — softmax scores sum to ~1.0, making it a no-op.
3. Down-proj weight layout mismatch (`i*E+e` vs `e*I+i` indexing) and per-expert biases collapsed to single sum.

**Patch:** `mlp_moe_patch.py`:

1. Extracts per-expert weights from flattened RowParallelLinear via reshape + permute
2. Computes per-expert down projections via `einsum('nei,eih->neh')`
3. Applies routing weights per expert, then sums across experts
4. `store_per_expert_down_bias()` helper preserves per-expert biases after weight copy

**Pitfall encountered (Pitfall 5, Case B):** Initial patch used `.float()` on einsum operands, biases, and routing weights. This produced error ratio 0.82 (< 1.0) — the target was artificially closer to fp32 reference than the bf16 HF reference. The HF `GptOssExperts.forward()` has zero `.float()` calls. Fixed by removing all `.float()` calls so computation stays in native dtype.

**Result:** Error ratio ~0.94 (within 1.2 threshold). Both test_05 and test_06 pass.

---

**Skill Version:** 2.1
**Based on:** GPT-OSS debugging sessions (February 2026) — YaRN RotaryEmbedding, RMSNorm, MLP/MoE
