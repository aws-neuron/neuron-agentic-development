# Stage 5+6: E2E Comparison + Distributional + Semantic (CODE)

> **Your role:** You are running scripts and recording results. Do NOT debug failures — record them and proceed. If a metric fails, log the value and continue to the next prompt/position. Recommend Stage 4 at the end if failures exist.

Verify the assembled model with real weights under teacher forcing. Covers Stage 5 (E2E R-ratio), Stage 6 Condition B (cosine), and Condition C (KL) in a single pass.

> **vLLM-Neuron targets:** this script loads the adapter, so its automatic `check_environment()` runs first and **exits early with an `EnvironmentError`** on a `vllm`/`vllm-neuron` version skew or a missing `vllm_neuron` import. That is intended fail-fast behavior, not a skill bug — fix the environment and re-run. Details in [STAGE0.md](STAGE0.md) and SKILL.md.

## Required Outputs

```
{EXP_DIR}/results/
└── teacher_forced.json    # Per-position R-ratio, cosine, KL for all prompts
```

## What the Design Doc Requires

For each prompt and each **teacher-forced position t**:

- R-ratio on output logits (three-tensor: source FP32, source BF16, target BF16)
- Cosine similarity cos(v_source, v_target) ≥ θ (Condition B)
- KL divergence D_KL(P_source ∥ P_target) ≤ δ (Condition C)
- Top-k agreement (1/5/10)

**Teacher forcing** means: at each position t, all three models receive the same prefix tokens (from the source FP32 model's greedy output), so logits are compared under identical contexts.

## Run

```bash
PYTHONPATH={SCRIPTS_DIR} python3 {SCRIPTS_DIR}/run_teacher_forced_comparison.py \
  --model-path {SOURCE_MODEL_PATH} \
  --compiled-model-path {COMPILED_MODEL_PATH} \
  --model-class {TARGET_MODELING_FILE}:{TARGET_CAUSAL_CLASS} \
  --config-class {TARGET_MODELING_FILE}:{TARGET_CONFIG_CLASS} \
  --num-tokens 32 \
  --output {EXP_DIR}/results/teacher_forced.json
```

## Pass Criteria

- R-ratio p95 < τ_R (1.2)
- Cosine similarity p5 ≥ θ (0.95)
- Top-1 agreement > 50%

## Interpretation

- **Stage 2 all-pass + Stage 5 fail** → compilation-induced divergence (operator fusion, kernel numerics). Not a porting bug.
- **Stage 2 fail + Stage 5 fail** → porting bug propagates to E2E. Fix via Stage 4 first.
- **Condition B pass + C fail** → logit directions agree but probability mass differs. Threshold calibration needed.
- **Stage 5 fail, components clean** → unmapped component, compiler issue, or different execution path. Run `detect_class_divergence.py` first (see Stage 0, Step 0.3). If divergences found, go back to Stage 2 and write device-algorithm tests. If no divergences and CPU E2E also passes, this could be a compiler issue — first analyze `log-neuron-cc.txt` for compiler errors or warnings, then escalate externally (see escalation criteria in [references/device-component-debugging.md](references/device-component-debugging.md)).

## Tensor Capture for Per-Layer Comparison

For per-layer intermediate comparison (beyond final logits), use `templates/diagnostic_forward_template.py` which provides hook-based capture at named module boundaries.

Key rules:

- Strip the `model.` prefix from source model names to get consistent names across both sides
- Device captures pad input to full `seq_len` — slice to match: `device_tensor[:, :ref_seq_len, :]`
- `self_attn` on device captures `cos_cache`, not hidden_states — use `post_attention_layernorm` as attention quality proxy
- `embed_tokens` has baseline_err = 0 (lookup, no computation) → error_ratio = inf — check cosine similarity instead
- `lm_head` on device outputs `[1, 1, vocab]` (last position only) — compare `hf[:, -1:, :]` vs `device[:, :, :]`
- Device `TensorCaptureConfig` requires `OnDeviceSamplingConfig` — without it, captured tensors are silently not returned

See [references/dump-tensors.md](references/dump-tensors.md) for the full tensor capture methodology including HF hook-based capture, device `TensorCaptureConfig` setup, and fallback strategies.
