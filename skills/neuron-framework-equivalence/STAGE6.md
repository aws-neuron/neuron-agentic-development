# Stage 6: Distributional and Semantic Verification

> **Your role:** You are running scripts and recording results. This stage is combined with Stage 5 — see STAGE5.md. Do NOT debug failures.

**Stage 6 is now combined with Stage 5.** Run `run_teacher_forced_comparison.py` which computes both E2E R-ratio (Stage 5) and per-position KL + cosine (Stage 6) in a single teacher-forced pass.

See [STAGE5.md](STAGE5.md) for the combined instructions.

## Condition B: Semantic Consistency

Per-position cosine similarity: `cos(v_source, v_target) ≥ θ`

- p5 percentile must be ≥ θ (default 0.95)
- Tail fraction below θ must be ≤ ρ (default 0.05)

## Condition C: Distributional Equivalence

Per-position KL divergence: `D_KL(P_source ∥ P_target) ≤ δ`

- p95 percentile must be ≤ δ
- Maximum must be ≤ δ_max

**Threshold note:** Defaults need calibration from known-good ports (design doc Appendix E). Use `scripts/run_calibration.py` with outputs from multiple verified ports.
