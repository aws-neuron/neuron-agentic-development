# Equivalence Report Template

Generate `{EXP_DIR}/EQUIVALENCE_REPORT.md` after all stages complete.

## Completion Gate (MANDATORY)

**Before generating the report, verify ALL of the following have concrete results — not "Pending":**

- [ ] Stage 1 smoke test ran with token match rate
- [ ] Stage 2 component tests ran on CPU with numeric R-ratios for all components
- [ ] Stage 3 fault localization ran (if any Stage 2 failures)
- [ ] Stage 4 debugging complete (if faults found) — all R < 1.2
- [ ] Stages 5+6 teacher-forced comparison ran with R-ratio, cosine, KL values
- [ ] Stage 7 downstream eval ran (if applicable)

**If any phase is missing:** go back and complete it. Do NOT generate a report with "Pending", "Not yet tested", "Requires X", or "Deferred" entries.

If a phase genuinely cannot complete (e.g., compilation fails), report the blocking error to the user and ask whether to generate a partial report.

## Report Structure

```markdown
# Equivalence Report: {MODEL_NAME}

## Executive Summary

- **Overall:** PASS / FAIL
- **Model:** {MODEL_NAME} ({num_params} parameters)
- **Source:** HuggingFace {MODEL_ID}
- **Target:** {TARGET_FRAMEWORK} implementation
- **Date:** {timestamp}

## Stage 1: Smoke Test

- Token match rate: {rate}%
- Cosine similarity (mean): {cos_mean}
- Verdict: {PASS/FAIL}

## Stage 2: Component-Level Results

| Component | R-ratio | Threshold | Result |
|-----------|---------|-----------|--------|
| rmsnorm | {r} | 1.2 | {PASS/FAIL} |
| embedding | {r} | 1.2 | {PASS/FAIL} |
| ... | ... | ... | ... |

- Components tested: {N}
- Passed: {P}/{N}
- Failed: {F}/{N}

## Stage 3: Fault Localization

{If Stage 2 had failures:}
- Primary fault: {component} (R={r}, pattern={spike/step})
- Root cause classification: {cause}

## Stage 4: Debug Summary

{If debugging was needed:}
| Patch | Component | R Before | R After |
|-------|-----------|----------|---------|
| {name} | {component} | {before} | {after} |

## Stages 5+6: E2E Teacher-Forced Comparison

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| R-ratio (p95) | {r} | 1.2 | {PASS/FAIL} |
| Cosine sim (p5) | {cos} | 0.95 | {PASS/FAIL} |
| KL divergence (p95) | {kl} | 0.1 | {PASS/FAIL} |
| Top-1 agreement | {pct}% | 50% | {PASS/FAIL} |

## Stage 7: Downstream Evaluation

{If run:}
| Task | HF Score | Target Score | Delta | Result |
|------|----------|-------------|-------|--------|
| {task} | {hf} | {target} | {delta} | {PASS/FAIL} |

## Artifacts

- Stage results: `{EXP_DIR}/results/`
- Component tests: `{EXP_DIR}/tests/`
- Patches (if any): `{EXP_DIR}/patches/`
```
