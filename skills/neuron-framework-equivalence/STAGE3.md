# Stage 3: Fault Localization (CODE — only if Stage 2 fails)

> **Your role:** You are an analyst. Run the bundled script and interpret its output. Do NOT write patches — just identify and rank the suspect components for Stage 4.

Analyzes the R-ratio curve from Stage 2 to identify where divergence originates and classify root cause.

## Automatic Activation

Stage 3 runs automatically when Stage 2 has failures. No manual action needed.

## Change-Point Detection

- **Spike**: High R at single point, returns to baseline → alignment artifact or transient error
- **Step**: High R that persists for subsequent points → functional bug at that point, error propagates

The earliest step-pattern point is the primary fault candidate.

## Root-Cause Classification

| R magnitude | Likely cause                             | Examples                                                          |
| ----------- | ---------------------------------------- | ----------------------------------------------------------------- |
| R >> 10     | Missing algorithm or wrong formula       | YaRN scaling absent from RoPE, MoE routing ignored, wrong masking |
| 1.2 < R < 3 | Precision ordering or missing multiplier | Variance in BF16 instead of FP32, attention scaling omitted       |
| R < 1       | Over-precision (unintended FP32 upcast)  | Extra `.float()` call not in reference                            |

## Output

Ranked suspect components with: component name, R-ratio, divergence pattern (spike/step), root cause label, description, and the mapped module paths.
