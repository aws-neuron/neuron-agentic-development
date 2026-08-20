---
name: equiv-concept
description: Foundational concepts for verifying numerical equivalence between a reference neural network implementation and a target implementation, covering the 3-way comparison method, perturbation-based baseline estimation, error metrics, and distribution analysis.
---

# Equivalence Checking Concepts

This skill defines the foundational concepts and methods for verifying that a target neural network implementation produces numerically equivalent outputs to a reference implementation.

---

## Overview

### What Equivalence Means

Two neural network implementations are **numerically equivalent** when they produce outputs that are meaningfully close to each other, given the same input and precision settings. We do **not** expect bitwise-identical outputs because:

- **Floating-point non-associativity:** `(a + b) + c != a + (b + c)` in floating-point arithmetic. Different implementations may evaluate the same mathematical expression in a different order, producing different rounding.
- **Operation ordering:** Compiler optimizations, fused kernels, and hardware-specific execution paths change the order of operations. When suspecting compiler-related issues, analyze `log-neuron-cc.txt` before escalating.
- **Hardware differences:** Different devices (CPU, GPU, Trainium 1/2/3) have different floating-point units with different rounding behaviors and intermediate precision.

### Reference and Target Implementations

- **Reference implementation:** The trusted baseline, typically an existing framework implementation (e.g., HuggingFace Transformers, a PyTorch reference). Usually runs on CPU or GPU.
- **Target implementation:** The implementation under test (e.g., a NeuronX port for AWS Trainium). Can run on CPU (for logic validation) or on device (for final deployment validation).

### Precision Settings

Equivalence is evaluated under matching precision configurations. Common data types:

| Data Type | Description                                             |
| --------- | ------------------------------------------------------- |
| fp32      | 32-bit IEEE float — used as high-precision ground truth |
| bf16      | bfloat16 — 16-bit with 8-bit exponent, 7-bit mantissa   |
| mxfp8     | Microscaling FP8 — 8-bit with block scaling factors     |

The reference and target are compared under the **same** target precision (e.g., both bf16). The fp32 reference serves as a separate high-precision anchor.

---

## The 3 Runs

The comparison method requires three outputs. These can come from live execution or from pre-computed tensor files stored on disk.

| Run       | Implementation | Precision                            | Purpose                                                                                 |
| --------- | -------------- | ------------------------------------ | --------------------------------------------------------------------------------------- |
| **out_1** | Reference      | fp32                                 | High-precision ground truth (usually on CPU or GPU)                                     |
| **out_2** | Reference      | Target precision (bf16, mxfp8, etc.) | Quantization baseline (usually on CPU or GPU)                                           |
| **out_3** | Target         | Target precision (bf16, mxfp8, etc.) | The implementation under test, on a user-specified platform (CPU, Trainium 1/2/3, etc.) |

The method determines whether **out_3** is numerically equivalent to **out_2**, using **out_1** as the high-precision anchor. The specific hardware platform for run 3 is a user-specified parameter — the method itself is hardware-agnostic.

**Note:** Pre-computed outputs are sufficient. The comparison does not require live execution if tensor files are available.

---

## Method 1: FP32-Baseline 3-Way Comparison

**When to use:** The reference implementation can run in fp32.

### Intuition

When the reference runs in both fp32 and the target precision (e.g., bf16), the difference between out_2 and out_1 represents the **inherent error from precision reduction** — this is the unavoidable cost of using lower precision. The target implementation should not introduce significantly more error than this baseline.

### Error Ratio

```
error_ratio = ||out_3 - out_1|| / ||out_2 - out_1||
```

This is a simplification of the ratio of relative errors:

```
error_ratio = (||out_3 - out_1|| / ||out_1||) / (||out_2 - out_1|| / ||out_1||)
```

where `||out_1||` cancels out.

- `|| . ||` denotes the Frobenius norm (L2 norm of the flattened tensor)
- **Numerator:** How far the target output is from the fp32 ground truth
- **Denominator:** How far the reference at target precision is from the fp32 ground truth (the baseline precision error)

### Interpretation

- **Ratio ~ 1.0:** The target introduces no more error than the precision downgrade alone
- **Ratio <= 1.2:** The target is within acceptable tolerance
- **Ratio >> 1.2:** The target likely has an implementation bug

---

## Method 2: Perturbation-Based Baseline

**When to use:** The reference implementation **cannot** run in fp32 — only the target precision is available.

### Intuition

When fp32 execution is unavailable, we cannot directly measure the inherent precision error. Instead, we estimate a baseline error by perturbing the input by machine epsilon. If the target's deviation from the reference is comparable to the deviation caused by a tiny input perturbation, the implementations are numerically equivalent.

### Setup

Run the reference implementation twice with the same precision:

| Run       | Input                   | Output                       |
| --------- | ----------------------- | ---------------------------- |
| **out_1** | Original input X        | Reference @ target precision |
| **out_2** | Perturbed input X + eps | Reference @ target precision |
| **out_3** | Original input X        | Target @ target precision    |

### Perturbation Details

- **eps** is scaled to the machine epsilon for the data type:
  - bf16: `eps ~ 2^-7`
  - fp32: `eps ~ 2^-23`
- **Integer inputs (token IDs):** Do not perturb the integer indices directly. Instead, perturb the floating-point representation **after** the embedding lookup operation. This ensures the perturbation acts on the continuous representation that the network actually processes.

### Error Ratio

The same formula applies:

```
error_ratio = ||out_3 - out_1|| / ||out_2 - out_1||
```

Here the denominator measures how sensitive the reference output is to a machine-epsilon input perturbation. If the target's error is on the same scale, the two implementations are equivalent.

---

## Elementwise Error Distribution Analysis

Beyond the scalar error ratio, we examine the **distribution** of elementwise errors to gain deeper insight into whether two outputs are meaningfully close.

### Error Tensors

For any 3-way comparison (Method 1 or Method 2):

- **err\_{2,1}** = out_2 - out_1 (elementwise difference tensor between the baseline and ground truth)
- **err\_{3,1}** = out_3 - out_1 (elementwise difference tensor between the target and ground truth)

Each element in these tensors is treated as a **sample of numerical error**. If out*2 and out_3 are meaningfully close to each other, err*{2,1} and err\_{3,1} should follow the **same distribution**.

### Visual Tools

| Tool          | What to look for                                                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **QQ Plot**   | Plot quantiles of err*{3,1} against quantiles of err*{2,1}. Points should fall on the **45-degree line** if the distributions match. |
| **Histogram** | Overlay both error distributions. The shapes should **overlap closely**.                                                             |

The QQ plot is particularly informative: systematic deviations from the diagonal indicate a distributional shift (e.g., the target has heavier tails, a different mean, or a different variance), which points to a specific type of implementation error.

### Implementation Reference

`tensor_compare.py` in `.claude/skills/equiv-concept/scripts/` provides:

- `compare_3tensors(out_1, out_2, out_3)` — returns a 12-key dictionary of normwise and elementwise metrics with `_2_1` and `_3_1` suffixes
- `compare_2tensors(tensor1, tensor2)` — returns a 6-key dictionary for pairwise comparison
- `_visualize_differences_two_series(elem_diff1, elem_diff2, ...)` — generates overlaid histograms and QQ plots

---

## Pass/Fail Criteria

### Error Ratio Thresholds

| Error Ratio  | Interpretation                                                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| ~ 1.0        | Excellent — target matches reference precision error                                                                                 |
| <= 1.1 - 1.2 | Good — within acceptable tolerance                                                                                                   |
| 1.2 - 2.0    | Marginal — may be acceptable with documented justification (e.g., known precision ordering differences at higher tensor parallelism) |
| >> 1.2       | Fail — implementation bug likely                                                                                                     |

### Distribution Criteria

- QQ plot points should lie on the 45-degree line
- Histograms of err*{2,1} and err*{3,1} should overlap

A passing error ratio with a failing QQ plot (or vice versa) warrants further investigation — the normwise metric can mask localized outliers that the distributional analysis reveals.

---

## Related Skills

This concepts skill provides the theoretical foundation. The following execution skills implement the workflow:

| Step                        | Skill                                                   | Purpose                                                                                                                                                                   |
| --------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Environment setup           | `env-setup`                                             | Docker container with proper mounts and dependencies                                                                                                                      |
| Model structure analysis    | `build-model-tree`, `component-mapping`                 | Understand and map module hierarchies between reference and target                                                                                                        |
| Component-level testing     | `component-testing`                                     | Build bottom-up equivalence tests using the 3-way comparison                                                                                                              |
| CPU component debugging     | `cpu-component-debugging`                               | Diagnose and fix failing component tests on CPU via monkey patches                                                                                                        |
| Device component debugging  | `device-component-debugging`                            | Diagnose and fix failing component tests on device using XLA-compatible patches                                                                                           |
| Intermediate tensor capture | `tensor-capture`                                        | Capture tensors at specific layers for targeted comparison                                                                                                                |
| Device execution            | `enable-model-run`                                      | Compile and run the target on Neuron devices                                                                                                                              |
| Compiler issues             | (analyze `log-neuron-cc.txt`, then escalate externally) | Analyze `log-neuron-cc.txt` for errors/warnings first, then file a [Neuron SDK GitHub issue](https://github.com/aws-neuron/aws-neuron-sdk/issues) with reproduction steps |
