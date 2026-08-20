# Debug Orchestration Workflow

When equivalence validation (Stages 1–3, 5–7) reports failures, this document defines the systematic debugging escalation. It sequences through component-level to E2E, CPU to device, until all tests pass.

**Core principle:** Fix bugs bottom-up (components before E2E) and inside-out (CPU before device). Every stage gate uses the 3-tensor R-ratio as the pass/fail criterion.

---

## Escalation Workflow

```
Validation reports failures
    │
    v
Stage 1: Component Debugging (CPU)
    │   Reference: references/cpu-component-debugging.md
    │   Gate: all component tests pass on CPU (R ≤ 1.2)
    │
    v
Stage 2: Component Debugging (Device)
    │   Reference: references/device-component-debugging.md
    │   Gate: all component tests pass on device (R ≤ 1.2)
    │   Escalate: suspect a compiler issue if code matches but device diverges
    │
    v
Stage 3: CPU E2E Debugging
    │   Reference: references/cpu-e2e-debugging.md
    │   Gate: TP=1 FP32, TP=1 BF16, TP>1 BF16 all pass
    │
    v
Stage 4: Device E2E Debugging
    │   Reference: references/device-e2e-debugging.md
    │   Gate: full model R ≤ 1.2, top-1 token match
    │
    v
Stage 5: Re-run validation for clean report
```

---

## Decision Table

| Symptom                                                 | Go to                                                      |
| ------------------------------------------------------- | ---------------------------------------------------------- |
| Component test fails on CPU                             | Stage 1: cpu-component-debugging                           |
| Component passes CPU, fails device                      | Stage 2: device-component-debugging                        |
| Component code matches reference, device still diverges | Suspected compiler issue                                   |
| CPU E2E fails at TP>1                                   | Stage 3: cpu-e2e-debugging (check mp.spawn, weight layout) |
| Device E2E diverges from CPU E2E                        | Stage 4: device-e2e-debugging (1-layer isolation)          |
| Compilation fails                                       | Suspected compiler issue                                   |
| All stages pass                                         | Re-run validation, generate clean report                   |

---

## Stage Gates

| Stage                | Pass criterion                                                      |
| -------------------- | ------------------------------------------------------------------- |
| 1. CPU Components    | All component R ≤ 1.2 on CPU                                        |
| 2. Device Components | All component R ≤ 1.2 on device                                     |
| 3. CPU E2E           | TP=1 FP32 rel_fro < 1e-5, TP=1 BF16 R < 1.2, TP>1 BF16 R < 1.2      |
| 4. Device E2E        | Full model R ≤ 1.2, top-1 token match, coherent 20-token generation |

---

## Subagent Delegation Pattern

For Stage 4 debugging, which can span dozens of turns per failing component, use subagent delegation to manage context:

1. **Spawn one subagent per failing component** with:
   - The failing test output (component name, R-ratio, error)
   - Both reference and target source paths
   - The relevant debugging reference document
   - Instructions to produce a monkey-patch and verification result

2. **Parent agent collects results:**
   - Aggregate patches from all subagents
   - Re-run the full test suite with all patches applied
   - If new failures appear (patch broke a downstream composite), iterate

3. **Escalation from subagent:**
   - If a subagent determines the failure is device-specific or compiler-level, it returns that assessment to the parent for escalation to the next debugging stage

---

## Reference: GPT-OSS 20B Timeline

| Stage            | Duration | What was found                                                                          |
| ---------------- | -------- | --------------------------------------------------------------------------------------- |
| CPU Components   | 2 weeks  | 4 patches: rmsnorm precision, YaRN rotary, MoE routing + weight layout, attention sinks |
| CPU E2E          | 1 week   | mp.spawn patch inheritance, weight layout fix, bias restoration                         |
| Device E2E       | 2 weeks  | SPMDRank for ParallelEmbedding (590→127), windowed attention path (127→0.97)            |
| **Final result** |          | R = 1.0005, top-1 match, coherent generation                                            |

---

Based on: debug-equiv-workflow from Equivalence-1 (GPT-OSS 20B, Feb-Apr 2026)
