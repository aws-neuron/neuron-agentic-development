# Stage 4: Debug and Patch (AGENTIC — only if Stage 3 finds faults)

> **Your role:** You are a debugger. This is the ONLY stage where you read source code, diagnose root causes, and write monkey patches. All fixes go in standalone patch files — never modify the original port.

Fix failing components with standalone monkey patches.

**Before debugging, READ the full reference:** [references/cpu-component-debugging.md](references/cpu-component-debugging.md) (17KB). This stage document is a summary. The reference contains the 6-category root cause classification, 5 detailed pitfalls with code examples, and 3 worked examples from GPT-OSS 20B.

## Error Magnitude Triage

Use this table for immediate triage when a component fails. (`run_stage3.py` automates this classification, but this table is essential for manual debugging context.)

| R-ratio range | Likely cause | Examples |
|---|---|---|
| 100x+ | Formula-level bug — wrong function or missing operation | YaRN scaling absent from RoPE, MoE routing ignored |
| 1–3x | Precision issue — dtype casting or operation ordering | Variance computed in BF16 instead of FP32, scaling applied after cast |
| < 1.0 | Over-precision — extra `.float()` calls not in reference | Target artificially closer to FP32 than the BF16 baseline |
| 1000x+ | Routing problem — wrong code path or CPU class running device logic | Factory function returning wrong class, dispatch path mismatch |

## Workflow

1. **Read** the fault localization output (component, R-ratio, root cause)
2. **Compare** HF and Neuron source code side-by-side, focusing on:
   - Config params consumed by HF but absent in Neuron
   - Operations present in one but absent in the other
   - dtype casting differences
3. **Write** a standalone monkey-patch file (never modify the original port)
4. **Re-run** the failing component test with the patch applied
5. **Verify** R drops to ≈ 1.0 (not just below 1.2)

## Patch Structure

```python
def apply_<name>_patch():
    from modeling_xxx import TargetClass
    if getattr(TargetClass, "_patched", False):
        return
    _original_init = TargetClass.__init__
    def _patched_init(self, config):
        _original_init(self, config)
        # Fix: compute corrected values
    def _patched_forward(self, *args, **kwargs):
        # Fix: use corrected computation
    TargetClass.__init__ = _patched_init
    TargetClass.forward = _patched_forward
    TargetClass._patched = True
```

## Common Pitfalls

1. **Config parameter gaps** — Neuron config missing fields HF config provides. Derive from known values.
2. **Precision ordering** — Scaling applied after BF16 cast instead of before. Reimplement forward inline.
3. **Buffer assignment** — `register_buffer("name", None)` resists direct assignment. Store on wrapper instead.
4. **Output shape conventions** — Match the target's shape format so downstream code works.
5. **Dtype must match reference exactly** — No extra `.float()`, no missing `.float()`.

## Loop

Repeat Steps 1-5 until all R < τ_R. If a patch fixes one module but breaks a downstream composite, it's an incomplete fix — re-run the full bottom-up suite.

## Detailed Debugging Guides

- [references/cpu-component-debugging.md](references/cpu-component-debugging.md) — Full CPU debugging workflow with pitfalls, examples, and patterns from real debugging sessions
- [references/device-component-debugging.md](references/device-component-debugging.md) — Device-specific debugging: XLA-compatible patch patterns (SPMDRank, index_select, _reduce), pre_shard_weights_hook injection, compiler log analysis (`log-neuron-cc.txt`), escalation to compiler debugging
- [references/device-e2e-debugging.md](references/device-e2e-debugging.md) — Device E2E debugging: 1-layer isolation technique, fix-compile-verify cycle, full model validation
- [references/cpu-e2e-debugging.md](references/cpu-e2e-debugging.md) — CPU E2E debugging: TP=1 FP32 baseline, mp.spawn patch inheritance, weight sharding pipeline, bias restoration
- [references/debugging-case-study-gptoss.md](references/debugging-case-study-gptoss.md) — Complete worked example from GPT-OSS 20B with specific error ratios, root causes, and patches
