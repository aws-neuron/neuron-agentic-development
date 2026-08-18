# Stage 1: Smoke Tests (CODE)

> **Your role:** You are a test runner. Run the bundled script, record the output, and report pass/fail. Do NOT debug failures or read source code to investigate — just record the token match rate and proceed to Stage 2.

Verify the port is alive and produces coherent output. Greedy token matching against an
FP32 HuggingFace reference, computed in `run_stage1.py` itself.

> **vLLM-Neuron targets:** this script loads the adapter, so its automatic `check_environment()` runs first and **exits early with an `EnvironmentError`** on a `vllm`/`vllm-neuron` version skew or a missing `vllm_neuron` import. That is intended fail-fast behavior, not a skill bug — fix the environment and re-run. Details in [STAGE0.md](STAGE0.md) and SKILL.md.

## Required Outputs

```
{EXP_DIR}/results/
└── stage1.json    # Token match rate (overall + per prompt)
```

## Run

```bash
python3 scripts/run_stage1.py \
  --model-path ${HF_MODEL_PATH} \
  --compiled-model-path ${COMPILED_MODEL_PATH} \
  --model-class ${PORT_MODELING_FILE}:${PORT_CAUSAL_CLASS} \
  --config-class ${PORT_MODELING_FILE}:${PORT_CONFIG_CLASS} \
  --num-tokens 32 \
  --output ${EXP_DIR}/results/stage1.json
```

Requires `{SCRIPTS_DIR}` on PYTHONPATH. No external validation package is needed.

**NxDI targets only:** the compiled model is loaded through
`scripts/nxdi_compiled_loader.py`, which rebuilds the `NeuronConfig` from the compiled
directory's `neuron_config.json` using public NxDI API only. This loader does **not** apply
to vLLM-Neuron targets — that adapter runs inference through `vllm.LLM` instead.

## What It Does

- 10-prompt greedy token matching against an FP32 HF reference (`AutoModelForCausalLM`)
- Writes `token_matching` (overall + per-prompt match rate) and `passed`

> **No distribution metrics here.** Cosine similarity, KL divergence and top-1 agreement
> are Stage 5/6 metrics, produced by `run_teacher_forced_comparison.py` — see
> [STAGE6.md](STAGE6.md). Relative L2 error is not computed by any current script.

## Pass Criteria

Token match rate > 30% (liveness threshold). This is NOT a correctness test.

## Interpretation

- 100% match on most prompts with a few divergences → normal BF16 precision drift
- < 30% match → catastrophic failure, proceed to Stage 2 for localization
- High cosine similarity (> 0.95) with low token match → margin-sensitive divergence (expected)
