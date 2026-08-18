# Stage 0: Structural Scaffolding

> **Your role:** You are a scaffolder. Run the bundled scripts, compare the output trees, and produce `component_mapping.json`. Do NOT write tree generation code or modify model source files.

Build model trees, construct component mapping, define alignment functions.

**Use the bundled `scripts/run_stage0.py` and `scripts/stage0_scaffolding.py` — do NOT write your own tree generation code.**

## Required Outputs

```
{EXP_DIR}/
├── model_tree/
│   ├── model_tree_source.json           # Compressed source tree
│   ├── model_tree_source_full.json      # Full (uncompressed) source tree
│   ├── model_tree_source_pretty.txt     # ASCII pretty-print
│   ├── model_tree_source_flat_paths.txt # Flat list of module paths
│   ├── model_tree_target.json           # Compressed target tree
│   ├── model_tree_target_full.json      # Full (uncompressed) target tree
│   ├── model_tree_target_pretty.txt     # ASCII pretty-print
│   └── model_tree_target_flat_paths.txt # Flat list of module paths
└── component_mapping.json               # Component mapping with reasoning
```

## Step 0.1 — Build Model Trees

```bash
source {VENV}/bin/activate
PYTHONPATH={SCRIPTS_DIR} python3 {SCRIPTS_DIR}/run_stage0.py \
  --source-model-path {SOURCE_MODEL_PATH} \
  --target-model-path {SOURCE_MODEL_PATH} \
  --target-module-file {TARGET_MODELING_FILE} \
  --target-inner-class {TARGET_INNER_CLASS} \
  --target-config-class {TARGET_CONFIG_CLASS} \
  --output-dir {EXP_DIR}/model_tree
```

The script generates all 8 tree artifacts and prints both trees. The target tree is built in CPU mode (`NXD_CPU_MODE=1`) with TP=1.

**vLLM-Neuron targets — automatic environment check (by design, not a bug):** `run_stage0.py` builds the target tree through the stack adapter, and `get_adapter()` runs `check_environment()` first. For the `vllm_neuron` adapter this **deliberately exits early with an `EnvironmentError`** if any of these hold:

1. `vllm_neuron` is not importable — fix by prepending `{VLLM_NEURON_DIR}` to `PYTHONPATH` (see SKILL.md → *vLLM-Neuron Targets*).
2. The installed `vllm` framework is off the pinned `0.24` line.
3. The installed `vllm-neuron` plugin is off the pinned `0.24` line.

This is intentional fail-fast behavior — it prevents a cryptic mid-stage `AssertionError: Current vLLM config is not set.` from a version skew. If you see this `EnvironmentError`, **do not treat it as a skill failure**: read the printed message, fix the environment (correct `vllm`/`vllm-neuron` versions or `PYTHONPATH`), and re-run. The pinned versions live in `scripts/adapters/vllm_neuron.py` (`PINNED_VLLM_VERSION`, `PINNED_VLLM_NEURON_VERSION`).

**Note:** The target tree shows CPU-mode classes (e.g., `LlamaRMSNorm` instead of `CustomRMSNorm`). The module hierarchy is identical to device mode — only certain leaf classes differ. See [references/expected_structural_diffs.md](references/expected_structural_diffs.md).

## Step 0.2 — Build Component Mapping

Compare the two printed trees and build `{EXP_DIR}/component_mapping.json`.

Rules:
1. Start from leaf modules, work upward
2. Inspect source code to verify semantic equivalence (same name ≠ same function)
3. Support one-to-one and one-to-many mappings (e.g., fused QKV → split Q/K/V)
4. Use indexed variables: `layers.{i}.self_attn.q_proj` with `i ∈ [0..N-1]`
5. Every unmapped component needs explicit reasoning

See [references/mapping_example.json](references/mapping_example.json) for the output format.

## Step 0.3 — Detect CPU vs Device Class Divergence

Scan the target modeling file for components that use different classes in CPU mode vs device mode:

```bash
python3 {SCRIPTS_DIR}/detect_class_divergence.py \
  --target-module-file {TARGET_MODELING_FILE} \
  --output {EXP_DIR}/class_divergence_report.json
```

The script detects three patterns:
1. **Factory functions** (`get_rmsnorm_cls()`, `get_attn_cls()`) that return different classes based on `NXD_CPU_MODE`
2. **Conditional assignments** (`self.norm = ClassA() if cpu else ClassB()`)
3. **NKI kernel imports** (`CustomRMSNorm` on device vs `LlamaRMSNorm` on CPU)

For each divergence found, the report specifies which CPU class and device class are used, and recommends writing dual tests in Stage 2 — one for the CPU class (standard validation) and one reimplementing the device class's math in pure PyTorch (algorithm validation).

**Why this matters:** Stage 2 runs in CPU mode. If a component uses `LlamaRMSNorm` on CPU but `CustomRMSNorm` (NKI kernel) on device, Stage 2 only validates `LlamaRMSNorm`. The device class may have different precision behavior, and Stage 5 E2E will fail even though all Stage 2 tests passed.

## Step 0.4 — Define Alignment Functions

For mapped components with shape/layout differences (fused operators, transposed weights, SPMD shards), document the alignment transform needed for tensor comparison in the mapping file.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `'NoneType' has no attribute 'windowed_context_encoding_size'` | Config validation requires `neuron_config` | Pass `neuron_config` to `from_pretrained()` |
| `intra_layer_model parallel group is not initialized` | Parallel state not initialized | `run_stage0.py` handles this — if running manually, call `init_process_group("gloo")` then `initialize_model_parallel(tp=1)` |
| `Please initialize parallel processing via 'torchrun'` | `world_size > 1` without torchrun | Use `tp_degree=1, world_size=1` for structure inspection |
| `No module named 'modeling_xxx'` | Missing sys.path entry | Check `--target-module-file` path is correct |
| HF model type not recognized | Transformers version too old | Check `transformers.__version__` supports the model |
| `from_pretrained` fails on config class | Config class uses non-standard constructor | Script falls back to two-arg constructor automatically |
