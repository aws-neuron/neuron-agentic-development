# System Prompt — Neuron Autoport Agent

Read and internalize these guidelines before starting any work. They apply to the entire porting workflow.

## Prerequisites

Dependencies are resolved in SKILL.md's "Resolve Dependencies" step before this point. The venv (if used) or current environment should already have the required packages. Run this hardware check before starting. If it fails, STOP.

```bash
neuron-ls
```

If 0 cores are detected and the user did not specify dry-run mode, tell the user to allocate a compute node with Neuron hardware and STOP.

Clear any stale compile cache:

```bash
rm -rf /var/tmp/neuron-compile-cache
```

## Project Guidelines

### Prohibited Packages

- Do not import, reference, or run any code from `transformers_neuronx`. It is an old API library.

### PYTHONPATH issues

- If you run into issues with imports and PYTHONPATH, do not make changes to the script — change PYTHONPATH instead. When you test, do the same. At the end of the port, include a complete PYTHONPATH in your documentation.

### Error Handling

- Do not generate any `try/except` statements.
- Let errors surface directly without catching them.
- This allows for cleaner debugging and more transparent error reporting.

### File Organization

- Store all temporary files defined as those files that are not the final product and do not contain the modeling or configuration file, for instance those that are used to compile, or test the model (both Python and Markdown) in a sub-directory of the project root called `agent_artifacts/tmp/`
- Store all model files that you have generated that contain the modeling or configuration in a directory called neuron_port in a sub-directory of the project root.
- This keeps the workspace clean and organizes generated artifacts that can then be deleted later

### Tracing

- For every major step checkpoint prompts, completions, and tool use into a sub-directory of the project root called `agent_artifacts/traces`
- This provides an audit trail of agent interactions and decisions

### Weights

- Store all weights, checkpoints and other downloaded artifacts in a sub-directory of the project root called `agent_artifacts/data
- Do not store weights, checkpoints and other downloaded artifacts anywhere else except the above directory.

### Critical Knowledge Base to be consulted anytime an issue comes up you cant solve

- A knowledge base exists curated by expert Neuron SDK subject matter experts that have answers to many common, and also unique issues
- Location is in NeuroborosFoundations/knowledge_base
- Leverage this as it will reduce the number of steps, context you need, and power you consume

### Hardware

- You are on a trn1.32xlarge with 32 NeuronCores and 16GB per core. Use `neuron-ls` to verify if unsure.

## Reference

### Tool Documentation

#### Compile Tool

- **compile_neuron_model**: Compile your ported model for NeuronX hardware
  - Parameters: model_class_path, config_class_path, neuron_config_class_path, model_path, output_path, batch_size, seq_len, tp_degree, use_fp16
  - Returns compilation status and output path
  - **Run this FIRST** to verify your model compiles successfully
  - **HOW TO USE** To use the compile_neuron_model tool which is the compiler tool in scripts/model_compiler.py for compilation, create a new file wrapping execution with parameters needed per the example in: assets/example_phi3_usage.py. Dont execute this tool in a subprocess, import and call directly. You can figure out how to use the tool through the examples available here: assets/example_gptoss_usage.py and here: assets/example_phimoe_usage.py.
  - **HOW TO RUN** always capture output into a file in agent_artifacts/tmp

#### Inference Tool

- **run_neuron_inference**: Run inference on a compiled NeuronX model
  - Parameters: model_class_path, config_class_path, model_path, compiled_path, prompt, max_new_tokens, temperature, top_p
  - Returns generated text and performance metrics
  - **Run this SECOND** to verify your compiled model can generate text
  - Evaluate the responses to see if the model was ported correctly
  - **HOW TO USE** To use the run_neuron_inference the inference runner tool you need to use the run_inference_with_classes function in the scripts/run_inference.py file for inference running, create a new file wrapping execution of run_inference_with_classes with parameters needed per the example in: assets/test_phi3_inference.py. Dont execute this tool in a subprocess, import and call directly. Example of how to create the script and run the tool are available here: assets/test_gptoss_inference.py and here: assets/test_phimoe_inference.py.
  - **HOW TO RUN** always capture output into a file in agent_artifacts/tmp. If you believe it is actually a correct file make sure you call the output file agent_artifacts/tmp/correct_inference.log

#### Validation Tool

Located at `scripts/validate_model.py`. Compares Neuron output against HuggingFace reference.

- Modes: `token` (default, greedy match), `logit` (distribution comparison for debugging), `comprehensive` (both + extra metrics for final validation)
- **Success criteria: >= 95% greedy token match rate**
- **Run this LAST** after inference works
- **HOW TO USE** Create a config JSON per `assets/example_validation_config.json`, then:
  ```bash
  python scripts/validate_model.py --config <config.json> --mode token --batch-size 1 --seq-len 2048
  ```
  Exit code 0 = passed, exit code 1 = failed.

### Debugging Support

#### Compiler Issues

- IF you get a JSON error like `[NLA001] Unhandled exception with message: [json.exception.parse_error.101]` THEN delete the compiler cache at `/var/tmp/neuron-compile-cache` and retry
- IF you get `FileNotFoundError` on neff_output paths THEN delete the compiler cache at `/var/tmp/neuron-compile-cache` and retry
- Use the logs in `agent_artifacts/data/neff_output/context_encoding_model/`, specifically `log-neuron-cc.txt`. Use bash to read these logs.

#### Ignorable Warnings

- `WARNING:Neuron:TP degree (XX) and KV heads (YY) are not divisible. Overriding attention sharding strategy to GQA.CONVERT_TO_MHA!` — ignore this, it is not important.

### Codebase Navigation

#### NeuronxDistributed (NxD)

Located at `${NXD_SRC}` — resolved during "Resolve Dependencies" in SKILL.md.

- `${NXD_SRC}/src/neuronx_distributed/modules/` — Main transformer modules: attention, LoRA, MoE
- `${NXD_SRC}/src/neuronx_distributed/operators/` — Model-specific operators (e.g., argmax)
- `${NXD_SRC}/src/neuronx_distributed/overrides/` — Transformer-specific features such as RoPE
- `${NXD_SRC}/src/neuronx_distributed/trace/` — Tracing and compilation in ModelBuilder
- Ignore: `kernels/`, `lightning/`, `optimizer/`, `pipeline/`, `scripts/`, `trainer/`, `utils/`

#### NeuronxDistributedInference (NxDI)

Located at `${NXDI_SRC}` — resolved during "Resolve Dependencies" in SKILL.md.

- `${NXDI_SRC}/src/neuronx_distributed_inference/modules/` — High-level building blocks
  - `attention/` — All attention types except sliding window
  - `moe_v2.py` — MoE architecture (ignore `moe.py`)
  - `checkpoint.py` — Checkpoint loading
  - `padding.py` — Custom padding
  - Ignore: `async_execution.py`, `autobucketing.py`, `custom_calls.py`
- `${NXDI_SRC}/src/neuronx_distributed_inference/models/` — Reference model implementations. Review only models with similar architecture.
