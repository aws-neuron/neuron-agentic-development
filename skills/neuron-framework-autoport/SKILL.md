---
name: neuron-framework-autoport
description: |
  Port a HuggingFace model to NeuronX Distributed Inference for AWS Trainium/Inferentia.
  Use when the user says "port model", "autoport", "convert to neuron",
  "compile for trainium", or invokes /neuron-framework-autoport. Handles the full workflow:
  knowledge base analysis, architecture analysis, NeuronX implementation,
  compilation, inference testing, and accuracy validation (95%+ token match).
argument-hint: "ModelName, pathToModelImplementationDirectory, nameOfImplementationFile, nameOfConfigurationFile, huggingFaceModelID, pathToModelWeightsDirectory, pathToVenv (optional)"
---

# Model Porting

## Overview

This document provides the agent direct instructions on how to port a model from pytorch and running on NVIDIA GPU to pytorch/Neuron running on Trainium. The agent will be expected to follow these steps below.

## Pay attention to the success criteria, do not stop, or declare that you are done until you meet the complete success criteria

## Dry-run

When the user specifies `dry-run`:

- **Skip** the "Resolve Dependencies" step
- **Run** these commands to activate the venv and resolve source paths:
  ```bash
  export PATH=<pathToVenv>/bin:$PATH
  NXDI_SRC=$(python3 -c "import neuronx_distributed_inference; print(neuronx_distributed_inference.__path__[0])")
  NXD_SRC=$(python3 -c "import neuronx_distributed; print(neuronx_distributed.__path__[0])")
  TRANSFORMERS_SRC=$(python3 -c "import transformers; print(transformers.__path__[0])")
  ```
- **Do not run any code** — no compilation, inference, or validation (no Trainium hardware available)

## Before You Start

### Set User Invocation Directory

```bash
USER_INVOCATION_DIR="$(pwd)"
```

### Resolve Dependencies

Follow `references/setup_flow.md`. It handles venv validation, install consent, and exit code recovery. Do not proceed until it completes successfully.

After success, retain the 3 resolved paths from the script output for use throughout the workflow:

| Variable              | Description                                  |
| --------------------- | -------------------------------------------- |
| `${NXDI_SRC}`         | Path to NeuronX Distributed Inference source |
| `${NXD_SRC}`          | Path to NeuronX Distributed source           |
| `${TRANSFORMERS_SRC}` | Path to HuggingFace Transformers source      |

### Read Project Guidelines

READ `references/systemPrompts/systemPrompt.md` in this skill directory. It contains prerequisites (including version checks), project guidelines, tool documentation, debugging support, codebase navigation, and hardware context. Run all prerequisite checks and follow all guidelines throughout the workflow.

## Porting Parameters

Extract these six required parameters from the user's request before starting:

| Parameter                            | Description                                                                             |
| ------------------------------------ | --------------------------------------------------------------------------------------- |
| `ModelName`                          | The HuggingFace model class name (e.g., `ArceeForCausalLM`)                             |
| `pathToModelImplementationDirectory` | Path to the model source directory (e.g., `transformers/src/transformers/models/arcee`) |
| `NameOfImplementationFile`           | The modeling file name (e.g., `modeling_arcee.py`)                                      |
| `NameOfConfigurationFile`            | The configuration file name (e.g., `configuration_arcee.py`)                            |
| `huggingFaceModelID`                 | The HuggingFace model ID (e.g., `arcee-ai/AFM-4.5B-Base`)                               |
| `pathToModelWeightsDirectory`        | Path to store/load model weights (e.g., `agent_artifacts/data`)                         |

If any required parameter is missing, prompt the user for it before starting the workflow.

## Porting Workflow

### Step 1: Analyze a project repository of existing knowledge and code

The skill's `references/knowledge_base/` directory contains porting guides, known issues, and implementation patterns. The NeuronSDK source code (NxDI and NxD) is accessed via the resolved dependency paths `${NXDI_SRC}` and `${NXD_SRC}` from the "Resolve Dependencies" step. The HuggingFace Transformers source is at `${TRANSFORMERS_SRC}`. All directories not created by you should be read only.

#### Analyze Knowledge Base First

Review `references/knowledge_base/` for porting patterns and common pitfalls relevant to the target model's architecture. Refer to `references/knowledge_base/NEURONX_PORTING_GUIDE.md` for porting patterns and `references/knowledge_base/MODEL_IMPLEMENTATION_GUIDE.md` for common implementation patterns. Come back to `references/knowledge_base/` whenever you hit an issue during porting.

#### Analyze code bases second

Then analyze the NeuronSDK source at `${NXDI_SRC}` and `${NXD_SRC}`. Pay particular attention to the model implementations and modules directories. When analyzing NeuronxDistributed and NeuronxDistributedInference understand the architecture of each model, and the various mechanisms used to build each model. including differences in attention, embedding, mlp, mixture of experts, encoder, decoder, generation and sampling, and parallelization, and sharding, please describe all of the architectural details.

#### Reflect to me what you have learned and retain it for later use

Please give me an architectural description of each of the existing supported models and also identify common traits amongst models including relationships between one model and another.

### Step 2: Port the model from NVIDIA GPU to pytorch Neuron

Based on your understanding, including all of the existing available components in the Neuron SDK you have analyzed in directories NeuronxDistributed and NeuronxDistributedInference, please now analyze a CUDA specific implementation of {{ modelName }} in the project root sub-directory {{ pathToModelImplementationDirectory }} it contains the implementation of the model in file {{ NameOfImplementationFile }} and configuration of the model in {{ NameOfConfigurationFile }} and then create a version of this model that works based on the neuronx_distributed_inference framework in this repository. In the same directory will be a configuration file which will provide you the configuration you need. Please pay attention to the configurations, particularly the quantization, torch_dtype. Do a basic implementation of {{ modelName }} model using no sharding be thorough in your explanation, and ensure that the code produced is well documented and refers to the functions and files in the project root sub-directory {{ pathToModelImplementationDirectory }} directory where possible.

#### Component and approach instructions

You will need to reuse all of the existing Neuron components and where you cant you need to flag them in comments. Before concluding a component cannot be reused, check if the base class supports override hooks or accepts None for optional parameters. When you port this model from huggingface please keep the names for each component of the model consistent with huggingface. If you can not please include a \_u in the name to indicate that it does not have a 1:1 mapping. Implement it a component at a time and test the individual components before proceeding onto the next step, dont skip tests if distributed initialization or other features of the framework are required. For instance if there is a MLP, or attention component, implement the MLP component and test it first by doing a forward pass before going onto the Attention component. Always read the helper function's signature before calling it, and use only the parameters it accepts. Leverage the documents in the references/knowledge_base/ directory as a guide to ensure you avoid all the past mistakes. When you create the new code only leverage the NeuronxDistributed and NeuronxDistributedInference frameworks and pytorch, and do not use anything outside of the NeuronxDistributed and NeuronxDistributedInference. Do not modify any of the existing framework code. Do not pip install any additional packages. When you create the resulting ported implementation please place it in neuron_port sub-directory in the project.

### Step 3: Compile

Use the **compile_neuron_model** tool. Download weights using {{ huggingFaceModelID }} to {{ pathToModelWeightsDirectory }} if they don't exist. Pay attention to the configuration files of the downloaded weights and make sure they match the model configurations and the usage for neuron configurations. Also review examples in `${NXDI_SRC}/examples` and `${NXDI_SRC}/test` to understand how models are configured, compiled, deployed and tested. Use print statements instead of python logging. Store compiled output in `agent_artifacts/data`. If compilation fails, debug using the Debugging Support in `references/systemPrompts/systemPrompt.md` and `references/knowledge_base/` before proceeding. You also have access to known issues through the tickety-mcp server (search_tickets, get_ticket).

**Do not proceed to inference until compilation succeeds.**

### Step 4: Test inference

Use the **run_neuron_inference** tool. Make sure when you load model weights you leverage `convert_hf_to_neuron_state_dict` — see examples in `${NXDI_SRC}/models`. You must re-use the inference tool, not create your own script.

**Do not proceed to validation until inference produces coherent output.**

### Step 5: Validate

Validate against the HuggingFace golden reference using the **Validation Tool**:

1. Create a validation config JSON at `agent_artifacts/tmp/validation_config.json` based on `assets/example_validation_config.json`. Map all fields to match your ported model.
2. Run validation with the same batch_size and seq_len used during compilation:
   ```bash
   python scripts/validate_model.py --config agent_artifacts/tmp/validation_config.json --mode token --batch-size 1 --seq-len <same_as_compilation> 2>&1 | tee agent_artifacts/tmp/validation.log
   ```

**Success criteria: >= 95% greedy token match rate.** Exit code 0 means passed.

**Iteration loop if validation fails (<95% match rate):**

1. Fix code
2. Delete compiled model: `rm -rf agent_artifacts/data/compiled_model && rm -rf /var/tmp/neuron-compile-cache`
3. Re-compile
4. Re-validate

Once validation passes, save a final port summary to `agent_artifacts/traces/port_summary.md` covering: model name, HF ID, architecture decisions, TP degree, seq_len, compilation parameters, validation match rate, and any issues encountered during the port.

**GATE: Do NOT declare success until validation passes with >= 95% greedy token match. Do not stop, or declare that you are done until you meet the complete success criteria.**
