# Getting Started with Neuron Agentic Development

Neuron Agentic Development provides a collection of AI agents and skills for developing custom NKI (Neuron Kernel Interface) kernels on AWS Trainium and Inferentia hardware. These capabilities equip coding agents in Kiro and Claude to author, debug, profile, and optimize NKI kernels — abstracting away the proprietary hardware knowledge required to work with Neuron.

## What are NKI Kernels?

NKI is the programming interface for writing custom compute kernels that run directly on AWS Trainium and Inferentia NeuronCores. NKI gives you fine-grained control over the hardware's tensor engines, vector engines, and DMA subsystem, enabling performance that exceeds what framework-level compilation alone can achieve.

Common use cases include fused attention kernels, custom normalization operations, quantized matrix multiplications, and any operation where the default compiler output leaves performance on the table.

## Prerequisites

| #   | Requirement                      | Details                                                  | Needed for                         |
| --- | -------------------------------- | -------------------------------------------------------- | ---------------------------------- |
| 1   | Trainium/Inferentia instance     | trn1, trn2, inf2 EC2 instance (AL2023 DLAMI recommended) | Compiling, profiling, optimization |
| 2   | Neuron SDK                       | `aws-neuronx-tools` (pre-installed on DLAMI)             | All on-device skills               |
| 3   | Python venv with Neuron packages | `neuronx-cc`, `torch-neuronx`, `neuron-explorer`         | Compilation, profiling, analysis   |
| 4   | Kiro or Claude Code              | Installed on the Trainium instance                       | Running agents and skills          |
| 5   | Anthropic API key                | For Claude model inference                               | Agent reasoning                    |

> **Important:** The agent runs on the same machine as the hardware. There is no laptop-to-remote-box file transfer — everything is co-located. Writing and documentation skills work anywhere (no hardware needed), but profiling and debugging require on-instance execution.

## Step 1: Launch and Verify Your Trainium Instance

Launch a `trn2.3xlarge` instance in `São Paulo (sa-east-1)` or `Melbourne (ap-southeast-4)` using the [Neuron Deep Learning AMI (DLAMI)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/dlami.html), then SSH into the instance.

Verify the Neuron devices are visible:

```bash
# Confirm Neuron devices are visible
neuron-ls

# Confirm neuron-explorer is available
which neuron-explorer && neuron-explorer --version
```

## Step 2: Activate Your Python Environment

The DLAMI comes with a pre-installed virtual environment. Activate it:

```bash
source ~/opt/aws_neuronx_venv_pytorch_2_9/bin/activate
```

## Step 3: Install Neuron Agentic Development

### Clone from GitHub (for customization or contribution):\*\*

```bash
git clone https://github.com/aws-neuron/neuron-agentic-development.git
cd neuron-agentic-development
pip install .
```

Then deploy to your preferred tool:

```bash
# For Kiro
deploy-neuron-agentic-development-to-kiro

# For Claude Code
deploy-neuron-agentic-development-to-claude
```

## Step 4: Install Kiro (if not already installed)

```bash
curl -fsSL https://cli.kiro.dev/install | bash
```

## Step 5: Start Using Neuron Agentic Development

```bash
kiro-cli chat --agent neuron-nki-agent
```

The `neuron-nki-agent` is the unified entry point. It automatically selects the right workflow based on your request and orchestrates the appropriate skills.

### Example Prompts

| What you want to do       | What to say                                    | Hardware needed?      |
| ------------------------- | ---------------------------------------------- | --------------------- |
| Write a new kernel        | "Write a fused softmax kernel for bf16 inputs" | No                    |
| Debug a compilation error | "Fix this kernel" (with error output)          | Yes                   |
| Profile a kernel          | "Profile my kernel and show me the metrics"    | Yes                   |
| Analyze a profile         | "What's the bottleneck in this kernel?"        | Yes (neuron-explorer) |

## Skills

The package provides five specialized skills that follow the natural kernel development pipeline: **write → debug → profile → analyze**.

| #   | Skill                         | Category      | Use when                                             |
| --- | ----------------------------- | ------------- | ---------------------------------------------------- |
| 1   | `neuron-nki-writing`          | Authoring     | Writing new kernels or modifying existing ones       |
| 2   | `neuron-nki-debugging`        | Debugging     | Resolving compiler errors or numerical mismatches    |
| 3   | `neuron-nki-docs`             | Documentation | Looking up API signatures, tutorials, error codes    |
| 4   | `neuron-nki-profiling`        | Profiling     | Capturing execution traces on hardware               |
| 5   | `neuron-nki-profile-querying` | Analysis      | SQL-based performance bounds and bottleneck analysis |

### Kernel Authoring (`neuron-nki-writing`)

Your starting point for creating NKI kernels. Translates PyTorch, NumPy, or natural language descriptions into correct NKI code. Covers tiling strategies respecting hardware constraints (e.g., 128 partition dimension, 512/4096 PSUM free dimension), memory access patterns, and efficiency guidelines for DMA sizing and SBUF reuse.

### Debugging (`neuron-nki-debugging`)

Systematic workflow for resolving NKI compilation and execution errors. Covers environment setup with correct `--target` flags, compiler error resolution with a categorized index of all 28 NCC error codes, and numerical validation against CPU-computed references.

### Profiling (`neuron-nki-profiling`)

Captures execution profiles on hardware: configures runtime inspection environment variables, runs the kernel, identifies the correct NEFF (Neuron Execution File Format), captures the trace with `neuron-explorer` including DGE notifications for DMA-level detail, and extracts JSON metrics.

### Profile Analysis (`neuron-nki-profile-querying`)

Ingests NEFF/NTFF files and runs SQL queries to compute performance bounds, identify bottleneck engines, and localize inefficiencies to specific NKI source lines. Supports three analysis approaches: the neuron-explorer API server, DuckDB directly on parquet, or Pandas for custom computation.

### Documentation (`neuron-nki-docs`)

Used across all stages of development. Provides API signatures and tutorials during authoring, explains error codes during debugging, and clarifies hardware architecture details during profiling. Ask about any `nisa.*` or `nl.*` API, look up error codes, find tutorials, or browse architecture guides for Trainium 1/2/3.

## Agents

| #   | Agent                               | Focus                | What it does                                                                                                                                                                 |
| --- | ----------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `neuron-nki-agent`                  | Full lifecycle       | Top-level entry point. Auto-selects the right workflow based on your request and orchestrates the appropriate skills.                                                        |
| 2   | `neuron-nki-writer-agent`           | Authoring            | Translates PyTorch, NumPy, or natural language descriptions into NKI code. Handles modifications to existing kernels.                                                        |
| 3   | `neuron-nki-debugger-agent`         | Debugging            | Autonomously analyzes compiler errors, searches documentation for fixes, and applies corrections. Tracks iterations (up to 10) and progressively simplifies when stuck.      |
| 4   | `neuron-nki-profile-analysis-agent` | Profiling + Analysis | Captures execution profiles on hardware, then runs SQL queries against profile data to compute performance bounds, identify bottleneck engines, and localize inefficiencies. |

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Your Trainium Instance                          │
│                                                  │
│  ┌──────────────┐    ┌───────────────────────┐  │
│  │ Claude Code  │───▶│ Neuron Agentic Dev    │  │
│  │ or Kiro      │    │  • write kernel       │  │
│  └──────────────┘    │  • compile + debug    │  │
│         │            │  • profile (NEFF→NTFF) │  │
│         │            │  • analyze (SQL/DuckDB)│  │
│         ▼            │  • optimize + iterate  │  │
│  Anthropic API       └───────────────────────┘  │
│  (LLM inference)              │                  │
│                               ▼                  │
│                  ┌────────────────────┐          │
│                  │ NeuronCores        │          │
│                  │ (compile + execute)│          │
│                  └────────────────────┘          │
└─────────────────────────────────────────────────┘
```

## Things to Know

- Profiling and debugging skills require execution on actual Trainium/Inferentia hardware. The writing and docs skills work anywhere.
- All skills target the current NKI Beta 0.3.0 API.
- Skills support Trainium 1 (gen2), Trainium 2 (gen3), and Trainium 3 (gen4) with appropriate `--target` flags.
- Skills and agents are designed to work together: the top-level agent automatically invokes profiling and debugging skills as needed.

## Sample Workflow: Custom Softmax Kernel

Here's a typical end-to-end workflow:

1. **Write** — Ask the agent: _"Write an NKI kernel that computes scaled softmax: softmax(x _ scale) along the last dimension, for input shape [batch, seq_len, hidden_dim] in bfloat16."\* The agent produces a complete kernel with proper tiling, hardware-accelerated exp, float32 accumulation, and bfloat16 output cast.

2. **Debug** — Ask the agent to run the kernel and verify numerical parity against a PyTorch reference. If compilation errors occur, the agent diagnoses and fixes them autonomously.

3. **Profile** — Point the agent at your kernel and ask it to profile. It compiles to a NEFF, benchmarks across multiple input shapes, and captures a hardware profile with `neuron-profile`.

4. **Analyze** — The agent runs a full bounds analysis on the captured profile and delivers a diagnosis: which engine is the bottleneck, what the stall rates are, and what optimization opportunities exist (e.g., DMA pipelining, double-buffering).

## Feedback

Give the Neuron Agentic Development capabilities a try and send feedback through [GitHub Issues](https://github.com/aws-neuron/neuron-agentic-development/issues) or your usual AWS Support contacts.
