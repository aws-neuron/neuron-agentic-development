# Neuron Agentic Development

This repository contains AI agents and skills for developing on [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/) (Trainium/Inferentia) hardware, including NKI kernel development, profiling and debugging. For an overview of Neuron Agentic Development and the tools it offers for agent-enabled workflows with Neuron, see [the overview of Neuron Agentic Development in the the public Neuron docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/agentic-development-overview.html).

## Installation

```bash
# from neuron pypi repository
pip install --upgrade neuron-agentic-development \
    --extra-index-url https://pip.repos.neuron.amazonaws.com


# from wheel
pip install neuron_agentic_development-1.0-py3-none-any.whl

# from local github clone
git clone https://github.com/aws-neuron/neuron-agentic-development.git
cd neuron-agentic-development

pip install .

# Then deploy to your preferred tool:

deploy-neuron-agentic-development-to-kiro
# or
deploy-neuron-agentic-development-to-claude
```

## Agents

| Agent                                                                            | Description                                                                                                                                                                                                                                                                      |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [neuron-nki-agent](agents/neuron-nki-agent.md)                                   | Unified NKI kernel development agent. Full lifecycle: writing kernels from PyTorch/NumPy/natural language, debugging compilation errors, profiling performance, optimizing bottlenecks, migrating between API versions, analyzing Perfetto traces, and NKI documentation lookup. |
| [neuron-nki-writer-agent](agents/neuron-nki-writer-agent.md)                     | NKI kernel authoring and modification. Translates from PyTorch/NumPy/natural language, adds shape/dtype support, refactors tiling strategies, and implements new features following Beta 3 API patterns.                                                                         |
| [neuron-nki-debugger-agent](agents/neuron-nki-debugger-agent.md)                 | Autonomous NKI kernel compilation error debugging. Analyzes compiler errors, searches documentation and code examples for fixes, applies corrections following simplicity over performance, and validates fixes.                                                                 |
| [neuron-nki-profile-analysis-agent](agents/neuron-nki-profile-analysis-agent.md) | Profile and analyze NKI kernels on Neuron hardware. Captures execution traces, computes performance bounds, identifies bottleneck engines, and runs investigations to localize inefficiencies to NKI source lines.                                                               |
| [neuron-framework-autoport-agent](agents/neuron-framework-autoport-agent.md)     | A model porting agent to port GPU-compatible models to functionally accurate implementation on Neuron. Executes the full porting workflow including architecture analysis, implementation, compilation, inference testing, and validation.                                       |

## Skills

| Skill                                                                        | Description                                                                                                                                                                                                                           |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [neuron-nki-writing](skills/neuron-nki-writing/SKILL.md)                     | Write and modify NKI kernels. Covers new kernel creation from PyTorch/NumPy/natural language, editing existing kernels, adding shape/dtype support, refactoring tiling strategies, and implementing new features.                     |
| [neuron-nki-debugging](skills/neuron-nki-debugging/SKILL.md)                 | Debug NKI compilation errors on Neuron hardware.                                                                                                                                                                                      |
| [neuron-nki-docs](skills/neuron-nki-docs/SKILL.md)                           | Research NKI documentation for API lookups, tutorials, error codes, and architecture details.                                                                                                                                         |
| [neuron-nki-profiling](skills/neuron-nki-profiling/SKILL.md)                 | Profile NKI kernels to analyze performance on Neuron hardware.                                                                                                                                                                        |
| [neuron-nki-profile-querying](skills/neuron-nki-profile-querying/SKILL.md)   | Query and analyze NKI kernel profile data from neuron-explorer parquet files via SQL and Python.                                                                                                                                      |
| [neuron-framework-autoport](skills/neuron-framework-autoport/SKILL.md)       | Port a GPU compatible model to NeuronX Distributed Inference for AWS Trainium/Inferentia. Handles the full workflow including architecture analysis, NeuronX implementation, compilation, inference testing, and accuracy validation. |
| [neuron-framework-equivalence](skills/neuron-framework-equivalence/SKILL.md) | Verifies functional equivalence between two implementations of the same model using a hierarchical algorithm.                                                                                                                         |

## Contributing

We are evaluating the external contribution process. All capabilities undergo internal verification to ensure technical accuracy, security, and architectural alignment. In the interim, we welcome feedback and feature requests via [Issues](https://github.com/aws-neuron/neuron-agentic-development/issues)

## License

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
