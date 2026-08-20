# NAD Contribution Model Reference

Complete validation criteria for contributions to the NeuronAgenticDevelopment package.

## Namespaces

| Namespace | Area |
|-----------|------|
| `neuron-nki` | NKI Library, Compiler, Optimization, Writing, Debugging |
| `neuron-framework` | NxDI, vLLM Neuron |
| `neuron-core` | Neuron Runtime, Collectives, Graph Compiler |
| `neuron-infra` | Kaizen, Slurm |
| `neuron-dev` | SDLC, CR |
| `neuron-training` | moduscope, TorchNeuronEager |

The `experimental-` prefix is for artifacts intended for limited usage by one internal team.

## Required Directory Structures

### Skill

```
skills/<namespace>-<skill-name>/
├── SKILL.md            # Required
├── NEURON_METADATA.md  # Required
├── scripts/            # Optional
├── references/         # Optional
├── assets/             # Optional
└── ...
```

### Agent

Agents live in `agents/` and require both formats:

**Claude Code** (`agents/<namespace>-<agent-name>.md` + YAML frontmatter):
```yaml
---
name: <namespace>-<agent-name>
description: |
  Description of what the agent does.
model: opus
color: green
tools: ["Read", "Write", "Edit", "Grep", "Glob", "Bash", "Task", "TodoWrite", "Skill"]
skills:
  - <namespace>-<skill-1>
  - <namespace>-<skill-2>
---
```

**Kiro** (`agents/<namespace>-<agent-name>.json`):
```json
{
  "schemaVersion": "1",
  "name": "<namespace>-<agent-name>",
  "config": {
    "description": "Description of what the agent does.",
    "systemPrompt": "{{ aim:include:agents/AGENT_NAME.md }}",
    "model": "claude-opus-4.6"
  },
  "dependencies": {
    "skills": {
      "skillNames": ["<namespace>-<skill-1>", "<namespace>-<skill-2>"]
    }
  },
  "clientConfig": {
    "kiroCli": {
      "tools": ["@builtin"],
      "allowedTools": ["fs_read", "fs_write", "grep", "glob", "execute_bash", "todo_list"],
      "resources": []
    }
  }
}
```

### AIM Benchmark Tests

```
tests/<namespace>-<agent-name>/
└── <test-id>.json
```

Test scenario schema:
```json
"scenarios": [
{
  "taskId": "<test-id>",
  "input": ["prompt text"],
  "expectedOutput": "LLM-as-judge scoring instructions",
  "expectedTools": { "required": [], "alternatives": {} },
  "metadata": {
    "category": "Category Name",
    "difficulty": "Easy|Medium|Hard",
    "useCase": ["what this test verifies"]
  },
  "expectedStrings": ["must appear in output"],
  "forbiddenStrings": ["must NOT appear in output"]
}
]
```

Registry entry in `tests/registry.json`:
```json
{
  "name": "<namespace>-<agent-name>",
  "version": "1.0",
  "judgeVersion": "2026-04-13",
  "description": "Description of test suite",
  "datasetPath": "./<namespace>-<agent-name>",
  "defaultJudge": "kiro-cli",
  "taskIdSubset": ["<test-id-1>", "<test-id-2>"]
}
```

### Evaluation Tests

```
evals/<artifact-type>/<artifact-name>/
├── data/
│   └── <eval-data>.json    # Input, output, accuracy scores
└── test/
    └── execute_tests.py    # Runs the skill and produces output
```

## NEURON_METADATA.md Schema

```yaml
---
owner_resolver_group: <ResolverGroupName>
release_category: internal | external
master_repository_uri: ssh://git.amazon.com/pkg/<Package>
---
```

## Requirements by Release Category

### Internal

| Requirement | Threshold |
|-------------|-----------|
| AIM benchmark tests | ≥ 1, 100% pass@k |
| Evaluation tests | ≥ 1, 70% pass@k |
| Product review | Not required |
| AppSec consultation | Not required |
| Open source review | Not required |
| Neuroboros CR reviewers | 1 (2 for new submissions) |

### External

| Requirement | Threshold |
|-------------|-----------|
| AIM benchmark tests | ≥ 5, 100% pass@k |
| Evaluation tests | ≥ 5, 70% pass@k |
| Neuroboros CR reviewers | 2 |
| No internal dependencies | Required |
| No `experimental-` prefix | Required |

## SKILL.md Frontmatter (agentskills.io spec)

Required fields:
- `name`: max 64 chars, lowercase + hyphens only, no leading/trailing/consecutive hyphens, must match directory name
- `description`: max 1024 chars, non-empty, describes what the skill does and when to use it

Optional fields:
- `license`, `compatibility`, `metadata`, `allowed-tools`

## CR Review Policy

- All CRs need ≥ 1 reviewer from the owning team
- New submissions to NAD require 2 Neuroboros reviewers
- Updates to existing artifacts require 1 Neuroboros reviewer
- External artifacts always require 2 Neuroboros reviewers

## Ongoing Maintenance Obligations

Owning teams commit to:
- Keep artifacts in sync with new Neuron releases
- Maintain AIM benchmarks at 100% pass@k
- Maintain evals at 70% pass@k
- Be responsive to CR requests
- Merge changes from source repo into NAD (pull model)
- Provide customer support for external artifacts
