---
name: neuron-dev-contribution-validation
description: |
  Validate agentic artifact contributions (skills, agents, hooks, tools) to the
  NeuronAgenticDevelopment (NAD) package against the contribution model requirements.
  Use when reviewing a CR for NAD, checking contribution readiness, or asking
  "does this contribution meet NAD requirements", "validate my NAD contribution",
  "review CR for contribution model compliance".
---

# NAD Contribution Validation

Validate contributions to the NeuronAgenticDevelopment package against the contribution model.

## Activation

When asked to validate a NAD contribution, review a CR for NAD compliance, or check if artifacts meet contribution requirements:

1. Identify the artifact(s) being contributed (skills, agents, hooks, tools, MCP)
2. Determine the release category (internal or external) from `NEURON_METADATA.md`
3. Run through the validation checklist below
4. Report results as PASS/FAIL per item with actionable guidance for failures

## Validation Checklist

### 1. Namespace Compliance

- [ ] Artifact name uses a valid namespace prefix
- [ ] Valid namespaces: `neuron-nki`, `neuron-framework`, `neuron-core`, `neuron-infra`, `neuron-dev`, `neuron-training`
- [ ] `experimental-` prefix used only for single-team internal artifacts
- [ ] Name is unique — no existing artifact with the same name in the repository

### 2. Skill Structure (if contributing a skill)

- [ ] `SKILL.md` exists with valid YAML frontmatter (`name`, `description` required)
- [ ] `name` field: lowercase, hyphens only, max 64 chars, matches directory name
- [ ] `description` field: non-empty, max 1024 chars, describes what and when
- [ ] `NEURON_METADATA.md` exists with required fields
- [ ] Directory follows the structure: `skills/<namespace>-<skill-name>/`

### 3. NEURON_METADATA.md Validation

- [ ] Contains YAML frontmatter with all three required fields:
  - `owner_resolver_group` — valid resolver group name
  - `release_category` — either `internal` or `external`
  - `master_repository_uri` — valid repository URI
- [ ] `release_category` matches the intended audience

### 4. Agent Structure (if contributing an agent)

- [ ] Claude Code YAML spec exists in `agents/` directory
- [ ] Kiro JSON spec exists in `agents/` directory
- [ ] Kiro spec `systemPrompt` references Claude Code markdown via aim:include directive
- [ ] Both specs reference the same set of skills
- [ ] Agent name uses correct namespace prefix

### 5. AIM Benchmark Tests

Determine minimum count based on `release_category`:
- **Internal**: ≥ 1 test, 100% pass@k
- **External**: ≥ 5 tests, 100% pass@k

Validate:
- [ ] Test scenario files exist in `tests/<namespace>-<agent-name>/`
- [ ] Each test has: `taskId`, `input`, `expectedOutput`, `metadata`
- [ ] Tests use `expectedStrings`/`forbiddenStrings` for determinism
- [ ] Test is registered in `tests/registry.json`
- [ ] Registry entry has: `name`, `version`, `datasetPath`, `defaultJudge`, `taskIdSubset`
- [ ] `taskIdSubset` lists all contributed test IDs
- [ ] Minimum test count met for the release category

### 6. Evaluation Tests

Determine minimum count based on `release_category`:
- **Internal**: ≥ 1 eval, 70% pass@k
- **External**: ≥ 5 evals, 70% pass@k

Validate:
- [ ] Eval artifacts exist in `evals/<artifact-type>/<artifact-name>/`
- [ ] `data/` subdirectory contains evaluation results
- [ ] `test/` subdirectory contains test execution script
- [ ] Minimum eval count met for the release category

### 7. External-Only Requirements (if `release_category: external`)

- [ ] No `experimental-` prefix used
- [ ] No dependencies on internal or IP-sensitive artifacts

### 8. CR Review Requirements

- [ ] At least 1 reviewer from the owning team
- [ ] Neuroboros reviewers: 1 for existing artifacts, 2 for new submissions
- [ ] External artifacts: 2 Neuroboros reviewers required

## Output Format

Present results as:

```
## NAD Contribution Validation Report

**Artifact**: <name>
**Type**: <skill|agent|hook|tool|mcp>
**Release Category**: <internal|external>
**Overall**: PASS | FAIL (N issues)

### Results

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Namespace compliance | ✅ PASS | |
| 2 | Skill structure | ✅ PASS | |
| ... | ... | ... | ... |

### Issues (if any)
1. <description of issue and how to fix>
```

## Reference

For the full contribution model details, see [references/contribution-model.md](references/contribution-model.md).
