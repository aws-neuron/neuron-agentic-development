# NAD Skill Benchmark Gate

A CI gate that evaluates this repo's agent skills by running them through the
`kiro-cli` agent in Docker and scoring the output against per-scenario rubrics.
A PR that changes `skills/`, `agents/`, `tests/`, or this `benchmark/` grader is
scored against the repo's own assets.

## Layout (single-repo)

Everything the gate needs lives in this repo:

- **Inputs** (at the repo root, scored by the gate):
  - `skills/` — the skills under test
  - `agents/` — agent specs (`*.agent-spec.json`) listing each agent's `skillNames`
  - `tests/` — scenario definitions per subset + `tests/registry.json` (the source
    of truth for which scenarios are *active/gated*)
- **Grader** (this `benchmark/` directory):
  - `run_skill_tests.py` — loads scenarios, runs the agent in Docker, scores, gates
  - `shell/{docker,common}.sh` — build the image + run `kiro-cli` in Docker
  - `environment/Dockerfile` — the run image (`python:3.11-slim` + kiro-cli)

## How it runs

The gate scores the repo's own `skills/` + `agents/` + `tests/` by running
`benchmark/run_skill_tests.py --nad-root=.`. No cross-repo references or tokens.

**Every PR is gated on demand via a label** (`.github/workflows/benchmark-external.yml`):

- A maintainer reviews a PR and adds the **`run-benchmark`** label; that triggers
  the gate on the PR. Applying a label requires the repo's Triage role or above, so
  a fork contributor (no role on this repo) can't opt themselves in — nothing runs
  until a maintainer does. Pushing new commits does not re-run it — the maintainer
  must re-apply the label (re-review).
- It uses `pull_request_target` so the run can access the `KIRO_API_KEY` secret
  even for PRs from external forks (a plain `pull_request` run gets no secrets on
  forks). The person who labels a PR is vouching for it: the job runs the PR's
  contributed skill code with the key present, so only label PRs you've reviewed.
  Note that `contents: read` restricts only the `GITHUB_TOKEN` — it does **not**
  shield `KIRO_API_KEY`, which is exposed to the untrusted PR code, so review the
  diff before labeling.
- **Requires** on the repo: a `KIRO_API_KEY` Actions secret (authenticates both the
  agent and the LLM judge; use a service-account key) and a `run-benchmark` label.

The same workflow also supports **manual** `workflow_dispatch` runs from the Actions
tab (inputs: `subset`, `count`, `gate`). Manual dispatch is restricted to users with
write access, so it checks out the selected branch directly (trusted) rather than an
untrusted PR head.

## Scoring

Per scenario:

- `pattern_score` (75%) — proportional match of `expectedStrings` (present) and
  `forbiddenStrings` (absent), matched against the output file + the agent's response.
- `judge_score` (25%) — an LLM judge (`kiro-cli`) verdict against the scenario's
  rubric: `1.0` pass / `0.0` fail (or `null` if no rubric).
- `combined_score = 0.75*pattern + 0.25*judge` (pattern-only if the judge is off).
- `passed = combined_score >= 0.90` (per-scenario override via `metadata.baseline_k`).

The gate fails if the overall pass rate is below `--gate` (default 90%).

> **Note on the judge:** the LLM judge is non-deterministic, so a scenario with a
> perfect `pattern_score=1.00` can still score `judge=0` on an unlucky verdict,
> pulling `combined` to 0.75. The overall pass rate can therefore vary run-to-run
> near the threshold independent of any code change.

## Reliability handling

- **No infra-error retry.** A failure is reported as-is — including a timeout that
  produced no output, which typically reflects the *agent* going down a bad path
  (e.g. fetching a dead URL and stalling). This is skill behavior the benchmark
  should measure, not hide by retrying until it happens to pass. For deliberate
  multi-attempt runs use `--count` (transparent Pass@k).
- **Salvage-on-timeout:** if the agent times out but has already written a non-empty
  `output.py`, that output is scored instead of being discarded (the agent often
  finishes the task and then hangs on a trivial final step). This scores what the
  agent actually produced — it does not grant a fresh attempt. A timeout that
  produced no output is reported as an infrastructure error (`run_status:
  infra_error`), logged with its exception type + container-output tail.

## Running it

```bash
# On a PR (primary path): a maintainer adds the `run-benchmark` label to the PR.

# Manually via CLI (workflow_dispatch on benchmark-external.yml):
gh workflow run benchmark-external.yml -f subset=all -f count=1 -f gate=90

# Locally (needs Docker + KIRO_API_KEY):
python benchmark/run_skill_tests.py --nad-root=. --subset=all --count=1 --judge --gate=90
```

`--subset` accepts `all`, a single subset name (e.g. `neuron-nki-agent`), or a
comma-separated list. Results are written to `benchmark/build/nad_benchmark_results.json`.
