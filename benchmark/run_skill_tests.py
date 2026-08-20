"""Run all NeuronAgenticDevelopment skill tests using skills-benchmarks framework.

Each scenario runs kiro-cli inside Docker (same flow as Claude in skills-benchmarks).
Parallelized with ProcessPoolExecutor — one Docker container per scenario.

Usage:
    python run_skill_tests.py --subset=neuron-nki-agent --count=2 --workers=4
    python run_skill_tests.py --subset=all --count=1 --judge   # all subsets
    python run_skill_tests.py --task-id=beta3-no-platform-target --count=1
    python run_skill_tests.py --subset=neuron-nki-agent --count=1 --judge
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# NAD benchmark inputs (tests/, skills/, agents/) live at the repo root and are
# resolved via --nad-root (default ".", handled in main via set_nad_root). In this
# single-repo layout the gate scores the repo's own assets, so --nad-root=. points
# at the checkout root. --nad-root can also point at any other NAD checkout to
# score that repo's native assets instead.
#
# FRAMEWORK_DIR (shell/, environment/Dockerfile, build/) is SEPARATE and always anchored to
# this file — only the three input dirs move behind --nad-root.
PROJECT_DIR = Path(__file__).parent.parent
TESTS_DIR = PROJECT_DIR / "tests"
SKILLS_DIR = PROJECT_DIR / "skills"
AGENTS_DIR = PROJECT_DIR / "agents"
FRAMEWORK_DIR = Path(__file__).parent
DOCKER_SH = FRAMEWORK_DIR / "shell" / "docker.sh"


def set_nad_root(nad_root: str) -> None:
    """Point the input paths (tests/, skills/, agents/) at a NAD checkout.

    Reassigns the module-level input dirs so both the main process and forked
    worker processes (ProcessPoolExecutor uses fork on Linux) see the same root.
    Call this BEFORE run_all() creates the executor. FRAMEWORK_DIR is unaffected.
    """
    global PROJECT_DIR, TESTS_DIR, SKILLS_DIR, AGENTS_DIR
    PROJECT_DIR = Path(nad_root).expanduser().resolve()
    TESTS_DIR = PROJECT_DIR / "tests"
    SKILLS_DIR = PROJECT_DIR / "skills"
    AGENTS_DIR = PROJECT_DIR / "agents"

PATTERN_WEIGHT = 0.75
JUDGE_WEIGHT = 0.25
DEFAULT_BASELINE_K = 0.90

# NOTE: There is deliberately NO infra-error retry. A timeout that produced no
# output.py reflects the AGENT going down a bad path (e.g. fetching a 404 URL,
# wandering, hanging) — that is skill behavior the benchmark should MEASURE, not
# mask by retrying until it happens to succeed. Retrying would turn "this skill
# fails ~1 in N runs" into a false 100% pass. Completed-but-hung runs are instead
# recovered honestly by the salvage path in _run_scenario_once (score the output
# the agent actually wrote). For deliberate multi-attempt runs, use --count
# (Pass@k), which is transparent in the reported metric.

# Max chars of the submitted file handed to the LLM judge. The old 4000 cap
# silently truncated large outputs (e.g. a ~25K-char autoport modeling file),
# so the judge only saw the first ~16% and reported the file "cut off" — a
# harness artifact, not a real failure. Set well above the largest expected
# submission so the judge sees the whole file.
JUDGE_SOURCE_LIMIT = 40000


def load_registry_active() -> dict:
    """Map subset name -> set of active (gated) taskIds, from registry.json.

    registry.json is the source of truth for which scenarios are ACTIVE. Some
    subsets have extra fixtures on disk that are not gated (e.g. autoport ships 4
    *.json but only 1 is active), so globbing files would over-count. Returns {}
    if the registry is missing (callers then fall back to all on-disk files).
    """
    reg_file = TESTS_DIR / "registry.json"
    if not reg_file.exists():
        return {}
    reg = json.loads(reg_file.read_text())
    return {s["name"]: set(s.get("taskIdSubset", [])) for s in reg.get("subsets", [])}


def load_scenarios(subset: str = None, task_id: str = None) -> list[dict]:
    """Load scenarios, optionally filtered.

    subset may be None/"all" (every subset), a single subset name, or a
    comma-separated list. Scenarios are restricted to the registry's active
    taskIds unless a specific --task-id is requested (which allows loading
    inactive fixtures for debugging).
    """
    # None / "" / "all" means every subset; otherwise accept a comma-sep list.
    wanted = None
    if subset and subset.strip().lower() != "all":
        wanted = {s.strip() for s in subset.split(",")}

    # Only gate to registry-active scenarios for normal runs; an explicit
    # --task-id may target an inactive fixture, so skip the filter in that case.
    active = load_registry_active() if not task_id else {}

    scenarios = []
    for subset_dir in sorted(TESTS_DIR.iterdir()):
        if not subset_dir.is_dir() or subset_dir.name.startswith("."):
            continue
        if wanted is not None and subset_dir.name not in wanted:
            continue
        active_ids = active.get(subset_dir.name)
        for json_file in sorted(subset_dir.glob("*.json")):
            data = json.loads(json_file.read_text())
            for s in data.get("scenarios", []):
                # Registry is the source of truth for active/gated scenarios.
                if active_ids is not None and s["taskId"] not in active_ids:
                    continue
                s["_subset"] = subset_dir.name
                scenarios.append(s)

    if task_id:
        task_ids = [t.strip() for t in task_id.split(",")]
        scenarios = [s for s in scenarios if s["taskId"] in task_ids]

    return scenarios


def get_agent_skills(agent_name: str) -> list[str]:
    """Get skill names from agent spec."""
    spec_file = AGENTS_DIR / f"{agent_name}.agent-spec.json"
    if not spec_file.exists():
        return []
    spec = json.loads(spec_file.read_text())
    return spec.get("dependencies", {}).get("skills", {}).get("skillNames", [])


def setup_workspace(skills: list[str], dockerfile_dir: Path) -> Path:
    """Create temp workspace with skills and Dockerfile for Docker execution."""
    test_dir = Path(tempfile.mkdtemp(prefix="skill_eval_"))

    # Install skills into .kiro/skills/ — copy full directory (references, examples, etc.)
    for skill_name in skills:
        skill_src = SKILLS_DIR / skill_name
        if not skill_src.exists():
            continue
        skill_dst = test_dir / ".kiro" / "skills" / skill_name
        shutil.copytree(skill_src, skill_dst)

    # Copy Dockerfile into test_dir (required by docker.sh build)
    shutil.copy(dockerfile_dir / "Dockerfile", test_dir / "Dockerfile")

    return test_dir


def run_kiro_in_docker(test_dir: Path, prompt: str, timeout: int = 120) -> str:
    """Run kiro-cli inside Docker container. Returns stdout+stderr."""
    cmd = ["bash", str(DOCKER_SH), "run-kiro", str(test_dir), prompt, "--timeout", str(timeout)]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout + 30, env=os.environ.copy(),
    )
    return result.stdout + result.stderr


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text.

    Covers SGR color codes (…m) AND cursor/mode codes like \\x1b[?25l (hide cursor,
    emitted per spinner frame) — the latter otherwise flood the log as '25l25l25l…'.
    """
    import re
    return re.sub(r'\x1b\[[?0-9;]*[a-zA-Z]', '', text)


def extract_agent_response(raw_transcript: str) -> str:
    """Extract the agent's own response prose from a kiro-cli transcript.

    kiro-cli prefixes the assistant's narration lines with "> "; everything else
    is tool output (files it read, shell results, skill reference docs). We match
    expectedStrings/forbiddenStrings against the agent's response ONLY — never the
    tool output — mirroring the internal harness, which matches its `actualOutput`
    (the agent's final response), not what the agent read.

    This distinction is load-bearing: 35 of 42 NKI forbiddenStrings (deprecated
    APIs like `nl.load`, `neuronxcc`, `mask=`) ALSO appear in the skill reference
    docs the agent reads. Matching the whole transcript would false-flag those as
    violations. Matching the "> " response lines does not.
    """
    text = _strip_ansi(raw_transcript)
    lines = [ln.strip()[2:] for ln in text.splitlines() if ln.strip().startswith("> ")]
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict | None:
    """Extract the first balanced {...} JSON object from arbitrary text.

    kiro-cli may pretty-print the verdict across multiple lines or wrap it with
    surrounding prose/footer, so neither json.loads(whole) nor a single-line
    scan is reliable. Walk the text tracking brace depth (ignoring braces inside
    strings) and try to parse each complete top-level object.
    """
    start = None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
            continue
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start = None  # not valid; look for the next object
    return None


def score_judge(source: str, rubric: str, dockerfile_dir: Path) -> tuple[float, str, str]:
    """Score via LLM judge using kiro-cli in Docker. Returns (score, reason, status).

    The judge runs through the same Docker path as the agent (docker.sh run-kiro)
    so it shares one kiro-cli install and the KIRO_API_KEY auth passthrough — the
    host runner does NOT need kiro-cli installed. The judge workspace has NO skills
    loaded, so it is a raw LLM call (no agent context).

    status is one of:
      "ok"      — a real verdict was parsed (score is 0.0 or 1.0)
      "skipped" — no rubric to judge against
      "error"   — the judge could not run or returned no parseable verdict
    """
    if not rubric:
        return 0.5, "no rubric", "skipped"

    prompt = (
        "You are an expert code evaluator. Judge whether the following code meets the rubric.\n\n"
        f"## Rubric\n{rubric}\n\n"
        f"## Code to evaluate\n```python\n{source[:JUDGE_SOURCE_LIMIT]}\n```\n\n"
        'Respond ONLY with JSON, no other text: {"passed": true/false, "reason": "brief explanation"}'
    )

    judge_dir = None
    try:
        # Empty workspace (no skills) so the judge is a plain LLM call, not an agent.
        judge_dir = setup_workspace([], dockerfile_dir)
        raw = run_kiro_in_docker(judge_dir, prompt, timeout=60)

        # Strip ANSI color codes and kiro-cli prompt markers before parsing
        output = _strip_ansi(raw).strip()
        cleaned_lines = []
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("> "):
                line = line[2:]
            cleaned_lines.append(line)
        output = "\n".join(cleaned_lines)

        # kiro-cli may pretty-print the JSON across lines or wrap it in prose,
        # so extract the first balanced {...} object rather than assuming it is
        # the whole output or on a single line.
        verdict = _extract_json_object(output)

        if verdict is None:
            return 0.5, f"no JSON in output: {output[:80]}", "error"

        passed = verdict.get("passed", False)
        reason = verdict.get("reason", "")
        return (1.0 if passed else 0.0), reason, "ok"
    except subprocess.TimeoutExpired:
        return 0.5, "judge timeout", "error"
    except Exception as e:
        return 0.5, f"judge error: {str(e)[:50]}", "error"
    finally:
        if judge_dir is not None:
            shutil.rmtree(judge_dir, ignore_errors=True)


def score_patterns(haystack: str, expected: list[str], forbidden: list[str]) -> tuple[float, list, list, list]:
    """Score expected/forbidden string matches, case-insensitively.

    `haystack` is what we match against — see evaluate(): the submitted file
    PLUS the agent's response prose (extract_agent_response, not raw tool output).
    Matching is against the agent's final response text (case-insensitively), not
    raw tool output.
    Some expectedStrings are not source tokens at all — e.g. the autoport
    scenario expects `neuron_port/modeling_`, an output *path* the agent names
    in its narration but that never appears inside the .py file. Matching the
    file alone caps such scenarios below pass; matching file+transcript does not.
    """
    if not haystack:
        return 0.0, [], expected, []
    total = len(expected) + len(forbidden)
    if total == 0:
        return 1.0, [], [], []
    hay = haystack.lower()
    hits = [p for p in expected if p.lower() in hay]
    misses = [p for p in expected if p.lower() not in hay]
    violations = [p for p in forbidden if p.lower() in hay]
    score = (len(hits) + (len(forbidden) - len(violations))) / total
    return score, hits, misses, violations


def evaluate(source: str, scenario: dict, use_judge: bool = False, dockerfile_dir: Path = None,
             agent_response: str = "") -> dict:
    """Evaluate output against scenario spec.

    `source` is the submitted file (output.py); `agent_response` is the agent's
    own response prose (see extract_agent_response — NOT the raw transcript).
    Pattern matching (approach "a") runs over BOTH so path-style expectedStrings
    that appear only in the agent's narration still match, while the LLM judge
    scores the SUBMITTED FILE only (the rubric explicitly judges "the final
    Python file the agent pasted").
    """
    expected = scenario.get("expectedStrings", [])
    forbidden = scenario.get("forbiddenStrings", [])
    baseline_k = scenario.get("metadata", {}).get("baseline_k", DEFAULT_BASELINE_K)

    # Match against file + agent response so narration-only strings (e.g. output
    # paths) are found; the file is included so genuine code tokens still match.
    # We deliberately exclude tool output (see extract_agent_response).
    haystack = source + "\n" + agent_response
    p_score, hits, misses, violations = score_patterns(haystack, expected, forbidden)

    j_score = 0.5
    j_reason = "skipped"
    j_status = "off"  # judge not requested
    if use_judge:
        rubric = scenario.get("expectedOutput", "")
        j_score, j_reason, j_status = score_judge(source, rubric, dockerfile_dir)

    # Only fold the judge into the combined score when it produced a real verdict.
    # A "skipped" (no rubric) or "error" (couldn't run) judge must NOT silently
    # degrade to pattern-only without surfacing why — see judge_status below.
    judge_scored = j_status == "ok"
    if judge_scored:
        combined = (PATTERN_WEIGHT * p_score) + (JUDGE_WEIGHT * j_score)
    else:
        combined = p_score

    # Always emit all three scores so a reader never has to guess what a single
    # "score" means. combined_score is the headline (75% pattern + 25% judge when
    # the judge ran; pattern-only otherwise — judge_status says which).
    return {
        "combined_score": combined,
        "pattern_score": p_score,
        "judge_score": j_score if judge_scored else None,
        "baseline_k": baseline_k,
        "passed": combined >= baseline_k,
        "judge_reason": j_reason if j_status in ("ok", "error") else None,
        "judge_status": j_status,
        "expected_hits": hits,
        "expected_misses": misses,
        "forbidden_violations": violations,
    }


def _run_scenario_once(scenario: dict, skills: list, dockerfile_dir: Path, use_judge: bool) -> dict:
    """One agent-run + scoring attempt for a scenario.

    Returns a result dict. On an infrastructure failure (Docker/agent timeout,
    etc.) returns run_status='infra_error' rather than raising, so the caller can
    decide whether to retry. A returned dict WITHOUT run_status is a real scored
    result (pass or genuine skill failure).
    """
    prompt = "\n".join(scenario["input"]) + "\n\nSave your output to output.py"
    test_dir = setup_workspace(skills, dockerfile_dir)
    try:
        # 300s per-scenario agent budget (matches docker.sh's default). Heavy
        # scenarios (e.g. the autoport model port) can still exceed this under
        # parallel load and time out; a timeout that already produced output.py is
        # salvaged below, and one that produced nothing is reported as infra_error.
        transcript = run_kiro_in_docker(test_dir, prompt, timeout=300)
        output_file = test_dir / "output.py"
        source = output_file.read_text() if output_file.exists() else ""
        # Match patterns against file + agent's response prose only (not tool
        # output) — see extract_agent_response for why that distinction matters.
        agent_response = extract_agent_response(transcript)
        return evaluate(source, scenario, use_judge=use_judge,
                        dockerfile_dir=dockerfile_dir, agent_response=agent_response)
    except subprocess.TimeoutExpired as e:
        # The agent/Docker call exceeded the wall-clock budget. Capture the type,
        # the actual timeout, and whatever the container printed before it was
        # killed (TimeoutExpired carries partial stdout/stderr).
        partial = ""
        for stream in (e.stdout, e.stderr):
            if stream:
                partial += stream.decode() if isinstance(stream, bytes) else stream
        partial = _strip_ansi(partial)

        # SALVAGE completed work: the agent often finishes the real task (writes
        # output.py) and then hangs on a trivial final step (e.g. a spinner after
        # "save the port summary"), timing out with a VALID output already on disk.
        # Killing that as infra_error throws away a real result and triggers a
        # pointless retry of already-done work. If output.py exists, score it —
        # the timeout was a tail-end hang, not a failure to produce output.
        output_file = test_dir / "output.py"
        if output_file.exists() and output_file.read_text().strip():
            source = output_file.read_text()
            agent_response = extract_agent_response(partial)  # partial transcript
            result = evaluate(source, scenario, use_judge=use_judge,
                              dockerfile_dir=dockerfile_dir, agent_response=agent_response)
            # Note the salvage so it's visible we scored a timed-out-but-complete run.
            result["salvaged_after_timeout"] = f"timed out after {e.timeout}s; scored existing output.py"
            return result

        # No usable output — a genuine infra failure; keep it as infra_error so the
        # retry loop can try again.
        return {
            "combined_score": 0,
            "passed": False,
            "run_status": "infra_error",
            "run_error_type": "TimeoutExpired",
            "run_error": f"timed out after {e.timeout}s (no output.py produced)",
            "run_error_detail": partial[-2000:],  # tail of container output
            "expected_misses": [],
            "forbidden_violations": [],
        }
    except Exception as e:
        # Any other infrastructure failure (Docker build/run error, kiro-cli
        # crash, etc.) — NOT a skill failure. Record the exception TYPE + a wide
        # message so the real cause is visible, not truncated to 120 chars.
        return {
            "combined_score": 0,
            "passed": False,
            "run_status": "infra_error",
            "run_error_type": type(e).__name__,
            "run_error": str(e)[:2000],
            "expected_misses": [],
            "forbidden_violations": [],
        }
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def run_single_scenario(args: tuple) -> dict:
    """Run a single scenario in Docker. Called by ProcessPoolExecutor.

    Pass@k loop: run up to `count` times and keep the first PASS (transparent
    Pass@k semantics). There is NO infra-error retry — a failure (including a
    timeout that produced no output) is a real result and is reported as-is; the
    agent going down a bad path is skill behavior the benchmark should measure,
    not hide. A completed-but-hung run is recovered by salvage in _run_scenario_once.
    """
    scenario, skills, dockerfile_dir, count, use_judge = args
    task_id_str = scenario["taskId"]
    subset_str = scenario.get("_subset")
    result = {"combined_score": 0, "passed": False, "expected_misses": [], "forbidden_violations": []}

    for _attempt in range(count):
        result = _run_scenario_once(scenario, skills, dockerfile_dir, use_judge)
        # Log infra errors so a failed run is legible to a human (Docker crash vs.
        # agent-wander vs. genuine skill fail) — visibility, not auto-retry.
        if result.get("run_status") == "infra_error":
            print(f"  [infra_error] {task_id_str} attempt {_attempt + 1}: "
                  f"[{result.get('run_error_type')}] {result.get('run_error')}", flush=True)
            if result.get("run_error_detail"):
                print(f"      container output (tail):\n{result['run_error_detail']}", flush=True)
        if result["passed"]:
            break  # keep the first pass (Pass@k)

    result.setdefault("run_status", "ok")
    return {"taskId": task_id_str, "subset": subset_str, **result}


def run_all(subset: str = None, task_id: str = None, count: int = 2, workers: int = 4, use_judge: bool = False):
    """Run all scenarios in parallel Docker containers."""
    scenarios = load_scenarios(subset, task_id)
    if not scenarios:
        print("No scenarios found.")
        return

    dockerfile_dir = FRAMEWORK_DIR / "environment"

    # Each subset maps to its own agent spec with its own skills. Resolve skills
    # PER SUBSET so a multi-subset run gives each scenario the right skills
    # (previously one agent's skills were used for every scenario).
    subsets_present = sorted({s["_subset"] for s in scenarios})
    skills_by_subset = {name: get_agent_skills(name) for name in subsets_present}

    judge_str = "pattern(75%) + judge(25%)" if use_judge else "pattern-only"
    for name in subsets_present:
        n = sum(1 for s in scenarios if s["_subset"] == name)
        skills = skills_by_subset[name]
        print(f"Agent: {name}  ({n} scenario(s))  skills: {', '.join(skills) if skills else 'none'}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Pass@{count} | Workers: {workers} | Docker: yes | Scoring: {judge_str}")
    print("=" * 80)

    # Pre-build Docker image once
    subprocess.run(
        ["bash", str(DOCKER_SH), "build", str(dockerfile_dir)],
        capture_output=True, timeout=300,
    )

    # Run in parallel processes — each scenario carries its own subset's skills.
    work = [(s, skills_by_subset[s["_subset"]], dockerfile_dir, count, use_judge) for s in scenarios]
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_scenario, w): w[0]["taskId"] for w in work}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            status = "✓" if r["passed"] else "✗"
            # Infra failure (Docker/agent error) is neither pass nor a real fail —
            # mark it distinctly so it's not read as "the skill produced bad code".
            if r.get("run_status") == "infra_error":
                status = "⚠"
            # combined_score is the headline; also show its pattern/judge parts.
            line = f"  {status} {r['taskId']:<45} combined={r['combined_score']:.2f}"
            if r.get("pattern_score") is not None:
                line += f"  pattern={r['pattern_score']:.2f}"
            if r.get("judge_score") is not None:
                line += f"  judge={r['judge_score']:.0f}"
            elif r.get("judge_status") == "error":
                # Judge was requested but could not run — make it loud, don't hide it.
                line += "  judge=ERR"
            if r.get("run_status") == "infra_error":
                # Show the exception TYPE + message so the cause is diagnosable
                # (e.g. TimeoutExpired vs. a Docker/kiro-cli crash), not opaque.
                etype = r.get("run_error_type", "")
                line += f"  INFRA_ERROR[{etype}]: {r.get('run_error', '')[:120]}"
            elif not r["passed"]:
                if r.get("expected_misses"):
                    line += f"  missing={r['expected_misses']}"
                if r.get("forbidden_violations"):
                    line += f"  forbidden={r['forbidden_violations']}"
            # Note when a timed-out-but-complete run was salvaged (scored its output.py).
            if r.get("salvaged_after_timeout"):
                line += "  (salvaged)"
            if r.get("judge_reason") and r.get("judge_status") in ("ok", "error"):
                print(line, flush=True)
                print(f"      judge: {r['judge_reason'][:100]}", flush=True)
            else:
                print(line, flush=True)

    # Summary
    total_passed = sum(1 for r in results if r["passed"])
    pass_rate = total_passed / len(scenarios) * 100
    print("=" * 80)
    print(f"RESULT: {total_passed}/{len(scenarios)} passed ({pass_rate:.0f}%)")
    print(f"TARGET: 90%+ | STATUS: {'✓ MEETS TARGET' if pass_rate >= 90 else '✗ BELOW TARGET'}")

    # Per-subset breakdown — meaningful when running more than one subset so a
    # single weak subset isn't hidden inside an aggregate pass rate.
    by_subset = sorted({r.get("subset") for r in results})
    if len(by_subset) > 1:
        for name in by_subset:
            rs = [r for r in results if r.get("subset") == name]
            p = sum(1 for r in rs if r["passed"])
            print(f"  {name}: {p}/{len(rs)} passed ({p / len(rs) * 100:.0f}%)")

    # Infra errors (Docker/agent failures) are counted here so they're not read
    # as skill failures. They still count against pass rate, but this makes the
    # cause visible — a low pass rate driven by flakes is not a skill regression.
    infra_errors = [r for r in results if r.get("run_status") == "infra_error"]
    if infra_errors:
        print(f"INFRA: {len(infra_errors)} scenario(s) hit an infrastructure error "
              f"(not skill failures): {[r['taskId'] for r in infra_errors]}")
        # Dump each infra error's type + captured detail so the actual cause is
        # visible in the log (e.g. what the container printed before a timeout).
        for r in infra_errors:
            print(f"  --- {r['taskId']} [{r.get('run_error_type', '?')}]: {r.get('run_error', '')}")
            if r.get("run_error_detail"):
                print(f"      container output (tail):\n{r['run_error_detail']}")

    # Salvaged runs: the agent timed out but had already written a valid output.py,
    # so it was scored rather than discarded. Surfaced for visibility.
    salvaged = [r for r in results if r.get("salvaged_after_timeout")]
    if salvaged:
        print(f"SALVAGED: {len(salvaged)} scenario(s) timed out but had output.py "
              f"scored: {[r['taskId'] for r in salvaged]}")

    # Judge health: if the judge was requested, surface how many scenarios it
    # actually scored vs. errored. A wholesale failure (0 scored) means the
    # combined scores silently fell back to pattern-only — flag it loudly.
    if use_judge:
        judged_ok = sum(1 for r in results if r.get("judge_status") == "ok")
        judged_err = sum(1 for r in results if r.get("judge_status") == "error")
        print(f"JUDGE: {judged_ok}/{len(results)} scored | {judged_err} errored")
        if judged_ok == 0:
            print("  ⚠ WARNING: judge ran on NO scenarios — scores are pattern-only. "
                  "Check kiro-cli/Docker/KIRO_API_KEY.")

    # Write machine-readable results for CI artifact upload / regression detection
    results_file = FRAMEWORK_DIR / "build" / "nad_benchmark_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results written to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        default="neuron-nki-agent",
        help="Subset(s) to run: a single name, a comma-separated list, or 'all' "
        "for every subset (nki + autoport + validation). Restricted to "
        "registry-active scenarios unless --task-id is given.",
    )
    parser.add_argument(
        "--nad-root",
        default=str(Path(__file__).parent.parent),
        help="Path to the NAD inputs root containing tests/, skills/, agents/. "
        "Defaults to the repo root (this is a single-repo layout, so the gate scores "
        "the repo's own assets). Point it at another NAD checkout to score that "
        "repo's native assets instead.",
    )
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge (25%% weight) via kiro-cli")
    parser.add_argument(
        "--gate",
        type=float,
        default=None,
        help="Exit non-zero if pass rate (0-100) is below this threshold. "
        "Use --gate=100 for a strict PR gate. Omit for informational runs.",
    )
    args = parser.parse_args()

    # Repoint input dirs before run_all() forks workers, so every process agrees.
    set_nad_root(args.nad_root)
    if not TESTS_DIR.exists():
        print(f"ERROR: no tests/ under --nad-root={PROJECT_DIR}", file=sys.stderr)
        sys.exit(2)

    results = run_all(
        subset=args.subset, task_id=args.task_id, count=args.count,
        workers=args.workers, use_judge=args.judge,
    )

    if args.gate is not None:
        results = results or []
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        pass_rate = (passed / total * 100) if total else 0.0
        if pass_rate < args.gate:
            print(f"GATE FAILED: {pass_rate:.0f}% < required {args.gate:.0f}%")
            sys.exit(1)
        print(f"GATE PASSED: {pass_rate:.0f}% >= required {args.gate:.0f}%")
