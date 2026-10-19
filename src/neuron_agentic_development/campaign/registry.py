"""Specialist identities and capability contracts shared by campaign operations."""

def _capability(agent_id, inputs, outputs, hardware=False, source_write=False):
    return {
        "agent_id": agent_id,
        "required_inputs": tuple(inputs),
        "required_payload": tuple(outputs),
        "hardware": hardware,
        "source_write": source_write,
    }


CAPABILITIES = {
    "parallelism_plan": _capability(
        "neuron-parallelism-planner-agent",
        ("run_spec",),
        ("objective", "candidates", "selected_configs", "buckets", "exclusions"),
    ),
    "hardware_entitlement": _capability(
        "neuron-hardware-entitlement-agent",
        ("candidate_id",),
        ("candidate_id", "floors", "binding", "reconciliation_residual", "feasible"),
    ),
    "profile": _capability(
        "neuron-framework-profiling-vllm-neuron-agent",
        ("config", "workload", "phase"),
        ("phase", "measurement_kind", "samples"),
        hardware=True,
    ),
    "profile_analysis": _capability(
        "neuron-framework-profile-analysis-vllm-neuron-agent",
        ("profile_attempt_id",),
        ("source_attempt_id", "queries", "bottlenecks", "falsification_conditions"),
    ),
    "autoport": _capability(
        "neuron-framework-autoport-vllm-agent",
        ("source_repo", "model_config"),
        ("baseline_revision", "mutation_manifest", "compile", "smoke"),
        source_write=True,
    ),
    "equivalence": _capability(
        "neuron-framework-equivalence-agent",
        ("source_attempt_id",),
        ("source_attempt_id", "checks", "passed"),
        hardware=True,
    ),
    "compiler_debug": _capability(
        "experimental-neuron-perf-compiler-artifact-debugging-agent",
        ("source_attempt_id", "failure"),
        ("source_attempt_id", "reproducer", "compiler", "classification", "workaround"),
        hardware=True,
    ),
    "nki_triage": _capability(
        "neuron-nki-agent",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "routes", "findings"),
    ),
    "nki_profile_analysis": _capability(
        "neuron-nki-profile-analysis-agent",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "findings", "recommendations"),
    ),
    "nki_cost_analysis": _capability(
        "nki-cost-specialist",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "findings", "recommendations"),
    ),
    "nki_compute_analysis": _capability(
        "nki-compute-specialist",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "findings", "recommendations"),
    ),
    "nki_separation_analysis": _capability(
        "nki-separation-specialist",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "findings", "recommendations"),
    ),
    "nki_roofline_analysis": _capability(
        "nki-roofline-specialist",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "findings", "recommendations"),
    ),
    "nki_dma_order_analysis": _capability(
        "nki-dma-order-specialist",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "findings", "recommendations"),
    ),
    "nki_sharding_analysis": _capability(
        "nki-sharding-specialist",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "findings", "recommendations"),
    ),
    "nki_bottleneck_analysis": _capability(
        "nki-bottleneck-specialist",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "findings", "recommendations"),
    ),
    "nki_gap_analysis": _capability(
        "nki-gap-specialist",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "findings", "recommendations"),
    ),
    "nki_offload_analysis": _capability(
        "nki-offload-specialist",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "findings", "recommendations"),
    ),
    "nki_efficiency_analysis": _capability(
        "nki-efficiency-specialist",
        ("profile_attempt_id", "target"),
        ("source_attempt_id", "findings", "recommendations"),
    ),
    "nki_implementation": _capability(
        "neuron-nki-writer-agent",
        ("analysis_attempt_id", "target"),
        ("source_attempt_id", "mutation_manifest", "correctness", "measurement"),
        hardware=True,
        source_write=True,
    ),
}

NKI_ANALYSES = tuple(
    name
    for name in CAPABILITIES
    if name.startswith("nki_") and name.endswith("_analysis")
)
