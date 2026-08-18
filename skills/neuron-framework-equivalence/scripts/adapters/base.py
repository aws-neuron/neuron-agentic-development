"""
Base adapter class for serving-stack abstraction.

The equivalence methodology (R-ratio, 3-tensor, 8 stages) is platform-agnostic.
Only 5 integration points differ per serving stack. Each adapter implements these.

Beyond the 5 core methods, adapters may provide enhanced comparison, reference
model setup, teacher-forced logit generation, tensor alignment, and debugging
tools (capture/replace). These are optional — the base class provides fallbacks
that use the equivalence framework's own utilities (R-ratio, manual forward).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class StackAdapter(ABC):
    """Abstract base for serving-stack adapters.

    Subclass this and implement all 5 methods to support a new stack.
    See references/adapter-contract.md for the full specification.
    """

    name: str = "base"

    @abstractmethod
    def init_distributed(self, tp_degree: int = 1) -> None:
        """Initialize distributed state for CPU-mode testing.

        Must set up process groups so that TP-aware modules can be
        instantiated at the given tp_degree. Called once before
        create_model().
        """

    @abstractmethod
    def create_model(
        self,
        target_module_file: str,
        target_class_name: str,
        target_config_name: str,
        hf_model_path: str,
        **kwargs,
    ) -> torch.nn.Module:
        """Instantiate the target model in CPU mode for structure inspection.

        Returns an unweighted model instance suitable for tree building
        and component mapping. Weights are NOT loaded here.
        """

    @abstractmethod
    def load_weights(
        self,
        model: torch.nn.Module,
        hf_model_path: str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.nn.Module:
        """Load HuggingFace weights into the target model.

        Handles all stack-specific transforms: transpositions, fusions,
        naming conventions, etc. Returns the model with weights loaded.
        """

    @abstractmethod
    def forward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run a forward pass and return logits.

        Handles stack-specific forward signature differences internally.
        The caller always passes (model, input_ids) and gets back logits
        as a float32 tensor of shape [batch, seq_len, vocab] or [seq_len, vocab].
        """

    @abstractmethod
    def device_inference(
        self,
        model_id: str,
        tp_size: int,
        prompts: List[str],
        max_tokens: int = 32,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run inference on actual Neuron hardware.

        Returns a list of dicts with at minimum:
          - "text": generated text
          - "tokens": list of token ids
        Optionally: "logits", "logprobs"
        """

    def cleanup(self) -> None:
        """Tear down distributed state. Override if needed."""
        pass

    def check_environment(self) -> None:
        """Validate the runtime environment before any stage runs.

        Default: no-op. Override in stacks that pin a specific dependency
        version (e.g. vLLM-Neuron pins vLLM) to fail fast with a clear,
        actionable message instead of crashing mid-stage on an API skew.

        Raise ``EnvironmentError`` to abort. Called automatically by
        ``get_adapter()`` right after the adapter is constructed.
        """
        pass

    # ── P0: Three-way comparison ──

    def compare_three_way(
        self,
        baseline: torch.Tensor,
        expected: torch.Tensor,
        actual: torch.Tensor,
        name: str = "tensor",
        **kwargs,
    ) -> "ThreeWayResult":
        """Three-way comparison: baseline (FP32) vs expected (dtype) vs actual (target).

        Default implementation uses R-ratio from tensor_compare.compare_3tensors.
        vLLM adapter overrides with BC/σ-ratio from assert_close_three_way.

        Returns a ThreeWayResult with uniform fields regardless of backend math.
        """
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from tensor_compare import compare_3tensors

        b = baseline.detach().cpu().float().flatten().numpy()
        e = expected.detach().cpu().float().flatten().numpy()
        a = actual.detach().cpu().float().flatten().numpy()
        metrics = compare_3tensors(b, e, a)
        r = metrics["r_ratio"]
        return ThreeWayResult(
            passed=r < kwargs.get("tau_r", 1.2),
            name=name,
            r_ratio=r,
            bc=None,
            sigma_ratio=None,
            linf_ratio=metrics["rel_matrix_inf_norm_dist_3_1"] / (metrics["rel_matrix_inf_norm_dist_2_1"] + 1e-30),
            l2_ratio=metrics["rel_matrix_fro_norm_dist_3_1"] / (metrics["rel_matrix_fro_norm_dist_2_1"] + 1e-30),
            raw_metrics=metrics,
        )

    def compare_three_way_multi(
        self,
        baselines: List[torch.Tensor],
        expecteds: List[torch.Tensor],
        actuals: List[torch.Tensor],
        name: str = "tensor",
        **kwargs,
    ) -> "ThreeWayResult":
        """Multi-input three-way comparison with aggregation.

        Default: worst-case R-ratio across inputs.
        vLLM adapter: aggregated BC/σ-ratio across all inputs.
        """
        worst_r = 0.0
        worst_linf = 0.0
        worst_l2 = 0.0
        for b, e, a in zip(baselines, expecteds, actuals):
            result = self.compare_three_way(b, e, a, name=name, **kwargs)
            worst_r = max(worst_r, result.r_ratio or 0.0)
            worst_linf = max(worst_linf, result.linf_ratio)
            worst_l2 = max(worst_l2, result.l2_ratio)
        return ThreeWayResult(
            passed=worst_r < kwargs.get("tau_r", 1.2),
            name=name,
            r_ratio=worst_r,
            bc=None,
            sigma_ratio=None,
            linf_ratio=worst_linf,
            l2_ratio=worst_l2,
            raw_metrics={"n_inputs": len(baselines), "worst_r_ratio": worst_r},
        )

    # ── P1: Reference model setup and teacher-forced logit generation ──

    def setup_reference_model(
        self,
        model_path: str,
        dtype: torch.dtype = torch.float32,
        config: Any = None,
    ) -> torch.nn.Module:
        """Load a HuggingFace model for reference logit generation.

        Default: basic AutoModelForCausalLM.from_pretrained.
        vLLM adapter: architecture-aware loading with SDPA/chunked attention.
        """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
        )
        model.eval()
        return model

    def generate_teacher_forced_logits(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        teacher_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Generate logits under teacher forcing with KV cache reuse.

        Default: naive per-step full-sequence forward (O(n²)).
        vLLM adapter: incremental KV-cached decoding (O(n)).

        Args:
            model: HF reference model (not the target).
            input_ids: Prompt input_ids [1, prompt_len].
            teacher_sequence: Token IDs to force [num_tokens] or [1, num_tokens].

        Returns:
            Logits tensor [num_tokens, 1, vocab_size] or [num_tokens, vocab_size].
        """
        if teacher_sequence.dim() == 2:
            teacher_sequence = teacher_sequence.squeeze(0)
        num_tokens = teacher_sequence.shape[0]
        all_logits = []
        current_input = input_ids.clone()

        with torch.no_grad():
            for t in range(num_tokens):
                out = model(current_input)
                logits = out.logits if hasattr(out, "logits") else out
                last_logits = logits[:, -1, :].float()
                all_logits.append(last_logits)
                next_token = teacher_sequence[t].unsqueeze(0).unsqueeze(0)
                current_input = torch.cat([current_input, next_token], dim=1)

        return torch.stack(all_logits, dim=0)

    def generate_reference_logits(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        output_length: int,
    ) -> torch.Tensor:
        """Generate greedy reference logits (for extracting teacher sequence).

        Default: naive per-step full-sequence forward.
        vLLM adapter: incremental KV-cached decoding.

        Returns:
            Logits tensor [output_length, 1, vocab_size].
        """
        all_logits = []
        current_input = input_ids.clone()

        with torch.no_grad():
            for _ in range(output_length):
                out = model(current_input)
                logits = out.logits if hasattr(out, "logits") else out
                last_logits = logits[:, -1, :].float()
                all_logits.append(last_logits)
                next_token = last_logits.argmax(dim=-1, keepdim=True)
                current_input = torch.cat([current_input, next_token], dim=1)

        return torch.stack(all_logits, dim=0)

    # ── P1: Tensor shape alignment ──

    def align_tensors(
        self,
        baseline: torch.Tensor,
        expected: torch.Tensor,
        actual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Align three tensors for comparison (batch dim, seq truncation, hidden truncation).

        Default: basic batch squeeze + seq min.
        vLLM adapter: delegates to align_and_truncate_hidden.
        """
        tensors = [baseline.float(), expected.float(), actual.float()]
        max_dim = max(t.dim() for t in tensors)
        if max_dim == 3:
            tensors = [t.unsqueeze(0) if t.dim() == 2 else t for t in tensors]
        tensors = [t.squeeze(0) if t.dim() == 3 and t.shape[0] == 1 else t for t in tensors]
        if all(t.dim() == 2 for t in tensors):
            min_seq = min(t.shape[0] for t in tensors)
            min_h = min(t.shape[1] for t in tensors)
            tensors = [t[:min_seq, :min_h] for t in tensors]
        match = all(t.shape == tensors[0].shape for t in tensors)
        return tensors[0], tensors[1], tensors[2], match

    # ── P0: Tensor capture and replacement (debugging) ──

    def create_tensor_capturer(self, model: torch.nn.Module, patterns: List[str], **kwargs) -> Any:
        """Wrap model for intermediate tensor capture on device.

        Only supported by stacks with torch.compile + hook infrastructure.
        """
        raise NotImplementedError(f"{self.name} adapter does not support tensor capture")

    def create_tensor_replacer(self, reference_captures: Any, prompt_token_ids: List[List[int]], **kwargs) -> Any:
        """Create a TensorReplacer for injecting HF reference values during device forward.

        Only supported by stacks with scheduler-driven forward passes.
        """
        raise NotImplementedError(f"{self.name} adapter does not support tensor replacement")

    # ── Optional diagnostic methods ──
    # These leverage stack-specific debugging tools when available.
    # Override in subclasses that have access to such tools.

    def run_accuracy_analysis(
        self,
        model_id: str,
        tp_size: int,
        eval_fn: Any = None,
        thresholds: Optional[Dict[str, float]] = None,
        input_task_results: Optional[str] = None,
        output_dir: str = "./accuracy_report",
        **kwargs,
    ) -> Dict[str, Any]:
        """Run task-level accuracy analysis with per-sample deviation tracking.

        Returns structured results with scores, pass/fail, and deviated prompts.
        """
        raise NotImplementedError(f"{self.name} adapter does not support accuracy analysis")

    def run_logit_validation(
        self,
        model_id: str,
        tp_size: int,
        prompts: List[str],
        output_length: int = 16,
        output_dir: str = "./accuracy_report",
        **kwargs,
    ) -> Dict[str, Any]:
        """Run per-token logit validation on compiled device model.

        Compares device logits against FP32 baseline and dtype reference.
        """
        raise NotImplementedError(f"{self.name} adapter does not support logit validation")

    def run_kv_cache_analysis(
        self,
        model_id: str,
        tp_size: int,
        prompts: List[str],
        output_length: int = 16,
        output_dir: str = "./accuracy_report",
        **kwargs,
    ) -> Dict[str, Any]:
        """Run three-way KV cache comparison (FP32 vs HF-dtype vs vLLM).

        Identifies which layers/positions have KV divergence.
        """
        raise NotImplementedError(f"{self.name} adapter does not support KV cache analysis")

    def run_prompt_diagnosis(
        self,
        model_id: str,
        tp_size: int,
        prompts: List[str],
        output_length: int = 16,
        output_dir: str = "./accuracy_report",
        plugins: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run prompt-level diagnosis combining logit validation and KV analysis.

        Orchestrates multiple analysis plugins against specific prompts.
        """
        raise NotImplementedError(f"{self.name} adapter does not support prompt diagnosis")


@dataclass
class ThreeWayResult:
    """Uniform result from three-way comparison, regardless of backend math.

    Adapters may populate different subsets of fields:
    - NxDI: r_ratio is primary, bc/sigma_ratio are None
    - vLLM: bc/sigma_ratio are primary, r_ratio is None
    """
    passed: bool
    name: str
    r_ratio: Optional[float] = None
    bc: Optional[float] = None
    sigma_ratio: Optional[float] = None
    linf_ratio: float = 0.0
    l2_ratio: float = 0.0
    raw_metrics: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts = [f"[{status}] {self.name}"]
        if self.r_ratio is not None:
            parts.append(f"  R-ratio: {self.r_ratio:.4f}")
        if self.bc is not None:
            parts.append(f"  BC: {self.bc:.4f}")
        if self.sigma_ratio is not None:
            parts.append(f"  sigma-ratio: {self.sigma_ratio:.3f}")
        parts.append(f"  L-inf ratio: {self.linf_ratio:.4f}  L2 ratio: {self.l2_ratio:.4f}")
        return "\n".join(parts)
