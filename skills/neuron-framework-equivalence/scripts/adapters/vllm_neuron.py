"""
vLLM-Neuron adapter.

Handles: vLLM distributed init, from_configs(), manual weight mapping
with transpositions/QKV fusion, attn_metadata construction, vllm.LLM API.

Wraps vllm_neuron.accuracy utilities for:
- Three-way comparison (BC/σ-ratio via assert_close_three_way)
- Reference model setup (SDPA/chunked attention, arch detection)
- Teacher-forced logit generation (KV-cached O(n) decoding)
- Tensor alignment (hidden dim truncation for Neuron padding)
- Tensor capture and replacement (divergence isolation)

Based on: Equivalence Framework Adaptation experiment (TinyLlama, Apr 2026).
"""
import gc
import glob
import importlib
import importlib.util
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import StackAdapter, ThreeWayResult
from . import register


# Pinned versions this adapter is written against. Two distinct numbers:
#   - PINNED_VLLM_VERSION: the upstream vLLM framework. vLLM changes its
#     distributed-init, VllmConfig, and v1 APIs across minor releases, so the
#     equivalence skill targets one specific minor line. The adapter was
#     updated for vLLM 0.24.0 (VllmConfig context requirement still applies;
#     verified against vLLM-Neuron 0.24 that set_current_vllm_config +
#     ParallelConfig(tensor_parallel_size=) + VllmConfig(parallel_config=) and
#     initialize_neuron_parallel_state(tp_global_ranks=, local_rank=) are
#     unchanged).
#   - PINNED_VLLM_NEURON_VERSION: the vLLM-Neuron plugin itself (dist name
#     "vllm-neuron"). Its model classes, parallel-state helpers, and accuracy
#     utilities are what the adapter actually calls; the plugin tracks vLLM but
#     versions independently (its own version is e.g. 0.24.0.1.1.0).
# Bump BOTH in lockstep with the adapter when moving to a new line.
PINNED_VLLM_VERSION = "0.24.0"
PINNED_VLLM_NEURON_VERSION = "0.24.0"


@register("vllm_neuron")
class VLLMNeuronAdapter(StackAdapter):
    name = "vllm_neuron"
    pinned_vllm_version = PINNED_VLLM_VERSION
    pinned_vllm_neuron_version = PINNED_VLLM_NEURON_VERSION

    def check_environment(self) -> None:
        """Verify the installed vLLM / vLLM-Neuron matches the pinned version.

        Raises ``EnvironmentError`` with an actionable message and forces an
        early exit when the environment is incompatible, rather than letting
        the user hit a cryptic ``AssertionError: Current vLLM config is not
        set.`` (or similar) deep inside a stage. Called automatically by
        ``get_adapter()`` right after the adapter is constructed.

        Three failure modes are surfaced clearly:
          1. vLLM-Neuron not importable — almost always the editable-install
             PYTHONPATH issue (the vLLM plugin entry point can't import
             ``vllm_neuron``). We point the user at the fix.
          2. Installed vLLM minor line != the pinned line — the adapter's
             API assumptions won't hold. We refuse to proceed.
          3. Installed vLLM-Neuron plugin minor line != the pinned line — the
             model classes / parallel helpers the adapter calls may differ.
        """
        # 1. vLLM-Neuron must be importable (catches the editable-install /
        #    PYTHONPATH problem before it manifests as a plugin import error).
        try:
            import vllm_neuron  # noqa: F401
        except ModuleNotFoundError as e:
            raise EnvironmentError(
                "vLLM-Neuron equivalence adapter: cannot import 'vllm_neuron'.\n"
                f"  Underlying error: {e}\n"
                "  This usually means the editable install is not on PYTHONPATH.\n"
                "  Fix: export PYTHONPATH so it includes the vllm-neuron\n"
                "  project root, e.g.\n"
                "    PYTHONPATH=/path/to/vllm-neuron:$PYTHONPATH python3 ...\n"
            ) from e

        # 2. Installed vLLM must match the pinned minor line.
        try:
            from vllm.version import __version__ as installed_vllm
        except Exception:
            try:
                from importlib.metadata import version as _pkg_version
                installed_vllm = _pkg_version("vllm")
            except Exception:
                installed_vllm = None

        if installed_vllm is None:
            raise EnvironmentError(
                "vLLM-Neuron equivalence adapter: vLLM is not installed or its\n"
                "  version could not be determined. This adapter requires vLLM "
                f"{self.pinned_vllm_version}."
            )

        if not _same_minor_line(installed_vllm, self.pinned_vllm_version):
            raise EnvironmentError(
                "vLLM-Neuron equivalence adapter: incompatible vLLM version.\n"
                f"  Installed:  vllm=={installed_vllm}\n"
                f"  Required:   vllm=={self.pinned_vllm_version} "
                f"(this skill is pinned to the {_minor_line(self.pinned_vllm_version)} line)\n"
                "  The adapter's distributed-init and forward() logic depend on\n"
                f"  vLLM {_minor_line(self.pinned_vllm_version)} APIs. Install the matching vLLM\n"
                "  (and rebuild vllm-neuron against it) before re-running.\n"
            )

        # 3. Installed vLLM-Neuron plugin must match the pinned minor line.
        #    Read the dist version ("vllm-neuron"); fall back to the module's
        #    __version__ if metadata is unavailable (e.g. editable w/o metadata).
        installed_plugin = None
        try:
            from importlib.metadata import version as _pkg_version
            installed_plugin = _pkg_version("vllm-neuron")
        except Exception:
            installed_plugin = getattr(vllm_neuron, "__version__", None)

        if installed_plugin is None:
            # Importable but unversioned (common for editable installs). Don't
            # hard-fail — the import already succeeded — but make it visible.
            print(
                "WARNING: vLLM-Neuron equivalence adapter: could not determine the\n"
                "  installed vllm-neuron plugin version; expected the "
                f"{_minor_line(self.pinned_vllm_neuron_version)} line. Proceeding."
            )
        elif not _same_minor_line(installed_plugin, self.pinned_vllm_neuron_version):
            raise EnvironmentError(
                "vLLM-Neuron equivalence adapter: incompatible vLLM-Neuron plugin.\n"
                f"  Installed:  vllm-neuron=={installed_plugin}\n"
                f"  Required:   vllm-neuron=={self.pinned_vllm_neuron_version} "
                f"(this skill is pinned to the {_minor_line(self.pinned_vllm_neuron_version)} line)\n"
                "  The adapter calls plugin-specific model classes and parallel\n"
                "  helpers that differ across plugin lines. Check out / install the\n"
                f"  vllm-neuron {_minor_line(self.pinned_vllm_neuron_version)} line before re-running.\n"
            )

        # 4. Reference-model helpers are optional — warn, never fail.
        self._warn_if_reference_helpers_missing()

    @staticmethod
    def _warn_if_reference_helpers_missing() -> bool:
        """Report whether the reference-model helpers are importable.

        ``setup_reference_model`` / ``generate_teacher_forced_logits`` live in the
        vLLM-Neuron repo's ``test/`` tree, which is **not** shipped in the wheel. Their
        absence is not fatal, so this warns rather than raising — but the user should
        know up front that Stage 5/6 will be degraded rather than discovering it
        mid-run.

        Returns True when the full comparison is available.
        """
        try:
            import test.vllm_neuron.utils.logit_test_utils  # noqa: F401
        except ModuleNotFoundError:
            print(
                "WARNING: vLLM-Neuron reference-model helpers are not importable\n"
                "  ('test.vllm_neuron.utils.logit_test_utils'). This is expected for a\n"
                "  wheel-only install — the test/ tree ships only with the repo.\n"
                "\n"
                "  Stage 5/6 will run in DEGRADED mode:\n"
                "    - Step 4 (run_teacher_forced_comparison.py) is UNAFFECTED. The\n"
                "      accuracy_debugger path builds its own FP32/dtype references.\n"
                "    - run_stage5.py falls back to per-token logit validation instead of\n"
                "      the three-tensor R-ratio, so it reports pass/fail rather than an R value.\n"
                "\n"
                "  For the full three-tensor comparison, put the vLLM-Neuron repo root\n"
                "  (VLLM_NEURON_DIR) on PYTHONPATH so 'test/' is importable."
            )
            return False
        return True

    def init_distributed(self, tp_degree: int = 1) -> None:
        """Initialize distributed + parallel state for CPU-mode testing.

        vLLM 0.24.0 requires ``initialize_model_parallel`` to run inside a
        ``set_current_vllm_config(VllmConfig(...))`` context, and the exact
        ``VllmConfig`` / ``ParallelConfig`` signatures change between vLLM
        releases. Rather than reconstruct that context by hand (the 0.19.0
        approach, which breaks on 0.21.0+ with "Current vLLM config is not
        set."), we delegate to vLLM-Neuron's own bootstrap helper
        ``initialize_neuron_parallel_state``. It builds the minimal
        ``VllmConfig``, sets the context, and calls
        ``initialize_model_parallel`` internally — matching exactly what the
        serving stack and MPExecutor test path do (see
        ``vllm_neuron.utils.executor`` and
        ``vllm_neuron.parallel.neuron_parallel_state``).

        Bootstrap order mirrors the repo's test path: raw
        ``dist.init_process_group`` first (NOT vLLM's
        ``init_distributed_environment`` — that pre-creates ``_WORLD`` and
        makes the neuron helper early-return before building TP groups), then
        ``initialize_neuron_parallel_state(tp_global_ranks=..., local_rank=0)``
        which creates ``_WORLD``, the vLLM model-parallel sub-groups, and the
        Neuron-specific groups in one shot.
        """
        os.environ["NXD_CPU_MODE"] = "1"
        os.environ.setdefault("WORLD_SIZE", str(tp_degree))
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "8099")
        os.environ.setdefault("RANK", "0")

        import torch.distributed as dist
        from vllm_neuron.parallel.neuron_parallel_state import (
            initialize_neuron_parallel_state,
            is_initialized,
        )

        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://{}:{}".format(
                    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]
                ),
                rank=0,
                world_size=tp_degree,
            )

        # Delegates VllmConfig construction + set_current_vllm_config context
        # + initialize_model_parallel to the repo (version-robust). Guarded so a
        # second init_distributed() is a no-op for the parallel groups.
        if not is_initialized():
            initialize_neuron_parallel_state(
                tp_global_ranks=list(range(tp_degree)),
                local_rank=0,
            )

        # initialize_neuron_parallel_state sets the vLLM config context only
        # transiently (a `with` block that exits when it returns). But model
        # __init__ (e.g. LlamaAttention) calls get_current_vllm_config() at
        # construction time — outside any stage's own engine context — and
        # fails with "Current vLLM config is not set." So open a PERSISTENT
        # config context here that stays live through create_model(), and tear
        # it down in cleanup(). Mirrors the executor's
        # `with set_current_vllm_config(vllm_config): model = load_fn()`.
        if getattr(self, "_vllm_config_ctx", None) is None:
            from vllm.config import (
                VllmConfig,
                ParallelConfig,
                set_current_vllm_config,
            )
            self._vllm_config = VllmConfig(
                parallel_config=ParallelConfig(tensor_parallel_size=tp_degree),
            )
            self._vllm_config_ctx = set_current_vllm_config(self._vllm_config)
            self._vllm_config_ctx.__enter__()

    def create_model(self, target_module_file, target_class_name, target_config_name,
                     hf_model_path, **kwargs):
        module = _import_module(target_module_file)
        ModelClass = getattr(module, target_class_name)

        # vLLM-Neuron uses a separate config module
        config_module_path = kwargs.get("target_config_module")
        if config_module_path:
            config_mod = _import_module(config_module_path)
            ConfigClass = getattr(config_mod, target_config_name)
        else:
            ConfigClass = getattr(module, target_config_name, None)
            if ConfigClass is None:
                # Try config.py in same directory
                config_path = os.path.join(os.path.dirname(target_module_file), "config.py")
                if os.path.exists(config_path):
                    config_mod = _import_module(config_path)
                    ConfigClass = getattr(config_mod, target_config_name)

        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        config = ConfigClass.from_configs(hf_config, neuron_config=None)
        return ModelClass(config)

    def load_weights(self, model, hf_model_path, dtype=torch.bfloat16):
        from safetensors.torch import load_file

        hf_state = {}
        for f in sorted(glob.glob(os.path.join(hf_model_path, "*.safetensors"))):
            hf_state.update(load_file(f))

        neuron_state = _map_hf_to_vllm_neuron(hf_state, model, dtype)
        model.load_state_dict(neuron_state, strict=False, assign=True)
        return model

    def forward(self, model, input_ids, **kwargs):
        """Forward pass for E2E comparison.

        For vLLM-Neuron, the native model forward requires attn_metadata with
        KV cache state that can't be properly maintained across incremental calls.
        Instead, we use the HF model (loaded via setup_reference_model) for
        teacher-forced logit generation, matching the NxDI pattern.

        For component-level tests, use the individual modules directly.
        For device E2E, use device_inference() via the vLLM engine.
        """
        # If this is an HF model (has .generate method with past_key_values),
        # use standard HF forward — this is the reference model path
        if hasattr(model, "generate") and hasattr(model, "config"):
            seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            attention_mask = kwargs.get("attention_mask", torch.ones_like(input_ids))
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

            with torch.no_grad():
                out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
                logits = out.logits if hasattr(out, "logits") else out
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
            return torch.nan_to_num(logits.float(), nan=0.0, posinf=1e6, neginf=-1e6)

        # Native vLLM-Neuron forward (for component tests only)
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)

        seq_len = input_ids.shape[0]
        positions = torch.arange(seq_len, dtype=torch.long)

        config = _get_model_config(model)
        num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 1))
        kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 1))
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden = getattr(config, "hidden_size", 128)
            n_heads = getattr(config, "num_attention_heads", 1)
            head_dim = hidden // n_heads

        block_size = 128
        num_blocks = (seq_len + block_size - 1) // block_size
        attn_metadata = {}

        # Do NOT assume every layer has a self_attn submodule. The original
        # loop emitted attn_metadata for layers.{i}.self_attn on EVERY layer,
        # which crashes hybrid models (mixed linear/full attention) where only
        # a subset of layers have self_attn. Use config.layer_types to emit
        # metadata only for full-attention layers. Non-hybrid models have no
        # layer_types, so _is_full_attention_layer() returns True for all of
        # them — exactly the original behaviour, unchanged.
        #
        # NOTE: this only stops the homogeneous-layer assumption from breaking
        # hybrid models — it is NOT full hybrid forward support. The linear/
        # recurrent layers' state inputs/outputs are not threaded through this
        # forward path. Untested: no hybrid model exists in the repo/weights.
        layer_types = _get_layer_types(config, num_layers)

        for i in range(num_layers):
            if not _is_full_attention_layer(layer_types, i):
                continue
            layer_name = f"layers.{i}.self_attn"
            k_cache = torch.zeros(num_blocks, kv_heads, block_size, head_dim, dtype=torch.bfloat16)
            v_cache = torch.zeros_like(k_cache)

            layer = _get_layer(model, i)
            if layer is not None:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    attn.k_cache = k_cache
                    attn.v_cache = v_cache

            attn_metadata[layer_name] = {
                "max_query_len": seq_len,
                "decode_token_threshold": 1,
                "slot_mapping": torch.arange(seq_len),
                "block_size": block_size,
                "block_table_tensor": torch.arange(num_blocks).unsqueeze(0),
                "cached_seq_len": torch.tensor([0]),
                "kv_segment_size": None,
                "prior_tokens": torch.tensor([0]),
            }

        # Defensive hook (UNVERIFIED): models with recurrent/linear-attention
        # state may expose bind_recurrent_state() to allocate it before forward.
        # We call it if present; non-hybrid models don't have it and are
        # skipped. This is best-effort and untested — no hybrid model is
        # available to exercise it, and the recurrent state is not otherwise
        # wired into this forward path.
        if hasattr(model, "bind_recurrent_state"):
            model.bind_recurrent_state(batch_size=1, device="cpu")

        sampling_positions = torch.arange(seq_len)

        with torch.no_grad():
            logits = model(input_ids, positions, attn_metadata=attn_metadata, sampling_positions=sampling_positions)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if hasattr(logits, "logits"):
                logits = logits.logits

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        return torch.nan_to_num(logits.float(), nan=0.0, posinf=1e6, neginf=-1e6)

    def device_inference(self, model_id, tp_size, prompts, max_tokens=32, **kwargs):
        os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_id,
            tensor_parallel_size=tp_size,
            max_model_len=kwargs.get("max_model_len", 256),
            max_num_seqs=kwargs.get("max_num_seqs", 4),
        )
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens,
            logprobs=kwargs.get("logprobs", None),
        )
        outputs = llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            text = output.outputs[0].text
            tokens = output.outputs[0].token_ids
            entry = {"text": text, "tokens": list(tokens)}
            if output.outputs[0].logprobs:
                entry["logprobs"] = output.outputs[0].logprobs
            results.append(entry)
        return results

    def cleanup(self):
        # Exit the persistent vLLM config context opened in init_distributed.
        ctx = getattr(self, "_vllm_config_ctx", None)
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._vllm_config_ctx = None
            self._vllm_config = None

        # Tear down Neuron-specific groups first, then vLLM model-parallel and
        # the distributed environment. Mirrors the inverse of init_distributed.
        try:
            from vllm_neuron.parallel.neuron_parallel_state import (
                destroy_neuron_parallel_state,
            )
            destroy_neuron_parallel_state()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
                destroy_distributed_environment,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass

    # ── P0: Three-way comparison via BC/σ-ratio ──

    def compare_three_way(self, baseline, expected, actual, name="tensor", **kwargs):
        from vllm_neuron.accuracy.testing import assert_close_three_way, ThreeWayAssertResult

        bc_threshold = kwargs.get("bc_threshold", 0.99)
        max_linf_ratio = kwargs.get("max_linf_ratio", 5.0)
        max_l2_ratio = kwargs.get("max_l2_ratio", 3.0)

        try:
            result = assert_close_three_way(
                baseline, expected, actual,
                name=name,
                bc_threshold=bc_threshold,
                max_linf_ratio=max_linf_ratio,
                max_l2_ratio=max_l2_ratio,
                plot_on_failure=kwargs.get("plot_on_failure", False),
                output_dir=kwargs.get("output_dir"),
            )
            return ThreeWayResult(
                passed=True,
                name=name,
                r_ratio=None,
                bc=result.bc,
                sigma_ratio=result.sigma_ratio,
                linf_ratio=result.linf_ratio,
                l2_ratio=result.l2_ratio,
                raw_metrics={
                    "base_linf": result.base_linf,
                    "tgt_linf": result.tgt_linf,
                    "base_l2": result.base_l2,
                    "tgt_l2": result.tgt_l2,
                    "n_inputs": result.n_inputs,
                },
            )
        except AssertionError:
            # assert_close_three_way raises on failure; extract result from the instance
            from vllm_neuron.accuracy.tensor_compare import compare_tensors_three_way, _compute_bc

            base_list = _to_tensor_list(baseline)
            exp_list = _to_tensor_list(expected)
            act_list = _to_tensor_list(actual)

            all_base_errors, all_tgt_errors = [], []
            worst_linf = worst_l2 = 0.0

            for b, e, a in zip(base_list, exp_list, act_list):
                r = compare_tensors_three_way(b, e, a, name=name)
                worst_linf = max(worst_linf, r.linf_ratio)
                worst_l2 = max(worst_l2, r.l2_ratio)
                if r.base_errors is not None:
                    all_base_errors.append(r.base_errors)
                    all_tgt_errors.append(r.tgt_errors)

            if all_base_errors:
                cat_base = np.concatenate(all_base_errors)
                cat_tgt = np.concatenate(all_tgt_errors)
                agg_bc = _compute_bc(cat_base, cat_tgt)
                base_rms = np.sqrt(np.mean(cat_base**2))
                tgt_rms = np.sqrt(np.mean(cat_tgt**2))
                sigma_ratio = tgt_rms / base_rms if base_rms > 0 else float("inf")
            else:
                agg_bc = 0.0
                sigma_ratio = float("inf")

            return ThreeWayResult(
                passed=False,
                name=name,
                r_ratio=None,
                bc=agg_bc,
                sigma_ratio=sigma_ratio,
                linf_ratio=worst_linf,
                l2_ratio=worst_l2,
                raw_metrics={"n_inputs": len(base_list)},
            )

    def compare_three_way_multi(self, baselines, expecteds, actuals, name="tensor", **kwargs):
        from vllm_neuron.accuracy.testing import assert_close_three_way

        bc_threshold = kwargs.get("bc_threshold", 0.99)
        max_linf_ratio = kwargs.get("max_linf_ratio", 5.0)
        max_l2_ratio = kwargs.get("max_l2_ratio", 3.0)

        try:
            result = assert_close_three_way(
                baselines, expecteds, actuals,
                name=name,
                bc_threshold=bc_threshold,
                max_linf_ratio=max_linf_ratio,
                max_l2_ratio=max_l2_ratio,
                plot_on_failure=kwargs.get("plot_on_failure", False),
                output_dir=kwargs.get("output_dir"),
            )
            return ThreeWayResult(
                passed=True,
                name=name,
                r_ratio=None,
                bc=result.bc,
                sigma_ratio=result.sigma_ratio,
                linf_ratio=result.linf_ratio,
                l2_ratio=result.l2_ratio,
                raw_metrics={"n_inputs": result.n_inputs},
            )
        except AssertionError:
            # Recompute metrics without raising
            return self.compare_three_way(baselines, expecteds, actuals, name=name, **kwargs)

    # ── P0: Tensor capture and replacement ──

    def create_tensor_capturer(self, model, patterns, **kwargs):
        from vllm_neuron.accuracy.tensor_capture import TensorCaptureModel, expand_patterns

        expanded = expand_patterns(patterns)
        return TensorCaptureModel(model, expanded)

    def create_tensor_replacer(self, reference_captures, prompt_token_ids, **kwargs):
        from vllm_neuron.accuracy.tensor_replacement import TensorReplacer

        return TensorReplacer(reference_captures, prompt_token_ids=prompt_token_ids)

    # ── P1: Reference model setup (arch-aware, SDPA/chunked) ──

    def setup_reference_model(self, model_path, dtype=torch.float32, config=None):
        from test.vllm_neuron.utils.logit_test_utils import setup_reference_model
        return setup_reference_model(model_path, dtype, config=config)

    # ── P1: Teacher-forced logit generation (KV-cached, O(n)) ──

    def generate_teacher_forced_logits(self, model, input_ids, teacher_sequence):
        from test.vllm_neuron.utils.logit_test_utils import generate_teacher_forced_logits

        if teacher_sequence.dim() == 2:
            teacher_sequence = teacher_sequence.squeeze(0)
        # logit_test_utils expects teacher_sequence[i].unsqueeze(1) to work → need [seq_len, 1]
        if teacher_sequence.dim() == 1:
            teacher_sequence = teacher_sequence.unsqueeze(1)
        return generate_teacher_forced_logits(model, input_ids, teacher_sequence)

    def generate_reference_logits(self, model, input_ids, output_length):
        from test.vllm_neuron.utils.logit_test_utils import generate_reference_logits
        return generate_reference_logits(model, input_ids, output_length)

    # ── P1: Tensor alignment (hidden dim truncation for Neuron padding) ──

    def align_tensors(self, baseline, expected, actual):
        from vllm_neuron.accuracy.tensor_alignment_utils import align_and_truncate_hidden
        return align_and_truncate_hidden(baseline, expected, actual)

    # ── Diagnostic methods (delegate to accuracy_debugger APIs) ──

    def run_accuracy_analysis(self, model_id, tp_size, eval_fn=None,
                              thresholds=None, input_task_results=None,
                              output_dir="./accuracy_report", **kwargs):
        import os
        from vllm_neuron.accuracy.accuracy_debugger import run_task_analysis
        from vllm_neuron.accuracy.accuracy_debugger.task_plugins.lm_eval_analyzer import LmEvalAnalyzer

        if input_task_results is None:
            if eval_fn is None:
                raise ValueError("Provide eval_fn or input_task_results")
            server_handle = kwargs.get("server_handle")
            started_server = False
            if server_handle is None:
                server_handle = self._start_server(model_id, tp_size, **kwargs)
                started_server = True
            try:
                input_task_results = os.path.join(output_dir, "eval_results")
                eval_kwargs = dict(kwargs.get("eval_kwargs") or {})
                # New default limit=None runs the whole dataset; caller controls
                # sample count by passing limit inside eval_kwargs.
                eval_fn(
                    base_url=server_handle.base_url,
                    model=server_handle.model,
                    results_dir=input_task_results,
                    **eval_kwargs,
                )
            finally:
                if started_server and hasattr(server_handle, "stop"):
                    server_handle.stop()

        result = run_task_analysis(
            LmEvalAnalyzer(),
            input_task_results=input_task_results,
            thresholds=thresholds,
            output_dir=output_dir,
        )
        return {
            "passed": result.passed,
            "scores": result.scores,
            "thresholds": result.thresholds,
            "deviated_prompts": result.deviated_prompts,
            "report_path": result.report_path,
        }

    def run_logit_validation(self, model_id, tp_size, prompts, output_length=16,
                             output_dir="./accuracy_report", **kwargs):
        from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.logit_val import LogitValPlugin

        return self._run_prompt_plugins(
            model_id, tp_size, prompts, output_length, output_dir,
            plugin_steps=[LogitValPlugin()], **kwargs,
        )

    def run_kv_cache_analysis(self, model_id, tp_size, prompts, output_length=16,
                              output_dir="./accuracy_report", **kwargs):
        from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.kv_cache import KvCachePlugin

        return self._run_prompt_plugins(
            model_id, tp_size, prompts, output_length, output_dir,
            plugin_steps=[KvCachePlugin()], **kwargs,
        )

    def run_prompt_diagnosis(self, model_id, tp_size, prompts, output_length=16,
                             output_dir="./accuracy_report", plugins=None, **kwargs):
        if plugins is None:
            from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.kv_cache import KvCachePlugin
            from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.logit_val import LogitValPlugin
            plugin_steps = [LogitValPlugin(), KvCachePlugin()]
        else:
            from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins import PLUGIN_REGISTRY
            unknown = [name for name in plugins if name not in PLUGIN_REGISTRY]
            if unknown:
                raise ValueError(f"Unknown plugins: {unknown}. Available: {list(PLUGIN_REGISTRY.keys())}")
            plugin_steps = [PLUGIN_REGISTRY[name]() for name in plugins]

        return self._run_prompt_plugins(
            model_id, tp_size, prompts, output_length, output_dir,
            plugin_steps=plugin_steps, **kwargs,
        )

    def _run_prompt_plugins(self, model_id, tp_size, prompts, output_length,
                            output_dir, plugin_steps, **kwargs):
        from vllm_neuron.accuracy.accuracy_debugger import run_prompt_analysis

        server_config = {
            "model": model_id,
            "tp_degree": tp_size,
            "max_model_len": kwargs.get("max_model_len", 8192),
            "batch_size": kwargs.get("batch_size", 1),
        }
        additional_config = kwargs.get("additional_config")
        if additional_config:
            server_config["additional_config"] = additional_config

        result = run_prompt_analysis(
            server_config=server_config,
            prompts=prompts,
            plugin_steps=plugin_steps,
            output_dir=output_dir,
            output_length=output_length,
        )
        return {
            "prompts": result.prompts,
            "plugin_results": result.plugin_results,
            "report_path": result.report_path,
        }

    def _start_server(self, model_id, tp_size, **kwargs):
        """Start a vLLM server for task-level analysis."""
        import shlex
        from test.utils.simple_server import start_server

        max_model_len = kwargs.get("max_model_len", 8192)
        if not isinstance(tp_size, int) or tp_size < 1:
            raise ValueError(f"tp_size must be a positive integer, got {tp_size}")
        if not isinstance(max_model_len, int) or max_model_len < 1:
            raise ValueError(f"max_model_len must be a positive integer, got {max_model_len}")

        # Ensure Neuron env vars are set for the server subprocess
        os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("PJRT_DEVICE", "NEURON")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MALLOC_ARENA_MAX", "64")

        # Ensure vllm CLI is on PATH (may be in a venv)
        venv_bin = os.path.dirname(sys.executable)
        if venv_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

        additional_config = kwargs.get("additional_config")
        cmd = f"vllm serve {shlex.quote(str(model_id))} --tensor-parallel-size {tp_size} --max-model-len {max_model_len}"
        if additional_config:
            import json as _json
            cmd += f" --additional-config '{_json.dumps(additional_config)}'"

        return start_server(cmd)


# ── Helpers ──

def _parse_version(v):
    """Parse a version string into a (major, minor, patch) int tuple.

    Tolerates suffixes like '0.24.0+neuron', '0.24.0rc1', and dev tags.
    """
    head = re.split(r"[^0-9.]", str(v).strip())[0]
    parts = head.split(".")
    nums = []
    for p in parts[:3]:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def _minor_line(v):
    """Return the 'major.minor' string for a version (e.g. '0.24')."""
    major, minor, _ = _parse_version(v)
    return f"{major}.{minor}"


def _same_minor_line(installed, pinned):
    """Whether two versions share the same major.minor line."""
    return _parse_version(installed)[:2] == _parse_version(pinned)[:2]


def _to_tensor_list(x):
    """Flatten nested tensors/lists into a flat list of tensors."""
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(_to_tensor_list(item))
        return out
    raise TypeError(f"Expected tensor or sequence, got {type(x)}")


def _import_module(filepath):
    module_file = os.path.abspath(filepath)
    module_dir = os.path.dirname(module_file)

    # Detect if file is inside a package (has __init__.py in parents)
    # and import properly to support relative imports
    parts = []
    check_dir = module_dir
    while os.path.exists(os.path.join(check_dir, "__init__.py")):
        parts.insert(0, os.path.basename(check_dir))
        check_dir = os.path.dirname(check_dir)

    if parts:
        if check_dir not in sys.path:
            sys.path.insert(0, check_dir)
        module_name = ".".join(parts) + "." + os.path.basename(module_file).replace(".py", "")
        return importlib.import_module(module_name)

    # Fallback: standalone file
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    mod_name = f"_target_{os.path.basename(module_file).replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(mod_name, module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _get_model_config(model):
    """Extract config from model, handling nested model.config patterns."""
    if hasattr(model, "config"):
        return model.config
    if hasattr(model, "model") and hasattr(model.model, "config"):
        return model.model.config
    return None


def _get_layer_types(config, num_layers):
    """Return a per-layer attention-type list, or None for non-hybrid models.

    Hybrid configs expose layer_types (e.g. ["linear_attention", ...,
    "full_attention"]) or layers_block_type. Returns None when the model is
    homogeneous (every layer is full attention) so callers can fall back to
    the original "all layers have self_attn" assumption.
    """
    if config is None:
        return None
    for attr in ("layer_types", "layers_block_type", "block_types"):
        types = getattr(config, attr, None)
        if types:
            return list(types)
    return None


def _is_full_attention_layer(layer_types, idx):
    """Whether layer idx is a full (self-)attention layer.

    Non-hybrid models pass layer_types=None → every layer is full attention.
    """
    if layer_types is None:
        return True
    if idx >= len(layer_types):
        return True
    return layer_types[idx] == "full_attention"


def _get_layer(model, idx):
    """Get decoder layer by index, handling nested model structures."""
    for attr in ("model.layers", "layers"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            try:
                return obj[idx]
            except (IndexError, TypeError):
                pass
    return None


def _map_hf_to_vllm_neuron(hf_state, model, dtype):
    """Map HF safetensors weights to vLLM-Neuron format.

    Key differences from HF:
    - All linear weights are TRANSPOSED (.t())
    - Q/K/V are fused into single qkv_proj_weight: cat([Q.t(), K.t(), V.t()], dim=-1)
    - Weight names use _weight suffix (bare Parameters), not .weight (nn.Linear)
    - Norms and embeddings are direct copy (no transpose)
    """
    neuron_state = {}
    target_keys = set(dict(model.named_parameters()).keys())

    # Direct mappings for non-linear weights (norms, embeddings)
    for key, tensor in hf_state.items():
        # Try direct mapping first
        if key in target_keys:
            neuron_state[key] = tensor.to(dtype)
            continue

        # Try _weight suffix mapping: model.layers.0.mlp.gate_proj.weight -> ...gate_proj_weight
        alt_key = key.replace(".weight", "_weight").replace(".bias", "_bias")
        if alt_key in target_keys:
            # Linear weights need transposition (except gate_weight which stays as-is in float32)
            if tensor.dim() == 2 and "norm" not in key and "embed" not in key and "gate_weight" not in alt_key:
                neuron_state[alt_key] = tensor.t().to(dtype)
            elif "gate_weight" in alt_key:
                neuron_state[alt_key] = tensor  # keep original shape and float32
            else:
                neuron_state[alt_key] = tensor.to(dtype)
            continue

    # Handle fused QKV if target has qkv_proj_weight
    _fuse_qkv_weights(hf_state, neuron_state, target_keys, dtype, model)

    # Handle MoE expert weight stacking
    _fuse_moe_expert_weights(hf_state, neuron_state, target_keys, dtype)

    return neuron_state


def _fuse_qkv_weights(hf_state, neuron_state, target_keys, dtype, model):
    """Fuse separate Q/K/V weights into combined qkv_proj_weight."""
    # Find all layer prefixes that have separate q/k/v
    qkv_layers = set()
    for key in hf_state:
        m = re.match(r"(.+\.self_attn)\.q_proj\.weight", key)
        if m:
            qkv_layers.add(m.group(1))

    for prefix in qkv_layers:
        fused_key = f"{prefix}.qkv_proj_weight"
        if fused_key not in target_keys:
            continue

        q = hf_state.get(f"{prefix}.q_proj.weight")
        k = hf_state.get(f"{prefix}.k_proj.weight")
        v = hf_state.get(f"{prefix}.v_proj.weight")
        if q is not None and k is not None and v is not None:
            neuron_state[fused_key] = torch.cat(
                [q.t(), k.t(), v.t()], dim=-1
            ).to(dtype)

        # Fuse biases if present
        fused_bias_key = f"{prefix}.qkv_proj_bias"
        if fused_bias_key in target_keys:
            qb = hf_state.get(f"{prefix}.q_proj.bias")
            kb = hf_state.get(f"{prefix}.k_proj.bias")
            vb = hf_state.get(f"{prefix}.v_proj.bias")
            if qb is not None and kb is not None and vb is not None:
                fused = torch.cat([qb, kb, vb]).to(dtype)
                target_param = dict(model.named_parameters()).get(fused_bias_key)
                if target_param is not None and target_param.dim() > fused.dim():
                    fused = fused.reshape(target_param.shape)
                neuron_state[fused_bias_key] = fused


def _fuse_moe_expert_weights(hf_state, neuron_state, target_keys, dtype):
    """Stack per-expert 2D weights into 3D tensors for vLLM-Neuron MoE layers."""
    expert_map = {"w1": "gate_proj_weights", "w3": "up_proj_weights", "w2": "down_proj_weights"}
    moe_prefixes = set()
    for key in hf_state:
        m = re.match(r"(.+\.block_sparse_moe)\.experts\.\d+\.w[123]\.weight", key)
        if m:
            moe_prefixes.add(m.group(1))

    for prefix in moe_prefixes:
        for hf_name, target_name in expert_map.items():
            target_key = f"{prefix}.{target_name}"
            if target_key not in target_keys:
                continue
            expert_weights = []
            idx = 0
            while f"{prefix}.experts.{idx}.{hf_name}.weight" in hf_state:
                expert_weights.append(hf_state[f"{prefix}.experts.{idx}.{hf_name}.weight"].t())
                idx += 1
            if expert_weights:
                neuron_state[target_key] = torch.stack(expert_weights, dim=0).to(dtype)
