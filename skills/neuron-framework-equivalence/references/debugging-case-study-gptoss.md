# Reference Case Study: GPT-OSS 20B Equivalence Debugging

Real-world debugging timeline from porting a 20B MoE model (HuggingFace → NeuronX).
Use this to pattern-match on failures you encounter.

## Timeline

| Stage                     | Duration | What Was Found                                                                                                                        |
| ------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Component debugging (CPU) | 2 weeks  | 4 patches: rmsnorm precision, YaRN rotary scaling, MoE routing + weight layout, attention sinks                                       |
| CPU E2E debugging         | 1 week   | mp.spawn patch inheritance, weight layout fix, bias restoration                                                                       |
| Device E2E debugging      | 2 weeks  | Root cause #1: ParallelEmbedding missing SPMDRank (590→127). Root cause #2: Windowed attention path missing sink injection (127→0.97) |

## Final Result

| Metric                 | Value                                                  |
| ---------------------- | ------------------------------------------------------ |
| Full model error_ratio | 1.0005                                                 |
| Top-1 token match      | Yes                                                    |
| 20-token generation    | Coherent ("a topic of much debate and speculation...") |
| Total patches          | 5 (3 unified CPU+device, 2 device-only)                |

---

## Component-Level Results (Stage 2)

| Component                    | Error Ratio | Threshold | Result | Root Cause                                       |
| ---------------------------- | ----------- | --------- | ------ | ------------------------------------------------ |
| RMSNorm                      | 1.85        | 2.0       | PASS\* | Neuron computes variance in bf16 instead of fp32 |
| Embedding                    | 1.00        | 1.1       | PASS   | Identical bf16 lookup                            |
| Rotary Embedding             | 130–133     | 1.2       | FAIL   | Missing YaRN scaling entirely                    |
| Linear Projections (Q/K/V/O) | 1.15–1.17   | 1.2       | PASS   | Identical linear math                            |
| Router                       | 0.57        | 1.2       | PASS   | Same linear + topk + softmax                     |
| Experts (with routing)       | 1774        | 1.2       | FAIL   | Neuron ignores routing entirely                  |
| Full MLP/MoE                 | 1809        | 1.2       | FAIL   | Cascaded routing + experts bugs                  |
| LM Head                      | 1.00        | 1.2       | PASS   | Identical linear math                            |

\*RMSNorm was hidden by a relaxed 2.0 tolerance. After patching, it dropped to 1.0.

---

## Bug #1: YaRN Rotary Embedding (R=130x)

**Symptom:** `test_02_rotary_emb.py` — error ratio 130x (cos) and 133x (sin).

**Diagnosis:**

- HF `GptOssRotaryEmbedding` uses `ROPE_INIT_FUNCTIONS["yarn"]` which blends interpolated/extrapolated `inv_freq` and applies `attention_scaling = 1.3466` to cos/sin.
- Neuron `NeuronGptOssRotaryEmbedding` wraps the framework's basic `RotaryEmbedding(dim, max_pos, base)` which only computes standard RoPE. The `rope_scaling` config is stored but never used.

**Patch:** `yarn_rotary_patch.py`:

1. `_patched_init`: computes YaRN `inv_freq` and `attention_scaling`, stores as `self._yarn_inv_freq` and `self.attention_scaling`
2. `_patched_forward`: computes rotary embeddings in fp32 using YaRN inv_freq, applies attention_scaling before dtype cast

**Pitfalls encountered:**

1. Neuron config's `rope_scaling` dict was missing `original_max_position_embeddings` (4096). Derived: `original = max_position_embeddings / factor`.
2. Framework's `register_buffer("inv_freq", None)` resisted direct assignment. Stored `_yarn_inv_freq` on the wrapper instead.
3. Applying `attention_scaling` after the framework cast cos/sin to bf16 caused sin error ratio of 13.7x. Fixed by reimplementing forward inline, applying scaling in fp32 before the cast.

**Result:** Error ratio 1.0000 for both cos and sin.

---

## Bug #2: RMSNorm Precision (R=1.85x)

**Symptom:** `test_00_rmsnorm.py` — error ratio 1.85x.

**Diagnosis:** Neuron `StandardRMSNorm` casts to `torch.bfloat16` before `pow(2).mean()`. HF `GptOssRMSNorm` casts to `torch.float32`. Computing variance in bf16 loses precision due to 7-bit mantissa.

**Patch:** `rmsnorm_patch.py` — patches the `get_rmsnorm_cls()` factory function. The patched forward casts to `float32` matching HF.

**Pitfall:** Patching the class returned by one `get_rmsnorm_cls()` call didn't affect subsequent calls because the function defines a new class each time. Fixed by patching the factory function itself.

**Result:** Error ratio 1.0000.

---

## Bug #3: MoE Routing + Weight Layout (R=1774x)

**Symptom:** `test_05_experts.py` — error ratio 1774x. `test_06_mlp_moe.py` — error ratio 1809x.

**Diagnosis:** Three bugs:

1. `NeuronGptOssExperts.forward()` ignores `router_indices` and `routing_weights` — all tokens go through all 32 experts instead of top-4.
2. `NeuronGptOssMLP.forward()` uses `router_scores.sum(dim=-1)` as multiplier — softmax scores sum to ~1.0, making it a no-op.
3. Down-proj weight layout mismatch (`i*E+e` vs `e*I+i` indexing) and per-expert biases collapsed to single sum.

**Patch:** `mlp_moe_patch.py`:

1. Extracts per-expert weights from flattened RowParallelLinear via reshape + permute
2. Computes per-expert down projections via `einsum('nei,eih->neh')`
3. Applies routing weights per expert, then sums across experts

**Pitfall (over-precision):** Initial patch used `.float()` on einsum operands. This produced error ratio 0.82 (< 1.0) — the target was artificially closer to fp32 than the bf16 HF reference. HF `GptOssExperts.forward()` has zero `.float()` calls. Fixed by removing all `.float()` calls.

**Result:** Error ratio ~0.94.

---

## CPU E2E Debugging (Stage 4 → E2E)

### Primary Issue: mp.spawn Doesn't Inherit Patches

`torch.multiprocessing.spawn()` creates brand new child processes. All monkey patches must be re-applied inside each worker function BEFORE model instantiation.

**Detection:** TP=1 passed, TP>1 produced garbage. Captured `_patched` flag: `True` in TP=1, `False` in TP>1 workers.

**Fix:** Added `apply_all_patches()` as first action in `_tp_worker()`.

### Secondary Issues at TP>1

| Issue                                                 | Fix                                                      |
| ----------------------------------------------------- | -------------------------------------------------------- |
| Down-proj weight layout incompatible with TP sharding | `fix_down_proj_layout()` before `get_sharded_checkpoint` |
| Attention sink parameters not TP-sharded              | Manual slicing: `sinks[rank*local:(rank+1)*local]`       |
| Biases removed by `get_sharded_checkpoint`            | Manual bias restoration with TP-aware slicing            |

### CPU E2E Final Results

| Test               | Metric       | Value    | Threshold | Result |
| ------------------ | ------------ | -------- | --------- | ------ |
| FP32 Direct TP=1   | rel_fro_norm | 3.986e-7 | 1e-5      | PASS   |
| 3-Tensor BF16 TP=1 | error_ratio  | 0.878    | 1.2       | PASS   |
| 3-Tensor BF16 TP=4 | error_ratio  | 0.850    | 1.2       | PASS   |
| 3-Tensor BF16 TP=8 | error_ratio  | 0.976    | 1.2       | PASS   |

---

## Device E2E Debugging (Stages 5+)

### Root Cause #1: ParallelEmbedding Missing SPMDRank (error_ratio 590 → 127)

`NeuronGptOssModel` created `ParallelEmbedding` without `use_spmd_rank=True`. All 8 TP ranks used rank-0 vocabulary boundaries, producing the sum of 8 wrong embeddings.

**Fix:** `embedding_patch_device.py` — recreate `embed_tokens` with `use_spmd_rank=True`, inject `embed_tokens.rank_util.rank = torch.arange(tp_degree)` in `pre_shard_weights_hook`.

### Root Cause #2: Attention Sinks on Windowed Path (error_ratio 127 → 0.97)

GPT-OSS layer 0 is `sliding_attention`, which dispatches to `perform_prefill_windowed_attn`. The original `attention_sink_patch.py` only patched `perform_prefill`, missing the windowed path entirely. This worked on CPU by coincidence (TP=1 always uses `perform_prefill`), but on device the windowed dispatch path was active.

**Fix:** Updated `attention_sink_patch.py` to patch both `perform_prefill` and `perform_prefill_windowed_attn`.

### Key Insight

The two device-specific root causes would never have been found by CPU testing alone:

- **SPMDRank:** `parallel_state.get_rank()` returns correct values on CPU but bakes as constant 0 during XLA tracing
- **Windowed attention:** CPU used the patched path by coincidence (TP=1 always uses `perform_prefill`), while device layer 0 used the unpatched windowed path

---

## Final Patch Inventory

| Patch                            | What It Fixes                          | CPU/Device |
| -------------------------------- | -------------------------------------- | ---------- |
| `yarn_rotary_patch.py`           | YaRN scaling in rotary embeddings      | Both       |
| `rmsnorm_patch.py`               | FP32 variance computation              | Both       |
| `mlp_moe_patch.py`               | MoE routing + weight layout            | Both       |
| `attention_sink_patch.py`        | Sinks on both attention paths          | Both       |
| `embedding_patch_device.py`      | SPMDRank for ParallelEmbedding         | Device     |
| `attention_bias_patch_device.py` | Enable attention biases in projections | Device     |
