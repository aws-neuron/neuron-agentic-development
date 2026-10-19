---
name: neuron-framework-autoport-vllm-neuron
description: Port a HuggingFace model architecture to the vLLM-Neuron Trainium2 backend. Handles architecture research, code generation, and validation.
allowed-tools: Read Write Edit Bash Grep Glob WebFetch Agent Skill
argument-hint: "<model-name> <hf-model-id> [--review]"
---

# vLLM-Neuron Model Porting Skill

Port the HuggingFace model `$1` to the vLLM-Neuron Trainium2 backend as `$0`.

## Conventions

- `MODEL_NAME` = `$0` (snake_case, e.g., `yi`, `deepseek_v2`)
- `HF_MODEL_ID` = `$1` (e.g., `01-ai/Yi-6B-Chat`)
- `--review` flag (if present in `$ARGUMENTS`): pause at Step 2 for user confirmation
- Derive `PascalName` from MODEL_NAME (e.g., `yi` → `Yi`, `gpt_neox` → `GPTNeoX`)
- Model code goes in `vllm_neuron/model`
- Example script goes in `examples/vllm_neuron/models/MODEL_NAME/`

## Dry-run

When the user specifies `dry-run`:
- **Skip** all agent-level prerequisites (package import checks, NeuronCore check via `neuron-ls`)
- **Activate** the provided venv and resolve source paths by filesystem lookup (do NOT use `import` — dependencies are not installed):
  ```bash
  export PATH=<pathToVenv>/bin:$PATH
  VLLM_NEURON_SRC=<path-to-vllm_neuron-source>   # e.g. the vllm_neuron/ dir inside the checkout
  TRANSFORMERS_SRC=$(python3 -c "import site; print(site.getsitepackages()[0])")/transformers
  ```
- **Do not run any code** — no compilation, inference, or validation (no Trainium hardware available)

## Before You Start

Ensure the agent-level prerequisites have been completed (venv activation, package verification, NeuronCore check). These steps are owned by the agent prompt and should not be re-executed here.

> **Cache hygiene (read before trusting any validation result):**
> - Do NOT proactively clear `/var/tmp/neuron-compile-cache` — it is a shared system directory and other processes or users may depend on it. Only clear it reactively on a `[NLA001]` JSON parse error or `FileNotFoundError` on neff_output paths.
> - When `$HOME` is on NFS/Lustre the vLLM runner falls back to a *separate* compile cache under `/tmp/vllm_neuron_wdir_$USER/neuron/compile_cache` (see `vllm_neuron/envs.py`, `os.path.expandvars("/tmp/vllm_neuron_wdir_$USER")`). The runner **logs the resolved path** on startup (`Default compile cache path ... is on a remote filesystem ... Falling back to <PATH>.`). **Do not guess the path** — `expandvars` leaves a literal `$USER` in the directory name when the `USER` env var is unset (common under Slurm/`srun`) but expands it when set, so a shell `rm -rf /tmp/vllm_neuron_wdir_$USER/...` can resolve to a *different* directory than the runner used. To clear it, copy the exact `Falling back to ...` path from the run log and `rm -rf '<that exact path>'/*` (single-quote to stop the shell re-expanding `$USER`) — and only reactively.
> - **A run that hits a stale/poisoned cache can silently PASS validation against the OLD graph.** After any change to `model.py`, before trusting a smoke/logit/equivalence result, confirm the graph was actually recompiled. The compiler logs one line per graph: a cache hit prints `Using a cached neff at <path>`, a fresh compile prints `Compilation Successfully Completed for <name>`. There is **no** aggregate summary line — count the individual lines. If the run shows only `Using a cached neff` lines after you edited the model, you validated the stale graph. When in doubt, clear the correct cache path (from the log line above) and re-run.

## Step 0: Parse & Detect State

1. Package check: Ensure the `vllm_neuron` package exists in the user's workspace (the containing repo/directory name varies by installation — locate it with `python3 -c "import vllm_neuron; print(vllm_neuron.__file__)"` or `find . -maxdepth 3 -type d -name vllm_neuron`). Also confirm the porting tree is present: `ls vllm_neuron/model/ vllm_neuron/functional/`. If the package or either tree is missing, STOP and inform the user.
2. Parse `$ARGUMENTS` — extract model-name, hf-model-id, and `--review` flag.
3. Check if `vllm_neuron/model/MODEL_NAME/` already exists:
   - If model.py + registry entry both exist → skip to **Phase C** (validation)
   - If model.py exists but not registered → resume at **Step 5** (register)
   - If directory doesn't exist → proceed to **Phase A**

---

## Phase A: Research & Analysis

### Step 1: Architecture Research

Gather all information needed to port the model. Do NOT write any files yet.

**1a. Fetch HF config:**

**Precheck: confirm the installed transformers registers the target architecture AND stays inside the vllm-neuron dependency window.** For a newly-released model the config class may not exist in the installed transformers, so `AutoConfig`/`from_pretrained` fails or falls back to a wrong class:
```bash
python3 -c "import transformers, vllm; print('transformers', transformers.__version__); print('vllm', vllm.__version__)"
python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('HF_MODEL_ID', trust_remote_code=True); print(type(c).__name__, c.architectures)"
```
If the arch is unrecognized, find the minimum transformers version that ships it — but do NOT upgrade blindly. vllm-neuron pins its dependencies, and transformers/vllm are tightly coupled. Verify the needed version falls inside the installed vllm-neuron's declared window BEFORE upgrading:
```bash
# Read the pins YOUR checkout declares — the vllm pin advances every release
# (0.21.0 at time of writing), so never carry one over from memory:
grep -rniE 'transformers' <vllm_neuron_repo>/requirements/core.txt
grep -rniE '^vllm==' <vllm_neuron_repo>/requirements/core.txt
# If requirements/core.txt has no transformers line, the release does not constrain it —
# fall back to what the pinned vllm itself requires.
# Confirm the transformers version that ships your arch satisfies BOTH
# (a) vllm-neuron's transformers pin and (b) the range the pinned vllm accepts:
python3 -m pip install "transformers==<needed>" --dry-run   # surfaces conflicts without mutating the env
```
If the arch requires a transformers version OUTSIDE vllm-neuron's window (e.g. a 4.x→5.x or 5.x→6.x jump), STOP: bumping transformers past the pin can silently break the pinned `vllm` (import errors, changed processor/config APIs) and is a project-level dependency decision, not a per-port one. Record the required version and the conflict in the Step 2 analysis and get sign-off before upgrading.

Then fetch the config:
```python
python3 -c "
from transformers import AutoConfig
import json
c = AutoConfig.from_pretrained('HF_MODEL_ID', trust_remote_code=True)
print(json.dumps(c.to_dict(), indent=2, default=str))
"
```

Extract and record:
- `architectures` field (exact string for registry, e.g., `"YiForCausalLM"`)
- `hidden_size`, `intermediate_size`, `num_hidden_layers`
- `num_attention_heads`, `num_key_value_heads`, `head_dim` (compute if missing: hidden_size / num_attention_heads)
- `vocab_size`, `max_position_embeddings`
- `rms_norm_eps` or `layer_norm_epsilon`
- `rope_theta`, `rope_scaling` (NOTE: some models use `rope_parameters` instead — check both)
- `hidden_act` (activation function)
- `tie_word_embeddings`
- Bias flags: `attention_bias`, `mlp_bias`, `lm_head_bias` (or check transformers source)
- Any model-specific fields: `sliding_window`, `num_local_experts`, `partial_rotary_factor`, etc.

Additionally, scan for these **runtime-surprise traits** — each changes the reference choice or a kernel contract and has silently broken past ports if missed:

- **MLA / latent attention:** `kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`. If present, the effective attention `head_dim` is NOT `hidden_size / num_attention_heads` — for MLA it is `qk_rope_head_dim + kv_lora_rank` (e.g. DeepSeek-V3.2 = 64 + 512 = 576). Compute the real per-layer head_dim from these.
- **Heterogeneous / hybrid layers:** `layer_types`, `full_attention_interval`, `linear_attention`, `sliding_window_pattern`. A model may mix sliding-window vs global attention (different head_dim AND kv-head counts per layer — Gemma4 has SWA head_dim=256/16-KV and global head_dim=512/4-KV) or mix full-attention with linear-attention/GatedDeltaNet layers (Qwen3.5-27B: 3:1 linear:full). Record the exact per-layer pattern.
- **Score/logit shaping:** `final_logit_softcapping`, `attn_logit_softcapping`, `query_pre_attn_scalar`, `use_qk_norm` (per-head Q/K RMSNorm). When QK-norm is present the attention scaling is often 1.0 (QK-norm replaces the 1/sqrt(head_dim) factor) — do not assume `scale = 1/sqrt(head_dim)`.
- **Attention output gating:** if `q_proj` output width is `2 * head_dim` per head, or the HF forward multiplies attention output by a `sigmoid`/`softplus` gate. NOT expressible by `NF.attention_decode`/`NF.flash_attention` — forces a hand-rolled decode path. Flag it now.

**CRITICAL — Cross-check AutoConfig against the raw config.json.** `AutoConfig.from_pretrained` populates config-class DEFAULT values for fields the model's own `config.json` omits (e.g. Gemma's `query_pre_attn_scalar=256` is a transformers text-config class default, NOT in the checkpoint's config.json). A default silently substituted for a missing field can produce a subtly wrong attention scaling / sliding window / softcap that passes smoke tests but fails accuracy validation. Always diff the AutoConfig dump against the raw `config.json` and treat the raw file as authoritative for any attention-scaling / window / softcap field:
```python
# Prefer the local checkpoint config.json (works on offline compute nodes):
python3 -c "import json; print(json.dumps(json.load(open('CHECKPOINT_DIR/config.json')), indent=2))"
# Or, only if the node has internet, fetch the raw file from HF:
python3 -c "import urllib.request; print(urllib.request.urlopen('https://huggingface.co/HF_MODEL_ID/raw/main/config.json').read().decode())"
```

**1b. Read HF transformers source:**
Find the modeling file for this architecture. Check:
- Exact weight names (for checkpoint key mapping in `load_weights`)
- Whether Q, K, V projections are fused or separate in the checkpoint
- Bias presence on each projection (Q, K, V, O, gate, up, down, layernorm)
- Activation function implementation
- Any non-standard behavior (e.g., parallel residual, special normalization)

**CRITICAL — Verify normalization type by reading the forward() implementation, not the class name.**
Some models name their norm class "RMSNorm" but actually implement full LayerNorm (with mean subtraction and bias). The difference:
- RMSNorm: `variance = x.pow(2).mean(); x = x * rsqrt(variance + eps)` — no mean subtraction, no bias
- LayerNorm: `mean = x.mean(); variance = (x - mean).pow(2).mean(); x = (x - mean) / sqrt(variance + eps)` — has mean subtraction, may have bias

Using the wrong normalization produces subtly wrong output that passes smoke tests but fails accuracy validation.

**Also read the reference forward() for two things beyond norm-type:**
- **Weight/compute dtype precision.** Note any place the reference upcasts to fp32 for a weight multiply or reduction (e.g. `x * self.weight.float()`, fp32 router gate, fp32 variance). Losing an fp32 cast produces high-cosine-but-large-absolute divergence (a DeepSeek RMSNorm bug reached max_abs=64 at cosine 0.99996) that smoke tests miss.
- **Novel/iterative algorithms.** Position-bias, Sinkhorn/normalized routing, hyper-connections, delta-net recurrences etc. must be read step-by-step (Sinkhorn is iterative row-AND-column normalization, not a single row-norm). Transcribe the exact iteration count and axes normalized — these are not in config.json and are a common source of post-hoc equivalence failures.

**1c. Read the vLLM-Neuron porting guide:**
Locate the model-onboarding guide in your checkout — it has been renamed between releases, so find it rather than assuming a path:
```bash
find docs -iname '*onboard*' -o -iname '*model_bringup*'   # currently docs/model-dev/onboarding-models.md
ls docs/design/parallelism/                                # TP, DP, EP design docs
```
The guide designates an in-tree model as the canonical annotated reference (the GPT-OSS implementation, `vllm_neuron/model/gpt_oss/`; confirm with `grep -rl 'PARALLELISM' vllm_neuron/model/*/model*.py`). In that reference, `# >>> PARALLELISM <<<` blocks are infrastructure to keep as-is when porting; `# <-- MODEL-SPECIFIC` blocks are what change per architecture. The canonical code is a structural template, not a spec for the target model — derive all model-specific decisions from the target's HF `modeling_*.py` and `config.json`, never copy MODEL-SPECIFIC sections blindly. Read the parallelism design docs the guide links before porting; misunderstanding a collective or shard dimension causes silent accuracy bugs.

**1d. Select best reference model:**

**List what your tree actually ships first.** The released plugin carries a deliberately small set of reference ports, and that set changes between releases (a model present in one release may be dropped from the next):
```bash
ls vllm_neuron/model/     # authoritative list of reference ports available to you
```
At the current release that is `gpt_oss/` (MoE + EP, the canonical annotated reference), `llama3/` (dense GQA + RoPE, plus FP8/MX and Eagle3 variants), `qwen3_vl/` (vision-language), and `synthetic/` (a config-driven test harness, not a porting reference).

Match the target's distinguishing trait to the closest available port. Most architectures do NOT have an exact in-tree match — pick the nearest structural relative and derive the rest from the target's HF `modeling_*.py`:

| If the model has... | Start from | Notes |
|---|---|---|
| Standard GQA + RoPE (most common) | `llama3/model.py` | the default starting point for any dense decoder |
| QKV bias | `llama3/` | add bias params to the QKV/O projections |
| Quantized weights (FP8 / MX) | `llama3/model_mx_fp8.py`, `model_static_fp8.py` | paired `weight_loaders_*` and `weight_pack_*` show the packing contract |
| Mixture of Experts (with or without EP) | `gpt_oss/model_bf16.py` | see the EP section below; `model_mxfp4.py` for quantized MoE |
| Vision-language / multimodal | `qwen3_vl/` | `vision_encoder_bf16.py`, `utils/mrope.py`, `utils/vision_block_packing.py` |
| Speculative decoding / draft model | `llama3/eagle3_model.py` | — |
| Q/K per-head RMSNorm | `llama3/` + `NF.attention_decode(rmsnorm_QK_*=...)` | the kernel takes QK-norm weights directly; check the signature |
| MLA / latent attention, hybrid or linear attention, ALiBi, learned position embeddings, non-gated MLP, attention output gates, head_dim > 128 | **no in-tree reference at this release** | confirm with the discovery greps below; hand-roll from the HF source and record the absence in the Step 2 analysis |

Before concluding a trait is unsupported, grep for it — the set grows release to release:
```bash
grep -rl 'kv_lora_rank'                vllm_neuron/model/   # MLA / latent attention
grep -rl 'Conv1d\|conv_state\|linear_attention' vllm_neuron/model/   # hybrid / GatedDeltaNet / Mamba
grep -rl 'alibi'                       vllm_neuron/model/   # ALiBi
grep -rl 'layer_types'                 vllm_neuron/model/   # heterogeneous per-layer attention
grep -rl 'attention_decode_kernel'     vllm_neuron/model/   # custom d-tiled kernel for head_dim > 128
```

Read the selected reference model's `model.py`, `config.py`, and `factory.py`. If NO in-tree port matches the target's distinguishing trait, say so explicitly in the Step 2 analysis rather than forcing a poor reference — and expect a hand-rolled block plus a recorded blocker instead of a kernel call.

**1e. Parallelism & memory sizing:**

A back-of-envelope sizing pass you do by hand from the HF `config.json` + the trn2
constants below — no scripts, no torch. It is a **theoretical lower bound** whose only job
is to point the first on-hardware run (real ITL/TTFT will be higher). Measure, then trust.

**trn2 constants (per LOGICAL NeuronCore = 2 physical, LNC2, bf16):**
`TE_PEAK = 79e12 FLOP/s` · `HBM_BW = 716e9 B/s` · `HBM_CAP = 24 GiB` (25769803776 B) ·
`cores_per_node = 64` · within-node link `1.28e12 B/s`, cross-node `400e9 B/s`.
Per-core memory budget = `0.90 × 24 GiB ≈ 21.6 GiB` (rest is runtime/fragmentation).
Let `d = dtype_bytes` (bf16 → 2). TP degrees map onto logical cores, so TP=32 is half a
node and TP=64 is one full node — NOT '2 nodes'. Confirm the on-node count with `neuron-ls`.

**1. TP feasibility — all 5 must hold; TP ceiling = `num_attention_heads`** (TP shards the
query projection; going past it splits a head's `head_dim`):
1. `num_attention_heads % TP == 0`
2. `(num_attention_heads / TP)` is **even** (NKI decode-megakernel constraint)
3. `num_key_value_heads % TP == 0` **OR** `TP % num_key_value_heads == 0` (GQA replication)
4. `intermediate_size % TP == 0`
5. `vocab_size % TP == 0` (or pad)

Models with heterogeneous per-layer head counts must satisfy all 5 for *every* layer type
(e.g. Laguna: 48 heads on full-attn, 72 on sliding — TP=4 works, 12 & 18 heads/rank both even; TP=8 fails, 72/8=9 is odd; TP=16 fails, 48/16=3 is odd). Cap TP at the ceiling and absorb the rest into EP/DP.

**2. KV replication (the non-obvious one).** Per-device KV divides by
`min(TP, n_kv_heads)`, **NOT** by `TP`. Once `TP > n_kv_heads` extra ranks hold *redundant*
KV copies (replication factor `TP / min(TP, n_kv_heads)`). Q heads shard by full TP; K/V
shard to 1/rank then replicate. Consequence: high-TP decode goes **KV-read-bound early** at
long context, and a high-TP prefill pool **replicates (wastes) the prefix cache**. The
binding number is the *smallest* `n_kv_heads` across attention groups.

**3. CP (context parallelism)** is carved from TP (`TP % CP == 0`, **adds no cores**). It
shards the KV cache sequence dim and prefill compute with **no** replication ceiling — the
long-context lever; only helps decode when KV-bound. **Read the DCP validation your build
actually applies before assuming a constraint** — it has changed direction between releases:
```bash
grep -rn 'decode_context_parallel_size\|dcp' vllm_neuron/vllm/platform.py
```
On the current release the plugin *relaxes* vLLM's upstream DCP check for non-MLA models
(`_apply_dcp_patch` monkey-patches `ModelConfig.verify_with_parallel_config` to skip the
assertion, because prefill DCP shards attention across the DCP sub-group rather than full TP);
other builds have instead *enforced* `tp_size >= dcp_size * num_kv_heads`. So do not hard-code
either rule. What holds regardless: `num_kv_heads` (= `get_total_num_kv_heads()`) is the binding
quantity, so heterogeneous-KV models with a small KV-head count are the ones that strain CP at
low TP. Read the validator, try the config, and fall back to CP=1 if it rejects.

**4. DP replicates** the whole model (full weights + KV per replica, zero cross-replica
comm): dense → free throughput scale-out (aggregate throughput ×DP). **EP (MoE only):**
routed experts shard across `world = TP·DP`; gate `num_local_experts % ep_degree == 0`,
`ep_degree = world / tp_sub`. Attention/dense-MLP still shard by `tp_sub`; ALL collectives
use the single world group (see EP section). Dense models have no experts → EP is a no-op.

**5. Per-device HBM fit** (replaces the vague rule of thumb). Fits if:
`weight_bytes/TP  +  KV_bytes/min(TP,n_kv_heads)  +  activation_headroom  ≤  0.90 × 24 GiB`
(for EP, expert weights divide by `world` instead of TP; attention/dense by `tp_sub`).
- `weight_bytes = embed_table + Σ_layers(attn + mlp)`, all ×`d`:
  - `embed_table = vocab·hidden·d` (×1 if `tie_word_embeddings`, else ×2 for lm_head)
  - GQA attn/layer = `hidden·n_q·head_dim + 2·hidden·n_kv·head_dim + n_q·head_dim·hidden`
  - SwiGLU MLP/layer = `3·hidden·intermediate`; MoE = `3·hidden·moe_inter·num_experts`
    (+ shared/dense path if present, run every token)
- `KV_bytes = 2 · n_kv_heads · head_dim · n_layers · seq · batch · kv_dtype_bytes`
  (mixed-attention: sum per group, capping sliding-window layers at `min(seq, window)`)
- `activation_headroom` ≈ `4 · hidden · tokens_per_step · d` (coarse; compile is the arbiter)

Pick the **smallest valid TP ≥ 2 that fits**; raise TP (then add CP for long context) if not.

**6. Decode ITL (roofline, HBM-bound):** one step re-reads per-rank weights + per-rank KV.

```
ITL_ms ≈ (weight_bytes/TP  +  KV_bytes/min(TP,n_kv_heads)) / 716e9 · 1000
```

(Divide KV additionally by `CP` when CP>1.) The interactivity lever: raising TP shrinks the
weight term ~1/TP but the KV term only to the `n_kv_heads` floor.

**7. Prefill TTFT (roofline, FLOP-bound, ~100% parallel-efficient):**

```
TTFT_ms ≈ (2 · num_params · prompt_tokens) / (TP · 79e12) · 1000
```

Prefill is compute-bound at every length; scale with cores via **CP** (PP=1 always).

**Worked example — gemma-4-31B (dense) on trn2, TP=8, seq=8192, bs=8:**
config: hidden 5376, inter 21504, 60 layers, 32 q-heads, mixed attn (50 sliding: 16 kv /
hd256 / win1024; 10 global: 4 kv / hd512), vocab 262144, tied embeddings.
- weights: blocks ≈29.4 B params (30.7 B total incl. embeddings) + tied embed table (262144·5376·2 = 2.82 GB), minus the
  phantom global v_proj (V aliases K on `attention_k_eq_v` layers) → **≈ 61.4 GB**.
  Per rank: `61.4/8 ≈ 7.7 GB`.
- KV per rank: sliding `50·2·16·256·2·1024·8 / min(8,16)` + global `10·2·4·512·2·8192·8 /
  min(8,4)` ≈ **2.18 GB**.
- fit: `7.7 + 2.18 + ~2 (act) ≈ 11.9 GB ≤ 21.6 GiB` → fits (TP=8 is the functional-minimum
  region; 3 cores just hold weights).
- ITL ≈ `(7.7e9 + 2.18e9)/716e9·1000 ≈ 13.8 ms`. TTFT (bs=1) ≈ `2·30.7e9·8192/(8·79e12)·1000
  ≈ 800 ms`; at TP=16 → ~400 ms. TP=16 is the interactivity pick — TP=32 would hit the 32-query-head
  ceiling but leaves 1 head/rank, which rule 2 rejects. Its global-layer KV already
  replicates 4× (`min(16,4)`) — pair with CP at long context.

**What this deliberately DROPS vs the full planner** (so treat it as a first cut only): the
config×bucket **sweep + objective-based select**, the topology-aware **comm/collective model**
(AllReduce/ring/all-to-all costs), and **speculative decoding**. It sizes and points the first
run — it is not a substitute for measuring on hardware.

**1f. Inspect checkpoint weight keys:**
```python
python3 -c "
from safetensors import safe_open
import glob
files = sorted(glob.glob('/path/to/weights/*.safetensors'))
for f in files:
    with safe_open(f, framework='pt') as sf:
        for k in sf.keys():
            if 'bias' in k or 'layer.0' in k.replace('layers.0', 'layer.0'):
                print(f'{k}: {sf.get_tensor(k).shape}')
"
```
This reveals which layers have biases, the exact key naming convention, and weight shapes. Essential for building the `load_weights` mapping correctly.

**1g. Evaluate Expert Parallelism (EP) need (MoE models only):**

If the model is MoE and no single TP size satisfies all 5 rules above while also fitting in memory, the model needs **Expert Parallelism (EP)**. EP uses a two-level parallelism: TP_sub for attention/dense layers, EP for distributing experts across groups.

Determine EP need:
- Compute `model_bytes = num_params * 2` (bf16)
- If `model_bytes / max_valid_tp` (plus KV cache and activation headroom) exceeds `0.90 × 24 GiB ≈ 21.6 GiB` per logical NeuronCore → EP required
- EP config: `world_size = 64` (full trn2.48xlarge), `ep_degree = world_size / tp_sub`
- Verify: `num_local_experts % ep_degree == 0` (experts divide evenly across EP groups)

**CRITICAL EP CONSTRAINT — Unified sp_group:**
The Neuron DGE cannot handle two different collective group sizes in the same NEFF. ALL collectives (attention, MLP, MoE, embedding, LM head, sampler) must use a single group — the full world group (64 ranks). Weight sizing uses the sub-group (tp_sub ranks) but collectives go through one unified group. Violating this causes `NEFF Warmup failed with status 1006`.

Reference: the in-tree EP port (`grep -rl 'ep_degree\|expert_parallel' vllm_neuron/model/*/model*.py`; commonly `gpt_oss/model_bf16.py`) for the working EP pattern.

**1h. Kernel-Feasibility & Runtime-Surprise Pre-Check:**

Using the traits from 1a/1b, fill this table BEFORE choosing a reference or writing code. Each 'yes' is a known runtime surprise that changes the plan; record the mitigation in the Step 2 analysis.

| Check | How | If triggered |
|---|---|---|
| **head_dim > 128** | Compute real head_dim (for MLA = `qk_rope_head_dim + kv_lora_rank`; for hybrid, per layer-type). | The shared attention kernels cap head_dim (128 at the time of writing) and do NOT behave uniformly on violation — do not assume a clean compile error. **Read the caps and gates out of your own tree before relying on any of this:** `grep -rn 'MAX_HEAD_DIM' vllm_neuron/functional/attention/` and `grep -rn -A30 '_can_use_attention_block_kernel' vllm_neuron/functional/attention/attention_decode.py`. The three distinct behaviours to expect: `NF.flash_attention` (prefill/CTE) **SILENTLY returns a torch fallback** (slow SDPA, no error) — the real accuracy/perf trap; `NF.segmented_attention` **RAISES `ValueError`** on the device path; `NF.attention_decode`'s Python gate screens `H % 128` and `d_head % 2` but NOT head_dim itself, so an oversized decode routes INTO the partition-tiled megakernel and fails later at kernel-contract validation. `NF.o_proj` is unaffected (it re-tiles via `min(128, ND)`). To run a real NKI attention kernel above the cap you need a custom d-tiled kernel — find one in your tree with `grep -rl 'attention_decode_kernel\|head_dim.*256' vllm_neuron/model/` (no released port ships one; internal ports have used a native 256-SWA / 512-global kernel requiring `head_dim % 128 == 0`). If nothing in-tree handles it, record the port as torch-fallback pending a custom kernel rather than claiming the kernel path. Do NOT assume a kernel is running just because the call compiles; treat any subsequent compile/OOM/kernel-contract symptom (`NCC_INKI016`, `NCC_EOOM002`) as a compiler-debugging hand-off. |
| **Hybrid / linear-attention layers** | `layer_types` contains `linear_attention` / GatedDeltaNet / Mamba. | No NF attention kernel exists for these; the released plugin ships no reference port for them (confirm: `grep -rl 'Conv1d\|conv_state\|linear_attention' vllm_neuron/model/`). Expect to hand-roll the conv + recurrent-state path from the HF source and say so in the Step 2 analysis. Expect the hybrid/Mamba KV-cache allocator to control `block_size` itself (the model hardcodes it and reads the effective value from `attn_metadata`) — verify the effective block_size at runtime rather than assuming a `block_size=` arg is honored. |
| **Attention output gate** | 1a flagged a sigmoid/softplus gate or `q_proj` width = 2×head_dim. | `NF.attention_decode`/`flash_attention` cannot express a multiplicative output gate — decode attention must be hand-rolled. Strip interleaved gate columns from any fused QKV before passing width to `NF.attention_decode` (it infers q-head count from W_qkv width). |
| **Custom router** | Router does sqrt-softplus, sigmoid+bias-for-selection-only, or logsigmoid-renorm. | `NF.router` accepts a **callable** `activation`, a `router_bias`, and several `RouterComputationOrder` modes (`functional/moe/router.py`) — try to express the router with these before hand-rolling. If the bias-vs-activation ordering genuinely does not map, hand-rolling is acceptable; record why. |
| **top-k in a Python loop** | Any `for _ in range(k)` argmax loop or `torch.topk`. | Data-independent-trip-count loops get Dynamo-unrolled into thousands of FX nodes; `torch.topk` lowers to an unsupported `sort` HLO. Follow the top-k rule in Step 4, section 4d. |

If any row triggers on an architecture the reference table has no clean row for, say so explicitly in the Step 2 analysis rather than proceeding as a 'standard' port.

### Step 2: Architecture Analysis

Present a summary comparing this model to the reference:

```
## Architecture Analysis: MODEL_NAME

Reference model: <selected reference>
HF architecture: <exact string>
Recommended TP: <size>

| Section | Reference | This Model | Change Needed |
|---------|-----------|------------|---------------|
| Norm    | RMSNorm   | ...        | ...           |
| RoPE    | Llama3    | ...        | ...           |
| Attention | No bias | ...        | ...           |
| MLP     | SwiGLU    | ...        | ...           |
| MoE     | N/A       | ...        | ...           |
| EP      | N/A       | ...        | ...           |
| Decoder | Pre-norm  | ...        | ...           |
| Backbone | Standard | ...        | ...           |
| LM Head | Tied      | ...        | ...           |

Known pitfalls: <list any that apply>
EP required: <yes/no — if yes, list ep_degree, tp_sub, world_size>
```

If `--review` flag is set: ask the user to confirm or override decisions.
Otherwise: print the analysis and proceed automatically.

### Step 2b: Kernel Selection Table (MANDATORY — produce before writing model.py)

Before generating `model.py`, you MUST fill in this table. Every compute block gets EITHER a concrete `NF.*` symbol / kernel path it will call, OR a one-line blocker reason. "I'll hand-roll it" without a blocker reason is NOT allowed. Fill the "Chosen kernel" column by first running the discovery commands in Phase B's "CRITICAL RULE: Search Before Implementing" against THIS environment's installed `vllm_neuron` — the pre-filled suggestions below are seed defaults, so if discovery exposes a newer/better-fitting kernel, use it here. Print this table in the Step 2 analysis; if `--review`, get user sign-off on any row whose choice is `HAND-ROLLED`.

| Block | Chosen kernel (call site) | If HAND-ROLLED, blocker reason |
|---|---|---|
| RMSNorm / LayerNorm | reference model norm class (shared) or NKI `rms_norm` | |
| QKV projection | `NF.qkv_proj` | |
| Prefill attention core | `NF.flash_attention` (or `NF.segmented_attention` / `NF.segmented_attention_cp` for chunked/CP) | |
| Decode attention core | `NF.attention_decode` | |
| Output projection | `NF.o_proj` | |
| MLP (SwiGLU / gated) | `NF.mlp` | |
| MoE expert compute (prefill) | `NF.moe_cte` + `NF.build_blockwise_mapping` | |
| MoE expert compute (decode) | `NF.moe_block_tkg` (SOFTMAX/SIGMOID routing only) or `NF.moe_cte` (custom routing) | |
| MoE router / top-k | `NF.router` (callable activation + `router_bias`) + `NF.rmsnorm_router_topk_tkg` | |
| Top-k (indexer / sampler) | `from vllm_neuron.functional.topk import topk` (NKI path when the kernel admits the shape, else `torch.topk`; NOT exported as `NF.topk`) | |
| Sampler | `NF.sample` / `vllm_neuron.nn.sampler.Sampler` | |

Rule: a row may be `HAND-ROLLED` ONLY with a concrete blocker reason that you verified by reading the chosen kernel's signature/asserts and confirming it rejects this model's config (an unsupported routing activation, an incompatible head/tile constraint, a required layout the checkpoint can't provide — see "Known reuse blockers" in Phase B). A raw `torch.einsum` / `torch.matmul(q,k^T)+softmax` / `F.silu(x@w)*...` / `for _ in range(k): argmax` with no such verified blocker is a defect to fix before Phase C, not a valid port. This table (and any `HAND-ROLLED` justification) is what the Step 7 gate is verified against — an unused `import vllm_neuron.functional as NF` does NOT satisfy it.

---

## Phase B: Code Generation

### CRITICAL RULE: Search Before Implementing

Before implementing ANY module (norm, RoPE, attention, MLP, etc.), search for an existing implementation in this priority order:

1. **Shared functional ops**: `vllm_neuron/functional/` — `qkv_proj`, `o_proj`, `flash_attention`, `attention_decode`, `mlp`, `sample`, etc.
2. **Shared nn modules**: `vllm_neuron/nn/` — `ColumnParallelLinear`, `RowParallelLinear`, `VocabDimShardedEmbedding`, `Sampler`
3. **Other ported models**: `vllm_neuron/model/*/model.py` — many models share identical implementations of norms, RoPE, etc.
4. **NKI pre-built kernels**: Search the installed `neuronxcc` package for pre-prod kernels (e.g., `python3 -c "import neuronxcc; print(neuronxcc.__path__)"` then look under `nki/_pre_prod_kernels/`): `rms_norm/`, `layernorm_tkg.py`, `mlp/`, `qkv.py`, `attention_token_gen*.py`, `attn_fwd*.py`, `output_proj.py`, `moe_token_gen.py`, `topk/`
5. **NKI production kernels**: Search the installed `neuronxcc` package under `nki/kernels/`

**Only implement from scratch if no existing implementation matches.**

#### First: DISCOVER what kernels the installed package actually exposes

Do NOT treat the table below as the complete set — the installed `vllm_neuron` is the source of truth, and it gains kernels across SDK versions. Before choosing kernels, enumerate what exists in THIS environment and note anything the table omits:

```bash
# Public functional API (authoritative op list; catches kernels added since this skill was written):
python3 -c "import vllm_neuron.functional as NF; print('\n'.join(sorted(getattr(NF, '__all__', dir(NF)))))"
# Submodule kernels not re-exported at top level (e.g. topk, router internals):
python3 -c "import vllm_neuron.functional as F, os, pkgutil; p=os.path.dirname(F.__file__); print('\n'.join(sorted(m.name for m in pkgutil.iter_modules([p]))))"
ls vllm_neuron/functional/ vllm_neuron/functional/*/ 2>/dev/null
# Reference-model usage — the best signal for how a new kernel is actually called:
grep -rhoE 'NF\.[a-z_]+\(' vllm_neuron/model/*/model*.py | sort | uniq -c | sort -rn
```

If discovery surfaces a symbol not in the seed table (or a better-fitting kernel than the one you'd pick), USE IT and note it in the Step 2b table. A kernel that exists in the installed package always beats hand-rolling, whether or not this skill lists it. Only fall back to the seed table's suggestion when discovery confirms no better match exists.

#### Seed op -> symbol map (verified against the source ports; reconcile against discovery above)

| You need... | Use | Import |
|---|---|---|
| QKV / output projection | `NF.qkv_proj`, `NF.o_proj` | `import vllm_neuron.functional as NF` |
| Prefill attention | `NF.flash_attention` | (same) |
| Chunked / context-parallel prefill | `NF.segmented_attention`, `NF.segmented_attention_cp` | (same) |
| Decode attention | `NF.attention_decode` (+ `NF.gen_attention_decode_mask`) | (same) |
| SwiGLU / gated MLP | `NF.mlp` | (same) |
| MoE experts, prefill | `NF.moe_cte` + `NF.build_blockwise_mapping` | (same) |
| MoE experts, decode | `NF.moe_block_tkg` | (same) |
| MoE router (custom activation/bias) | `NF.router` (accepts `activation=<callable>`, `router_bias=`, `router_computation_order=RouterComputationOrder.*`) | `from vllm_neuron.functional.moe.router import router, RouterComputationOrder` |
| Fused RMSNorm+router+topk (decode) | `NF.rmsnorm_router_topk_tkg` | `import vllm_neuron.functional as NF` |
| Top-k across a TP-sharded dim (NKI rotational when the kernel admits the shape; falls back to `torch.topk` otherwise — verify with `_can_use_nki_topk`) | `topk` | `from vllm_neuron.functional.topk import topk`  (NOTE: NOT exported as `NF.topk`) |
| Depthwise causal conv1d + state (Mamba/GatedDeltaNet/short-conv) | copy the pattern in | no released reference port — check `grep -rl 'Conv1d\|conv_state' vllm_neuron/model/` first, else hand-roll: conv1d groups=dim, padding=k-1 for prefill; per-tap decode loop over conv_state |
| Sampler | `Sampler` / `NF.sample` | `from vllm_neuron.nn.sampler import Sampler` |

`NF.router` supports a Callable activation (e.g. `activation=lambda x: F.softplus(x).sqrt()`) and multiple `RouterComputationOrder` modes — check these BEFORE concluding a custom router is required.

This table is a seed, not the whole API. If the discovery step above lists a symbol not here, it is still a valid (and preferred-over-hand-rolling) choice — record it in the Step 2b table.

#### Known reuse blockers (valid reasons to hand-roll instead of reusing a shared kernel)

If you hand-roll an op, record which blocker below applies. If your case isn't listed, verify the blocker by reading the kernel's signature/asserts before hand-rolling — do not hand-roll silently, and do not repeatedly attempt a swap that provably can't work.

- **Per-head multiplicative output gate between attention and o_proj** (e.g. Laguna `softplus(g)*attn`, Qwen3.5 `sigmoid(gate)`): `NF.attention_decode` fuses attn->o_proj and exposes `bias_qkv` / `sink` / `bias_out` but NO multiplicative gate hook (verified in `vllm_neuron/functional/attention/attention_decode.py`, `def attention_decode`). Decode legitimately hand-rolled.
- **Biased-select / unbiased-weight routing** (DeepSeek-v3/v4 style: bias added for top-k SELECTION only, weights taken from unbiased scores): did not map to any `RouterComputationOrder` mode as of writing (the enum carried only pre-norm → linear → {topk,act} orderings). **Enumerate the modes your tree actually offers before hand-rolling** — the enum gains members over time: `grep -nE '^\s+[A-Z_]+ = ' vllm_neuron/functional/moe/router.py`. Router hand-rolled. But FIRST try `NF.router(activation=<callable>, router_bias=...)` — a simple bias and custom activations (e.g. sqrt-softplus) DO map via those params.
- **head_dim > 128**: shared attention kernels only support d_head <= 128 and otherwise fall back to torch or error out — see the head_dim pre-check in Step 1h for exact per-kernel behavior. Reuse an in-tree custom d-tiled decode kernel if one exists (`grep -rl 'attention_decode_kernel' vllm_neuron/model/`); the released plugin ships none, so record torch-fallback explicitly rather than silently accepting it.
- **Custom additive score bias** (e.g. relative-position-logit bias with no RoPE, Inkling): `NF.flash_attention` takes `scale` + `causal_mask`, and `NF.attention_decode` takes `softmax_scale` + `attention_mask` — neither accepts an arbitrary additive bias tensor. Verify the signature before hand-rolling.

For top-k, do NOT reach for an iterative `for _ in range(k): argmax` loop as a first choice, and do NOT assume `torch.topk` is a valid drop-in — see the top-k rule in Step 4.

### Shared imports (ALWAYS use these, never reimplement):

```python
# TP layers
from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear
# Embedding
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
# Sampler
from vllm_neuron.nn.sampler import Sampler
# Functional ops (import as NF)
import vllm_neuron.functional as NF
# KV cache
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
# Weight loaders
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader, fused_qkv_weight_loader,
    sharding_weight_loader, sharding_weight_loader_with_padding,
    last_dim_padding_weight_loader, set_weight_loader,
)
# EP parallel state (only needed for EP models)
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_ep_degree, get_neuron_ep_rank,
    get_neuron_ep_group, get_neuron_ep_tp_group,
)
```

### Whether the stock NKI MLP kernel activates, vs. falling back to PyTorch

`functional/mlp.py::_can_use_kernel` decides, **per TP degree**, whether the fused nkilib
MLP kernel (gate+up+SiLU+mul+down, one launch, no HBM round-trip between stages) is used,
or whether the model silently falls back to a slower, unfused PyTorch MLP. This is decided
by tiling arithmetic on the **per-rank** intermediate size, not the full model's
`intermediate_size` — so the same model can be kernel-accelerated at one TP degree and
PyTorch-fallback at another, with no error or warning either way. Concretely:

```
I_per_rank = intermediate_size // tensor_parallel_size
tiles = ceil(I_per_rank / 512)
_can_use_kernel returns True  iff  tiles <= 8   (and other gates: T > _TKG_BS_SEQLEN_THRESHOLD
                                                   selects CTE vs TKG branch; can_shard_on_i
                                                   requires the FULL hidden_size >= 7168,
                                                   independent of TP)
```

Real example (`intermediate_size=14336`, `TP=16` → `I_per_rank=896` → `tiles=ceil(896/512)=2`
→ kernel **active**): recompute this per model/TP rather than assuming a specific TP degree
generalizes — **do not assume kernel activation is TP-invariant.** Verify empirically at the TP degree you're actually
shipping by inspecting the compiled FX graph
(`/tmp/vllm_cache_*/.../fxgraph.txt`, grep for `torch.ops.higher_order.nki_kernel_wrapper`
around the MLP call) rather than assuming from a "should tile cleanly" mental estimate —
the tiling condition is a `ceil()` boundary check, not a simple "power of 128" rule, and a
close-to-boundary `I_per_rank` is easy to get wrong by hand. If the kernel is NOT active
(pure PyTorch fallback in the graph — visible as `torch.ops.aten.silu`/manual
elementwise-multiply instead of the `nki_kernel_wrapper` call), that fallback path's own
correctness gates apply (see the SiLU/GELU fusion note in "Confirmed levers" in
AutoBridge.md-style workflows, and check `_get_activation_fn` for whether it's using
fused or decomposed activations on that fallback path).

### Step 3: Generate Boilerplate

Generate these 3 files using the templates in this skill's `templates/` directory, filled with values from Step 1:

1. `vllm_neuron/model/MODEL_NAME/config.py` — from `templates/config.py.template`
2. `vllm_neuron/model/MODEL_NAME/factory.py` — from `templates/factory.py.template`
3. `vllm_neuron/model/MODEL_NAME/__init__.py` — from `templates/init.py.template`

### Step 4: Generate model.py

Copy the selected reference model's `model.py` and modify **section by section**:

**Section 1 (Normalization)**: Use RMSNorm or LayerNorm as needed. Match eps value. If the model has layernorm biases, add a `bias` parameter and include it in `load_weights` mappings.

**Section 2 (RoPE)**:
Match the model's rotary embedding variant. Remove scaling if standard.

**CRITICAL — Meta tensor avoidance:** vLLM creates models on `torch.device("meta")` first, then loads weights. Any tensor computation in `__init__` (like RoPE inv_freq via `ROPE_INIT_FUNCTIONS`) creates meta tensors that fail later with "Cannot copy out of meta tensor; no data!". Solution: store only scalar config values in `__init__`, compute inv_freq lazily in a `_compute_inv_freq(device)` method called from `forward()`. This matches the Llama3 pattern.

**CRITICAL — No `.to(device)` in forward path:** Never call `.to(device)` on tensors inside the forward path. On CPU simulator this is a no-op, but on real Neuron hardware, `torch.compile` traces the graph and `.to(device)` becomes a cross-device copy (`xla:0 → neuron:N`) which raises `NotImplementedError: unimplemented _copy_from`. All tensors passed into `forward()` are already on the correct device.

**CRITICAL — No `.item()` in forward path:** Calling `.item()`, `.max().item()`, or any scalar extraction breaks `torch.compile` with "Unsupported Tensor.item() call with capture_scalar_outputs=False". Use static values or tensor operations instead.

**Section 3 (Attention)**:
Add/remove bias. Handle sliding window, GQA/MQA head counts.

**CRITICAL — Bias shapes for NKI kernels:** The `NF.attention_decode` megakernel requires bias tensors to be 2D `[1, size]`, not 1D `(size,)`. When passing `bias_qkv` or `bias_out`, always unsqueeze:
```python
bias_qkv=self.qkv_proj_bias.unsqueeze(0) if self.qkv_proj_bias is not None else None
```
This passes on CPU simulator (which accepts 1D) but fails on real Neuron hardware with: "Bias shape must be [1, I], got (768,), expected (1, 768)".

For the prefill path, biases can be applied manually after the matmul (1D is fine for `torch.matmul` + bias addition). Only the decode megakernel has the 2D requirement.

**Section 4 (MLP)**: Match activation function and gating. SwiGLU (gate+up+down) vs non-gated (up+down).

For MoE models: keep router gate weights in float32, not bf16. The softmax over num_experts is extremely sensitive to precision — a tiny difference in bf16 can select a completely different pair of experts. In the model init:
```python
self.gate_weight = nn.Parameter(torch.empty(..., dtype=torch.float32))
```
In `load_weights`, preserve float32 for router weights:
```python
if 'gate_weight' in name:
    rank_sharded[name] = tensor.to(torch.float32)
```

**Section 4b (MoE with EP)**: If EP is required, implement the unified sp_group pattern:
```python
# In every module that does collectives:
ep_degree = get_neuron_ep_degree()
if ep_degree > 1:
    self.tp_group = get_neuron_ep_tp_group()   # sub-group, for WEIGHT SIZING only
    self.sp_group = get_tp_group()              # full world, for ALL COLLECTIVES
else:
    self.tp_group = get_tp_group()
    self.sp_group = self.tp_group
```
- Use `self.tp_group.world_size` for parameter dimensions (num_heads_per_rank, etc.)
- Use `self.sp_group` for every all_gather, reduce_scatter, all_reduce call
- This applies to ALL modules: Attention, MLP, MoE, Embedding, LM Head, Sampler
- MoE forward: loop over LOCAL experts only (`num_local_experts = num_experts / ep_degree`)
- Expert weight loading: filter by EP rank — only load `range(ep_rank * num_local_experts, (ep_rank+1) * num_local_experts)`
- After loading weights, scale replicated layers: `o_proj_weight.div_(ep_degree)`, `down_proj_weight.div_(ep_degree)` for non-MoE layers. Do NOT scale MoE expert weights.
- Reference: the in-tree EP port for the complete working EP pattern (locate it with `grep -rl 'ep_degree\|expert_parallel' vllm_neuron/model/*/model*.py`; commonly `gpt_oss/model_bf16.py`).

**Section 4c: MoE kernel capability matrix (pick the expert-compute + router path up front, not reactively):**

| Kernel | Path | Routing | Use when |
|---|---|---|---|
| `NF.moe_cte` + `NF.build_blockwise_mapping` | prefill (and decode fallback) | affinities computed EXTERNALLY (any router) | default expert compute; ALSO use for decode when your router doesn't fit `moe_block_tkg` (DeepSeek-V4 pattern: moe_cte for both paths) |
| `NF.moe_block_tkg` | decode | routing computed INTERNALLY, `RouterActFnType.SOFTMAX` or `SIGMOID` ONLY | decode only when routing is a plain softmax/sigmoid top-k |
| `NF.router` | either | `activation` = str OR Callable, optional `router_bias`, `RouterComputationOrder.*` | computing expert affinities; try a callable (e.g. sqrt-softplus) before hand-rolling |
| `NF.rmsnorm_router_topk_tkg` | decode | fused norm+router+topk | decode router with standard norm+topk |

Do NOT hand-roll expert compute with `torch.einsum` or a per-expert matmul loop — raw einsum expert math is a known `NCC_ITIN902` trigger, and a Python per-expert loop Dynamo-unrolls into thousands of FX nodes. Feed externally-computed `expert_affinities_masked` into `NF.moe_cte` instead. Pad the token dimension to `block_size` before calling `moe_cte` (unpadded token counts cause a DGE out-of-bound).

**Section 4d: Avoid `torch.topk` / `sort` in the forward path (router top-k, indexer top-k, sampler).** On trn2 these lower to a `sort` HLO that is unsupported/buggy (surfaces as `NCC_EVRF029` "HLO sort unsupported", or a bare `IGAA901` wrapped in `NCC_INAS001` in decode graphs). Options, in order:
1. Where routing fits its contract, use the in-tree router ops that do top-k internally (`NF.router`, `NF.rmsnorm_router_topk_tkg`).
2. For sharded vocab/logit top-k, use the distributed `from vllm_neuron.functional.topk import topk` (module-level function; NOT exported as `NF.topk`; silently falls back to `torch.topk` for inputs the kernel can't handle).
3. Iterative-argmax + `-inf` mask is the compiler-mandated correctness fallback used across shipping models (find current examples with `grep -rl 'argmax' vllm_neuron/model/*/model*.py`), but Dynamo-unrolls into ~3*k FX nodes per call — do NOT use it for large k.

If a `sort`/topk compile error still appears, hand off to the compiler-debugging agent.

**Section 5 (Decoder)**: Match norm placement and residual connection pattern.
**Section 6 (Backbone)**: Handle vocab padding for TP alignment. Handle position embeddings if non-RoPE.
**Section 7 (LM Head)**: Handle tied vs untied embeddings. If the model has `lm_head_bias=True`, create a separate `nn.Parameter` for the bias (not inside `ColumnParallelLinear`) and add it to the logits in `forward()`. Map it in `load_weights` with the correct checkpoint key (e.g., `"lm_head_bias" → "lm_head.bias"`).

**Weight Loading**: Build complete checkpoint key mapping. Every `nn.Parameter` must have a mapping. Use `fused_qkv_weight_loader` for fused QKV weights. For fused QKV biases (1D tensors), create a custom bias loader since `fused_qkv_weight_loader` expects 2D slices. Attach `sharding_weight_loader` to TP-sharded params. Verify every mapping against the actual checkpoint (`model.safetensors.index.json` or `safe_open`) — do not assume the target follows the reference model's key conventions; a wrong transpose produces a model that runs and generates fluent-looking garbage.

**`load_weights_lite()` (CPU Compilation)**: During CPU Compilation (`VLLM_NEURON_CPU_COMPILE=1`) the model is instantiated on the `meta` device and `load_weights_lite()` is called instead of `load_weights()`. Implement it if the checkpoint has tensors that must be baked into the compiled graph as compile-time constants (e.g., FP8 dequant scale tensors) — read just those tensors on CPU, leave everything else on meta. If the model has no such requirement, the base-class no-op default is fine.

Rename ALL classes with the model's PascalCase prefix (e.g., `LlamaRMSNorm` → `YiRMSNorm`).

### Step 5: Register the Model

Edit `vllm_neuron/model/registry.py`:
1. Add import: `from .MODEL_NAME import PascalNameForCausalLM`
2. Add tuple to `get_models()`: `("HFArchitectureString", PascalNameForCausalLM),`

**CRITICAL: The HF architecture string must EXACTLY match the `architectures` field from config.json, including capitalization.** Example:
- HF config: `"architectures": ["PhiMoEForCausalLM"]`
- Registry: `("PhiMoEForCausalLM", PhiMoEForCausalLM)` ← correct
- Registry: `("PhimoeForCausalLM", PhiMoEForCausalLM)` ← WRONG, model won't load

Always verify with:
```bash
python3 -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('HF_MODEL_ID'); print(c.architectures)"
```

### Step 6: Generate Example & Docs

1. Create `examples/vllm_neuron/models/MODEL_NAME/run.py` from `templates/run.py.template`
2. Create `examples/vllm_neuron/models/MODEL_NAME/results/results.md` from `templates/results.md.template`
3. Create `vllm_neuron/model/MODEL_NAME/README.md` per the model bringup guide located in Step 1c (Step 10 of that guide). It must include an architecture parameter table, key differences from the reference model, and a feature status table (TP/SP/DP/EP, Eagle3, FP8 KV cache, etc. with ✅/❌/N/A and notes) — every parallelism block that was kept, removed, or deferred must be accounted for.

Every ❌/⚠️ in the feature status table MUST be qualified as exactly one of:
- **compile-blocked** — does not compile/run on device at all (hard blocker),
- **torch-fallback** — runs correctly but via a slow torch path because no matching kernel is wired in (works, unoptimized), or
- **kernel-unavailable** — a kernel is needed but none exists in-tree.

Do not write a bare ❌ that conflates these. For any torch-fallback or kernel-unavailable entry, name the candidate kernel or reference model to wire in — and verify the named symbol exists in the tree you are working in before citing it (e.g. 'head_dim=256 > 128 cap on NF.flash_attention/NF.attention_decode → decode currently falls back to torch; a custom d-tiled decode kernel would handle head_dim>128, but none ships in this release'). A claim like 'On-device compilation ❌' must be empirically verified — if the model actually runs via a fallback, it is 'torch-fallback', not 'compile-blocked'.

### Step 7: Self-Review

Before reporting completion, verify:
- [ ] Every `nn.Parameter` in model.py has a corresponding key in `load_weights` mappings
- [ ] Bias shapes are `[1, size]` (2D) for any param passed to `NF.attention_decode` (`bias_qkv`, `bias_out`)
- [ ] No `.to(device)` calls in the forward path (tensors are already on device)
- [ ] No `.item()` calls in the forward path (breaks torch.compile)
- [ ] RoPE inv_freq is computed lazily in `forward()`, not in `__init__()` (avoids meta tensor)
- [ ] Class names are consistent across config.py, factory.py, __init__.py, model.py
- [ ] `__init__.py` exports the factory's ForCausalLM (not the model's — they have the same name)
- [ ] Registry uses the exact HF architecture string from config.json (case-sensitive)
- [ ] `from_configs` classmethod properly maps all model-specific config fields
- [ ] **Weight loaders and TP layers are imported, not reimplemented** (`ColumnParallelLinear`/`RowParallelLinear`, `fused_qkv_weight_loader`, `sharding_weight_loader`, `VocabDimShardedEmbedding`, `Sampler`)
- [ ] **Per-block kernel accounting (fill this table; do not skip):** for each compute block, record the NF/nn symbol it CALLS, or an explicit one-line blocker reason. An `import vllm_neuron.functional as NF` with no call sites does NOT satisfy this.

  | Block | NF/nn symbol called | If hand-rolled: blocker reason |
  |---|---|---|
  | QKV proj | `NF.qkv_proj` / `ColumnParallelLinear` | |
  | Attention (prefill) | `NF.flash_attention` / `NF.segmented_attention` | |
  | Attention (decode) | `NF.attention_decode` | |
  | O proj | `NF.o_proj` | |
  | MLP / SwiGLU | `NF.mlp` | |
  | MoE experts | `NF.moe_cte` (+`NF.build_blockwise_mapping`) / `NF.moe_block_tkg` | |
  | MoE router | `NF.router` / `NF.rmsnorm_router_topk_tkg` | |
  | Top-k select | `topk` (NOTE: not re-exported as `NF.topk` — use `from vllm_neuron.functional.topk import topk`) | |
  | Sampler | `NF.sample` / `nn.Sampler` | |

  Verify with a grep for call sites, not the import line. Match ANY `NF.*` call (open pattern — so a kernel added since this skill was written still counts), then confirm each name against the discovered API rather than a frozen allow-list:
  ```bash
  # Every NF.* call site in the ported model (open-ended — do not hardcode a closed kernel list):
  grep -nE 'NF\.[a-z_]+\(' vllm_neuron/model/MODEL_NAME/model.py
  # Cross-check that each called name is a real public kernel in THIS install (catches typos AND new kernels):
  python3 -c "import vllm_neuron.functional as NF; print('\n'.join(sorted(getattr(NF,'__all__',dir(NF)))))"
  # top-k is imported directly from the submodule (intentionally NOT an NF.* export):
  grep -nE 'from vllm_neuron\.functional\.topk import topk|[^.]topk\(' vllm_neuron/model/MODEL_NAME/model.py
  ```
  If a block is hand-rolled, the blocker reason must be a concrete capability gap (e.g. 'attention_decode has no multiplicative output-gate hook', 'head_dim=256 > 128 kernel cap') — not 'future work'. Do NOT cite 'moe_tkg is bf16-incapable': `moe_tkg` supports bfloat16, MXFP4 (uint16), and STATIC_MX FP8 (uint32) — verify the actual dtype/shape constraint in the kernel source before claiming it as a blocker.

  A grep hit for a hand-rolled idiom (`einsum`, `matmul(...transpose)+softmax`, `range()+argmax`, `F.silu(...)*`) is an investigation signal, not an automatic defect — iterative-argmax is the mandated top-k workaround, manual matmul+softmax decode is legitimate alongside `NF.flash_attention` prefill, and MLA einsum / per-expert `F.silu(gate)*up` are intended in shipping models. It is a defect only if the block lacks a verified blocker in the Step 2b table.
- [ ] For MoE models: router gate weights are float32, not bf16
- [ ] Normalization type matches HF source (read forward(), don't trust class name)
- [ ] **Any framework-fit adaptation that changes the model's OUTPUT is recorded as a
      profiling watch-item.** When porting into (or adapting a pre-ported bundle to) a
      tree that has diverged from the bundle's original design, note every deviation —
      especially anything that changes what crosses the device↔host boundary or moves an
      op on/off device (pooling, sampling, normalization, returning full `[T,H]` hidden
      states vs a pooled/reduced output). These are locally-correct integration choices
      that frequently become the *binding* performance cost later (see the callout below).
- [ ] **No dead modules / stale comments:** every `nn.Module` (and every `nn.Parameter`) declared in `__init__` is actually invoked in `forward()`; docstrings/comments describe the code path actually taken. After any compiler-driven rewrite (e.g. replacing a rejected op with a manual equivalent), delete or update the abandoned module and its docstring. Spot-check: `grep -n 'self\.<module_name>(' model.py` for each declared submodule; a submodule whose only references are `.weight`/`.bias` reads is dead code to remove or document. If the dead module holds a weight referenced in `load_weights`, re-home it as an `nn.Parameter` and update mappings rather than deleting outright.

**Additional EP checks (MoE models with Expert Parallelism only):**
- [ ] ALL collectives use a single group (sp_group) — no mixed group sizes in the NEFF
- [ ] `tp_group` used only for weight sizing, `sp_group` for all collective ops
- [ ] O_proj and dense MLP down_proj weights scaled by `1/ep_degree` after loading
- [ ] MoE expert weights NOT scaled (each EP partition has different experts)
- [ ] Expert weight loading filtered by EP rank (`range(local_start, local_start + num_local)`)
- [ ] Router gate weight is replicated (full copy on every rank), not EP-sharded
- [ ] Embedding and LM head use sp_group.world_size for vocab sharding
- [ ] run.py uses `tensor_parallel_size=world_size`, `enable_expert_parallel=True`, `ep_degree` in neuron_config

Print summary: files created/modified, line counts, and any warnings.

> **Integration adaptations are a first-class source of downstream perf bugs — flag them,
> don't bury them.** When the target tree has diverged from the bundle's original design,
> the locally-correct adaptation you make to fit it can become the binding performance
> cost in the optimization phase. Real incident (`generic-encoder-only-model`): the bundle
> pooled on-device, but the target tree already had a CPU-side `DispatchPooler`, so the
> port was adapted to return full `[T,H]` hidden states and pool on host. That adaptation
> was invisible in CPU equivalence (math identical) and in the device trace
> (`nc_exec_running`), but forced a 4MB device→host `nrt_tensor_read` per request (~2ms)
> that pushed the E2E latency over SLA. The fix was to move pooling back on-device. Lesson:
> any adaptation that changes the model's output shape or moves an op host-side must be
> listed in the completion summary as an explicit **"profiling watch-item"** so the
> optimization phase checks it first — the integration and optimization phases look
> independent but are not.

### Step 7.5: Independent code review of model.py (before compiling)

Get a SECOND, independent read of the generated `vllm_neuron/model/MODEL_NAME/model.py` (and `config.py`) before spending compile time. Use whatever review capability the current harness ships — do NOT let the generating agent review its own output in the same context:
- **On a coding harness with built-in review support:** use its code-review skill or command to review the local, uncommitted changes, or dispatch its built-in review agent. Prefer the shipped reviewer because it already knows how to read a diff and report findings.
- **On any other coding harness (Cursor, Copilot, Windsurf, Aider, etc.):** invoke that harness's own code-review agent / command / skill.
- **If no dedicated review capability exists:** dispatch a fresh general-purpose subagent with the review brief below.

Run the review in TWO passes (both feed findings back before Step 8):

**Pass 1 — port-specific checklist (NOT a generic style review):**
- **Kernel reuse:** does every compute block call an `NF.*`/`nn` symbol, or carry a verified blocker reason? Flag any hand-rolled norm/RoPE/attention/MoE/router/top-k with no blocker.
- **Dead code:** any `nn.Module`/`nn.Parameter` declared in `__init__` but never called in `forward()`?
- **Weight mapping:** does every `nn.Parameter` have a `load_weights` key, with correct transpose/QKV-fusion?
- **Forward-path hazards:** any `.to(device)`, `.item()`, meta-tensor computation in `__init__`, or 1D bias passed to a decode megakernel?
- **Precision:** are fp32 upcasts (RMSNorm variance, router gate) preserved?

**Pass 2 — compare against the sibling models already in the repo.** Dispatch a subagent (or add to the reviewer's brief) to diff `model.py` against the 1–2 closest in-tree ports — the reference model chosen in Step 1d, plus the nearest architectural match (e.g. an existing MoE port for a MoE model, or whichever port your tree ships for head_dim>128 / hybrid-gated attention — use the Step 1d discovery commands to pick them). Point it at `vllm_neuron/model/*/model*.py`. The existing ports are the ground truth for the conventions a fresh port silently gets wrong:
- **Kernel call conventions:** does the port call each `NF.*` with the same argument shapes/dtypes/layouts the sibling uses (e.g. is `expert_affinities_masked` fed to `moe_cte` the same way; is the QKV width stripped of gate columns before `attention_decode` as the sibling does)?
- **Weight-loader shape:** does `load_weights` follow the sibling's transpose/fusion/EP-filtering pattern, or diverge without reason?
- **Forward signature & KV-cache handling:** do `forward()` / attention-metadata / block-table / `slot_mapping` handling match the sibling's contract (a mismatch here is a silent runtime-garbage or `status=5` bug, not a compile error)?
- **Structural divergence:** flag anything this port does that NO sibling does (a bespoke attention loop where every sibling uses a kernel, a novel weight layout) — divergence is allowed but must trace to a recorded blocker, not an oversight.

Give the subagent this brief: *"Compare `vllm_neuron/model/MODEL_NAME/model.py` against `<reference>/model.py` and `<nearest-arch-match>/model.py`. Report where the new port diverges from their kernel-call conventions, weight-loader patterns, forward/KV-cache contracts, and structure — and whether each divergence is justified or a likely defect."*

Feed all findings back as edits before Step 8. Record in results.md that an independent review ran, which sibling models it compared against, and what it caught.

**Check kernel LNC before the first compile, including at TP=1.** The
vLLM-Neuron 0.29 V2 runner derives compiler LNC from `NEURON_LOGICAL_NC_CONFIG`.
If the selected dense QKV path in `functional/attention/qkv.py` launches
`wrapped_kernel[2]`, export `NEURON_LOGICAL_NC_CONFIG=2` and remove conflicting
`--logical-nc-config=1` / `--lnc=1` options from `NEURON_CC_FLAGS` and
`neuron_config.compiler_args`. Otherwise compilation fails with `NCC_ENKI003`.
Inspect the selected prefill/decode kernels; paths supporting LNC=1 can keep it.

### Step 7.6: Reduced-layer bring-up ladder (large / novel-op models)

For any model that (a) is too large to fit+boot on the available node count, or (b) has novel ops (custom attention/router/conv, MLA/latent attention, GatedDeltaNet) whose numerics you have not yet verified, do NOT go straight to full-model smoke. Bring up in stages:

1. **CPU compile at reduced layers.** Make a truncated config (`num_hidden_layers = 2-4`; for hybrid/heterogeneous-layer models also hand-set the per-layer type list — `layer_types` and `mlp_layer_types` — so at least one of each layer kind is present). Compile on CPU with `VLLM_NEURON_CPU_COMPILE=1` **and** `NEURON_PLATFORM_TARGET_OVERRIDE` set to the target instance family (required — with no device present the platform can't be auto-detected). Shakes out meta-tensor, weight-mapping, and lowering errors in minutes. Note: CPU compile only confirms the pipeline succeeds — it CANNOT run inference or produce numerics.
2. **Numerics on CPU vs HF.** For each novel op, compare the reduced-layer forward against the HF reference on CPU using CPU *inference* (`VLLM_NEURON_CPU_MODE=1`, mutually exclusive with `VLLM_NEURON_CPU_COMPILE`) or a standalone module-level test — not the CPU-compile run above. This is where Sinkhorn/normalization/precision bugs surface cheaply.
3. **Reduced-layer device run.** Run the truncated model on the smallest world size that fits; confirm HTTP 200 / coherent tokens.
4. **Scale to full layers/world** only after 1-3 pass. If the full model cannot both fit and bootstrap (e.g. a world-size / collective-bootstrap ceiling), the reduced-layer device run is the validated deliverable — record this explicitly in results.md rather than reporting the full model as validated.

A truncated checkpoint is built by symlinking a subset of safetensors shards and patching `config.json` (`num_hidden_layers`, and for hybrid models the per-layer type list `layer_types`, e.g. `["linear_attention", "linear_attention", "linear_attention", "full_attention"]`).

---

## Phase C: Validation

### Step 8: Smoke Test

Run the example script:
```bash
# Standard (non-EP) models:
NEURON_SKIP_EFA_AFFINITY=1 python examples/vllm_neuron/models/MODEL_NAME/run.py

# EP models (requires VLLM_NEURON_SWITCH_CC for contiguous collective topology):
NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_SWITCH_CC=1 python examples/vllm_neuron/models/MODEL_NAME/run.py
```

Note: `NEURON_SKIP_EFA_AFFINITY=1` is needed on trn2 instances where the PCI topology doesn't match the hardcoded BDF-to-EFA mapping. Safe for single-node TP.

Check for:
- Successful weight loading (no missing key errors)
- Successful compilation (no NKI shape mismatch errors)
- Reasonable generated text (not garbage or degenerate repetition)

**Completion gate — a real generation MUST be produced.** Component-level and single-decoder-layer equivalence (Step 11) passing is NOT sufficient to call the port done. Before printing 'Port Complete' you must have run the assembled full model end-to-end (Step 8 smoke or Step 9 online) and inspected actual generated text for coherence — component tests with synthetic inputs do not exercise KV-cache assembly, cross-layer error compounding, or the sampler.

**Diagnosing common failures:**

| Symptom | Likely Cause | Fix |
|---|---|---|
| `Cannot copy out of meta tensor` | RoPE or other tensor computed in `__init__` | Move computation to `forward()`, compute lazily |
| `Bias shape must be [1, I], got (N,)` | 1D bias passed to NKI decode megakernel | `.unsqueeze(0)` before passing to `NF.attention_decode` |
| `unimplemented _copy_from xla:0 neuron:N` | `.to(device)` call in forward path | Remove `.to(device)`, tensors are already on device |
| `Unsupported Tensor.item()` | `.item()` call in forward path | Use static values or tensor ops |
| `Checkpoint key(s) not found` | Weight mapping mismatch | Check `load_weights` mappings vs checkpoint keys |
| `size mismatch for lm_head.bias` | lm_head bias not mapped correctly | Use separate `nn.Parameter` + explicit mapping |
| `nrt_tensor_allocate status=4` | HBM out of memory | Increase TP size |
| `No EFA device found` | PCI topology mismatch | `NEURON_SKIP_EFA_AFFINITY=1` |
| `neuronx-cc compilation failed with 70` / any `NCC_*` code / `_pad` ICE / f64 / `head_dim>128` kernel-contract / `status=5`/`101`/`1206` | A compile, lowering, kernel-contract, or device-runtime failure — NOT a `model.py` typo | STOP. Do not guess or recompile blindly. Dispatch the compiler-debugging agent (see the hand-off block below) |
| Degenerate output (`the the the...`) | Decode path bug (bias, KV cache, or norm) | Test prefill only (max_tokens=1), then debug decode |
| Registry `AttributeError: no attribute 'from_configs'` | Architecture string mismatch in registry | Verify exact HF architecture string |
| `torch.topk` compile error / IGAA901 in decode / silent wrong routing | `torch.topk` lowers to a sort HLO, unsupported/buggy on trn2 | Use `from vllm_neuron.functional.topk import topk`; iterative argmax is a correctness-only fallback that Dynamo-unrolls badly for large k |

**EP-specific failures (MoE models with Expert Parallelism):**
- `NEFF Warmup failed with status 1006` (collective / replica-group mismatch) → Mixed collective group sizes in one NEFF. Fix: unify ALL collectives to one group (sp_group). This one is model code — fix it here, do NOT route it to the compiler-debugging agent.
- DGE scatter/gather out-of-bound / `status=5` (DGE OOB, often from an unclamped `slot_mapping`/`block_table` with `PAD_SLOT_ID=-1`) / other runtime status codes → these are compile/runtime-artifact failures. First re-check the model-code causes in this list; if none apply, **dispatch the compiler-debugging agent** rather than hand-grepping FX dumps — it owns replica-group / collective-topology localization (read from the compiled graph's `replica_groups`, NOT Python group names), DGE-OOB / `PAD_SLOT_ID` clamping (`torch.where(slot>=0, slot, garbage_slot)`, not `clamp(min=0)` which corrupts block0), and runtime status-code triage.
- OOM during weight loading → Expert weights not filtered by EP rank. Check that `load_weights` only maps `range(local_expert_start, local_expert_start + num_local_experts)`.
- Garbage output but no crash → Check O_proj/down_proj scaling (missing `div_(ep_degree)`?). Check router gate is replicated (NOT EP-sharded). Check shared expert division by ep_degree after collective.
- Compilation OOM/timeout → With many experts in a loop, the compiler unrolls all iterations. For production, switch to `NF.moe_cte` (prefill) and `NF.moe_block_tkg` (decode) kernels. If switching kernels still OOMs or times out at compile, dispatch the compiler-debugging agent (it covers `NCC_EOOM002` compile-time graph-I/O blowup, distinct from runtime HBM).
- Run EP models with: `NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_SWITCH_CC=1 python run.py`

**Compiler-debugging agent hand-off (compile / lower / trace / runtime failures).**

Dispatch the `experimental-neuron-autoport-compiler-debugging-agent` (a subagent equipped with the `experimental-neuron-autoport-compiler-debugging-vllm-neuron` skill) and hand it the failing run's compile-cache dir, `model.py`, and the run/worker log; apply the fix it returns. On a harness without agent dispatch, invoke the `experimental-neuron-autoport-compiler-debugging-vllm-neuron` skill directly instead.

STOP local guessing and hand off when you see any of:
- `neuronx-cc compilation failed with 70` / driver "error 70" (overloaded — the agent reads per-rank `log-neuron-cc.txt` for the real code).
- Any `NCC_*` code (`NCC_ITIN902`, `NCC_IBCG901`, `NCC_ISMP902`, `NCC_ESPP004`, `NCC_ISCH719`, `NCC_INLA001`, `NCC_INKI016`, `NCC_EOOM002`, `NCC_EVRF029`), a bare `IGAA901`/`INAS001`, or an internal-compiler-error / `_pad` Simplifier ICE.
- f64 / `float64` legalization errors, or a graph that compiles clean on CPU simulator but fails on device.
- Kernel-contract failures: `head_dim > 128`, FP8, or bucket/TKG-vs-CTE mismatches.
- Dynamo trace traps: `torch._dynamo.exc.Unsupported`, `xla::unselect`, data-dependent slice, native segfault in torch_xla lowering.
- Runtime status codes on a graph that DID compile: `status=5` (DGE OOB / `PAD_SLOT_ID`), `status=101`, `status=1206` (fabric/collectives), or coherent-looking garbage text.

Do NOT route to the agent (fix locally): `status=1006` "mixed collective group size" — a model-code EP bug owned by THIS skill (unify all collectives to `sp_group`; see the unified sp_group constraint in Step 1g and the EP bullets above).

Fallback when the agent/skill is unavailable: do not multi-hour-recompile-guess. Record the exact error code + per-rank `log-neuron-cc.txt` excerpt in results.md, apply only the model-code fixes in the Step 8 tables, and flag the failure as a suspected compiler bug requiring the compiler-debugging skill.

The agent owns error-code meanings, good-vs-bad fxgraph clustering, layer-count bisection, minimal repro, sibling-HLO compare, cache-hygiene wipe procedure, and ticket escalation.

Update `results/results.md` with offline inference results.

### Step 9: Online Serving & APC

**Before running at TP>1 or with multiple replicas, read the runtime feasibility gates
below — they are real crash modes on trn2, not theoretical, and cost multiple hours to
rediscover if hit blind.** See
`neuron-parallelism-planner/references/tp-feasibility-and-kv-replication.md`
§6 for the full detail; summarized here for convenience:

- **`NEURON_VISIBLE_DEVICES` must list exactly `tensor_parallel_size` entries** — one ID
  per logical core (trn2.48xlarge = 32 logical cores at LNC=2). Fewer/more entries than
  `tensor_parallel_size` fails at worker init with
  `AssertionError: len(neuron_visible_devices) == num_local_ranks` — this is **not**
  evidence of a lower hardware core-count ceiling, it's a request/device-list size
  mismatch. Don't infer a smaller max-TP from this error.
- **Device index must equal rank** — a device window must be zero-based in its own
  process (`0..TP-1`). Two independent TP replicas cannot both use non-zero-based windows
  in the same launch (e.g. replica A on `0-3`, replica B on `12-15` fails with
  `RuntimeError: Invalid device index ... must match the rank`). A second replica CAN use
  a contiguous non-zero block like `16-31` as long as it's zero-based *within its own
  process* — this is how DP=2×TP=16 is actually deployed (two separate processes, each
  internally addressing its slice as local ranks 0..15).
- **The device set must form one valid contiguous collective mesh** — a set spanning two
  disjoint physical rings/NUMA nodes (check `neuron-ls` adjacency) crashes all ranks with
  `encd_mesh_add_wr_barrier: Assertion 'event->evt_type == EVT_SYNC' failed`. Pick a
  topologically contiguous block, don't just pick any N device IDs that add up to
  `tensor_parallel_size`.
- **Never force LNC=1 (or any `_NOMODULAR` variant) at `tensor_parallel_size > 1`.**
  NEFFs compiled with `--logical-nc-config=1` are runtime-incompatible with the default
  LNC=2 collective topology TP>1 needs (`Failed to load collectives for model` at
  warmup). Those flags are TP=1-only workarounds for all-PyTorch/no-NKI-kernel graphs;
  drop them the moment `tensor_parallel_size` goes above 1 — the compile error they work
  around (`NCC_IXRO002`) is specific to TP=1 graphs with no collectives to guide the
  compiler's modular-flow partitioner.

Start the vLLM API server:
```bash
NEURON_SKIP_EFA_AFFINITY=1 python3 -m vllm.entrypoints.openai.api_server \
    --model HF_MODEL_ID \
    --tensor-parallel-size TP_SIZE \
    --max-model-len 256
```

Test battery:
1. Basic completion: `POST /v1/completions` with "The capital of France is"
2. Batch: 4 prompts in one request
3. Streaming: `stream=true`, verify SSE chunks
4. Counting: "1 2 3 4 5 ", verify continuation

Performance: 10 requests, 64 tokens each, measure latency and throughput.

APC: Restart with `--enable-prefix-caching`, re-run all tests.

Update `results/results.md` with online serving results.

### Step 10: Logit Validation

Generate the canonical logit validation test from this skill's `templates/test_logit_validation.py.template` and run it.

1. Create the test directory following the current layout convention: `test/vllm_neuron/model/MODEL_NAME/bf16/e2e/` (precision tier, then `e2e/`) — see existing tests like `test/vllm_neuron/model/gpt_oss/bf16/e2e/test_logits.py` for the pattern (`ls test/vllm_neuron/model/*/*/e2e/test_logits.py` to find one in your tree).
2. Generate `test/vllm_neuron/model/MODEL_NAME/bf16/e2e/test_logits.py` from the template, filling in all `{{...}}` variables based on the model's architecture (hidden_size, TP sizes, batch sizes, HF model ID, etc.). Reference existing tests in `test/vllm_neuron/model/` for examples of how other models fill in these variables.
3. Run the sanity tests first (they're fast):
```bash
NEURON_SKIP_EFA_AFFINITY=1 pytest test/vllm_neuron/model/MODEL_NAME/bf16/e2e/test_logits.py -m "offline_serving" -v --timeout=3600
```

4. If sanity tests pass, run the online serving test at a small seq_len:
```bash
NEURON_SKIP_EFA_AFFINITY=1 pytest test/vllm_neuron/model/MODEL_NAME/bf16/e2e/test_logits.py -m "online_serving and seq256" -v --timeout=7200
```

**Interpreting results:** The logit validation compares Neuron output logits against a HuggingFace CPU reference at various top-k levels (5, 50, 1000, all). Results are reported as sigma values:
- **< 3 sigma**: Excellent — within normal numerical noise
- **3-5 sigma**: Acceptable — minor numerical differences, usually from bf16 quantization
- **5-10 sigma**: Investigate — may indicate a real issue but could also be model-specific
- **> 10 sigma**: Likely a bug in the port

**IMPORTANT:** If logit validation fails, do NOT treat it as a hard blocker. Report the sigma values and which top-k levels failed, explain what they mean, and continue to completion. The user can decide whether the accuracy is acceptable for their use case. Some models naturally have higher variance at certain top-k levels.

### Step 11: Deep Equivalence Validation

**Step 11 is a MANDATORY gate — the port is NOT complete until equivalence has been RUN and its result recorded.** Do not report "Port Complete" with equivalence skipped, deferred, or "pending". If you are on a harness with agent dispatch, hand the port off to the **`neuron-framework-equivalence` agent/skill** (it owns the 8-stage pipeline; do not re-implement any of it inline or hand-roll one-off needle/tolerance tests — see its CRITICAL RULES). Collect and pass all its Required Inputs up front (`SOURCE_MODEL_PATH`, `COMPILED_MODEL_PATH`, `TARGET_MODELING_FILE`, `TARGET_INNER_CLASS`, `TARGET_CAUSAL_CLASS`, `TARGET_CONFIG_CLASS`, `VENV`, `MODEL_VALIDATION_DIR`, `EXP_DIR`, and `VLLM_NEURON_DIR` — the vLLM-Neuron project root, which must also be prepended to `PYTHONPATH` on every stage command) and `--target-stack vllm_neuron`.

The ONLY acceptable non-run outcomes, each of which MUST be written into results.md with its reason:
- **Blocked**: the model cannot compile/run on the available hardware at all (e.g. world-size bootstrap ceiling). Record what blocked it and run equivalence at the largest feasible reduced-layer configuration (see Step 7.6) — a reduced-layer equivalence run is still required, not skipped.
- **Infeasible reference**: no runnable HF reference exists at any usable size. Record why; rare, must be justified.

"I ran out of turns", "the port smoke-tested fine", or "component tests passed" are NOT acceptable reasons to skip Step 11.

After logit validation, run the full equivalence pipeline for rigorous component-level and E2E verification. This step is **always run** — it is part of the standard porting pipeline.

**Invoke the `neuron-framework-equivalence` skill** to verify the port. The equivalence framework supports multiple serving stacks via adapters — **use the vLLM-Neuron adapter** by passing `--target-stack vllm_neuron` to every stage script that accepts it (`run_stage0.py`, `run_stage1.py`, `run_stage5.py`, `run_stage7.py`, `run_teacher_forced_comparison.py`). The adapter handles vLLM-specific concerns: distributed init, `from_configs()` instantiation, weight mapping with transpositions/QKV fusion, attn_metadata construction, and the `vllm.LLM` API. (The adapter is also auto-detected from `vllm_neuron` imports in the modeling file, but pass it explicitly to be safe.)

Provide the equivalence skill these inputs, mapped from this port:

```
target_stack: vllm_neuron
SOURCE_MODEL_PATH: {HF_MODEL_ID weights path}
COMPILED_MODEL_PATH: {compiled model path from Step 8, if available}
TARGET_MODELING_FILE: vllm_neuron/model/MODEL_NAME/model.py
TARGET_CONFIG_CLASS: {PascalName}Config  (from config.py)
TARGET_INNER_CLASS: {PascalName}Model  (backbone class in model.py)
TARGET_CAUSAL_CLASS: {PascalName}ForCausalLM  (from model.py)
VENV: {active venv path}
EXP_DIR: agent_artifacts/equiv_{MODEL_NAME}
TP_SIZE: {recommended TP from Step 1e}
```

The equivalence skill runs the 8-stage pipeline:
- Stage 0: Build model trees, component mapping
- Stage 2: Component-level R-ratio tests (per submodule)
- Stages 3-4: Fault localization and debugging (if failures found)
- Stages 5+6: Teacher-forced E2E comparison (R-ratio, cosine, KL)
- Step 6: Generate EQUIVALENCE_REPORT.md

For vLLM-Neuron-specific behaviors (weight transpositions, forward signature, TP detection, KV cache shapes), the equivalence skill's `references/vllm-neuron-adaptation.md` is the authority — defer to it if anything conflicts.

**Interpreting results:**
- All R < 1.2 and E2E passes → port is verified at component level (component-level pass alone is NOT a complete port — the Step 8 real-generation gate still applies)
- Component failures found → equivalence report includes patches showing what's wrong
- Use the patches as a guide to fix the actual model code in `vllm_neuron/model/MODEL_NAME/model.py`

**Interpreting equivalence numbers:** high cosine similarity with large absolute divergence is a known trap. Per-layer bf16 error can compound across depth so final-layer `max_abs` is large (e.g. 64) while cosine stays ~0.9999 — cosine is scale/shift-insensitive and looks 'excellent' even with a real precision bug (missing fp32 weight cast, wrong normalization). Do NOT accept high-cosine-only as proof; require the real-generation coherence check from Step 8, and if `max_abs` is large investigate for a precision bug (fp32 weight multiply, router gate dtype, missing normalization) before shipping.

**When Step 11 also applies (do not decline it in these cases):**
- **After a kernel swap** (swapping a hand-rolled block for `NF.*`, or wiring in a new attention kernel): make the swap in `model.py` and re-run the equivalence pipeline — the swapped model is the new 'target'; the HuggingFace model at `SOURCE_MODEL_PATH` remains the 'reference' (the framework always loads the reference via HF `AutoModelForCausalLM.from_pretrained`; there is no facility to use the pre-swap Neuron model as the reference). Re-running catches a swap that silently changes numerics; do NOT hand-roll a one-off needle test.
- **At a new context length / KV-dtype** (e.g. 64K, or fp8 KV): still use the equivalence framework rather than bespoke tolerances, but note the stage scripts default to short contexts — the teacher-forced path hardcodes `max_model_len=256` and `--num-tokens 32`, and the component KV cache defaults to bf16. To validate at a longer length or different KV dtype you must adjust that run configuration (or use the Step 10 logit-validation harness at the target seq_len); keep the standard 3-way error-ratio criteria (R < 1.2) and document any deviation instead of inventing new pass thresholds. If a full HF reference at the target length is genuinely impractical, record that and validate at the largest feasible length rather than skipping equivalence.

If you conclude the equivalence skill genuinely does not fit, record the specific reason in results.md and state what you validated instead (do not silently skip it).

---

## Completion

Print final status:
```
## Port Complete: MODEL_NAME (HF_MODEL_ID)

Phase A: Research     ✓
Phase B: Code Gen     ✓ (N files, M lines)
Phase C: Validation   ✓/✗
  Step 10 Logit validation:   <ran / sigma summary>       [REQUIRED]
  Step 11 Equivalence:        <ran: EQUIVALENCE_REPORT.md path / blocked+reason / infeasible+reason>   [REQUIRED — a port with this blank is INCOMPLETE]

Files created:
  vllm_neuron/model/MODEL_NAME/config.py
  vllm_neuron/model/MODEL_NAME/factory.py
  vllm_neuron/model/MODEL_NAME/__init__.py
  vllm_neuron/model/MODEL_NAME/model.py
  vllm_neuron/model/MODEL_NAME/README.md
  vllm_neuron/model/registry.py (modified)
  examples/vllm_neuron/models/MODEL_NAME/run.py
  examples/vllm_neuron/models/MODEL_NAME/results/results.md
  test/vllm_neuron/model/MODEL_NAME/bf16/e2e/test_logits.py
```
