# Worked example: a mixed-attention dense model on trn2

This walks the three capabilities end to end on a run-spec consumed by
`neuron-parallelism-planner` (a ~31B dense model with both sliding-window and
full-context attention groups, served on trn2: ~158 TFLOP/s bf16, ~716 GB/s HBM, 24 GiB per
logical NeuronCore at LNC2). One physical NeuronCore-v3 supplies 79 TFLOP/s; an LNC2 logical
rank owns two. The ridge point is `158e12 / 716e9 ≈ 221 FLOP/byte`.

Reproduce every table below with `python3.12` from `scripts/`. Set `RUNSPEC` to the Gemma4
run-spec produced for the project; it must describe the model and hardware assumptions above.
The numbers come straight from the scripts, and reconciliation residuals are printed on each
run.

```bash
RUNSPEC=/path/to/project/agent_artifacts/tmp/gemma4-31b-run.json
```

## 1. Decode floor — uniformly memory-bound

```bash
python3.12 operator_entitlement.py \
    --runspec "$RUNSPEC" \
    --tp 8 --bucket 8192 --bs 8
```

Every operator sits at arithmetic intensity ≈ 8 (projections) or lower (`sdpa`), all well
below the 221 ridge → **all memory-bound**:

| op | group | inst/dev | compute | mandatory HBM (wt / kv) | act interface | AI | binding |
|---|---|---|---|---|---|---|---|
| gate/up/down_proj | sliding | 50 | 0.073 ms | 2.018 ms (2.02 / –) | 6.2 MiB | 8.0 | memory |
| sdpa | full | 10 | 0.034 ms | 1.875 ms (– / 1.87) | 0.6 MiB | 4.0 | memory |
| sdpa | sliding | 50 | 0.011 ms | 1.172 ms (– / 1.17) | 1.6 MiB | 2.0 | memory |
| q/o_proj | sliding | 50 | 0.028 ms | 0.769 ms (0.77 / –) | 4.9 MiB | 8.0 | memory |
| lm_head | head | 1 | 0.018 ms | 0.492 ms (0.49 / –) | 0.6 MiB | 8.0 | memory |

This is the textbook small-batch decode signature: weight reuse is low (bs=8 amortizes each
weight read over only 8 tokens), so projections land at AI ≈ 8 and `sdpa` even lower (2–4)
because it has **no weights to amortize** — its operator HBM term is pure KV. Operator
interfaces are shown separately; the component records below replace them with each unique
external input/output transfer.

The planner's analytical model-floor path costs and reports four component-variant body
records. The cost model applies `max(TE, HBM)` to each record:

| component record | multiplicity | TE ms | HBM ms | floor ms | binding |
|---|---:|---:|---:|---:|---|
| attention/sliding | 50 | 0.094 | 3.490 | 3.490 | weight load |
| attention/global | 10 | 0.059 | 2.569 | 2.569 | KV load |
| ffn/dense | 60 | 0.263 | 7.280 | 7.280 | weight load |
| lm_head | 1 | 0.018 | 0.493 | 0.493 | weight load |
| **component floor** | **121 components / 4 reported rows** | **0.435** | **13.832** | **13.832** | **HBM** |

All four cost records are HBM-bound here, so the serialized component floor equals
the global HBM total. The component-body graph cut contributes 0.030 ms of that HBM total.
Collectives are excluded from this table and added by the config-level entitlement.

Reconciliation: compute, weight-read, and KV-read are all **0.00%** vs the planner cell.

## 2. Prefill floor — compute-bound, with structural exceptions

```bash
python3.12 operator_entitlement.py \
    --runspec "$RUNSPEC" \
    --tp 32 --cp 4 --bucket 131072 --bs 1 --prefill
```

The whole-prompt token count pushes AI to 131,072 for the ordinary projections → **all
compute-bound**, except the weight-bearing lm_head:

| op | group | compute | mandatory HBM | act interface | AI | binding |
|---|---|---|---|---|---|---|
| sdpa | full | 1113.438 ms | 3.749 ms | 2.5 GiB | 65536.5 | compute |
| gate/up/down_proj | sliding | 299.696 ms | 0.505 ms | 73.8 GiB | 131072 | compute |
| q/k/v/o_proj | sliding | 114.170 ms | 0.192 ms | 68.8 GiB | 131072 | compute |
| **lm_head** | **head** | **0.001 ms** | **0.123 ms** | **<0.1 MiB** | **1.0** | **memory** |

Projection AI is `S` because each weight is reused across all 131,072 prompt tokens. Full
SDPA instead executes the causal triangle, whose average active key count is
`(S + 1) / 2 = 65,536.5`; no LNC-sharing factor is involved.

The lm_head projects sampled rows only, so a 131k-token prefill does not create a full-prompt
lm_head activation interface. The model's full-context layers alias V to K, so no phantom
global `v_proj` row appears.

The same four body records produce four reported rows:

| component record | multiplicity | TE ms | HBM ms | floor ms | binding |
|---|---:|---:|---:|---:|---|
| attention/sliding | 50 | 500.004 | 197.670 | 500.004 | TE |
| attention/global | 10 | 1250.442 | 43.345 | 1250.442 | TE |
| ffn/dense | 60 | 1078.907 | 238.010 | 1078.907 | TE |
| lm_head | 1 | 0.001 | 0.123 | 0.123 | HBM |
| **component floor** | **121 components / 4 reported rows** | **2829.353** | **479.148** | **2829.475** | **mixed** |

The global engine diagnostic is `max(2829.353, 479.148) = 2829.353 ms`. The serialized
component floor is 2829.475 ms because the HBM-bound lm_head remains a separate component
after the TE-bound attention and FFN records. This small difference demonstrates why the
engine applies the roofline before component records are added. The component-body graph cut
contributes 472.386 ms of the HBM total: standard-prefill attention and FFN bodies each read
one gathered prompt tensor and write one pre-reduce-scatter partial. Those HBM terms remain
hidden under the larger mandatory TE term here.

Reconciliation: compute, weight-read, and KV-read are all **0.00%**.

## 3. Single-config entitlement over a bucket-regime

```bash
python3.12 config_entitlement.py --runspec "$RUNSPEC"
```

Pins one parallelism config and evaluates every cell. The global diagnostics
(`weight_load`, `kv_load`, `activation_load`, `matmul`, and `comm`) remain useful, while
latency comes from the serialized component records above plus communication. Representative
regenerated
rows are:

| cell | component floor ms | communication ms | step ms | global-total diagnostic |
|---|---:|---:|---:|---|
| 8k decode, bs8 | 13.832 | 2.701 | 16.533 | weight HBM read |
| 8k decode, bs32 | 23.060 | 3.243 | 26.302 | KV HBM read |
| 8k prefill, bs8 | 3269.583 | 1482.270 | 4751.854 | matmul |

Decode therefore moves from weight-dominated to KV-dominated as batch grows; it does not flip
to matmul in this run. Prefill remains matmul-dominated. Each cell carries its operator-level
reconciliation, and the script flags any cell whose reconciliation drifts past a few percent.

## 4. Actual-vs-floor HBM attribution

This run uses a fresh Gemma4 decode capture compiled at TP32, batch 1, maximum context
66,560, with BF16 weights and an FP8 E4M3 KV cache. Export the binary graph's
`host_program_shape` as a text ENTRY manifest (see
[actual-vs-floor-hbm.md](actual-vs-floor-hbm.md)), and use a copy of the Gemma4 run-spec
with `"kv_dtype_bytes": 1` so the analytic floor describes the same serving precision:

```bash
python3.12 operator_entitlement.py \
    --runspec /path/to/gemma4-31b-fp8kv-run.json \
    --project-root /path/to/project \
    --tp 32 --bucket 66560 --bs 1 \
    -o /path/to/project/agent_artifacts/tmp/operator_entitlement_decode.json

python3.12 hbm_attribution.py \
    --entry-hlo /path/to/captured_entry_manifest.hlo \
    --floor-json /path/to/project/agent_artifacts/tmp/operator_entitlement_decode.json \
    --runspec /path/to/gemma4-31b-fp8kv-run.json \
    --project-root /path/to/project \
    -o /path/to/project/agent_artifacts/tmp/actual_vs_floor.json
```

The parser finds 901 resident ENTRY tensors totaling 21.9674 GiB on the compiled rank. The
ranked table from that run is:

```
op_class            actual_GiB  floor_GiB  overage_GiB       x
kv_cache_sliding         9.9359     0.0244       9.9115  406.98  ← paged-pool sizing (not op overage)
kv_cache_full            9.9359     0.6348       9.3011   15.65  ← paged-pool sizing (not op overage)
embed_or_lm_head         0.1641     0.0820       0.0820    2.00
qkv_proj_full            0.1538     0.1025       0.0513    1.50
norm_or_bias             0.0025     0.0000       0.0025      --
qkv_proj_sliding         0.3845     0.3845       0.0000    1.00
o_proj_sliding           0.1282     0.1282       0.0000    1.00
gate_up_proj             0.8075     0.8075       0.0000    1.00
down_proj                0.4037     0.4037       0.0000    1.00
o_proj_full              0.0513     0.0513       0.0000    1.00
```

Takeaways (matched breakdown, no root-causing):

- **MLP and ordinary attention projection weights are at entitlement** (×1.00) — gate/up,
  down, both o-projection groups, and sliding QKV.
- `embed_or_lm_head` is ×2.00 because the graph carries two resident table tensors while the
  model floor assumes the tied embedding/output table is resident once.
- `qkv_proj_full` is ×1.50 because the fused tensor stores Q + K + V, while the model's
  `attention_k_eq_v` floor removes an independent global V projection.
- The two **KV rows dominate absolute bytes** but cross a config boundary (paged-pool
  reservation vs analytic per-request) — flagged, not an operator overage.
- Only 252 bytes fall into unattributed shape classes. Activation-interface metadata is not
  part of this resident-HBM comparison; the floor side remains mandatory weight + KV bytes.
