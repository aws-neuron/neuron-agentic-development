# Actual-vs-floor HBM attribution

## Contents

- [Goal and scope](#goal-and-scope)
- [Where the resident bytes actually live](#where-the-resident-bytes-actually-live)
- [Capturing the resident manifest](#capturing-the-resident-manifest)
- [Attribution by shape signature](#attribution-by-shape-signature)
- [The diff and how to read it](#the-diff-and-how-to-read-it)

## Goal and scope

The operator floor (`operator_entitlement.py`) gives the **theoretical** per-operator HBM
floor — the minimum bytes each operator MUST hold. `hbm_attribution.py` takes the **actual**
per-device resident manifest from a deployed graph, attributes each resident tensor to the
same operator classes, and diffs actual vs floor into a ranked overage table.

This is a **matched breakdown only**: it answers "which operators use more HBM than their
floor, and by how much," not "why." Root-causing an overage is a separate step that starts
*from* this table.

## Where the resident bytes actually live

This is the load-bearing fact. A compiler's memory-analysis allocation dumps (the per-pass,
per-core DRAM CSVs) track only the **coloring-allocator-managed** DRAM scratch — collectives,
matmul temporaries, spills — typically a few tens of MB. They do **not** track the model
weights or the paged KV pool.

The reason: weights and the KV pool enter the compiled graph as **fixed-address parameter
INPUTS**, not coloring-allocated DRAM. The HBM-usage compiler pass calls these "Unallocated
Tensors." The authoritative resident manifest is therefore the compiled graph's **ENTRY
signature** — every weight + KV tensor with its exact (TP-sharded) shape. Summing the ENTRY
signature reproduces the "Unallocated Tensors" figure from the HBM-usage log. So the
attribution parses the ENTRY manifest, **not** the allocation CSVs.

A static-HBM over-budget compile aborts at the HBM-usage assert before producing artifacts.
To capture the manifest of an over-budget config, disable that assert (and the dynamic-peak
checker) so the host-side compile finishes and emits the graph dump, then harvest the ENTRY
signature from the post-canonicalization HLO. The over-budget graph is exactly the one worth
inspecting — that is where the entitlement question is sharpest.

## Capturing the resident manifest

Capture is deployment-specific, but the shape is always:

1. Run the smallest bucket / smallest batch that still reproduces the footprint of interest.
   The smallest bucket holds the activation working set at its floor, so the captured
   resident footprint is dominated by weights + KV — the cleanest attribution target.
2. Add the compiler flags that dump the graph and disable the OOM/HBM-usage asserts (so an
   over-budget config still finishes the host compile and writes the dump). Gate this behind
   an env var so the normal serving path is untouched.
3. Ensure the dump survives cache promotion and bypass any compile cache (a cache hit skips
   the compiler and produces no dump).
4. Harvest the ENTRY signature from the canonicalized HLO into a manifest file.

`hbm_attribution.py` then parses that manifest:

```bash
python3.12 hbm_attribution.py \
    --entry-hlo captured_entry_manifest.hlo \
    --floor-json agent_artifacts/tmp/operator_entitlement_decode.json \
    --runspec agent_artifacts/tmp/run-spec.json \
    -o agent_artifacts/tmp/actual_vs_floor.json
```

It prints the parsed tensor count and total GB; the total should match the HBM-usage log's
"Unallocated Tensors" figure, confirming the manifest is complete.

## Attribution by shape signature

The deployed graph carries raw parameter tensors as **anonymous ENTRY inputs** — no
`Linear`/parameter names survive — so attribution keys on the **shape signature**. The
signatures are **derived from the model's `Arch` and the floor's TP**, not hardcoded, so the
same script attributes any model the run-spec describes.

Two facts drive the fold:

- **Fused vs split.** A deployed graph often fuses projections the oracle reports separately
  (a fused `qkv` = q + k + v; a fused `gate_up` = gate + up). The attribution registers both
  the fused signature and each split signature for the same class, and folds the matching
  oracle floors together so the diff is apples-to-apples. Orientation matters: a
  hidden-consuming projection `(hidden, shard)` and a hidden-producing one `(shard, hidden)`
  share a dim pair but are different operators — the signature is an *ordered* pair.
- **Tie assumptions.** The oracle may fold the embedding table into the lm_head expecting a
  weight tie. A graph that ships a separate embedding *and* lm_head surfaces as a ×2 overage
  on that class — a real, expected finding, not a bug.

The fused-qkv weight always carries q + 2·kv columns. A KV-alias optimization (where V
aliases K on some groups) is a **floor-side** correction, not a weight-storage one, so it
never shrinks the resident signature — expect a small (~×1.33) overage on an aliased group's
fused-qkv class where the stored weight rounds up vs the oracle's exact split.

## The diff and how to read it

Per operator class the diff reports `actual_bytes`, `resident_floor_bytes` (= oracle weight +
KV bytes/device for the folded ops), `overage = actual − floor`, and the ratio, ranked by
absolute overage.

- **Weights diff apples-to-apples.** A projection class at ratio ≈ 1.0 is at entitlement.
  A ratio above 1.0 is a real weight overage (untied tables, fused-shard rounding) and is the
  signal to investigate.
- **KV does NOT diff apples-to-apples — every KV row is flagged.** The actual KV figure is
  the **paged-pool reservation** (the block-table size, governed by the memory-utilization
  policy, config-independent within a run), while the floor KV is the **analytic per-request
  KV** for the floor's bucket/bs. The KV rows usually dominate the absolute byte ranking but
  represent a sizing/policy gap, not a per-operator inefficiency. Read them as "how much KV
  the pool reserved," not "how much this operator wasted."
