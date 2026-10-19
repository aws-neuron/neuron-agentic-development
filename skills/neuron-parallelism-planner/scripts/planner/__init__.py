"""Parallelism-planner engine (see ../../references/design.md).

Three-layer pipeline, one concern per module:
  segments -- §1 aggregation: build per-class Segment lists (weights / KV / FLOPs).
  shard    -- §2 sharding: weight/kv_cache/flop divisor per (class, type) key -- the single owner.
  cost     -- §3 totals + cost: per_device_totals -> component triples; step_time -> roofline+serialize.
  build    -- wiring: run-spec -> (Arch, HW, segments, meta) and a one-call per-cell entitlement.
  plan     -- §4 grid x bucket sweep + selection (single-node colocated).

Reuses Arch/AttnGroup/MLAParams/arch_from_hf/HW/comm_total_ms from entitlement_formulas.py
(one parse of the model config -- shared with the meta-device oracle). Orthogonal concerns
(oracle validation, speculative decoding, figures) plug in on top of these layers; see §4.
"""
