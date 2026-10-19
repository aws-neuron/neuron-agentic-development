# TP ceiling and KV-cache replication

The single most error-prone part of planning TP for a GQA/MQA model: **the TP ceiling is
the query-head count, and the KV cache does not shard past the KV-head count — it
replicates.** Getting this wrong over- or under-counts per-device KV bytes by the
replication factor, exactly in the high-TP regime where it matters most. The worked
example is gemma-4-31B on trn2, but the rule is general.

## 1. The TP ceiling is the query-head count, not the KV-head count

A common challenge: "we shard TP over attention heads and the model only has N KV heads —
isn't TP capped at N?" No. TP shards the **query** projection (one query head per rank at
TP = n_query_heads), so:

```
TP ceiling = n_query_heads      (going beyond would split an individual head's head_dim)
```

For gemma-4-31B: **32 query heads → TP ceiling 32.** The "16" in the challenge is the KV
head count of the sliding-window layers, a different quantity. So `TP_CHOICES` topping out
at 32 is correct, not a misread of the config.

## 2. KV heads shard to their count, then REPLICATE

The vendor sharding (gemma4 `model.py`) is explicit:

```python
num_attention_heads_per_rank = num_attention_heads // world_size      # 32 // TP
if world_size >= num_key_value_heads:
    num_key_value_heads_per_rank = 1
    num_kv_replicas = world_size // num_key_value_heads               # KV REPLICATES
```

Query heads divide by TP; KV heads shard down to 1/rank and then **replicate** once
`TP ≥ n_kv_heads`. So a layer-group's per-device KV cache divides by
**`min(TP, n_kv_heads)`, not `TP`** — beyond `n_kv_heads`, extra ranks hold redundant
copies. Replication factor = `TP / min(TP, n_kv_heads)`.

gemma-4-31B has two attention groups:
- **50 sliding-window layers** — 16 KV heads, capped at a 1024-token window.
- **10 global (`attention_k_eq_v`) layers** — 4 KV heads, context-growing (V aliases K).

The global layers' **4 KV heads** are the binding number for long context: **any TP>4
replicates the context-growing cache** — TP8=2×, TP16=4×, TP32=8×.

## 3. The bug this guards against

A naive engine divides the KV cache by `TP·PP·CP` unconditionally — silently assuming KV
keeps sharding past `n_kv_heads`. The error only appears when `TP > n_kv_heads` (i.e. the
high-TP regime under scrutiny), so it hides until exactly the configs you most want to
trust. The fix (`planner/shard.py` (kv_cache_divisor)):

```python
def kv_bytes_per_device(a, S, bs, TP, PP, CP):
    total = 0.0
    for g in a.attn_groups:
        eff_ctx = S if g.kv_seq_cap is None else min(S, g.kv_seq_cap)
        group_bytes = g.count * 2 * g.n_kv_heads * g.head_dim * a.dtype_bytes * eff_ctx * bs
        kv_shard = min(TP, g.n_kv_heads)        # replication ceiling, not TP
        total += group_bytes / (kv_shard * PP * CP)
    return total
```

PP (fewer layers/device) and CP (sequence shard) **do** divide on top — those are genuine,
not capped by head count. `kv_load_ms` and `peak_hbm_bytes` both source this.

## 4. Two consequences for selection

**(a) High-TP decode goes KV-read-bound early.** With KV correctly *not* sharding past the
head count, the per-device KV cache at TP=32 is larger than a naive model assumes, flipping
long-context decode cells from matmul-bound to KV-read-bound. The remedy is **CP**, which
shards the sequence (and thus the cache) with no replication ceiling. Worked example:

| Config | KV-read | ITL | Binding |
|---|---|---|---|
| TP=32 CP=1 | 1.504 ms | 1.504 ms | **kv_load** |
| TP=32 CP=2 | 0.752 ms | 0.799 ms | matmul |

So a "TP=32 doubles throughput and ITL for free" claim is **too strong** wherever the cache
is large — reaching the same interactivity costs an added CP (more cores), not zero. TP=32
is still the right interactivity call; it just isn't free once context grows.

**(b) High-TP prefill replicates the prefix cache.** On the colocated replica the
spare HBM is meant to hold a prefix cache; replication buys zero cache capacity while
multiplying footprint. This is why the prefill selector defaults to **non-replicating**
configs (`TP ≤ min_kv_heads`). See [methodology.md](methodology.md) §prefix-cache.

## 5. The general rule (any model)

- `TP_CHOICES` ceiling = **n_query_heads** (`min_kv_heads(arch)` is a *different*, smaller
  number — the non-replicating TP ceiling).
- Per-device KV divides by **`min(TP, n_kv_heads)` per attention group**, then PP·CP on top.
- The **smallest** KV-head count across groups (`min_kv_heads(arch)`) is the TP past which
  *some* group replicates — for gemma-4-31B that is 4 (the global layers).
- For long context, prefer **CP over TP** to grow KV sharding without the replication
  ceiling. Memory rarely binds; the KV *read* time is what high TP inflates.
