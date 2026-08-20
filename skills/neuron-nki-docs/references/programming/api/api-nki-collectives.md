# nki.collectives — Collective Communication

> **Module**: nki.collectives
> **Functions**: 10

## Overview

Collective Communication instructions.

## Functions

### nki.collectives.all_gather {#nki-collectives-all_gather}

`nki.collectives.all_gather(srcs, dsts, replica_group, collective_dim, priority, name)`

**Engine:** DMA Engine

**Signature:**

```python
collectives.all_gather(srcs, dsts, replica_group, collective_dim, priority=None, name=None)
```

Perform an all-gather on the given replica group and input/output tensors.

The `srcs` and `dsts` parameters accept lists of tensors to support coalesced
collective communication, which allows multiple tensors to be gathered in a single
collective operation for improved efficiency.

Tensors can reside on either HBM or SBUF. However, mixing memory spaces is not
supported: all tensors must be on HBM or all must be on SBUF. Coalesced collective
communication (multiple tensors) is only supported when tensors are on HBM.

- **srcs** — List of input tensors to gather
- **dsts** — List of output tensors to store results
- **replica_group** — ReplicaGroup defining rank groups for the collective
- **collective_dim** — Dimension along which output tensors are concatenated.
  Currently only 0 is supported for HBM tensors. For SBUF tensors, 0 or 1 is
  supported as SBUF collectives currently only operate on 2D tensors with a
  single free dimension.
- **priority** — DMA quality-of-service priority level 0-3 where lower is higher
  priority (NeuronCore-v4+ only)
- **name** — (optional) name for the instruction.

---

### nki.collectives.all_reduce {#nki-collectives-all_reduce}

`nki.collectives.all_reduce(srcs, dsts, replica_group, op, priority, name)`

**Engine:** DMA Engine

**Signature:**

```python
collectives.all_reduce(srcs, dsts, replica_group, op, priority=None, name=None)
```

Perform an all-reduce on the given replica group and input/output tensors.

The `srcs` and `dsts` parameters accept lists of tensors to support coalesced
collective communication, which allows multiple tensors to be reduced in a single
collective operation for improved efficiency.

Tensors can reside on either HBM or SBUF. However, mixing memory spaces is not
supported: all tensors must be on HBM or all must be on SBUF. Coalesced collective
communication (multiple tensors) is only supported when tensors are on HBM.

- **srcs** — List of input tensors to reduce
- **dsts** — List of output tensors to store results
- **replica_group** — ReplicaGroup defining rank groups for the collective
- **op** — The reduction operation to perform (`nl.add`, `nl.minimum`, or `nl.maximum`)
- **priority** — DMA quality-of-service priority level 0-3 where lower is higher
  priority (NeuronCore-v4+ only)
- **name** — (optional) name for the instruction.

---

### nki.collectives.all_to_all {#nki-collectives-all_to_all}

`nki.collectives.all_to_all(srcs, dsts, replica_group, collective_dim, priority, name)`

**Engine:** DMA Engine

**Signature:**

```python
collectives.all_to_all(srcs, dsts, replica_group, collective_dim, priority=None, name=None)
```

Perform an all-to-all on the given replica group and input/output tensors.

The `srcs` and `dsts` parameters accept lists of tensors to support coalesced
collective communication, which allows multiple tensors to be redistributed in a
single collective operation for improved efficiency.

Tensors must reside on HBM. SBUF is not currently supported for all-to-all.

- **srcs** — List of input tensors to redistribute
- **dsts** — List of output tensors to store results
- **replica_group** — ReplicaGroup defining rank groups for the collective
- **collective_dim** — Dimension along which input tensors are split and output tensors are concatenated.
  Currently only 0 is supported.
- **priority** — DMA quality-of-service priority level 0-3 where lower is higher
  priority (NeuronCore-v4+ only)
- **name** — (optional) name for the instruction.

---

### nki.collectives.all_to_all_v {#nki-collectives-all_to_all_v}

`nki.collectives.all_to_all_v(srcs, dsts, replica_group, metadata_tensor, recv_counts_known, has_rdispls, priority, name)`

**Engine:** DMA Engine

**Signature:**

```python
collectives.all_to_all_v(srcs, dsts, replica_group, metadata_tensor, recv_counts_known=False, has_rdispls=False, priority=None, name=None)
```

Executes an all-to-all collective where each rank can send
a different number of elements, known only at execution time (rather
than at compile time).

Unlike `all_to_all` which splits/concatenates along a collective
dimension, `all_to_all_v` treats tensors as flat element buffers.
Per-rank send/recv counts and displacements are supplied via a uint32
metadata tensor, making per-rank payload sizes dynamic.

**Current restrictions:**

On instances with a NeuronSwitch fabric (see `Trn3 architecture
<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html>`\_),
`all_to_all_v` requires LNC=2 and more than one participating
device. Multiple ranks per device are supported, but for every
replica-group rank-list, every device participating in that
rank-list must have all of its ranks (4 under LNC=2) included in
the same rank-list — each rank-list is a set of sequential ranks
in the world (e.g. `[[1, 2, 3, 4], [5, 6, 7, 8]]`). To exclude a
rank, keep it in the replica group and set its `send_count` to 0.

On other instances, `all_to_all_v` currently supports only
inter-node replica groups: each rank-list contains same-indexed
ranks from different nodes (a node refers to a different Trn EC2
instance).

- **srcs** — Input tensor list. Currently supports exactly one tensor.
  Must be HBM-backed.
- **dsts** — Output tensor list. Currently supports exactly one tensor.
  Must be HBM-backed. `src` and `dst` element counts can be
  different; sizes are validated against the metadata at execution
  time.
- **replica_group** — ReplicaGroup defining which ranks participate.
- **metadata_tensor** — `uint32` tensor laid out contiguously in
  memory. Shape depends on backing buffer, where `rows` is 3 when
  `has_rdispls=False` and 4 when `has_rdispls=True`:
  - HBM: `(rows, replica_group_size)`.
  - SBUF: `(1, rows, replica_group_size)` — the whole buffer must
    live on a single partition, so a trivial partition dim is
    prepended.

  For each other rank `r` in the replica group, the rows are:
  - Row 0 `send_counts[r]`: number of elements sent to rank `r`.
    Always an input.
  - Row 1 `send_displs[r]`: offset in elements within `src` where
    the chunk destined for rank `r` begins. Always an input.
  - Row 2 `recv_counts[r]`: number of elements received from rank
    `r`. Controlled by `recv_counts_known` — see that flag.
  - Row 3 `recv_displs[r]`: offset in elements within `dst` where
    the chunk from rank `r` is written. Only present when
    `has_rdispls=True`.

- **recv_counts_known** —
  Controls whether row 2 is populated by the collective during
  execution. Row 2 is never read as input.
  - `True`: row 2 is left untouched, avoiding a small per-rank
    writeback.
  - `False` (default): row 2 is an **output** — per-rank received
    counts are written during execution, and can be read after the
    op to learn received sizes.

- **has_rdispls** —
  - `True`: row 3 is an **input**; recv_displs must be populated.
    The chunk from sender rank `r` is written at
    `dst[recv_displs[r] : recv_displs[r] + recv_counts[r]]`.
  - `False` (default): row 3 may be omitted from `metadata_tensor` (pass a
    3-row tensor). Incoming chunks are laid out equally-spaced at
    `recv_displs[r] = (dst.total_elements / replica_group_size) * r`,
    regardless of the actual recv_count per rank.

- **priority** — DMA QoS priority level 0-3 where lower is higher
  priority (NeuronCore-v4+ only).
- **name** — (optional) name for the instruction.

---

### nki.collectives.collective_permute {#nki-collectives-collective_permute}

`nki.collectives.collective_permute(srcs, dsts, source_target_pairs, priority, name)`

**Engine:** DMA Engine

**Signature:**

```python
collectives.collective_permute(srcs, dsts, source_target_pairs, priority=None, name=None)
```

Send and receive data between ranks based on explicitly defined source-target pairs.

Each pair `(source, target)` specifies that data from the source rank
should be sent to the target rank. This gives you full control over the
communication pattern (e.g., pairwise swaps, arbitrary shuffles).

Prefer `collective_permute_implicit` when the communication
follows a ring topology, as the hardware can optimize that pattern.

Tensors must reside on HBM. SBUF is not currently supported for collective_permute.

Coalesced collective communication (multiple tensors) is not currently supported;
each list parameter must contain exactly one tensor.

- **srcs** — List of source tensors to send
- **dsts** — List of destination tensors to receive into
- **source_target_pairs** — List of (source, target) rank ID pairs
- **priority** — DMA quality-of-service priority level 0-3 where lower is higher
  priority (NeuronCore-v4+ only)
- **name** — (optional) name for the instruction.

---

### nki.collectives.collective_permute_implicit {#nki-collectives-collective_permute_implicit}

`nki.collectives.collective_permute_implicit(srcs_by_channel, dsts_by_channel, replica_group, channel_ids, priority, name)`

**Engine:** DMA Engine

**Signature:**

```python
collectives.collective_permute_implicit(srcs_by_channel, dsts_by_channel, replica_group, channel_ids=[0], priority=None, name=None)
```

Send and receive data between ranks in a ring, where sources and destinations are
implicitly determined by the ring structure during runtime.

Each rank sends data to its successor and receives from its predecessor in the ring.
This differs from `collective_permute` where users explicitly specify source-target pairs.

Since the sources and destinations are implicitly determined, use
`collective_permute_implicit_current_processing_rank_id` to get the rank ID
whose data is currently being processed.

The outer dimension of `srcs_by_channel` and `dsts_by_channel` corresponds to channels.
For each channel, the inner list contains exactly one tensor (coalesced collective
communication is not currently supported).

**Channels**: Multiple channels enable overlapping communication, allowing concurrent data
transfers. The number of available channels depends on the replica group and system
connectivity (see
[Neuron Collectives](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/about/collectives.html#system-connectivity)).
The maximum number of channels is 4 for replica groups containing all devices inside a node
and 2 for other supported replica groups.

- **srcs_by_channel** — List of source tensor lists, one per channel. Each inner list must contain exactly one tensor.
- **dsts_by_channel** — List of destination tensor lists, one per channel. Each inner list must contain exactly one tensor.
- **replica_group** — ReplicaGroup defining rank groups for the collective
- **channel_ids** — List of channel IDs to use for communication (default [0] for single channel).
  Currently must be consecutive integers starting from 0.
- **priority** — DMA quality-of-service priority level 0-3 where lower is higher
  priority (NeuronCore-v4+ only)
- **name** — (optional) name for the instruction.

---

### nki.collectives.collective_permute_implicit_current_processing_rank_id {#nki-collectives-collective_permute_implicit_current_processing_rank_id}

`nki.collectives.collective_permute_implicit_current_processing_rank_id(iteration_id, replica_group, channel_id, name)`

**Signature:**

```python
collectives.collective_permute_implicit_current_processing_rank_id(iteration_id, replica_group, channel_id=0, name=None)
```

Returns the rank ID of the data to be processed in the current ring iteration.

This function is intended to be used in conjunction with
`collective_permute_implicit` or `collective_permute_implicit_reduce`.
Since the sources and destinations are implicitly determined in ring algorithms,
the rank ID of received data can only be determined at runtime.

At iteration 0, this returns the current rank's own ID (processing local data).
In subsequent iterations, it returns the rank ID of data received from predecessors,
progressing around the ring.

The returned rank ID is a scalar register. To determine the offset of the received
data chunk within a tensor, use register ALU operations (e.g., multiply the rank ID
by chunk size), then use dynamic access pattern (`tensor.ap()`) in ISA compute
operations (e.g., `nisa.nc_matmul()`).

**Typical usage pattern**: In each iteration of a ring algorithm, the compute kernel
uses this function to identify which rank's data is being processed, computes on that
data while concurrently triggering the next communication step to send already-computed
chunks to the successor.

**Channels**: Multiple channels enable overlapping communication, allowing concurrent data
transfers. The number of available channels depends on the replica group and system
connectivity (see
[Neuron Collectives](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/about/collectives.html#system-connectivity)).
The maximum number of channels is 4 for replica groups containing all devices inside a node
and 2 for other supported replica groups.

- **iteration_id** — Current ring step (typically the loop counter).
- **replica_group** — ReplicaGroup defining the ring topology
- **channel_id** — Channel ID for the communication (0 to num_channels-1)
- **name** — (optional) name for the instruction.
  **Returns:** Scalar register containing the rank ID of the data to be processed

---

### nki.collectives.collective_permute_implicit_reduce {#nki-collectives-collective_permute_implicit_reduce}

`nki.collectives.collective_permute_implicit_reduce(srcs0_by_channel, srcs1_by_channel, dsts_by_channel, replica_group, op, channel_ids, priority, name)`

**Engine:** DMA Engine

**Signature:**

```python
collectives.collective_permute_implicit_reduce(srcs0_by_channel, srcs1_by_channel, dsts_by_channel, replica_group, op, channel_ids=[0], priority=None, name=None)
```

Perform an implicit collective permute with reduction in a ring, where sources and
destinations are implicitly determined by the ring structure during runtime.

Combines `collective_permute_implicit` with a reduction operation.
Each rank reduces its local sources using `op(srcs0_by_channel[i], srcs1_by_channel[i])`,
sends the result to its successor, and receives its predecessor's reduced result into
`dsts_by_channel[i]`.

Since the sources and destinations are implicitly determined, use
`collective_permute_implicit_current_processing_rank_id` to get the rank ID
whose data is currently being processed.

The outer dimension of `srcs0_by_channel`, `srcs1_by_channel`, and `dsts_by_channel`
corresponds to channels. For each channel, the inner list contains exactly one tensor
(coalesced collective communication is not currently supported).

**Channels**: Multiple channels enable overlapping communication, allowing concurrent data
transfers. The number of available channels depends on the replica group and system
connectivity (see
[Neuron Collectives](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/about/collectives.html#system-connectivity)).
The maximum number of channels is 4 for replica groups containing all devices inside a node
and 2 for other supported replica groups.

- **srcs0_by_channel** — List of source tensor lists (left operand of reduction), one per channel. Each inner list must contain exactly one tensor.
- **srcs1_by_channel** — List of source tensor lists (right operand of reduction), one per channel. Each inner list must contain exactly one tensor.
- **dsts_by_channel** — List of destination tensor lists to receive predecessor's reduced result, one per channel. Each inner list must contain exactly one tensor.
- **replica_group** — ReplicaGroup defining rank groups for the collective
- **op** — The reduction operation to perform (`nl.add`, `nl.minimum`, or `nl.maximum`)
- **channel_ids** — List of channel IDs to use for communication (default [0] for single channel).
  Currently must be consecutive integers starting from 0.
- **priority** — DMA quality-of-service priority level 0-3 where lower is higher
  priority (NeuronCore-v4+ only)
- **name** — (optional) name for the instruction.

---

### nki.collectives.rank_id {#nki-collectives-rank_id}

`nki.collectives.rank_id(name)`

**Signature:**

```python
collectives.rank_id(name=None)
```

Get the rank ID of the current rank.

- **name** — (optional) name for the instruction.
  **Returns:** The rank ID of the current rank within the collective group

---

### nki.collectives.reduce_scatter {#nki-collectives-reduce_scatter}

`nki.collectives.reduce_scatter(srcs, dsts, replica_group, collective_dim, op, priority, name)`

**Engine:** DMA Engine

**Signature:**

```python
collectives.reduce_scatter(srcs, dsts, replica_group, collective_dim, op, priority=None, name=None)
```

Perform a reduce-scatter on the given replica group and input/output tensors.

The `srcs` and `dsts` parameters accept lists of tensors to support coalesced
collective communication, which allows multiple tensors to be reduced and scattered
in a single collective operation for improved efficiency.

Tensors can reside on either HBM or SBUF. However, mixing memory spaces is not
supported: all tensors must be on HBM or all must be on SBUF. Coalesced collective
communication (multiple tensors) is only supported when tensors are on HBM.

- **srcs** — List of input tensors to reduce and scatter
- **dsts** — List of output tensors to store results
- **replica_group** — ReplicaGroup defining rank groups for the collective
- **collective_dim** — Dimension along which input tensors are split.
  Currently only 0 is supported for both HBM and SBUF tensors.
- **op** — The reduction operation to perform (`nl.add`, `nl.minimum`, or `nl.maximum`)
- **priority** — DMA quality-of-service priority level 0-3 where lower is higher
  priority (NeuronCore-v4+ only)
- **name** — (optional) name for the instruction.

---

### nki.collectives.all_gather_v {#nki-collectives-all_gather_v}

`nki.collectives.all_gather_v(srcs, dsts, replica_group, metadata_tensor, recv_counts_known, has_rdispls, priority, name)`

**Engine:** DMA Engine

**Signature:**

```python
collectives.all_gather_v(srcs, dsts, replica_group, metadata_tensor, recv_counts_known=False, has_rdispls=False, priority=None, name=None)
```

Perform a variable-length all-gather on the given replica group.

Unlike `all_gather` which concatenates along a collective dimension,
`all_gather_v` treats tensors as flat element buffers. Each rank
contributes a single chunk `src[send_displ : send_displ + send_count]`
that is broadcast to every rank in the replica group — the same chunk
is sent to all destinations. Each rank `r`'s `dst` is partitioned
into equal-sized slots (one per source rank); the chunk from sender
`s` lands at `dst[s * slot_elems : s * slot_elems + slot_elems]`,
where `slot_elems = dst.total_elements / replica_group_size`.

The send side is uniform, not per-destination: `send_count` and
`send_displ` are single values that define the one chunk broadcast
to every destination. (Contrast `all_to_all_v`, where each
destination gets its own count/displacement.) Although rows 0/1 are
sized with one column per rank, only the first column is read.

Recv-side counts and displacements remain per-src-rank and live in
rows 2/3.

**Current restrictions:**

- `has_rdispls=True` is not supported.
- Only LNC=2 is supported.
- Each replica subgroup must have exactly 4 ranks (intra-chip).

- **srcs** — Input tensor list. Currently supports exactly one tensor.
  Must be HBM-backed.
- **dsts** — Output tensor list. Currently supports exactly one tensor.
  Must be HBM-backed. `src` and `dst` element counts are free to
  differ; sizes are validated against the metadata at execution time.
- **replica_group** — ReplicaGroup defining which ranks participate.
- **metadata_tensor** — `uint32` tensor laid out contiguously in
  memory. Shape depends on backing buffer, where `rows` is 3 when
  `has_rdispls=False` and 4 when `has_rdispls=True`:
  - HBM: `(rows, replica_group_size)`.
  - SBUF: `(1, rows, replica_group_size)` — the whole buffer must
    live on a single partition, so a trivial partition dim is
    prepended.

  Rows 0/1 are single-valued for all-gather: only their first
  column is read.

  The rows are:
  - Row 0 `send_count`: number of elements in the chunk broadcast
    to every rank. Only the first column is read; the same count
    applies to all destinations. Always an input.
  - Row 1 `send_displ`: offset in elements within `src` where the
    broadcast chunk begins. Only the first column is read; the same
    displacement applies to all destinations. Always an input.
  - Row 2 `recv_counts[r]`: number of elements received from rank
    `r`. Per-src-rank. Controlled by `recv_counts_known` — see
    that flag.
  - Row 3 `recv_displs[r]`: offset in elements within `dst` where
    the chunk from rank `r` is written. Per-src-rank. Only present
    when `has_rdispls=True`.

- **recv_counts_known** —
  Controls whether row 2 is populated by the collective during
  execution. Row 2 is never read as input.
  - `True`: row 2 is left untouched, avoiding a small per-rank
    writeback.
  - `False` (default): row 2 is an **output** — per-rank received
    counts are written during execution, and can be read after the
    op to learn received sizes.

- **has_rdispls** —
  - `True`: row 3 is an **input**; recv_displs must be populated.
    The chunk from sender rank `r` is written at
    `dst[recv_displs[r] : recv_displs[r] + recv_counts[r]]`.
  - `False` (default): row 3 may be omitted from `metadata_tensor`
    (pass a 3-row tensor). Incoming chunks are laid out
    equally-spaced at
    `block_offset(r) = dst.total_elements / replica_group_size * r`,
    regardless of the actual recv_count per rank.

- **priority** — DMA QoS priority level 0-3 where lower is higher
  priority (NeuronCore-v4+ only).
- **name** — (optional) name for the instruction.

---
