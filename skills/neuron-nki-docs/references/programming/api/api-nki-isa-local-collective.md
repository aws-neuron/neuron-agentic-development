# nki.isa — Logical NeuronCore (LNC)

> **Module**: nki.isa
> **Functions**: 2

## Overview

Logical NeuronCore (LNC) instructions.

## Functions

### nki.isa.core_barrier {#nki-isa-core_barrier}

`nki.isa.core_barrier(data, cores, engine, name)`

**Engine:** GpSimd Engine

**Signature:**

```python
isa.core_barrier(data, cores, engine=engine_enum.gpsimd, name=None)
```

Synchronize execution across multiple NeuronCores by implementing a barrier mechanism.

> **Note:**
> Available only on NeuronCore-v3 or newer.

This instruction creates a synchronization point where all specified NeuronCores must
reach before any can proceed. The barrier is implemented using a semaphore-based protocol
where each NeuronCore writes a semaphore to each other core (remote semaphore update)
and then waits for the other cores' semaphores before continuing execution (local semaphore wait).

The use case is when two NeuronCores both need to write to disjoint portions of a
shared HBM tensor (`data`) and they both need to consume the tensor after both cores
have finished writing into the tensor. In this case, both cores can perform the write to
`data` in HBM using `nisa.dma_copy`, and then signal to each other when the write operation is complete
using `nisa.core_barrier`.

This instruction is only allowed in NeuronCore-v3 or newer when
[LNC (Logical NeuronCore)](../lnc.md)
is enabled. Currently only `cores=(0, 1)` is supported. This allows synchronization between exactly
two NeuronCores that share the same HBM stack.

The `data` parameter represents the shared data that all cores need to synchronize on.
This must be data in shared HBM that multiple cores are accessing.

The `engine` parameter allows specifying which engine inside the NeuronCores should execute the barrier
instruction (that is, the remote semaphore update and local semaphore wait). The barrier will block
execution on this engine, other engines will not be blocked.

- **data** — the shared data that all cores need to synchronize on; must be data in shared HBM
- **cores** — a tuple of core indices to synchronize; only `(0, 1)` is supported when LNC2 is enabled
- **engine** — the engine to execute the barrier instruction on; defaults to GpSimd Engine

Example:

```python
# Synchronize between two cores after each core writes to half of shared tensor
shared_tensor = nl.ndarray((batch_size, hidden_dim), dtype=nl.float32, buffer=nl.shared_hbm)

# Each core writes to half of the tensor
if core_id == 0:
    # Core 0 writes to first half
    core0_data = nl.ndarray((batch_size // 2, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=shared_tensor[:batch_size // 2, :], src=core0_data)
else:
    # Core 1 writes to second half
    core1_data = nl.ndarray((batch_size // 2, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=shared_tensor[batch_size // 2:, :], src=core1_data)

nisa.core_barrier(data=shared_tensor, cores=(0, 1))

# Now both cores can safely read the complete tensor
```

---

### nki.isa.sendrecv {#nki-isa-sendrecv}

`nki.isa.sendrecv(src, dst, send_to_rank, recv_from_rank, pipe_id, dma_engine, name)`

**Engine:** DMA Engine

**Signature:**

```python
isa.sendrecv(src, dst, send_to_rank, recv_from_rank, pipe_id, dma_engine=dma_engine_enum.dma, name=None)
```

Perform point-to-point communication between NeuronCores by sending and receiving data
simultaneously using DMA engines.

> **Note:**
> Available only on NeuronCore-v3 or newer.

This instruction enables bidirectional data exchange between two NeuronCores within a
Logical NeuronCore (LNC) configuration.
The current NeuronCore sends its `src` tile to the `dst` location of the target
NeuronCore specified by `send_to_rank`,
while simultaneously receiving data from `recv_from_rank` into its own `dst` tile.

The use case is when NeuronCores need to exchange data for distributed computation patterns,
such as all-gather communication or other collective operations where cores need to
coordinate their computations by exchanging tiles.

This instruction is only allowed in NeuronCore-v3 or newer when
[LNC (Logical NeuronCore)](../lnc.md)
is enabled. The communication occurs between NeuronCores that share the same HBM stack within the LNC configuration.
Therefore, `send_to_rank` and `recv_from_rank` must be either 0 or 1.

The `pipe_id` parameter provides synchronization control by grouping sendrecv operations. Operations with the same
`pipe_id` form a logical group where all operations in the group must complete before any can proceed. Operations
with different `pipe_id` values can progress independently without blocking each other.

The `dma_engine` parameter specifies which DMA transfer mechanism to use:

- `nisa.dma_engine.dma` (default): Uses the standard DMA engine with CoreBarrier synchronization.
  Can be triggered from any engine.
- `nisa.dma_engine.gpsimd_dma`: Uses the GPSIMD's internal DMA engine for low-latency
  SB-to-SB swaps in LNC=2. Implies GPSIMD as the trigger engine. This mode restricts the data size
  per partition to not exceed:
  - 1024 bytes for 32-bit types
  - 512 bytes for 16-bit types
  - 256 bytes for 8-bit types

**Constraints.**

- `src` and `dst` tiles must both be in SBUF.
- `src` and `dst` must have the same data type, but they can be any supported data types in NKI.
- `src` and `dst` must have the same shape and layout.
- `src` and `dst` must have the same partition dimension size and the same number of elements per partition.

- **src** — the source tile on the current NeuronCore to be sent to the target NeuronCore
- **dst** — the destination tile on the current NeuronCore where received data will be stored
- **send_to_rank** — rank ID of the target NeuronCore to send data to
- **recv_from_rank** — rank ID of the source NeuronCore to receive data from
- **pipe_id** — synchronization identifier that groups sendrecv operations; operations with the same pipe_id are synchronized
- **dma_engine** — the DMA transfer mode; defaults to `nisa.dma_engine.dma`

Example:

```python
# Exchange data between two cores in a ring pattern
num_cores = 2
current_rank = nl.program_id()
next_rank = (current_rank + 1) % num_cores
prev_rank = (current_rank - 1) % num_cores

# Data to send and buffer to receive
send_data = nl.ndarray((batch_size, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
recv_buffer = nl.ndarray((batch_size, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)

# Perform bidirectional exchange
nisa.sendrecv(
    src=send_data,
    dst=recv_buffer,
    send_to_rank=next_rank,
    recv_from_rank=prev_rank,
    pipe_id=0
)

# Now recv_buffer contains data from the previous core
```

---
