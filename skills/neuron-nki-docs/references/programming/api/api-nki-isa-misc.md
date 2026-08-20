# NKI ISA - Miscellaneous

> **Module**: nki.isa
> **Total Functions**: 15

## Overview

Other ISA functions.

## Functions

### nki.isa.dge_mode {#nki-isa-dge_mode}

# nki.isa.dge_mode

nki.isa.dge_mode

*class *nki.isa.dge*mode(\_value*)[[source]](../../../\_modules/nki/isa.html#dge_mode)
Neuron Descriptor Generation Engine Mode

Attributes

| unknown | Unknown DGE mode, i.e., let compiler decide the DGE mode |
| ------- | -------------------------------------------------------- |
| swdge   | Software DGE                                             |
| hwdge   | Hardware DGE                                             |
| none    | Not using DGE                                            |

---

### nki.isa.engine {#nki-isa-engine}

# nki.isa.engine

nki.isa.engine

*class *nki.isa.engine(_value_)[[source]](../../../\_modules/nki/isa.html#engine)
Neuron Device engines

Attributes

| tensor  | Tensor Engine  |
| ------- | -------------- |
| vector  | Vector Engine  |
| scalar  | Scalar Engine  |
| gpsimd  | GpSIMD Engine  |
| dma     | DMA Engine     |
| sync    | Sync Engine    |
| unknown | Unknown Engine |

---

### nki.isa.quantize_mx {#nki-isa-quantize_mx}

`nki.isa.quantize_mx(dst, src, dst_scale, name)`

**Engine:** Tensor Engine

**Signature:**

```python
isa.quantize_mx(dst, src, dst_scale, name=None)
```

Quantize FP16/BF16 data to MXFP8 tensors (both data and scales) using Vector Engine.

> **Note:**
>
> Available only on NeuronCore-v4 and newer.

The resulting `dst` and `dst_scale` tensors use the MXFP8 element and scale data types as defined in the
[OCP Microscaling standard](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
This instruction calculates the required scales for each group of 32 values in `src`, divides them by the calculated scale,
and casts to the target MXFP8 element data type.

The scale calculation differs from the sample conversion algorithm described in the OCP
specification: this instruction uses a block scale that is two times larger, reserving
additional range for rounding the largest values in each group without saturation. This
remains OCP MX-compliant.

The output layout is suitable for direct consumption by the
`nisa.nc_matmul_mx` API running on Tensor Engine.

**Memory types.**

All input `src` and output tiles (`dst` and `dst_scale`) must be in SBUF.

**Data types.**

The input `src` tile must be float16 or bfloat16. The output `dst` tile must be float8_e5m2_x4 or
float8_e4m3fn_x4 (4-packed FP8 data types). The `dst_scale` tile must be float8_e8m0fnu or uint8
(prefer `float8_e8m0fnu`: OCP MX standard; uint8 accepted for backward compatibility).

The 4-packed data types (float8_e5m2_x4/float8_e4m3fn_x4) are 32-bit data types that pack four 8-bit
float8_e5m2/float8_e4m3fn values.

**Layout.**

The quantization operates on groups of 32 elements from the input `src` tile, where each group consists of
8 partitions × 4 elements per partition. For each 32-element group, the instruction produces:

- Quantized FP8 data in `dst`
- One shared scale value in `dst_scale` per group

**Tile size.**

- The partition dimension size of `src` must be a multiple of 32 and must not exceed 128.
- The free dimension size of `src` must be a multiple of 4 and must not exceed the physical size of each SBUF
  partition.
- The `dst` tile has the same partition dimension size as `src` but a free dimension size
  that is 1/4 of `src` free dimension size due to the special 4-packed FP8 data types.

**Scale calculation.**

For a group :math:`V` of 32 values, let

.. math::

a*{\max} = \max*{V_i \in V} |V_i|

Let :math:`E_{\max}` be the maximum unbiased exponent of the destination
element data type: 8 for `float8_e4m3fn` and 15 for `float8_e5m2`.

The block scale :math:`X` is calculated as

.. math::

X =
2^{
\left\lfloor \log*2(a*{\max}) \right\rfloor - (E\_{\max} - 1)
}

For an all-zero group, where :math:`a_{\max} = 0`, the block scale is set to
:math:`X = 2^{-127}`, the minimum value representable by `float8_e8m0fnu`.

- **dst** — the quantized MXFP8 output tile
- **src** — the input FP16/BF16 tile to be quantized
- **dst_scale** — the MXFP8 output scale tile (float8_e8m0fnu or uint8)

---

### nki.isa.rand2 {#nki-isa-rand2}

`nki.isa.rand2(dst, min, max, name)`

**Engine:** Vector Engine

**Signature:**

```python
isa.rand2(dst, min, max, name=None)
```

Generate pseudo random numbers with uniform distribution using Vector Engine.

> **Note:**
>
> Available only on NeuronCore-v4 and newer.

This instruction generates pseudo random numbers and stores them into SBUF/PSUM.
The generated values follow a uniform distribution within the specified [min, max] range.

Key features:

- Uses XORWOW PRNG algorithm for high-quality random number generation
- Generates FP32 random values with uniform distribution
- Supports output conversion to various data types

**Memory types.**

The output `dst` tile can be in SBUF or PSUM.

**Data types.**

The output `dst` tile can be any of: float8_e4m3, float8_e5m2, float16, bfloat16, float32,
tfloat32, int8, int16, int32, uint8, uint16, or uint32.

**Tile size.**

The partition dimension size of `dst` must not exceed 128. The number of
elements per partition of `dst` must not exceed the physical size of each SBUF/PSUM partition.

**Constraints.**

- Supported arch versions: NeuronCore-v4+.
- Supported engines: Vector.
- min < max for valid range.

- **dst** — the destination tensor to write random values to
- **min** — minimum value for uniform distribution range (FP32), can be a scalar or vector value
- **max** — maximum value for uniform distribution range (FP32), can be a scalar or vector value

---

### nki.isa.rand_get_state {#nki-isa-rand_get_state}

`nki.isa.rand_get_state(dst, engine, name)`

**Engine:** GpSimd Engine

**Signature:**

```python
isa.rand_get_state(dst, engine=engine_enum.gpsimd, name=None)
```

Store the current pseudo random number generator (PRNG) states from the engine.

This instruction stores the current PRNG states cached inside the engine to SBUF/PSUM.
Each partition in the output tensor holds the PRNG states for the corresponding compute lane
inside the engine.

**Memory types.**

The output `dst` tile must be in SBUF (NeuronCore-v3) or SBUF/PSUM (NeuronCore-v4+).

**Data types.**

The output `dst` tile must be uint32.

**Tile size.**

- dst element count for XORWOW must be 6 elements (GpSimd) or 24 elements (Vector).

**Constraints.**

- Supported arch versions: NeuronCore-v3+.
- Supported engines: NeuronCore-v3: GpSimd. NeuronCore-v4+: GpSimd, Vector.
- Since GpSimd Engine cannot access PSUM, `dst` must be in SBUF when using GpSimd Engine.

- **dst** — the destination tensor to store PRNG state values; must be a 2D uint32 tensor
- **engine** — specify which engine to use: `nki.isa.engine.gpsimd` (default)
  or `nki.isa.engine.vector` (NeuronCore-v4+)

---

### nki.isa.rand_set_state {#nki-isa-rand_set_state}

`nki.isa.rand_set_state(src_seeds, engine, name)`

**Engine:** GpSimd Engine

**Signature:**

```python
isa.rand_set_state(src_seeds, engine=engine_enum.gpsimd, name=None)
```

Seed the pseudo random number generator (PRNG) inside the engine.

This instruction initializes the PRNG state for future random number generation operations.
Each partition in the source tensor seeds the PRNG states for the corresponding compute lane
inside the engine.

The PRNG state is cached inside the engine as a persistent state during the rest of NEFF
execution. However, the state cannot survive TPB resets or Runtime reload.

**Memory types.**

The input `src_seeds` tile must be in SBUF.

**Data types.**

The input `src_seeds` tile must be uint32.

**Tile size.**

- src_seeds element count for XORWOW must be 6 elements (GpSimd) or 24 elements (Vector).

**Constraints.**

- Supported arch versions: NeuronCore-v3+.
- Supported engines: NeuronCore-v3: GpSimd. NeuronCore-v4+: GpSimd, Vector.
- `src_seeds` must be in SBUF.

- **src_seeds** — the source tensor containing seed values for the PRNG; must be a 2D uint32 tensor
  with the partition dimension representing the compute lanes and the free dimension
  containing the seed values
- **engine** — specify which engine to use: `nki.isa.engine.gpsimd` (default)
  or `nki.isa.engine.vector` (NeuronCore-v4+)

---

### nki.isa.reduce_cmd {#nki-isa-reduce_cmd}

# nki.isa.reduce_cmd

nki.isa.reduce_cmd

*class *nki.isa.reduce*cmd(\_value*)[[source]](../../../\_modules/nki/isa.html#reduce_cmd)
Engine Register Reduce commands

Attributes

| idle         | Not using the accumulator registers                                                                                        |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| reset        | Resets the accumulator registers to its initial state                                                                      |
| reduce       | Keeps accumulating over the current value of the accumulator registers                                                     |
| reset_reduce | Resets the accumulator registers then immediately accumulate the results of the current instruction into the accumulators  |
| load_reduce  | Loads a value into the accumulator registers, then accumulate the results of the current instruction into the accumulators |

---

### nki.isa.register_alloc {#nki-isa-register_alloc}

`nki.isa.register_alloc(x)`

**Engine:** GpSimd Engine

**Signature:**

```python
isa.register_alloc(x=None)
```

Allocate a virtual register and optionally initialize it with a value.

Each engine sequencer (Tensor/Scalar/Vector/GpSimd/Sync Engine) within a NeuronCore maintains its own set of
physical registers for scalar operations (64x 32-bit registers per engine sequencer in NeuronCore v2-v4).
This API conceptually allocates a register within a virtual register space.
Users do not need to explicitly free a register through nisa APIs. The NKI compiler
handles physical register allocation (and deallocation) across the appropriate engine sequencers
based on the dynamic program flow.

NKI provides the following APIs to manipulate allocated registers:

- `nisa.register_move`: Move a constant integer or another register's value into a register
- `nisa.register_load`: Load a scalar (32-bit) value from HBM/SBUF into a register
- `nisa.register_store`: Store register contents to HBM/SBUF

In the current NKI release, these registers are primarily used to specify dynamic loop boundaries and
while loop conditions. The NKI compiler compiles such dynamic looping constructs to branching instructions
executed by engine sequencers. For additional details, see `nl.dynamic_range`. For more information
on engine sequencer and its capabilities, see
[Trainium/Inferentia2 architecture guide](../../architecture/trainium_inferentia2_arch.md).

- **x** — optional initialization value. Can be one of:

          - `None` (default): allocate an uninitialized register
          - `int`: allocate a register initialized with this immediate integer value

Example:

Three ways to allocate a register initialized to zero:

```python
# Approach 1: Using an immediate value
reg1 = nisa.register_alloc(0)

# Approach 2: Two-step with register_load
zero_tensor = nl.zeros([1, 1], dtype=nl.int32, buffer=nl.sbuf)
reg2 = nisa.register_alloc(None)
nisa.register_load(reg2, zero_tensor)
```

---

### nki.isa.register_load {#nki-isa-register_load}

`nki.isa.register_load(dst, src)`

**Signature:**

```python
isa.register_load(dst, src)
```

Load a scalar value from memory (HBM or SBUF) into a virtual register.

This instruction reads a single scalar value (up to 32-bit) from a memory location (HBM or SBUF)
and stores it in the specified virtual register. The source must be a NKI tensor with exactly
one element (shape [1] or [1, 1]). This enables dynamic loading of values computed at
runtime into registers for use in control flow operations.

The virtual register system allows the NKI compiler to allocate physical registers across
different engine sequencers as needed. See `nisa.register_alloc` for more details on
virtual register allocation.

- **dst** — the destination virtual register (allocated via `nisa.register_alloc`)
- **src** — the source tensor containing a single scalar value to load

Example:

```python
# Load a computed value into a register
computed_bound = nl.ones([1], dtype=nl.int32, buffer=nl.sbuf)  # bound of 1 in SBUF
loop_reg = nisa.register_alloc()
nisa.register_load(loop_reg, computed_bound)
```

---

### nki.isa.register_move {#nki-isa-register_move}

`nki.isa.register_move(dst, src)`

**Signature:**

```python
isa.register_move(dst, src)
```

Move a value into a virtual register.

This instruction loads a value into the specified virtual register. The source can be
either a compile-time constant integer or another virtual register.

The virtual register system allows the NKI compiler to allocate physical registers across
different engine sequencers as needed. See `nisa.register_alloc` for more details on
virtual register allocation.

This instruction operates on virtual registers only and does not access SBUF, PSUM, or HBM.

- **dst** — the destination virtual register (allocated via `nisa.register_alloc`)
- **src** — source value - either a compile-time constant integer or a VirtualRegister

Example:

```python
# Allocate a register and initialize it with a constant
loop_count = nisa.register_alloc()
nisa.register_move(loop_count, 10)  # Set register to 10

# Copy from another register
reg2 = nisa.register_alloc()
nisa.register_move(reg2, loop_count)  # Copy value from loop_count
```

---

### nki.isa.register_store {#nki-isa-register_store}

`nki.isa.register_store(dst, src)`

**Signature:**

```python
isa.register_store(dst, src)
```

Store the value from a virtual register into memory (HBM/SBUF).

This instruction writes the scalar value (up to 32-bit) stored in a virtual register to a memory location
(HBM or SBUF). The destination must be a tensor with exactly one element (shape [1] or [1, 1]).
This enables saving register values back to memory for later use or for output purposes.

The virtual register system allows the NKI compiler to allocate physical registers across
different engine sequencers as needed. See `nisa.register_alloc` for more details on
virtual register allocation.

- **dst** — the destination tensor with a single element to store the register value
- **src** — the source virtual register (allocated via `nisa.register_alloc`)

Example:

```python
# Store a register value back to memory
counter_reg = nisa.register_alloc(0)
# ... perform operations that modify counter_reg ...
result_tensor = nl.ndarray([1], dtype=nl.int32, buffer=nl.sbuf)
nisa.register_store(result_tensor, counter_reg)
```

---

### nki.isa.rng {#nki-isa-rng}

`nki.isa.rng(dst, engine, name)`

**Engine:** GpSimd Engine

**Signature:**

```python
isa.rng(dst, engine=engine_enum.vector, name=None)
```

Generate pseudo random numbers using the Vector or GpSimd Engine.

This instruction generates 32 random bits per element and writes them to the
destination tensor. Depending on the size of the dtype, the instruction truncates
each 32-bit random value to the specified data type, taking the least significant bits.

Example use case:
To generate random FP32 numbers between 0.0 and 1.0, follow the Rng instruction
with a normalization instruction (e.g., write 16 random bits as UINT16, then
divide by (2^16-1) to get a random FP32 number between 0.0 and 1.0).

**Memory types.**

The output `dst` tile can be in SBUF or PSUM.

**Data types.**

The output `dst` tile must be an integer type: int8, int16, int32, uint8, uint16, or uint32.

**Tile size.**

The partition dimension size of `dst` must not exceed 128. The number of
elements per partition of `dst` must not exceed the physical size of each SBUF/PSUM partition.

**Constraints.**

- Supported arch versions: NeuronCore-v2+.
- Supported engines: NeuronCore-v2: Vector. NeuronCore-v3+: GpSimd, Vector.
- Since GpSimd Engine cannot access PSUM, `dst` must be in SBUF when using GpSimd Engine.

- **dst** — the destination tensor to write random values to
- **engine** — specify which engine to use: `nki.isa.engine.vector` (default)
  or `nki.isa.engine.gpsimd` (NeuronCore-v3+)

---

### nki.isa.set_rng_seed {#nki-isa-set_rng_seed}

`nki.isa.set_rng_seed(src_seeds, name)`

**Engine:** Vector Engine

**Signature:**

```python
isa.set_rng_seed(src_seeds, name=None)
```

Seed the pseudo random number generator (PRNG) inside the Vector Engine.

The PRNG state is cached inside the engine as a persistent state during the rest of NEFF
execution. However, the state cannot survive TPB resets or Runtime reload.

Using the same seed will generate the same sequence of random numbers when used
together with the `nisa.rng()` on the Vector Engine.

**Memory types.**

The input `src_seeds` must be in SBUF or PSUM.

**Data types.**

The input `src_seeds` must be a 32-bit value.

**Tile size.**

The input `src_seeds` must be a [1,1] tensor.

- **src_seeds** — a [1,1] tensor on SBUF or PSUM with a 32-bit value to be used as the seed

---

### nki.isa.exponential {#nki-isa-exponential}

`nki.isa.exponential(dst, src, max_value, reduce_res, reduce_cmd, reduce_init, name)`

**Engine:** Vector Engine

**Signature:**

```python
isa.exponential(dst, src, max_value=0.0, reduce_res=None, reduce_cmd=reduce_cmd_enum.idle, reduce_init=0.0, name=None)
```

Apply exponential function to each element after subtracting a max_value using Vector Engine.

> **Note:**
> Available only on NeuronCore-v4 and newer.

This instruction computes `exp(src - max_value)` for each element. The instruction can
optionally maintain a running sum of the exponential values using shared internal reduction
registers in the Vector Engine.

The exponential operation is performed as:

```
dst[i] = exp(src[i] - max_value)

```

When accumulation is enabled through `reduce_cmd`, the instruction also computes:

```
reduce_res[i] = sum(dst[i])

```

The Vector Engine performs the computation in float32 precision internally and can
output results in various data types as specified by the `dst` dtype field.

**Constraints**

- Supported engines: Vector.
- `src`, `dst` must have the same number of elements in the partition dimension.
- `src`, `dst` must have the same number of elements in the free dimensions.
- `src`, `dst` can be up to 4D tensor.
- `reduce_init` should be unset or set to `0.0` when `reduce_cmd` is not `load_reduce`.

**Tensor indirection.**

`dst` and `src` support tensor indirection (gather/scatter) by passing a view
created with `.indirect(index)`. `max_value`, `reduce_res`, and
`reduce_init` do **not** support tensor indirection. Runs on the Vector engine.
(`exponential` itself is only available on NeuronCore-v4 and later.)

When operands are manually allocated, their base partitions must satisfy:

- the `index` of every `.indirect()` view starts on a quadrant boundary
  (a multiple of 32);
- if `src` uses `.indirect()`, `src` starts on the same partition as
  its `index`;
- if `dst` uses `.indirect()` and `src` is in SBUF, `dst`'s `index`
  starts on the same partition as `src`;
- if `dst` uses `.indirect()` and `src` is in PSUM and uses
  `.indirect()`, `dst`'s `index` starts on the same partition as
  `src`'s `index`.

- **dst** — The output tile with exponential function applied. Supported buffers: SBUF, PSUM. Supported dtypes: float8_e4m3, float8_e5m2, float16, bfloat16, float32, tfloat32, int8, int16, int32, uint8, uint16.
- **src** — The input tile to apply exponential function on. Supported buffers: SBUF, PSUM. Supported dtypes: float8_e4m3, float8_e5m2, float16, bfloat16, float32, int8, int16, int32, uint8, uint16, uint32.
- **max_value** — The maximum value to subtract from each element before applying exponential (for numerical stability). Can be a scalar or vector of shape `(src.shape[0], 1)`. Supported dtypes: float32.
- **reduce_res** — Optional tile to store reduction results (sum of exponentials). Must have shape `(src.shape[0], 1)`. Supported buffers: SBUF, PSUM. Supported dtypes: float8_e4m3, float8_e5m2, float16, bfloat16, float32, tfloat32.

  Pass `None` to keep the reduction result in the Vector Engine's internal
  accumulator without writing it out. This is useful when chaining multiple
  calls that reduce into the same accumulator — only the final call needs to
  pass a tile to retrieve the accumulated result.

- **reduce_cmd** — Control the state of reduction registers for accumulating exponential results. Supported: `idle`, `reset_reduce`, `reduce`, `load_reduce`.
- **reduce_init** — Initial value for reduction when using `reduce_cmd.load_reduce`. Supported dtypes: float32.

**Accumulator behavior:**

The Vector Engine maintains internal accumulator registers that can be controlled via the `reduce_cmd` parameter:

- `reduce_cmd.reset_reduce`: Reset accumulators to 0, then accumulate the current results.
- `reduce_cmd.reduce`: Continue accumulating without resetting (useful for multi-step reductions).
- `reduce_cmd.load_reduce`: Load the values from `reduce_init` into the accumulator, then accumulate the current result on top of it.
- `reduce_cmd.idle`: (default) No accumulation performed, accumulator state unknown.

> **Note:**
> Even when `reduce_cmd` is set to `idle`, the accumulator state may still be modified.
> Always use `reset_reduce` after any Vector Engine operation that ran with `idle` mode to ensure
> consistent behavior.
>
> **Note:**
> The accumulator registers are shared across Vector Engine accumulation instructions including
> nki.isa.range_select , nki.isa.select_reduce ,
> nki.isa.tensor_scalar_reduce , and
> nki.isa.tensor_scalar_cumulative .

**Behavior**

```python
# Initialize reduction if requested
if reduce_cmd == reduce_cmd.reset_reduce:
    accumulator = 0
elif reduce_cmd == reduce_cmd.load_reduce:
    accumulator = reduce_init
elif reduce_cmd == reduce_cmd.idle:
    accumulator = undefined  # Not used

# Process each element
for i in range(num_elements):
    dst[i] = exp(src[i] - max_value)

    # Update reduction if active
    if reduce_cmd != reduce_cmd.idle:
        accumulator += dst[i]
```

---

### nki.isa.nonzero_with_count {#nki-isa-nonzero_with_count}

`nki.isa.nonzero_with_count(dst, src, index_offset, padding_val, name)`

**Engine:** GpSimd Engine

**Signature:**

```python
isa.nonzero_with_count(dst, src, index_offset=0, padding_val=-1, name=None)
```

Find indices of nonzero elements in an input tensor and their total count using GpSimd Engine.

> **Note:**
>
> Available only on NeuronCore-v3 and newer.

NOTE: this instruction only operates on partitions [0, 16, 32, ..., 112] of the input tile
and writes to partitions [0, 16, 32, ..., 112] of the destination tile. The data in other
partitions of the destination tile are not modified, including the last 'extra' slot for count.

This behavior is due to the physical connectivity of GpSimd engine. Each of the eight GpSimd cores
connects to 16 contiguous SBUF partitions (e.g., core[0] connects to partitions[0:16]).
In nonzero_with_count, each GpSimd core reads from and writes to its 0-th partition only.

This instruction takes an input array and produces an output array containing the indices of all
nonzero elements, followed by padding values, and ending with the count of nonzero elements found.

The output tensor has one more element in the free dimension than the input tensor:

- **First N elements**: 0-indexed positions of nonzero elements, offset by `index_offset`
- **Next T-N elements**: Filled with `padding_val`
- **Last element**: Count `N` of nonzero elements found

The `index_offset` parameter is useful when processing arrays in tiles, allowing
indices to be relative to the original array position rather than the tile.

Example for one partition of the tensor:

```
Input array (T=8): [0, 1, 1, 0, 0, 1, 0, 0]
index_offset = 16
padding_val = -1

Output (T+1=9): [17, 18, 21, -1, -1, -1, -1, -1, 3]

Where:

- 17, 18, 21 are the indices (1, 2, 5) plus offset 16
- -1 is the padding value for unused slots
- 3 is the count of nonzero elements

```

**Constraints**

- Supported arch versions: NeuronCore-v3+.
- Supported engines: GpSimd.
- Parameters `src`, `dst` must have the same number of elements in the partition dimension.
- Destination tensor must have exactly 1 more element than the source tensor in the free dimension.
- Only accesses the 0-th partition for each GpSimd core (i.e., [0, 16, 32, ..., 112]).
- `src` must be in SBUF with dtype float32 or int32.
- `dst` must be in SBUF with dtype int32.
- `index_offset` and `padding_val` must be int32.

- **src** — Input tensor to find nonzero indices from. Only partitions [0, 16, 32, ..., 112] are read from. Supported buffers: SBUF. Supported dtypes: float32, int32.
- **dst** — Output tensor containing nonzero indices, padding, and count. Only partitions [0, 16, 32, ..., 112] are written to. It must have one extra element than src in the free dimension. Supported buffers: SBUF. Supported dtypes: int32.
- **index_offset** — Offset to add to the found indices (useful for tiled processing). Supported dtypes: int32.
- **padding_val** — Value to use for padding unused output elements. Supported dtypes: int32.

**Behavior**

```python
# Find all nonzero elements in input
nonzero_indices = []
for i in range(len(input_array)):
    if input_array[i] != 0:
        nonzero_indices.append(i + index_offset)

# Build output array
output = []
# Add found indices
for idx in nonzero_indices:
    output.append(idx)
# Add padding for remaining slots
for _ in range(len(input_array) - len(nonzero_indices)):
    output.append(padding_val)
# Add count as last element
output.append(len(nonzero_indices))

```

**Example**

```python
def nonzero_with_count_kernel(in_tensor):
    in_shape = in_tensor.shape
    assert len(in_tensor.shape) == 2, "expected 2D tensor"

    in_tile = nl.ndarray(in_shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=in_tile, src=in_tensor)

    out_tile = nl.ndarray((in_shape[0], in_shape[1] + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=out_tile, src=in_tile, index_offset=0, padding_val=-1)

    out_tensor = nl.ndarray(out_tile.shape, dtype=out_tile.dtype, buffer=nl.hbm)
    nisa.dma_copy(dst=out_tensor, src=out_tile)

    return out_tensor
```

---
