# NKI ISA - Scalar Engine

> **Module**: nki.isa
> **Total Functions**: 4

## Overview

Scalar Engine instructions.

## Functions

### nki.isa.activation {#nki-isa-activation}

`nki.isa.activation(dst, op, data, bias, scale, reduce_op, reduce_res, reduce_cmd, name)`

**Engine:** Scalar Engine

**Signature:**

```python
isa.activation(dst, op, data, bias=None, scale=1.0, reduce_op=None, reduce_res=None, reduce_cmd=reduce_cmd_enum.idle, name=None)
```

Apply an activation function on every element of the input tile using Scalar Engine, with an optional scale/bias operation
before the activation and an optional reduction operation after the activation in the same instruction.

The activation function is specified in the `op` input field (see [Supported Activation Functions for NKI ISA](nki.api.shared.md#nki-act-func) for a list of
supported activation functions and their valid input ranges).

`nisa.activation` can optionally multiply the input `data` by a scalar or vector `scale`
and then add another vector `bias` before the activation function is applied.

After the activation function
is applied, Scalar Engine can also reduce along the free dimensions of the activated data per lane, using
`reduce_op` operation. `reduce_op` must be `nl.add`.

The reduction result is then either stored into or reduced on top of a set of internal engine registers
called `reduce_regs` (one 32-bit register per compute lane, 128 registers in total), controlled by the
`reduce_cmd` field:

- `nisa.reduce_cmd.reset`: Reset `reduce_regs` to zero only.
- `nisa.reduce_cmd.idle`: Do not modify `reduce_regs`.
- `nisa.reduce_cmd.reduce`: Reduce activated data over existing values in `reduce_regs`.
- `nisa.reduce_cmd.reset_reduce`: Reset `reduce_regs` to zero and then store the reduction result
  of the activated data.

`nisa.activation` can also emit another instruction to read out `reduce_regs` by
passing an SBUF/PSUM tile in the `reduce_res` arguments.
The `reduce_regs` state can persist across multiple `nisa.activation` instructions without the need to
be evicted back to SBUF/PSUM (`reduce_res` tile).

The following is the pseudo code for `nisa.activation`:

```python
output = op(data * scale + bias)

if reduce_cmd == nisa.reduce_cmd.reset or reduce_cmd == nisa.reduce_cmd.reset_reduce:
    reduce_regs = 0

result = reduce_op(reduce_regs, reduce_op(output, axis=<FreeAxis>))

if reduce_cmd == nisa.reduce_cmd.reduce or reduce_cmd == nisa.reduce_cmd.reset_reduce:
    reduce_regs += result

if reduce_res:
    reduce_res = reduce_regs

```

All these optional operations incur no further performance penalty compared to only applying the activation function,
except reading out `reduce_regs` into `reduce_res` will have a small overhead due to an extra instruction.

**Memory types.**

The input `data` tile can be an SBUF or PSUM tile. Similarly, the instruction
can write the output `dst` tile into either SBUF or PSUM.

**Data types.**

Both input `data` and output `dst` tiles can be in any valid NKI data type
(see [Supported Data Types](nki.api.shared.md#nki-dtype) for more information).
The Scalar Engine always performs the math operations in float32 precision.
Therefore, the engine automatically casts the input `data` tile to float32 before
performing multiply/add/activate specified in the activation instruction.
The engine is also capable of casting the float32 math results into another
output data type in `dst` at no additional performance cost.
The `scale` parameter must
have a float32 data type, while the `bias` parameter can be any supported dtype except tfloat32.

**Layout.**

The `scale` can either be a compile-time constant scalar or a
`[N, 1]` vector from SBUF/PSUM. `N` must be the same as the partition dimension size of `data`.
In NeuronCore-v2, the `bias` must be a `[N, 1]` vector, but starting NeuronCore-v3, `bias` can either be
a compile-time constant scalar or a `[N, 1]` vector similar to `scale`.

When the `scale` (or similarly, `bias`) is a scalar, the scalar
is broadcasted to all the elements in the input `data` tile to perform the computation.
When the `scale` (or `bias`) is a vector, the `scale` (or `bias`) value in each partition is broadcast
along the free dimension of the `data` tile.

**Tile size.**

The partition dimension size of input `data` and output `dst` tiles must be the same and must not exceed 128.
The number of elements per partition of `data` and `dst` tiles must be the same and must not
exceed the physical size of each SBUF partition.

**Tensor indirection.**

On NeuronCore-v4 and later, `dst` and `data` support tensor indirection
(gather/scatter) by passing a view created with `.indirect(index)`. `bias`,
`scale`, and `reduce_res` do **not** support tensor indirection. Runs on the
Scalar engine.

When operands are manually allocated, their base partitions must satisfy:

- the `index` of every `.indirect()` view starts on a quadrant boundary
  (a multiple of 32);
- if `data` uses `.indirect()`, `data` starts on the same partition as
  its `index`;
- if `dst` uses `.indirect()` and `data` is in SBUF, `dst`'s `index`
  starts on the same partition as `data`;
- if `dst` uses `.indirect()` and `data` is in PSUM and uses
  `.indirect()`, `dst`'s `index` starts on the same partition as
  `data`'s `index`.

In addition, on the Scalar engine a scattered `dst` cannot be in PSUM.

> **Note:**
> `sin`, `arctan`, `log`, `sqrt`, `rsqrt`, and `reciprocal`
> have limited valid input ranges. See [Supported Activation Functions for NKI ISA](nki.api.shared.md#nki-act-func) for their
> ranges and out-of-range behavior.

- **dst** — the activation output
- **op** — an activation function (see [Supported Activation Functions for NKI ISA](nki.api.shared.md#nki-act-func) for supported functions).
- **data** — the input tile; layout: (partition axis <= 128, free axis)
- **scale** — a scalar or a vector for multiplication
- **bias** — a scalar (NeuronCore-v3 or newer) or a vector for addition
- **reduce_op** — the reduce operation to perform on the free dimension of the activated data
- **reduce_res** — a tile of shape `(data.shape[0], 1)` to hold the final state of `reduce_regs`.

  Pass `None` to keep the reduction result in the Scalar Engine's internal
  accumulator without writing it out. This is useful when chaining multiple
  calls that reduce into the same accumulator — only the final call needs to
  pass a tile to retrieve the accumulated result.

- **reduce_cmd** — an enum member from `nisa.reduce_cmd` to control the state of `reduce_regs`.

---

### nki.isa.activate2 {#nki-isa-activate2}

`nki.isa.activate2(dst, op, data, imm0, imm1, op0, op1, relu_param, reverse0, reverse1, reduce_op, reduce_res, reduce_cmd, name)`

**Engine:** Scalar Engine

**Signature:**

```python
isa.activate2(dst, op, data, imm0, imm1, op0, op1, relu_param=0.0, reverse0=False, reverse1=False, reduce_op=None, reduce_res=None, reduce_cmd=reduce_cmd_enum.idle, name=None)
```

Perform tensor activation with configurable tensor-scalar operations and optional reduction
using Scalar Engine.

> **Note:**
> Available only on NeuronCore-v4 and newer.

This instruction provides a three-stage pipeline per partition:

1. Tensor-scalar operations: `(data op0 imm0) op1 imm1`
2. Activation function application via `op`
3. Optional internal reduction controlled by `reduce_op` and `reduce_cmd`

The tensor-scalar stage supports six `(op0, op1)` combinations:

- `(nl.multiply, nl.add)` — scale and bias
- `(nl.multiply, nl.subtract)` — scale and negative bias
- `(nl.multiply, nl.bypass)` — scale only
- `(nl.add, nl.bypass)` — bias only
- `(nl.subtract, nl.bypass)` — subtract only
- `(nl.bypass, nl.bypass)` — no tensor-scalar operation

When `reverse0=True`, the first operation computes `imm0 <op0> data` instead of
`data <op0> imm0`. Similarly, `reverse1=True` computes `imm1 <op1> result`.

The Scalar Engine always performs math in float32 precision, automatically casting
input data to float32 before computation and casting results to the output dtype
at no additional performance cost.

**Constraints**

- Supported engines: Scalar.
- `data` and `dst` must have the same partition dimension size (at most 128).
- `data` and `dst` must have the same number of elements in the free dimensions.
- All immediates (`imm0`, `imm1`) must have the same dtype when both are tensors.
- `op1` requires `op0` to be set.
- `reverse0` requires `op0` to be set; `reverse1` requires `op1` to be set.

**Tensor indirection.**

`dst` and `data` support tensor indirection (gather/scatter) by passing a view
created with `.indirect(index)`. `imm0`, `imm1`, `relu_param`, and
`reduce_res` do **not** support tensor indirection. Runs on the Scalar engine.
(`activate2` itself is only available on NeuronCore-v4 and later.)

When operands are manually allocated, their base partitions must satisfy:

- the `index` of every `.indirect()` view starts on a quadrant boundary
  (a multiple of 32);
- if `data` uses `.indirect()`, `data` starts on the same partition as
  its `index`;
- if `dst` uses `.indirect()` and `data` is in SBUF, `dst`'s `index`
  starts on the same partition as `data`;
- if `dst` uses `.indirect()` and `data` is in PSUM and uses
  `.indirect()`, `dst`'s `index` starts on the same partition as
  `data`'s `index`.

In addition, on the Scalar engine a scattered `dst` cannot be in PSUM.

> **Note:**
> `sin`, `arctan`, `log`, `sqrt`, `rsqrt`, and `reciprocal`
> have limited valid input ranges. See [Supported Activation Functions for NKI ISA](nki.api.shared.md#nki-act-func) for their
> ranges and out-of-range behavior.

- **dst** — the activation output tile. Supported buffers: SBUF, PSUM.
- **op** — an activation function (see [Supported Activation Functions for NKI ISA](nki.api.shared.md#nki-act-func) for supported functions).
- **data** — the input tile; layout: (partition axis <= 128, free axis). Supported buffers: SBUF, PSUM.
- **imm0** — scalar or `[N, 1]` vector value for the first tensor-scalar operation.
  `N` must match the partition dimension size of `data`.
- **imm1** — scalar or `[N, 1]` vector value for the second tensor-scalar operation.
  `N` must match the partition dimension size of `data`.
- **op0** — first ALU operation in tensor-scalar pipeline. Must be an arithmetic operator
  (e.g., `nl.multiply`, `nl.add`, `nl.subtract`) or `nl.bypass` for no operation.
- **op1** — second ALU operation in tensor-scalar pipeline. Must be an arithmetic operator
  (e.g., `nl.add`, `nl.subtract`) or `nl.bypass` for no operation.
- **relu_param** — scalar or vector parameter for parameterized activation functions (e.g., PReLU).
  Defaults to `0.0`.
- **reverse0** — reverse operand order for `op0`. When `True`, computes
  `imm0 <op0> data` instead of `data <op0> imm0`. Requires `op0` to be set.
- **reverse1** — reverse operand order for `op1`. When `True`, computes
  `imm1 <op1> result` instead of `result <op1> imm1`. Requires `op1` to be set.
- **reduce_op** — the reduce operation to perform on the free dimension of the activated data.
  Supported: `nl.add`, `nl.maximum`, `nl.minimum`, `nl.abs_max`, `nl.abs_min`.
- **reduce_res** — a tile of shape `(data.shape[0], 1)` to hold the final state of the
  reduction registers. Supported buffers: SBUF, PSUM.

  Pass `None` to keep the reduction result in the Scalar Engine's internal
  accumulator without writing it out. This is useful when chaining multiple
  calls that reduce into the same accumulator — only the final call needs to
  pass a tile to retrieve the accumulated result.

- **reduce_cmd** — an enum member from `nisa.reduce_cmd` to control the state of the
  reduction registers.

**Accumulator behavior:**

The Scalar Engine maintains internal accumulator registers (one FP32 value per lane, 128 total)
that can be controlled via the `reduce_cmd` parameter:

- `reduce_cmd.reset_reduce`: Reset accumulators to the identity value for `reduce_op`, then
  reduce the current activation results into the accumulators.
- `reduce_cmd.reduce`: Continue accumulating on top of existing accumulator values.
- `reduce_cmd.reset`: Reset accumulators only, without reducing current elements.
- `reduce_cmd.idle`: (default) Do not modify accumulator state.

When `reduce_res` is provided, an additional instruction is emitted to read the accumulator
values into the output tile.

> **Note:**
> The accumulator registers are shared across Scalar Engine accumulation instructions including
> nki.isa.activation and `nki.isa.activate2`.

**Example**

```python
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest

@nki.jit
def activate2_scale_bias_kernel(data_tensor):
    '''Apply scale-and-bias followed by GELU activation using activate2.'''
    out = nl.ndarray(data_tensor.shape, dtype=nl.float32, buffer=nl.shared_hbm)

    # Load input from HBM to SBUF
    x = nl.ndarray(data_tensor.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=data_tensor)

    # activate2: multiply by 2.0, add 0.5, then apply GELU
    result = nl.ndarray(data_tensor.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.activate2(
        dst=result,
        op=nl.gelu,
        data=x,
        imm0=2.0,
        imm1=0.5,
        op0=nl.multiply,
        op1=nl.add,
    )

    nisa.dma_copy(dst=out, src=result)
    return out

```

**Behavior**

```python
for i in range(num_elements_per_partition):
    # Stage 1: tensor-scalar operations
    val = data[i]
    if op0 is not bypass:
        val = op0(val, imm0)       # or op0(imm0, val) if reverse0
    if op1 is not bypass:
        val = op1(val, imm1)       # or op1(imm1, val) if reverse1

    # Stage 2: activation function
    dst[i] = op(val, relu_param=relu_param)

    # Stage 3: optional reduction
    if reduce_cmd in (reset_reduce, reduce):
        accumulator = reduce_op(accumulator, dst[i])
```

---

### nki.isa.activation_reduce {#nki-isa-activation_reduce}

`nki.isa.activation_reduce(dst, op, data, reduce_op, reduce_res, bias, scale, name)`

**Engine:** Scalar Engine

**Signature:**

```python
isa.activation_reduce(dst, op, data, reduce_op, reduce_res, bias=None, scale=1.0, name=None)
```

Perform the same computation as `nisa.activation` and also a reduction along the free dimension of the
`nisa.activation` result using Scalar Engine. The results for the reduction is stored
in the reduce_res.

This API is equivalent to calling `nisa.activation` with
`reduce_cmd=nisa.reduce_cmd.reset_reduce` and passing in reduce_res. This API is kept for
backward compatibility, we recommend using `nisa.activation` moving forward.

Refer to nisa.activation for semantics of `op/data/bias/scale`.

In addition to nisa.activation computation, this API also performs a reduction
along the free dimension(s) of the nisa.activation result, at a small additional
performance cost. The reduction result is written into `reduce_res`, which must be a
SBUF/PSUM tile with the same partition axis size as the input tile `data` and one element per partition.
On NeuronCore-v2, the `reduce_op` must be `nl.add`.

There are 128 registers on the scalar engine for storing reduction results, corresponding
to the 128 partitions of the input. These registers are shared between `activation` and `activation_accu` calls.
This instruction first resets those
registers to zero, performs the reduction on the value after activation function is applied,
stores the results into the registers,
then reads out the reduction results from the register, eventually store them into `reduce_res`.

Note that `nisa.activation` can also change the state of the register. It's user's
responsibility to ensure correct ordering. It's the best practice to not mixing
the use of `activation_reduce` and `activation`.

Reduction axis is not configurable in this API. If the input tile has multiple free axis, the API will
reduce across all of them.

Mathematically, this API performs the following computation:

```python
output = op(data * scale + bias)
reduce_res = reduce_op(output, axis=<FreeAxis>)

```

**Tensor indirection.**

On NeuronCore-v4 and later, `dst` and `data` support tensor indirection
(gather/scatter) by passing a view created with `.indirect(index)`. `bias`,
`scale`, and `reduce_res` do **not** support tensor indirection. Runs on the
Scalar engine.

When operands are manually allocated, their base partitions must satisfy:

- the `index` of every `.indirect()` view starts on a quadrant boundary
  (a multiple of 32);
- if `data` uses `.indirect()`, `data` starts on the same partition as
  its `index`;
- if `dst` uses `.indirect()` and `data` is in SBUF, `dst`'s `index`
  starts on the same partition as `data`;
- if `dst` uses `.indirect()` and `data` is in PSUM and uses
  `.indirect()`, `dst`'s `index` starts on the same partition as
  `data`'s `index`.

In addition, on the Scalar engine a scattered `dst` cannot be in PSUM.

> **Note:**
> `sin`, `arctan`, `log`, `sqrt`, `rsqrt`, and `reciprocal`
> have limited valid input ranges. See [Supported Activation Functions for NKI ISA](nki.api.shared.md#nki-act-func) for their
> ranges and out-of-range behavior.

- **dst** — output tile of the activation instruction; layout: same as input `data` tile
- **op** — an activation function (see [Supported Activation Functions for NKI ISA](nki.api.shared.md#nki-act-func) for supported functions).
- **data** — the input tile; layout: (partition axis <= 128, free axis)
- **reduce_op** — the reduce operation to perform on the free dimension of the activation result
- **reduce_res** — a tile of shape `(data.shape[0], 1)`, where data.shape[0]
  is the partition axis size of the input `data` tile. The result of `sum(ReductionResult)`
  is written into the tensor.

                Pass `None` to keep the reduction result in the Scalar Engine's internal
                accumulator without writing it out. This is useful when chaining multiple
                calls that reduce into the same accumulator — only the final call needs to
                pass a tile to retrieve the accumulated result.

- **bias** — a vector with the same partition axis size as `data`
  for broadcast add (after broadcast multiply with `scale`)
- **scale** — a scalar or a vector with the same partition axis size as `data`
  for broadcast multiply

---

### nki.isa.dropout {#nki-isa-dropout}

# nki.isa.dropout

nki.isa.dropout

nki.isa.dropout(_dst_, _data_, _prob_, _name=None_)[[source]](../../../\_modules/nki/isa.html#dropout)
Randomly replace some elements of the input tile `data` with zeros
based on input probabilities using Vector Engine.
The probability of replacing input elements with zeros (i.e., drop probability)
is specified using the `prob` field:

- If the probability is 1.0, all elements are replaced with zeros.
- If the probability is 0.0, all elements are kept with their original values.

The `prob` field can be a scalar constant or a tile of shape `(data.shape[0], 1)`,
where each partition contains one drop probability value.
The drop probability value in each partition is applicable to the input
`data` elements from the same partition only.

Data type of the input `data` tile can be any valid NKI data types
(see [Supported Data Types](nki.api.shared.md#nki-dtype) for more information).
However, data type of `prob` has restrictions based on the data type of `data`:

- If data type of `data` is any of the integer types (e.g., int32, int16),
  `prob` data type must be float32

- If data type of data is any of the float types (e.g., float32, bfloat16),
  `prob` data can be any valid float type

The output data type `dst.dtype` must match the input data type `data.dtype`.

Parameters:

- **dst** – an output tile of the dropout result

- **data** – the input tile

- **prob** – a scalar or a tile of shape `(data.shape[0], 1)` to indicate the
  probability of replacing elements with zeros

---

### nki.isa.reciprocal {#nki-isa-reciprocal}

# nki.isa.reciprocal

nki.isa.reciprocal

nki.isa.reciprocal(_dst_, _data_, _name=None_)[[source]](../../../\_modules/nki/isa.html#reciprocal)
Compute element-wise reciprocal (1.0/x) of the input `data` tile using Vector Engine.

**Memory types.**

Both the input `data` and output `dst` tiles can be in SBUF or PSUM.

**Data types.**

The input `data` tile can be any valid NKI data type (see [Supported Data Types](nki.api.shared.md#nki-dtype) for more information).
The Vector Engine automatically casts the input data type to float32 and performs the reciprocal
computation in float32 math. The float32 results are cast to the data type of `dst`.

**Layout.**

The partition dimension of the input `data` is considered the parallel compute dimension.

**Tile size.**

The partition dimension size of input `data` and output `dst` tiles must be the same
and must not exceed 128. The number of elements per partition of `dst` must match
that of `data` and must not exceed the physical size of each SBUF partition.

Parameters:

- **dst** – the output tile

- **data** – the input tile

---
