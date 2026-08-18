# API Symbol Lookup Index

Quick reference for finding NKI API function and symbol documentation. Symbols are organized alphabetically within their respective modules.

---

## Quick Module Reference

| Module | Description | Documentation |
|--------|-------------|---------------|
| `nki` | Top-level NKI module | [nki](../programming/api/nki.md) |
| `nki.language` | High-level language APIs | [nki.language](../programming/api/nki.language.md) |
| `nki.isa` | Low-level ISA instructions | [nki.isa](../programming/api/nki.isa.md) |
| `nki.collectives` | Collective communication (all_gather, all_reduce, ...) | [nki.collectives](../programming/api/api-nki-collectives.md) |
| `nki.api.shared` | Shared data types and operators | [nki.api.shared](../programming/api/nki.api.shared.md) |

---

## A

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `abs` | nki.language | Op specifier for abs. | [nki.language.abs](../programming/api/api-nki-language-operators.md#nki-language-abs) |
| `abs_max` | nki.language | Element-wise absolute maximum (trn3 only) | [nki.api.shared](../programming/api/nki.api.shared.md) |
| `abs_min` | nki.language | Element-wise absolute minimum (trn3 only) | [nki.api.shared](../programming/api/nki.api.shared.md) |
| `activate2` | nki.isa | Two-stage tensor-scalar + activation in one instruction (trn3 only) | [nki.isa.activate2](../programming/api/api-nki-isa-scalar.md#nki-isa-activation) |
| `activation` | nki.isa | Apply activation function with optional scale/bias | [nki.isa.activation](../programming/api/api-nki-isa-scalar.md#nki-isa-activation) |
| `activation_reduce` | nki.isa | Activation with free-dimension reduction | [nki.isa.activation_reduce](../programming/api/api-nki-isa-scalar.md#nki-isa-activation_reduce) |
| `add` | nki.language | Op specifier for add. | [nki.language.add](../programming/api/api-nki-language-operators.md#nki-language-add) |
| `affine_range` | nki.language | Loop iterator (legacy alias for `range`) | [nki.language.affine_range](../programming/api/nki.language.md) |
| `affine_select` | nki.isa | Select elements using affine predicate | [nki.isa.affine_select](../programming/api/api-nki-isa-utility.md#nki-isa-affine_select) |
| `all` | nki.language | Whether all elements along the specified axis (or axes) evaluate to True. | [nki.language.all](../programming/api/api-nki-language-misc.md#nki-language-all) |
| `all_gather` | nki.collectives | Perform an all-gather on the given replica group and input/output tensors. | [nki.collectives.all_gather](../programming/api/api-nki-collectives.md#nki-collectives-all_gather) |
| `all_gather_v` | nki.collectives | Perform a variable-length all-gather on the given replica group. | [nki.collectives.all_gather_v](../programming/api/api-nki-collectives.md#nki-collectives-all_gather_v) |
| `all_reduce` | nki.collectives | Perform an all-reduce on the given replica group and input/output tensors. | [nki.collectives.all_reduce](../programming/api/api-nki-collectives.md#nki-collectives-all_reduce) |
| `all_to_all` | nki.collectives | Perform an all-to-all on the given replica group and input/output tensors. | [nki.collectives.all_to_all](../programming/api/api-nki-collectives.md#nki-collectives-all_to_all) |
| `all_to_all_v` | nki.collectives | Executes an all-to-all collective where each rank can send | [nki.collectives.all_to_all_v](../programming/api/api-nki-collectives.md#nki-collectives-all_to_all_v) |
| `ap` | nki.tensor | Low-level access pattern override (escape hatch). | [NkiTensor.ap](../programming/api/api-nki-tensor.md#nki-tensor-ap) |
| `arctan` | nki.language | Op specifier for arctan. | [nki.language.arctan](../programming/api/api-nki-language-operators.md#nki-language-arctan) |
| `average` | nki.language | Op specifier for average. | [nki.language.average](../programming/api/api-nki-language-operators.md#nki-language-average) |

---

## B

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `bfloat16` | nki.language | BF16 data type (1S,8E,7M) | [nki.language.bfloat16](../programming/api/api-nki-language-types.md#nki-language-bfloat16) |
| `bitwise_and` | nki.language | Op specifier for bitwise_and. | [nki.language.bitwise_and](../programming/api/api-nki-language-operators.md#nki-language-bitwise_and) |
| `bitwise_or` | nki.language | Op specifier for bitwise_or. | [nki.language.bitwise_or](../programming/api/api-nki-language-operators.md#nki-language-bitwise_or) |
| `bitwise_xor` | nki.language | Op specifier for bitwise_xor. | [nki.language.bitwise_xor](../programming/api/api-nki-language-operators.md#nki-language-bitwise_xor) |
| `bn_aggr` | nki.isa | Aggregate batch norm statistics | [nki.isa.bn_aggr](../programming/api/api-nki-isa-vector.md#nki-isa-bn_aggr) |
| `bn_stats` | nki.isa | Compute batch norm statistics | [nki.isa.bn_stats](../programming/api/api-nki-isa-vector.md#nki-isa-bn_stats) |
| `bool_` | nki.language | Boolean data type | [nki.language.bool_](../programming/api/api-nki-language-types.md#nki-language-bool_) |
| `broadcast` | nki.tensor | Expand a size-1 dimension to `size` by repeating elements. | [NkiTensor.broadcast](../programming/api/api-nki-tensor.md#nki-tensor-broadcast) |
| `broadcast_to` | nki.language | Broadcast a tile to a new shape following numpy broadcasting rules. | [nki.language.broadcast_to](../programming/api/api-nki-language-misc.md#nki-language-broadcast_to) |
| `bypass` | nki.language | Op specifier for bypass. | [nki.language.bypass](../programming/api/api-nki-language-operators.md#nki-language-bypass) |

---

## C

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `ceil` | nki.language | Op specifier for ceil. | [nki.language.ceil](../programming/api/api-nki-language-operators.md#nki-language-ceil) |
| `collective_permute` | nki.collectives | Send and receive data between ranks based on explicitly defined source-target pa | [nki.collectives.collective_permute](../programming/api/api-nki-collectives.md#nki-collectives-collective_permute) |
| `collective_permute_implicit` | nki.collectives | Send and receive data between ranks in a ring, where sources and destinations ar | [nki.collectives.collective_permute_implicit](../programming/api/api-nki-collectives.md#nki-collectives-collective_permute_implicit) |
| `collective_permute_implicit_current_processing_rank_id` | nki.collectives | Returns the rank ID of the data to be processed in the current ring iteration. | [nki.collectives.collective_permute_implicit_current_processing_rank_id](../programming/api/api-nki-collectives.md#nki-collectives-collective_permute_implicit_current_processing_rank_id) |
| `collective_permute_implicit_reduce` | nki.collectives | Perform an implicit collective permute with reduction in a ring, where sources a | [nki.collectives.collective_permute_implicit_reduce](../programming/api/api-nki-collectives.md#nki-collectives-collective_permute_implicit_reduce) |
| `copy` | nki.language | Op specifier for copy. | [nki.language.copy](../programming/api/api-nki-language-operators.md#nki-language-copy) |
| `core_barrier` | nki.isa | Synchronize across NeuronCores | [nki.isa.core_barrier](../programming/api/api-nki-isa-tensor.md#nki-isa-core_barrier) |
| `cos` | nki.language | Op specifier for cos. | [nki.language.cos](../programming/api/api-nki-language-operators.md#nki-language-cos) |

---

## D

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `device_print` | nki.language | Print debug output from kernel | [nki.language.device_print](../programming/api/nki.language.md) |
| `dge_mode` | nki.isa | DMA Descriptor Generation Engine mode enum | [nki.isa.dge_mode](../programming/api/nki.isa.md) |
| `divide` | nki.language | Op specifier for divide. | [nki.language.divide](../programming/api/api-nki-language-operators.md#nki-language-divide) |
| `dma_compute` | nki.isa | Math operations using DMA engines (replaces dma_copy RMW) | [nki.isa.dma_compute](../programming/api/api-nki-isa-memory.md#nki-isa-dma_compute) |
| `dma_copy` | nki.isa | Copy data using DMA engines | [nki.isa.dma_copy](../programming/api/api-nki-isa-memory.md#nki-isa-dma_copy) |
| `dma_engine` | nki.isa | DMA engine enum (dma, gpsimd_dma) | [nki.isa.dma_engine](../programming/api/nki.isa.md) |
| `dma_transpose` | nki.isa | Transpose using DMA engines | [nki.isa.dma_transpose](../programming/api/api-nki-isa-memory.md#nki-isa-dma_transpose) |
| `dropout` | nki.isa | Apply dropout to tensor | [nki.isa.dropout](../programming/api/api-nki-isa-scalar.md#nki-isa-dropout) |
| `dropout` | nki.language | Randomly zeroes some of the elements of the input tile given a probability rate. | [nki.language.dropout](../programming/api/api-nki-language-misc.md#nki-language-dropout) |
| `ds` | nki.language | Dynamic slice for tensor indexing | [nki.language.ds](../programming/api/nki.language.md) |
| `dynamic_range` | nki.language | Create a sequence for **dynamic** loop iteration. | [nki.language.dynamic_range](../programming/api/api-nki-language-dims.md#nki-language-dynamic_range) |

---

## E

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `empty_like` | nki.language | Create a new tensor with the same shape and type as a given tensor. | [nki.language.empty_like](../programming/api/api-nki-language-creation.md#nki-language-empty_like) |
| `engine` | nki.isa | Neuron Device engine enum | [nki.isa.engine](../programming/api/nki.isa.md) |
| `equal` | nki.language | Op specifier for equal. | [nki.language.equal](../programming/api/api-nki-language-operators.md#nki-language-equal) |
| `erf` | nki.language | Op specifier for erf. | [nki.language.erf](../programming/api/api-nki-language-operators.md#nki-language-erf) |
| `erf_dx` | nki.language | Op specifier for erf_dx. | [nki.language.erf_dx](../programming/api/api-nki-language-operators.md#nki-language-erf_dx) |
| `exp` | nki.language | Op specifier for exp. | [nki.language.exp](../programming/api/api-nki-language-operators.md#nki-language-exp) |
| `expand_dim` | nki.tensor | Insert a new dimension of size 1 at position `dim`. | [NkiTensor.expand_dim](../programming/api/api-nki-tensor.md#nki-tensor-expand_dim) |
| `expand_dims` | nki.language | Expand the shape of a tile. | [nki.language.expand_dims](../programming/api/api-nki-language-misc.md#nki-language-expand_dims) |
| `exponential` | nki.isa | Dedicated exponential instruction (Trn3/NeuronCore-v4 only) | [nki.isa.exponential](../programming/api/nki.isa.md) |

---

## F

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `flatten_dims` | nki.tensor | Merge a contiguous range of dimensions into one. | [NkiTensor.flatten_dims](../programming/api/api-nki-tensor.md#nki-tensor-flatten_dims) |
| `float16` | nki.language | FP16 data type | [nki.language.float16](../programming/api/api-nki-language-types.md#nki-language-float16) |
| `float32` | nki.language | FP32 data type | [nki.language.float32](../programming/api/api-nki-language-types.md#nki-language-float32) |
| `float4_e2m1fn_x4` | nki.language | 4x packed float4 for MXFP matmul | [nki.language.float4_e2m1fn_x4](../programming/api/api-nki-language-types.md#nki-language-float4_e2m1fn_x4) |
| `float8_e4m3` | nki.language | FP8 E4M3 data type | [nki.language.float8_e4m3](../programming/api/api-nki-language-types.md#nki-language-float8_e4m3) |
| `float8_e4m3fn` | nki.language | Data type constant `float8_e4m3fn` for tensor element types. | [nki.language.float8_e4m3fn](../programming/api/api-nki-language-types.md#nki-language-float8_e4m3fn) |
| `float8_e4m3fn_x4` | nki.language | 4x packed FP8 E4M3 for MXFP matmul | [nki.language.float8_e4m3fn_x4](../programming/api/api-nki-language-types.md#nki-language-float8_e4m3fn_x4) |
| `float8_e5m2` | nki.language | FP8 E5M2 data type | [nki.language.float8_e5m2](../programming/api/api-nki-language-types.md#nki-language-float8_e5m2) |
| `float8_e5m2_x4` | nki.language | 4x packed FP8 E5M2 for MXFP matmul | [nki.language.float8_e5m2_x4](../programming/api/api-nki-language-types.md#nki-language-float8_e5m2_x4) |
| `floor` | nki.language | Op specifier for floor. | [nki.language.floor](../programming/api/api-nki-language-operators.md#nki-language-floor) |
| `fmod` | nki.language | Op specifier for fmod. | [nki.language.fmod](../programming/api/api-nki-language-operators.md#nki-language-fmod) |
| `fori_loop` | nki.language | Structured for loop with dynamic bounds. | [nki.language.fori_loop](../programming/api/api-nki-language-dims.md#nki-language-fori_loop) |

---

## G

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `gather_flattened` | nki.language | Gather elements from data tensor using indices after flattening. | [nki.language.gather_flattened](../programming/api/api-nki-language-creation.md#nki-language-gather_flattened) |
| `gelu` | nki.language | Op specifier for gelu. | [nki.language.gelu](../programming/api/api-nki-language-operators.md#nki-language-gelu) |
| `gelu_apprx_sigmoid` | nki.language | Op specifier for gelu_apprx_sigmoid. | [nki.language.gelu_apprx_sigmoid](../programming/api/api-nki-language-operators.md#nki-language-gelu_apprx_sigmoid) |
| `gelu_apprx_sigmoid_dx` | nki.language | Op specifier for gelu_apprx_sigmoid_dx. | [nki.language.gelu_apprx_sigmoid_dx](../programming/api/api-nki-language-operators.md#nki-language-gelu_apprx_sigmoid_dx) |
| `gelu_apprx_tanh` | nki.language | Op specifier for gelu_apprx_tanh. | [nki.language.gelu_apprx_tanh](../programming/api/api-nki-language-operators.md#nki-language-gelu_apprx_tanh) |
| `gelu_dx` | nki.language | Op specifier for gelu_dx. | [nki.language.gelu_dx](../programming/api/api-nki-language-operators.md#nki-language-gelu_dx) |
| `get_nc_version` | nki.isa | Get NeuronCore version | [nki.isa.get_nc_version](../programming/api/api-nki-isa-tensor.md#nki-isa-get_nc_version) |
| `get_pattern` | nki.tensor | Return the view's access pattern as `[[stride, count], . | [NkiTensor.get_pattern](../programming/api/api-nki-tensor.md#nki-tensor-get_pattern) |
| `greater` | nki.language | Op specifier for greater. | [nki.language.greater](../programming/api/api-nki-language-operators.md#nki-language-greater) |
| `greater_equal` | nki.language | Op specifier for greater_equal. | [nki.language.greater_equal](../programming/api/api-nki-language-operators.md#nki-language-greater_equal) |

---

## H

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `hbm` | nki.language | HBM memory buffer (alias of private_hbm) | [nki.language.hbm](../programming/api/api-nki-language-memory.md#nki-language-hbm) |

---

## I

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `indirect` | nki.tensor | Create an indirect tensor view for Tensor Indirection (TI). | [NkiTensor.indirect](../programming/api/api-nki-tensor.md#nki-tensor-indirect) |
| `int16` | nki.language | 16-bit signed integer | [nki.language.int16](../programming/api/api-nki-language-types.md#nki-language-int16) |
| `int32` | nki.language | 32-bit signed integer | [nki.language.int32](../programming/api/api-nki-language-types.md#nki-language-int32) |
| `int8` | nki.language | 8-bit signed integer | [nki.language.int8](../programming/api/api-nki-language-types.md#nki-language-int8) |
| `invert` | nki.language | Op specifier for invert. | [nki.language.invert](../programming/api/api-nki-language-operators.md#nki-language-invert) |
| `iota` | nki.isa | Generate constant literal pattern | [nki.isa.iota](../programming/api/api-nki-isa-utility.md#nki-isa-iota) |
| `is_contiguous` | nki.tensor | Return True if the view covers storage contiguously (row-major order). | [NkiTensor.is_contiguous](../programming/api/api-nki-tensor.md#nki-tensor-is_contiguous) |
| `is_indirect` | nki.tensor | Return True if this view already uses indirect addressing. | [NkiTensor.is_indirect](../programming/api/api-nki-tensor.md#nki-tensor-is_indirect) |

---

## L

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `left_shift` | nki.language | Op specifier for left_shift. | [nki.language.left_shift](../programming/api/api-nki-language-operators.md#nki-language-left_shift) |
| `less` | nki.language | Op specifier for less. | [nki.language.less](../programming/api/api-nki-language-operators.md#nki-language-less) |
| `less_equal` | nki.language | Op specifier for less_equal. | [nki.language.less_equal](../programming/api/api-nki-language-operators.md#nki-language-less_equal) |
| `load` | nki.language | Load a tensor from device memory (HBM) into on-chip memory (SBUF). | [nki.language.load](../programming/api/api-nki-language-creation.md#nki-language-load) |
| `load_transpose2d` | nki.language | Load a tensor from device memory (HBM) and 2D-transpose the data before storing | [nki.language.load_transpose2d](../programming/api/api-nki-language-creation.md#nki-language-load_transpose2d) |
| `local_gather` | nki.isa | Gather SBUF data using indices | [nki.isa.local_gather](../programming/api/api-nki-isa-memory.md#nki-isa-local_gather) |
| `log` | nki.language | Op specifier for log. | [nki.language.log](../programming/api/api-nki-language-operators.md#nki-language-log) |
| `logical_and` | nki.language | Op specifier for logical_and. | [nki.language.logical_and](../programming/api/api-nki-language-operators.md#nki-language-logical_and) |
| `logical_not` | nki.language | Op specifier for logical_not. | [nki.language.logical_not](../programming/api/api-nki-language-operators.md#nki-language-logical_not) |
| `logical_or` | nki.language | Op specifier for logical_or. | [nki.language.logical_or](../programming/api/api-nki-language-operators.md#nki-language-logical_or) |
| `logical_xor` | nki.language | Op specifier for logical_xor. | [nki.language.logical_xor](../programming/api/api-nki-language-operators.md#nki-language-logical_xor) |

---

## M

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `matmul` | nki.language | x @ y matrix multiplication of x and y. | [nki.language.matmul](../programming/api/api-nki-language-misc.md#nki-language-matmul) |
| `max` | nki.language | Maximum of elements along the specified axis (or axes) of the input. | [nki.language.max](../programming/api/api-nki-language-misc.md#nki-language-max) |
| `max8` | nki.isa | Find 8 largest values per partition | [nki.isa.max8](../programming/api/api-nki-isa-utility.md#nki-isa-max8) |
| `maximum` | nki.language | Op specifier for maximum. | [nki.language.maximum](../programming/api/api-nki-language-operators.md#nki-language-maximum) |
| `mean` | nki.language | Arithmetic mean along the specified axis (or axes) of the input. | [nki.language.mean](../programming/api/api-nki-language-misc.md#nki-language-mean) |
| `memset` | nki.isa | Initialize tensor with constant value | [nki.isa.memset](../programming/api/api-nki-isa-memory.md#nki-isa-memset) |
| `min` | nki.language | Minimum of elements along the specified axis (or axes) of the input. | [nki.language.min](../programming/api/api-nki-language-misc.md#nki-language-min) |
| `minimum` | nki.language | Op specifier for minimum. | [nki.language.minimum](../programming/api/api-nki-language-operators.md#nki-language-minimum) |
| `mish` | nki.language | Op specifier for mish. | [nki.language.mish](../programming/api/api-nki-language-operators.md#nki-language-mish) |
| `mod` | nki.language | Op specifier for mod. | [nki.language.mod](../programming/api/api-nki-language-operators.md#nki-language-mod) |
| `multiply` | nki.language | Op specifier for multiply. | [nki.language.multiply](../programming/api/api-nki-language-operators.md#nki-language-multiply) |

---

## N

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `nc_find_index8` | nki.isa | Find indices of 8 values in data | [nki.isa.nc_find_index8](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_find_index8) |
| `nc_match_replace8` | nki.isa | Replace values and optionally return indices | [nki.isa.nc_match_replace8](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_match_replace8) |
| `nc_matmul` | nki.isa | Matrix multiplication on Tensor Engine | [nki.isa.nc_matmul](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_matmul) |
| `nc_matmul_mx` | nki.isa | MXFP quantized matrix multiplication | [nki.isa.nc_matmul_mx](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_matmul_mx) |
| `nc_n_gather` | nki.isa | Gather elements using indices | [nki.isa.nc_n_gather](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_n_gather) |
| `nc_stream_shuffle` | nki.isa | Cross-partition data shuffle | [nki.isa.nc_stream_shuffle](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_stream_shuffle) |
| `nc_transpose` | nki.isa | 2D transpose between P and F axes | [nki.isa.nc_transpose](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_transpose) |
| `nc_version` | nki.isa | NeuronCore version enum | [nki.isa.nc_version](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_version) |
| `ndarray` | nki.language | Create tensor on specified buffer | [nki.language.ndarray](../programming/api/nki.language.md) |
| `negative` | nki.language | Op specifier for negative. | [nki.language.negative](../programming/api/api-nki-language-operators.md#nki-language-negative) |
| `nonzero_with_count` | nki.isa | Find indices of nonzero elements and count (NeuronCore-v3+) | [nki.isa.nonzero_with_count](../programming/api/api-nki-isa-misc.md#nki-isa-nonzero_with_count) |
| `not_equal` | nki.language | Op specifier for not_equal. | [nki.language.not_equal](../programming/api/api-nki-language-operators.md#nki-language-not_equal) |
| `num_programs` | nki.language | Number of SPMD programs in grid | [nki.language.num_programs](../programming/api/nki.language.md) |

---

## O

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `ones` | nki.language | Create a new tensor of given shape and dtype on the specified buffer, filled wit | [nki.language.ones](../programming/api/api-nki-language-creation.md#nki-language-ones) |

---

## P

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `permute` | nki.tensor | Reorder tensor dimensions. | [NkiTensor.permute](../programming/api/api-nki-tensor.md#nki-tensor-permute) |
| `power` | nki.language | Op specifier for power. | [nki.language.power](../programming/api/api-nki-language-operators.md#nki-language-power) |
| `prelu` | nki.language | Op specifier for prelu. | [nki.language.prelu](../programming/api/api-nki-language-operators.md#nki-language-prelu) |
| `private_hbm` | nki.language | Private HBM memory buffer | [nki.language.private_hbm](../programming/api/api-nki-language-memory.md#nki-language-private_hbm) |
| `prod` | nki.language | Product of elements along the specified axis (or axes) of the input. | [nki.language.prod](../programming/api/api-nki-language-misc.md#nki-language-prod) |
| `program_id` | nki.language | Index of current SPMD program | [nki.language.program_id](../programming/api/nki.language.md) |
| `program_ndim` | nki.language | Number of dimensions in SPMD grid | [nki.language.program_ndim](../programming/api/nki.language.md) |
| `psum` | nki.language | PSUM memory buffer | [nki.language.psum](../programming/api/api-nki-language-memory.md#nki-language-psum) |

---

## Q

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `quantize_mx` | nki.isa | Quantize to MXFP8 format | [nki.isa.quantize_mx](../programming/api/api-nki-isa-tensor.md#nki-isa-quantize_mx) |

---

## R

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `rand` | nki.language | Create a new tensor of given shape and dtype on the specified buffer, filled wit | [nki.language.rand](../programming/api/api-nki-language-creation.md#nki-language-rand) |
| `rand2` | nki.isa | Generate uniform random numbers | [nki.isa.rand2](../programming/api/nki.isa.md) |
| `rand_get_state` | nki.isa | Get PRNG state from engine | [nki.isa.rand_get_state](../programming/api/nki.isa.md) |
| `rand_set_state` | nki.isa | Set PRNG state in engine | [nki.isa.rand_set_state](../programming/api/nki.isa.md) |
| `range_select` | nki.isa | Select elements based on range comparison | [nki.isa.range_select](../programming/api/api-nki-isa-utility.md#nki-isa-range_select) |
| `rank_id` | nki.collectives | Get the rank ID of the current rank. | [nki.collectives.rank_id](../programming/api/api-nki-collectives.md#nki-collectives-rank_id) |
| `rearrange` | nki.tensor | Rearrange tensor dimensions using einops-style patterns. | [NkiTensor.rearrange](../programming/api/api-nki-tensor.md#nki-tensor-rearrange) |
| `reciprocal` | nki.isa | Compute element-wise 1/x | [nki.isa.reciprocal](../programming/api/api-nki-isa-scalar.md#nki-isa-reciprocal) |
| `reciprocal` | nki.language | Op specifier for reciprocal. | [nki.language.reciprocal](../programming/api/api-nki-language-operators.md#nki-language-reciprocal) |
| `reduce_cmd` | nki.isa | Engine register reduce commands enum | [nki.isa.reduce_cmd](../programming/api/nki.isa.md) |
| `reduce_scatter` | nki.collectives | Perform a reduce-scatter on the given replica group and input/output tensors. | [nki.collectives.reduce_scatter](../programming/api/api-nki-collectives.md#nki-collectives-reduce_scatter) |
| `register_alloc` | nki.isa | Allocate virtual register | [nki.isa.register_alloc](../programming/api/nki.isa.md) |
| `register_load` | nki.isa | Load scalar from memory to register | [nki.isa.register_load](../programming/api/nki.isa.md) |
| `register_move` | nki.isa | Move value from source register to destination register | [nki.isa.register_move](../programming/api/nki.isa.md) |
| `register_store` | nki.isa | Store register value to memory | [nki.isa.register_store](../programming/api/nki.isa.md) |
| `relu` | nki.language | Op specifier for relu. | [nki.language.relu](../programming/api/api-nki-language-operators.md#nki-language-relu) |
| `reshape` | nki.tensor | Reshape the tensor to a new shape without copying data. | [NkiTensor.reshape](../programming/api/api-nki-tensor.md#nki-tensor-reshape) |
| `reshape_dim` | nki.tensor | Split a single dimension into multiple dimensions. | [NkiTensor.reshape_dim](../programming/api/api-nki-tensor.md#nki-tensor-reshape_dim) |
| `right_shift` | nki.language | Op specifier for right_shift. | [nki.language.right_shift](../programming/api/api-nki-language-operators.md#nki-language-right_shift) |
| `rms_norm` | nki.language | Apply Root Mean Square Layer Normalization. | [nki.language.rms_norm](../programming/api/api-nki-language-misc.md#nki-language-rms_norm) |
| `rng` | nki.isa | Generate pseudo random numbers | [nki.isa.rng](../programming/api/nki.isa.md) |
| `rsqrt` | nki.language | Op specifier for rsqrt. | [nki.language.rsqrt](../programming/api/api-nki-language-operators.md#nki-language-rsqrt) |

---

## S

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `sbuf` | nki.language | State Buffer memory | [nki.language.sbuf](../programming/api/api-nki-language-memory.md#nki-language-sbuf) |
| `scalar_tensor_tensor` | nki.isa | Two-op sequence with scalar broadcast | [nki.isa.scalar_tensor_tensor](../programming/api/api-nki-isa-tensor.md#nki-isa-scalar_tensor_tensor) |
| `select` | nki.tensor | Select a single element along a dimension, removing it. | [NkiTensor.select](../programming/api/api-nki-tensor.md#nki-tensor-select) |
| `select_reduce` | nki.isa | Conditional copy with optional reduction | [nki.isa.select_reduce](../programming/api/api-nki-isa-utility.md#nki-isa-select_reduce) |
| `sendrecv` | nki.isa | Point-to-point NeuronCore communication | [nki.isa.sendrecv](../programming/api/nki.isa.md) |
| `sequence_bounds` | nki.isa | Compute sequence bounds from segment IDs | [nki.isa.sequence_bounds](../programming/api/api-nki-isa-utility.md#nki-isa-sequence_bounds) |
| `sequential_range` | nki.language | Loop iterator (legacy alias for `range`) | [nki.language.sequential_range](../programming/api/nki.language.md) |
| `set_rng_seed` | nki.isa | Seed Vector Engine PRNG | [nki.isa.set_rng_seed](../programming/api/nki.isa.md) |
| `shared_constant` | nki.language | Create a tensor in shared HBM initialized with constant data. | [nki.language.shared_constant](../programming/api/api-nki-language-creation.md#nki-language-shared_constant) |
| `shared_hbm` | nki.language | Shared HBM across kernel instances | [nki.language.shared_hbm](../programming/api/api-nki-language-memory.md#nki-language-shared_hbm) |
| `sigmoid` | nki.language | Op specifier for sigmoid. | [nki.language.sigmoid](../programming/api/api-nki-language-operators.md#nki-language-sigmoid) |
| `sign` | nki.language | Op specifier for sign. | [nki.language.sign](../programming/api/api-nki-language-operators.md#nki-language-sign) |
| `silu` | nki.language | Op specifier for silu. | [nki.language.silu](../programming/api/api-nki-language-operators.md#nki-language-silu) |
| `silu_dx` | nki.language | Op specifier for silu_dx. | [nki.language.silu_dx](../programming/api/api-nki-language-operators.md#nki-language-silu_dx) |
| `simulate` | nki | Run NKI kernel on CPU without NeuronDevice (experimental) | [nki.simulate](../programming/api/api-nki-tools.md#nki-simulate) |
| `sin` | nki.language | Op specifier for sin. | [nki.language.sin](../programming/api/api-nki-language-operators.md#nki-language-sin) |
| `slice` | nki.tensor | Slice along a single dimension. | [NkiTensor.slice](../programming/api/api-nki-tensor.md#nki-tensor-slice) |
| `softmax` | nki.language | Softmax activation function on the input, element-wise. | [nki.language.softmax](../programming/api/api-nki-language-misc.md#nki-language-softmax) |
| `softplus` | nki.language | Op specifier for softplus. | [nki.language.softplus](../programming/api/api-nki-language-operators.md#nki-language-softplus) |
| `sqrt` | nki.language | Op specifier for sqrt. | [nki.language.sqrt](../programming/api/api-nki-language-operators.md#nki-language-sqrt) |
| `square` | nki.language | Op specifier for square. | [nki.language.square](../programming/api/api-nki-language-operators.md#nki-language-square) |
| `squeeze_dim` | nki.tensor | Remove a dimension of size 1. | [NkiTensor.squeeze_dim](../programming/api/api-nki-tensor.md#nki-tensor-squeeze_dim) |
| `static_range` | nki.language | Loop iterator (legacy alias for `range`) | [nki.language.static_range](../programming/api/nki.language.md) |
| `store` | nki.language | Store into a tensor on device memory (HBM) from on-chip memory (SBUF). | [nki.language.store](../programming/api/api-nki-language-creation.md#nki-language-store) |
| `subtract` | nki.language | Op specifier for subtract. | [nki.language.subtract](../programming/api/api-nki-language-operators.md#nki-language-subtract) |
| `sum` | nki.language | Sum of elements along the specified axis (or axes) of the input. | [nki.language.sum](../programming/api/api-nki-language-misc.md#nki-language-sum) |

---

## T

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `tan` | nki.language | Op specifier for tan. | [nki.language.tan](../programming/api/api-nki-language-operators.md#nki-language-tan) |
| `tanh` | nki.language | Op specifier for tanh. | [nki.language.tanh](../programming/api/api-nki-language-operators.md#nki-language-tanh) |
| `tensor_copy` | nki.isa | Copy tensor within on-chip SRAM | [nki.isa.tensor_copy](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_copy) |
| `tensor_copy_predicated` | nki.isa | Conditional element copy | [nki.isa.tensor_copy_predicated](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_copy_predicated) |
| `tensor_partition_reduce` | nki.isa | Reduce across partitions | [nki.isa.tensor_partition_reduce](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_partition_reduce) |
| `tensor_reduce` | nki.isa | Reduce along free axes | [nki.isa.tensor_reduce](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_reduce) |
| `tensor_scalar` | nki.isa | Tensor-scalar operations with broadcasting | [nki.isa.tensor_scalar](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_scalar) |
| `tensor_scalar_cumulative` | nki.isa | Tensor-scalar with cumulative reduction | [nki.isa.tensor_scalar_cumulative](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_scalar_cumulative) |
| `tensor_scalar_reduce` | nki.isa | Tensor-scalar with free-dim reduction | [nki.isa.tensor_scalar_reduce](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_scalar_reduce) |
| `tensor_tensor` | nki.isa | Element-wise operation on two tensors | [nki.isa.tensor_tensor](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_tensor) |
| `tensor_tensor_scan` | nki.isa | Scan operation on two tensors | [nki.isa.tensor_tensor_scan](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_tensor_scan) |
| `tfloat32` | nki.language | TF32 data type (1S,8E,10M) | [nki.language.tfloat32](../programming/api/api-nki-language-types.md#nki-language-tfloat32) |
| `tile_size` | nki.language | Tile size constants | [nki.language.tile_size](../programming/api/nki.language.md) |
| `topk` | nki.isa | Find the K largest values and their indices from a source tile using GpSIMD Engi | [nki.isa.topk](../programming/api/api-nki-isa-tensor.md#nki-isa-topk) |
| `transpose` | nki.language | Transposes a 2D tile between its partition and free dimension. | [nki.language.transpose](../programming/api/api-nki-language-misc.md#nki-language-transpose) |
| `trunc` | nki.language | Op specifier for trunc. | [nki.language.trunc](../programming/api/api-nki-language-operators.md#nki-language-trunc) |

---

## U

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `uint16` | nki.language | 16-bit unsigned integer | [nki.language.uint16](../programming/api/api-nki-language-types.md#nki-language-uint16) |
| `uint32` | nki.language | 32-bit unsigned integer | [nki.language.uint32](../programming/api/api-nki-language-types.md#nki-language-uint32) |
| `uint8` | nki.language | 8-bit unsigned integer | [nki.language.uint8](../programming/api/api-nki-language-types.md#nki-language-uint8) |

---

## V

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `var` | nki.language | Variance along the specified axis (or axes) of the input. | [nki.language.var](../programming/api/api-nki-language-misc.md#nki-language-var) |
| `vector_select` | nki.tensor | Per-partition indirect addressing using a vector of offsets. | [NkiTensor.vector_select](../programming/api/api-nki-tensor.md#nki-tensor-vector_select) |
| `view` | nki.tensor | Reinterpret the tensor's data as a different dtype. | [NkiTensor.view](../programming/api/api-nki-tensor.md#nki-tensor-view) |

---

## W

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `where` | nki.language | Return elements chosen from x or y depending on condition. | [nki.language.where](../programming/api/api-nki-language-misc.md#nki-language-where) |
| `while_loop` | nki.language | Structured while loop with a register condition. | [nki.language.while_loop](../programming/api/api-nki-language-dims.md#nki-language-while_loop) |

---

## Z

| Symbol | Module | Description | Documentation |
|--------|--------|-------------|---------------|
| `zeros` | nki.language | Create zero-filled tensor | [nki.language.zeros](../programming/api/nki.language.md) |
| `zeros_like` | nki.language | Create a new tensor of zeros with the same shape and type as a given tensor. | [nki.language.zeros_like](../programming/api/api-nki-language-misc.md#nki-language-zeros_like) |

---

## Symbols by Category

### Tensor Creation
| Symbol | Documentation |
|--------|---------------|
| `nki.language.ndarray` | [Link](../programming/api/nki.language.md) |
| `nki.language.zeros` | [Link](../programming/api/nki.language.md) |

### Memory Buffers
| Symbol | Documentation |
|--------|---------------|
| `nki.language.sbuf` | [Link](../programming/api/api-nki-language-memory.md#nki-language-sbuf) |
| `nki.language.psum` | [Link](../programming/api/api-nki-language-memory.md#nki-language-psum) |
| `nki.language.hbm` | [Link](../programming/api/api-nki-language-memory.md#nki-language-hbm) |
| `nki.language.private_hbm` | [Link](../programming/api/api-nki-language-memory.md#nki-language-private_hbm) |
| `nki.language.shared_hbm` | [Link](../programming/api/api-nki-language-memory.md#nki-language-shared_hbm) |

### Loop Iterators
| Symbol | Documentation |
|--------|---------------|
| `range` (recommended) | Standard Python range |
| `nki.language.static_range` | [Link](../programming/api/nki.language.md) (legacy alias for `range`) |
| `nki.language.affine_range` | [Link](../programming/api/nki.language.md) (legacy alias for `range`) |
| `nki.language.sequential_range` | [Link](../programming/api/nki.language.md) (legacy alias for `range`) |

### Data Types
| Symbol | Documentation |
|--------|---------------|
| `nki.language.bool_` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.int8` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.int16` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.int32` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.uint8` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.uint16` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.uint32` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.float16` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.float32` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.bfloat16` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.tfloat32` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.float8_e4m3` | [Link](../programming/api/api-nki-language-types.md) |
| `nki.language.float8_e5m2` | [Link](../programming/api/api-nki-language-types.md) |

### Matrix Operations (Tensor Engine)
| Symbol | Documentation |
|--------|---------------|
| `nki.isa.nc_matmul` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_matmul) |
| `nki.isa.nc_matmul_mx` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_matmul_mx) |
| `nki.isa.nc_transpose` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_transpose) |

### Vector Operations (Vector Engine)
| Symbol | Documentation |
|--------|---------------|
| `nki.isa.tensor_tensor` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_tensor) |
| `nki.isa.tensor_tensor_scan` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_tensor_scan) |
| `nki.isa.tensor_scalar` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_scalar) |
| `nki.isa.tensor_reduce` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_reduce) |
| `nki.isa.bn_stats` | [Link](../programming/api/api-nki-isa-vector.md#nki-isa-bn_stats) |
| `nki.isa.bn_aggr` | [Link](../programming/api/api-nki-isa-vector.md#nki-isa-bn_aggr) |
| `nki.isa.reciprocal` | [Link](../programming/api/api-nki-isa-scalar.md#nki-isa-reciprocal) |

### Scalar Operations (Scalar Engine)
| Symbol | Documentation |
|--------|---------------|
| `nki.isa.activation` | [Link](../programming/api/api-nki-isa-scalar.md#nki-isa-activation) |
| `nki.isa.activation_reduce` | [Link](../programming/api/api-nki-isa-scalar.md#nki-isa-activation_reduce) |
| `nki.isa.dropout` | [Link](../programming/api/api-nki-isa-scalar.md#nki-isa-dropout) |

### DMA Operations
| Symbol | Documentation |
|--------|---------------|
| `nki.isa.dma_copy` | [Link](../programming/api/api-nki-isa-memory.md#nki-isa-dma_copy) |
| `nki.isa.dma_transpose` | [Link](../programming/api/api-nki-isa-memory.md#nki-isa-dma_transpose) |
| `nki.isa.dma_compute` | [Link](../programming/api/api-nki-isa-memory.md#nki-isa-dma_compute) |

### Copy Operations
| Symbol | Documentation |
|--------|---------------|
| `nki.isa.tensor_copy` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_copy) |
| `nki.isa.tensor_copy_predicated` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-tensor_copy_predicated) |

### Utility Functions
| Symbol | Documentation |
|--------|---------------|
| `nki.isa.iota` | [Link](../programming/api/api-nki-isa-utility.md#nki-isa-iota) |
| `nki.isa.memset` | [Link](../programming/api/api-nki-isa-memory.md#nki-isa-memset) |
| `nki.isa.affine_select` | [Link](../programming/api/api-nki-isa-utility.md#nki-isa-affine_select) |
| `nki.isa.range_select` | [Link](../programming/api/api-nki-isa-utility.md#nki-isa-range_select) |
| `nki.isa.select_reduce` | [Link](../programming/api/api-nki-isa-utility.md#nki-isa-select_reduce) |
| `nki.isa.max8` | [Link](../programming/api/api-nki-isa-utility.md#nki-isa-max8) |
| `nki.isa.sequence_bounds` | [Link](../programming/api/api-nki-isa-utility.md#nki-isa-sequence_bounds) |

### Gather/Scatter Operations
| Symbol | Documentation |
|--------|---------------|
| `nki.isa.local_gather` | [Link](../programming/api/api-nki-isa-memory.md#nki-isa-local_gather) |
| `nki.isa.nc_n_gather` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_n_gather) |
| `nki.isa.nc_find_index8` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_find_index8) |
| `nki.isa.nc_match_replace8` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_match_replace8) |

### Quantization
| Symbol | Documentation |
|--------|---------------|
| `nki.isa.quantize_mx` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-quantize_mx) |

### Random Number Generation
| Symbol | Documentation |
|--------|---------------|
| `nki.isa.rng` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.rand2` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.rand_set_state` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.rand_get_state` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.set_rng_seed` | [Link](../programming/api/nki.isa.md) |

### Multi-Core/SPMD
| Symbol | Documentation |
|--------|---------------|
| `nki.language.program_id` | [Link](../programming/api/nki.language.md) |
| `nki.language.num_programs` | [Link](../programming/api/nki.language.md) |
| `nki.language.program_ndim` | [Link](../programming/api/nki.language.md) |
| `nki.isa.core_barrier` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-core_barrier) |
| `nki.isa.sendrecv` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.nc_stream_shuffle` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_stream_shuffle) |

### Register Operations
| Symbol | Documentation |
|--------|---------------|
| `nki.isa.register_alloc` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.register_load` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.register_move` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.register_store` | [Link](../programming/api/nki.isa.md) |

### Enums and Constants
| Symbol | Documentation |
|--------|---------------|
| `nki.isa.engine` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.reduce_cmd` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.dge_mode` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.dma_engine` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.oob_mode` | [Link](../programming/api/nki.isa.md) |
| `nki.isa.nc_version` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-nc_version) |
| `nki.isa.get_nc_version` | [Link](../programming/api/api-nki-isa-tensor.md#nki-isa-get_nc_version) |
| `nki.language.tile_size` | [Link](../programming/api/nki.language.md) |

---

## See Also

- [API Reference Index](../programming/api/index.md) - Complete API documentation
- [nki.language Module](../programming/api/nki.language.md) - Language-level APIs
- [nki.isa Module](../programming/api/nki.isa.md) - ISA-level APIs
- [Shared APIs](../programming/api/nki.api.shared.md) - Shared data types and operators
