# GNN Facade CLI - Kani Formal Verification

**CISA Secure by Design Compliance - January 2026**

Bit-precise formal verification of memory safety for the Rust/CUDA FFI boundary.

## Test Summary

| Type | Count | Status |
|------|-------|--------|
| Unit Tests | 76 | ✓ PASS |
| Kani Proof Harnesses | 19 | ✓ PASS |
| In-tree FFI C Boundary Proofs | 55 | ✓ PASS |
| In-tree CUDA Backend Proofs | 42 | ✓ PASS |
| In-tree OpenCL Backend Proofs | 42 | ✓ PASS |
| **Total Verifications** | **234** | ✓ PASS |

## Quick Start

```bash
# Run unit tests
cargo test

# Run formal verification (requires Kani)
cargo kani
```

## Kani Installation (one-time)

```bash
cargo install --locked kani-verifier
kani setup
```

## Verified Properties

| Category | Unit Tests | Kani Proofs |
|----------|------------|-------------|
| Graph Construction | 4 | - |
| Node Feature Access | 12 | 3 |
| Edge Operations | 14 | 5 |
| Adjacency List | 9 | 3 |
| Node Mask Manager | 7 | 2 |
| Edge Mask Manager | 8 | 2 |
| Buffer Validator | 14 | 4 |
| Edge Features | 3 | - |
| Boundary Values | 3 | - |
| Stress Tests | 2 | - |

## What This Proves

1. **Panic-Free FFI Boundary** - Operations never trigger panics that escape to CUDA/OpenCL
2. **Buffer Overflow Protection** - Edge/node indexing never accesses out-of-bounds memory  
3. **Nondeterministic Safety** - Logic is safe for any possible input value
4. **Boundary Conditions** - MAX_NODES and MAX_EDGES limits properly enforced
5. **FFI C Boundary Safety** - All data crossing the extern "C" boundary (ffi.rs) is validated
6. **CUDA Backend Safety** - Buffer sizes, grid/block dims, index validity for CUDA kernels
7. **OpenCL Backend Safety** - Buffer sizes, global/local work sizes, index validity for OpenCL kernels

### 16. FFI C Boundary Safety (Category 16)

Located in `src/kani/ffi_c_boundary.rs`, 55 Kani proofs + unit tests across 15 categories:

#### A. Unsigned Integer Validation
- `verify_ffi_cuint_positive_rejects_zero` - Zero c_uint rejected where positive required
- `verify_ffi_cuint_as_usize_always_safe` - c_uint -> usize always safe (no sign bit)
- `verify_ffi_cuint_max_enforced` - Upper bound validation
- `verify_ffi_len_validates_range` - Array length range validation

#### B. Output Buffer Overflow Prevention
- `verify_ffi_output_write_bounded_by_capacity` - Write never exceeds buffer capacity
- `verify_ffi_predict_output_bounded` - Predict output bounded
- `verify_ffi_zero_buffer_len_rejected` - Zero buffer length rejected for string output
- `verify_ffi_string_copy_bounded` - String copy bounded by buffer

#### C. NaN/Infinity Parameter Rejection
- `verify_ffi_f32_param_rejects_special_values` - NaN/Inf rejected at boundary
- `verify_ffi_learning_rate_rejects_nan/infinity/negative` - LR validation
- `verify_ffi_learning_rate_accepts_valid` - Valid LR accepted
- `verify_ffi_dropout_rate_validated` - Dropout [0,1] range validated
- `verify_ffi_damping_factor_validated` - PageRank damping [0,1] validated
- `verify_ffi_node_feature_nan/inf_rejected` - Node feature NaN/Inf rejected

#### D. Backend Enum Validation
- `verify_ffi_backend_enum_validation` - Backend int validated (0-2)
- `verify_ffi_backend_negative_handled` - Negative backend rejected

#### E. GNN Create Preconditions
- `verify_ffi_create_rejects_zero_*` - Zero sizes rejected for all 4 params
- `verify_ffi_create_rejects_oversized/excessive_*` - Upper bounds enforced
- `verify_ffi_create_pipeline_all_inputs` - End-to-end create validation

#### F. Array Length Validation
- `verify_ffi_feature_array_len_bounded` - Feature arrays bounded at 4096
- `verify_ffi_train_target_len_bounded` - Train target bounded at 1M
- `verify_ffi_predict_output_len_bounded` - Predict output bounded

#### G. Graph Structure Bounds
- `verify_ffi_node/edge_index_bounded` - Index bounds checked
- `verify_ffi_add_edge_validates_node_bounds` - Edge endpoint validation
- `verify_ffi_neighbor_access_bounded` - Neighbor access safe

#### H. Node/Edge Mask Validation
- `verify_ffi_node/edge_mask_oob_safe` - OOB mask access returns safe default
- `verify_ffi_edge_mask_add_respects_limit` - MAX_EDGES limit enforced

#### I. No-Panic Guarantee
- `verify_ffi_all_validators_no_panic` - All validators safe for any input

#### J. ABI Type Compatibility
- `verify_ffi_f32/u32/i32_abi_compatibility` - Size and alignment for C ABI

#### K. Input Array NaN/Infinity Detection
- `verify_ffi_nan/inf_in_f32_array_detectable` - NaN/Inf detectable in arrays

#### L. Resource Limits
- `verify_ffi_feature/train_allocation_bounded` - Memory bounded
- `verify_ffi_graph_node/edge_limit_enforced` - Graph limits enforced

#### M. Setter Value Validation
- `verify_ffi_setter_rejects_nan/infinity` - NaN/Inf rejected
- `verify_ffi_setter_accepts_valid_f32` - Valid values accepted
- `verify_ffi_dropout_setter_rejects_over_one/accepts_valid` - Range enforced

#### N. End-to-End Pipeline Validation
- `verify_ffi_complete_predict/train/create/page_rank_pipeline` - Full pipelines

#### O. Buffer Validator Proofs
- `verify_ffi_buffer_validator_node/edge_index` - Index validation
- `verify_ffi_buffer_validator_feature/embedding_offset_safe` - Offset calculations

### 17. CUDA Backend FFI Safety (Category 17)

Located in `src/kani/ffi_cuda_boundary.rs`, 42 Kani proofs + unit tests across 15 categories:

#### A. Layer Buffer Size Correctness
- Weight, bias, and gradient buffer size verification

#### B. CUDA Grid/Block Dimension Safety
- Forward, backward, input grad grid dims all cover neurons

#### C. Weight Index Validity for Flat Buffers
- Flat index always within allocated buffer for weights and messages

#### D. Transfer Size Non-Zero and Alignment
- All CUDA transfers are non-zero and f32-aligned

#### E. f32 Alignment for CUDA Transfers
- f32 and i32 size/alignment match CUDA expectations

#### F. Node Embedding Buffer Sizing
- MAX_NODES * hidden_size buffer and index validity

#### G. Message Buffer Sizing
- MAX_EDGES * hidden_size buffer and index validity

#### H. Kernel Launch Parameter Overflow Prevention
- Weight count overflow check, thread coverage validation

#### I. Aggregation Buffer Sizing
- Aggregated message buffer and index validity

#### J. Graph Readout Buffer Safety
- Graph embedding buffer size and readout grid dims

#### K. Gradient Buffer Sizing
- Output gradient and target buffer sizes match

#### L. Neighbor Offset/Count Buffer Safety
- Neighbor count buffer size and offset accumulation overflow check

#### M. No-Panic Guarantee for Dimension Calculations
- All grid and buffer size calculations are panic-free

#### N. ABI Type Compatibility for CUDA Interop
- f32, i32, u32 size and alignment for CUDA ABI

#### O. End-to-End Forward Pass Buffer Chain
- Message/update layer input sizes, readout→output chain, MSE gradient grid

### 18. OpenCL Backend FFI Safety (Category 18)

Located in `src/kani/ffi_opencl_boundary.rs`, 42 Kani proofs + unit tests across 15 categories:

#### A. Layer Buffer Size Correctness for clCreateBuffer
- Weight, bias, and gradient buffer size verification

#### B. OpenCL Global/Local Work Size Safety
- Forward, backward, input grad work sizes cover all work items, divisible by local

#### C. Weight Index Validity for Flat Buffers
- Flat index always within OpenCL buffer for weights and messages

#### D. Transfer Size Non-Zero and Alignment for Enqueue
- All clEnqueueRead/WriteBuffer sizes are non-zero and aligned

#### E. cl_float/cl_int Alignment for OpenCL Interop
- f32 and i32 size/alignment match cl_float/cl_int

#### F. Node Embedding Buffer Sizing
- MAX_NODES * hidden_size buffer and index validity for OpenCL

#### G. Message Buffer Sizing
- MAX_EDGES * hidden_size buffer and index validity for OpenCL

#### H. Work Group Size Properties
- Local work size is power of two and within device limits

#### I. Aggregation Buffer Sizing
- Aggregated message buffer and index validity for OpenCL

#### J. Graph Readout Buffer Safety
- Graph embedding buffer and readout work size

#### K. Gradient Buffer Sizing
- Output gradient and target buffer sizes for OpenCL

#### L. Neighbor Offset/Count Buffer Safety
- Neighbor count buffer and offset accumulation for OpenCL

#### M. No-Panic Guarantee for Work Size Calculations
- All global work size and buffer calculations are panic-free

#### N. ABI Type Compatibility for OpenCL Interop
- f32, i32, u32 match cl_float, cl_int, cl_uint

#### O. End-to-End Forward Pass Buffer Chain
- Message/update layer input sizes, readout→output chain, MSE gradient work size

## Author

Matthew Abbott <mattbachg@gmail.com>

## License

MIT License
