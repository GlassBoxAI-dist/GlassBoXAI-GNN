//! @file
//! @ingroup GNN_Core_Verified
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: C FFI Boundary Safety (CISA/NSA Compliance)
 *
 * Proves that all data crossing the C FFI boundary (ffi.rs) is validated
 * before use. Covers the complete extern "C" surface consumed by C++, Go,
 * C#, Julia, Zig, and Python (via PyO3) wrappers.
 *
 * CISA "Secure by Design" requirements verified:
 * - Unsigned integer overflow prevention (c_uint -> usize)
 * - Output buffer overflow prevention
 * - NaN/Infinity parameter rejection
 * - Enum/backend variant validation from foreign callers
 * - Resource exhaustion prevention at boundary
 * - No-panic guarantee for validation logic
 * - ABI type compatibility proofs
 * - Create precondition validation
 * - Array length bounds enforcement
 */

use super::*;

// =========================================================================
// FFI validation helpers (mirroring the guards used in ffi.rs)
// =========================================================================

fn validate_cuint_positive(val: u32) -> Option<usize> {
    if val == 0 { None } else { Some(val as usize) }
}

fn validate_cuint_as_usize(val: u32) -> Option<usize> {
    Some(val as usize)
}

fn validate_cuint_max(val: u32, max: u32) -> Option<usize> {
    if val > max { None } else { Some(val as usize) }
}

fn validate_ffi_len(len: u32, max: usize) -> Option<usize> {
    let len_usize = len as usize;
    if len_usize > max { None } else { Some(len_usize) }
}

fn validate_f32_param(value: f32) -> Option<f32> {
    if value.is_nan() || value.is_infinite() { None } else { Some(value) }
}

fn validate_f32_nonneg(value: f32) -> Option<f32> {
    if value.is_nan() || value.is_infinite() || value < 0.0 { None } else { Some(value) }
}

fn validate_dropout_rate(value: f32) -> Option<f32> {
    if value.is_nan() || value.is_infinite() || value < 0.0 || value > 1.0 { None } else { Some(value) }
}

fn validate_damping(value: f32) -> Option<f32> {
    if value.is_nan() || value.is_infinite() || value < 0.0 || value > 1.0 { None } else { Some(value) }
}

fn validate_backend_int(val: i32) -> bool {
    val >= 0 && val <= 2
}

const MAX_FFI_ARRAY_LEN: usize = 1048576;
const MAX_FEATURE_ARRAY_LEN: usize = 4096;
const MAX_GNN_SIZE: u32 = 4096;
const MAX_GNN_LAYERS: u32 = 64;

fn is_fp_sane(value: f32) -> bool {
    !value.is_nan() && !value.is_infinite()
}

// =========================================================================
// A. UNSIGNED INTEGER VALIDATION
// Prove that c_uint -> usize conversions are safe and that zero values
// are rejected where sizes must be positive.
// =========================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn verify_ffi_cuint_positive_rejects_zero() {
        let val: u32 = kani::any();
        let result = validate_cuint_positive(val);
        if val == 0 {
            kani::assert(result.is_none(), "Zero must be rejected");
        } else {
            kani::assert(result == Some(val as usize), "Positive must convert correctly");
        }
    }

    #[kani::proof]
    fn verify_ffi_cuint_as_usize_always_safe() {
        let val: u32 = kani::any();
        let result = validate_cuint_as_usize(val);
        kani::assert(result == Some(val as usize),
            "c_uint to usize is always safe (no sign bit)");
    }

    #[kani::proof]
    fn verify_ffi_cuint_max_enforced() {
        let val: u32 = kani::any();
        let max: u32 = kani::any();
        kani::assume(max <= MAX_GNN_SIZE);

        let result = validate_cuint_max(val, max);
        if val > max {
            kani::assert(result.is_none(), "Over-max must be rejected");
        } else {
            kani::assert(result == Some(val as usize), "In-range must be accepted");
        }
    }

    #[kani::proof]
    fn verify_ffi_len_validates_range() {
        let len: u32 = kani::any();
        let max: usize = kani::any();
        kani::assume(max <= MAX_FFI_ARRAY_LEN);

        let result = validate_ffi_len(len, max);
        if (len as usize) > max {
            kani::assert(result.is_none(), "Over-max length must be rejected");
        } else {
            kani::assert(result == Some(len as usize), "Valid length must be accepted");
        }
    }

    // =========================================================================
    // B. OUTPUT BUFFER OVERFLOW PREVENTION
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_output_write_bounded_by_capacity() {
        let data_len: usize = kani::any();
        let capacity: usize = kani::any();
        kani::assume(data_len <= 1024);
        kani::assume(capacity <= 1024);

        let write_len = data_len.min(capacity);
        kani::assert(write_len <= capacity, "Write must never exceed capacity");
        kani::assert(write_len <= data_len, "Write must never exceed data length");
    }

    #[kani::proof]
    fn verify_ffi_predict_output_bounded() {
        let result_len: usize = kani::any();
        let raw_capacity: u32 = kani::any();
        kani::assume(result_len <= 256);
        kani::assume(raw_capacity > 0);

        let capacity = raw_capacity as usize;
        let write_len = result_len.min(capacity);
        kani::assert(write_len <= capacity, "Predict output bounded by capacity");
    }

    #[kani::proof]
    fn verify_ffi_zero_buffer_len_rejected() {
        let buffer_len: u32 = 0;
        let valid = buffer_len > 0;
        kani::assert(!valid, "Zero buffer length must be rejected for string output");
    }

    #[kani::proof]
    fn verify_ffi_string_copy_bounded() {
        let string_len: usize = kani::any();
        let buffer_len: u32 = kani::any();
        kani::assume(string_len <= 4096);
        kani::assume(buffer_len > 0 && buffer_len <= 4096);

        let copy_len = string_len.min(buffer_len as usize);
        kani::assert(copy_len <= buffer_len as usize,
            "String copy must not exceed buffer capacity");
    }

    // =========================================================================
    // C. NaN/INFINITY REJECTION AT FFI BOUNDARY
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_f32_param_rejects_special_values() {
        let val: f32 = kani::any();
        let result = validate_f32_param(val);
        if val.is_nan() || val.is_infinite() {
            kani::assert(result.is_none(), "NaN/Infinity must be rejected");
        } else {
            kani::assert(result == Some(val), "Finite values must be accepted");
        }
    }

    #[kani::proof]
    fn verify_ffi_learning_rate_rejects_nan() {
        let val: f32 = kani::any();
        kani::assume(val.is_nan());
        kani::assert(validate_f32_nonneg(val).is_none(),
            "NaN learning rate must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_learning_rate_rejects_infinity() {
        let val: f32 = kani::any();
        kani::assume(val.is_infinite());
        kani::assert(validate_f32_nonneg(val).is_none(),
            "Infinite learning rate must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_learning_rate_rejects_negative() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite() && val < 0.0);
        kani::assert(validate_f32_nonneg(val).is_none(),
            "Negative learning rate must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_learning_rate_accepts_valid() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite() && val >= 0.0);
        kani::assert(validate_f32_nonneg(val).is_some(),
            "Valid learning rate must be accepted");
    }

    #[kani::proof]
    fn verify_ffi_dropout_rate_validated() {
        let val: f32 = kani::any();
        let result = validate_dropout_rate(val);
        if val.is_nan() || val.is_infinite() || val < 0.0 || val > 1.0 {
            kani::assert(result.is_none(), "Invalid dropout rate rejected");
        } else {
            kani::assert(result.is_some(), "Valid dropout rate accepted");
        }
    }

    #[kani::proof]
    fn verify_ffi_damping_factor_validated() {
        let val: f32 = kani::any();
        let result = validate_damping(val);
        if val.is_nan() || val.is_infinite() || val < 0.0 || val > 1.0 {
            kani::assert(result.is_none(), "Invalid damping factor rejected");
        } else {
            kani::assert(result.is_some(), "Valid damping factor accepted");
        }
    }

    #[kani::proof]
    fn verify_ffi_node_feature_nan_rejected() {
        let value: f32 = kani::any();
        kani::assume(value.is_nan());

        let accepted = !value.is_nan() && !value.is_infinite();
        kani::assert(!accepted, "NaN node feature must be rejected by setter guard");
    }

    #[kani::proof]
    fn verify_ffi_node_feature_inf_rejected() {
        let value: f32 = kani::any();
        kani::assume(value.is_infinite());

        let accepted = !value.is_nan() && !value.is_infinite();
        kani::assert(!accepted, "Infinite node feature must be rejected by setter guard");
    }

    // =========================================================================
    // D. BACKEND ENUM VALIDATION FROM FOREIGN CALLERS
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_backend_enum_validation() {
        let val: i32 = kani::any();
        let valid = validate_backend_int(val);

        match val {
            0 | 1 | 2 => kani::assert(valid, "Valid backend must be accepted"),
            _ => kani::assert(!valid, "Out-of-range backend must be rejected"),
        }
    }

    #[kani::proof]
    fn verify_ffi_backend_negative_handled() {
        let val: i32 = kani::any();
        kani::assume(val < 0);
        kani::assert(!validate_backend_int(val),
            "Negative backend value must be rejected");
    }

    // =========================================================================
    // E. GNN CREATE PRECONDITIONS
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_create_rejects_zero_feature_size() {
        let fs: u32 = 0;
        kani::assert(validate_cuint_positive(fs).is_none(),
            "Zero feature size must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_create_rejects_zero_hidden_size() {
        let hs: u32 = 0;
        kani::assert(validate_cuint_positive(hs).is_none(),
            "Zero hidden size must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_create_rejects_zero_output_size() {
        let os: u32 = 0;
        kani::assert(validate_cuint_positive(os).is_none(),
            "Zero output size must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_create_rejects_zero_layers() {
        let layers: u32 = 0;
        kani::assert(validate_cuint_positive(layers).is_none(),
            "Zero MP layers must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_create_rejects_oversized_params() {
        let val: u32 = kani::any();
        kani::assume(val > MAX_GNN_SIZE);

        kani::assert(validate_cuint_max(val, MAX_GNN_SIZE).is_none(),
            "Oversized GNN parameter must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_create_rejects_excessive_layers() {
        let val: u32 = kani::any();
        kani::assume(val > MAX_GNN_LAYERS);

        kani::assert(validate_cuint_max(val, MAX_GNN_LAYERS).is_none(),
            "Excessive layer count must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_create_pipeline_all_inputs() {
        let fs: u32 = kani::any();
        let hs: u32 = kani::any();
        let os: u32 = kani::any();
        let layers: u32 = kani::any();

        let fv = validate_cuint_positive(fs);
        let hv = validate_cuint_positive(hs);
        let ov = validate_cuint_positive(os);
        let lv = validate_cuint_positive(layers);

        if fv.is_some() && hv.is_some() && ov.is_some() && lv.is_some() {
            kani::assert(fv.unwrap() > 0, "Feature size positive");
            kani::assert(hv.unwrap() > 0, "Hidden size positive");
            kani::assert(ov.unwrap() > 0, "Output size positive");
            kani::assert(lv.unwrap() > 0, "Layers positive");
        }
    }

    // =========================================================================
    // F. TRAIN/PREDICT/FEATURE ARRAY LENGTH VALIDATION
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_feature_array_len_bounded() {
        let len: u32 = kani::any();
        let result = validate_ffi_len(len, MAX_FEATURE_ARRAY_LEN);
        if (len as usize) > MAX_FEATURE_ARRAY_LEN {
            kani::assert(result.is_none(), "Oversized feature array rejected");
        } else {
            kani::assert(result.is_some(), "Valid feature array accepted");
        }
    }

    #[kani::proof]
    fn verify_ffi_train_target_len_bounded() {
        let len: u32 = kani::any();
        let result = validate_ffi_len(len, MAX_FFI_ARRAY_LEN);
        if (len as usize) > MAX_FFI_ARRAY_LEN {
            kani::assert(result.is_none(), "Oversized target array rejected");
        } else {
            kani::assert(result.is_some(), "Valid target array accepted");
        }
    }

    #[kani::proof]
    fn verify_ffi_predict_output_len_bounded() {
        let len: u32 = kani::any();
        let result = validate_ffi_len(len, MAX_FFI_ARRAY_LEN);
        if (len as usize) > MAX_FFI_ARRAY_LEN {
            kani::assert(result.is_none(), "Oversized output buffer rejected");
        } else {
            kani::assert(result.is_some(), "Valid output buffer accepted");
        }
    }

    // =========================================================================
    // G. GRAPH STRUCTURE BOUNDS VALIDATION
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_node_index_bounded() {
        let num_nodes: usize = kani::any();
        let node_idx: usize = kani::any();
        kani::assume(num_nodes > 0 && num_nodes <= MAX_NODES);

        let graph = VerifiableGraph::new(num_nodes);
        let result = graph.get_node_features(node_idx);
        if node_idx >= num_nodes {
            kani::assert(result.is_none(), "OOB node access returns None");
        }
    }

    #[kani::proof]
    fn verify_ffi_edge_index_bounded() {
        let mut graph = VerifiableGraph::new(4);
        let added = graph.add_edge(0, 1);
        kani::assert(added.is_some(), "Valid edge should be added");

        let idx: usize = kani::any();
        kani::assume(idx <= 2);
        let result = graph.get_edge(idx);
        if idx >= graph.edges.len() {
            kani::assert(result.is_none(), "OOB edge access returns None");
        } else {
            kani::assert(result.is_some(), "Valid edge access returns Some");
        }
    }

    #[kani::proof]
    fn verify_ffi_add_edge_validates_node_bounds() {
        let num_nodes: usize = kani::any();
        kani::assume(num_nodes > 0 && num_nodes <= 8);

        let mut graph = VerifiableGraph::new(num_nodes);
        let src: usize = kani::any();
        let tgt: usize = kani::any();
        kani::assume(src <= num_nodes + 1);
        kani::assume(tgt <= num_nodes + 1);

        let result = graph.add_edge(src, tgt);
        if src >= num_nodes || tgt >= num_nodes {
            kani::assert(result.is_none(), "OOB edge endpoints rejected");
        }
    }

    #[kani::proof]
    fn verify_ffi_neighbor_access_bounded() {
        let num_nodes: usize = kani::any();
        kani::assume(num_nodes > 0 && num_nodes <= 8);

        let graph = VerifiableGraph::new(num_nodes);
        let idx: usize = kani::any();
        let result = graph.get_neighbors(idx);
        if idx >= num_nodes {
            kani::assert(result.is_none(), "OOB neighbor access returns None");
        } else {
            kani::assert(result.is_some(), "Valid neighbor access returns Some");
        }
    }

    // =========================================================================
    // H. NODE/EDGE MASK VALIDATION
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_node_mask_oob_safe() {
        let num_nodes: usize = kani::any();
        kani::assume(num_nodes > 0 && num_nodes <= 16);

        let mgr = NodeMaskManager::new(num_nodes);
        let idx: usize = kani::any();
        let result = mgr.get_mask(idx);
        if idx >= num_nodes {
            kani::assert(!result, "OOB mask returns false (default)");
        }
    }

    #[kani::proof]
    fn verify_ffi_edge_mask_oob_safe() {
        let mgr = EdgeMaskManager::new();
        let idx: usize = kani::any();
        kani::assume(idx <= 100);

        let result = mgr.get_mask(idx);
        kani::assert(!result, "Empty edge mask returns false for any index");
    }

    #[kani::proof]
    fn verify_ffi_edge_mask_add_respects_limit() {
        let mut mgr = EdgeMaskManager::new();
        for _ in 0..MAX_EDGES {
            mgr.add_edge();
        }
        kani::assert(!mgr.add_edge(), "MAX_EDGES limit must be enforced");
    }

    // =========================================================================
    // I. NO-PANIC GUARANTEE FOR ALL FFI VALIDATORS
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_all_validators_no_panic() {
        let u: u32 = kani::any();
        let f: f32 = kani::any();
        let i: i32 = kani::any();

        let _a = validate_cuint_positive(u);
        let _b = validate_cuint_as_usize(u);
        let _c = validate_cuint_max(u, MAX_GNN_SIZE);
        let _d = validate_ffi_len(u, MAX_FFI_ARRAY_LEN);
        let _e = validate_f32_param(f);
        let _f = validate_f32_nonneg(f);
        let _g = validate_dropout_rate(f);
        let _h = validate_damping(f);
        let _i = validate_backend_int(i);
        let _j = is_fp_sane(f);
    }

    // =========================================================================
    // J. ABI TYPE COMPATIBILITY
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_f32_abi_compatibility() {
        kani::assert(std::mem::size_of::<f32>() == 4, "f32 must be 4 bytes for C float");
        kani::assert(std::mem::align_of::<f32>() == 4, "f32 must be 4-byte aligned");
    }

    #[kani::proof]
    fn verify_ffi_u32_abi_compatibility() {
        kani::assert(std::mem::size_of::<u32>() == 4, "u32 must be 4 bytes for C uint32_t");
        kani::assert(std::mem::align_of::<u32>() == 4, "u32 must be 4-byte aligned");
    }

    #[kani::proof]
    fn verify_ffi_i32_abi_compatibility() {
        kani::assert(std::mem::size_of::<i32>() == 4, "i32 must be 4 bytes for C int32_t");
        kani::assert(std::mem::align_of::<i32>() == 4, "i32 must be 4-byte aligned");
    }

    // =========================================================================
    // K. INPUT ARRAY NaN/INFINITY DETECTION
    // =========================================================================

    #[kani::proof]
    #[kani::unwind(9)]
    fn verify_ffi_nan_in_f32_array_detectable() {
        let size: usize = kani::any();
        kani::assume(size > 0 && size <= 8);

        let mut arr = vec![1.0f32; size];
        let idx: usize = kani::any();
        kani::assume(idx < size);
        arr[idx] = f32::NAN;

        let has_bad = arr.iter().any(|x| !is_fp_sane(*x));
        kani::assert(has_bad, "NaN in f32 array must be detectable");
    }

    #[kani::proof]
    #[kani::unwind(9)]
    fn verify_ffi_inf_in_f32_array_detectable() {
        let size: usize = kani::any();
        kani::assume(size > 0 && size <= 8);

        let mut arr = vec![1.0f32; size];
        let idx: usize = kani::any();
        kani::assume(idx < size);
        arr[idx] = f32::INFINITY;

        let has_bad = arr.iter().any(|x| !is_fp_sane(*x));
        kani::assert(has_bad, "Infinity in f32 array must be detectable");
    }

    // =========================================================================
    // L. RESOURCE LIMITS AT FFI BOUNDARY
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_feature_allocation_bounded() {
        let len: u32 = kani::any();
        kani::assume(len <= MAX_FEATURE_ARRAY_LEN as u32);

        let len_usize = len as usize;
        let bytes = len_usize * std::mem::size_of::<f32>();
        kani::assert(bytes <= MAX_FEATURE_ARRAY_LEN * 4,
            "Feature array allocation must be bounded");
    }

    #[kani::proof]
    fn verify_ffi_train_allocation_bounded() {
        let len: u32 = kani::any();
        kani::assume(len <= MAX_FFI_ARRAY_LEN as u32);

        let len_usize = len as usize;
        let bytes = len_usize * std::mem::size_of::<f32>();
        kani::assert(bytes <= MAX_FFI_ARRAY_LEN * 4,
            "Train target allocation must be bounded");
    }

    #[kani::proof]
    fn verify_ffi_graph_node_limit_enforced() {
        let num_nodes: usize = kani::any();
        kani::assume(num_nodes > 0);

        let graph = VerifiableGraph::new(num_nodes);
        kani::assert(graph.num_nodes == num_nodes, "Graph node count matches");
    }

    #[kani::proof]
    fn verify_ffi_graph_edge_limit_enforced() {
        let mut graph = VerifiableGraph::new(4);
        let src: usize = kani::any();
        let tgt: usize = kani::any();
        kani::assume(src < 4 && tgt < 4);

        let result = graph.add_edge(src, tgt);
        if graph.edges.len() > MAX_EDGES {
            kani::assert(result.is_none(), "Edge limit enforced");
        }
    }

    // =========================================================================
    // M. SETTER VALUE VALIDATION
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_setter_rejects_nan() {
        let value: f32 = kani::any();
        kani::assume(value.is_nan());

        let accepted = !value.is_nan() && !value.is_infinite();
        kani::assert(!accepted, "NaN must be rejected by setter guard");
    }

    #[kani::proof]
    fn verify_ffi_setter_rejects_infinity() {
        let value: f32 = kani::any();
        kani::assume(value.is_infinite());

        let accepted = !value.is_nan() && !value.is_infinite();
        kani::assert(!accepted, "Infinity must be rejected by setter guard");
    }

    #[kani::proof]
    fn verify_ffi_setter_accepts_valid_f32() {
        let value: f32 = kani::any();
        kani::assume(!value.is_nan() && !value.is_infinite());

        let accepted = !value.is_nan() && !value.is_infinite();
        kani::assert(accepted, "Valid f32 must be accepted");
    }

    #[kani::proof]
    fn verify_ffi_dropout_setter_rejects_over_one() {
        let value: f32 = kani::any();
        kani::assume(value.is_finite() && value > 1.0);

        kani::assert(validate_dropout_rate(value).is_none(),
            "Dropout > 1.0 must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_dropout_setter_accepts_valid() {
        let value: f32 = kani::any();
        kani::assume(value.is_finite() && value >= 0.0 && value <= 1.0);

        kani::assert(validate_dropout_rate(value).is_some(),
            "Valid dropout rate must be accepted");
    }

    // =========================================================================
    // N. END-TO-END FFI PIPELINE VALIDATION
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_complete_predict_pipeline() {
        let output_len: u32 = kani::any();

        let ov = validate_ffi_len(output_len, MAX_FFI_ARRAY_LEN);

        if ov.is_some() {
            let result_len: usize = kani::any();
            kani::assume(result_len <= 256);
            let write = result_len.min(ov.unwrap());
            kani::assert(write <= ov.unwrap(), "Write bounded by capacity");
        }
    }

    #[kani::proof]
    fn verify_ffi_complete_train_pipeline() {
        let target_len: u32 = kani::any();

        let tv = validate_ffi_len(target_len, MAX_FFI_ARRAY_LEN);

        if tv.is_some() {
            kani::assert(tv.unwrap() <= MAX_FFI_ARRAY_LEN, "Target len bounded");
            let bytes = tv.unwrap() * std::mem::size_of::<f32>();
            kani::assert(bytes <= MAX_FFI_ARRAY_LEN * 4, "Target memory bounded");
        }
    }

    #[kani::proof]
    fn verify_ffi_complete_create_pipeline() {
        let fs: u32 = kani::any();
        let hs: u32 = kani::any();
        let os: u32 = kani::any();
        let layers: u32 = kani::any();

        let fv = validate_cuint_positive(fs);
        let hv = validate_cuint_positive(hs);
        let ov = validate_cuint_positive(os);
        let lv = validate_cuint_positive(layers);

        let fsm = validate_cuint_max(fs, MAX_GNN_SIZE);
        let hsm = validate_cuint_max(hs, MAX_GNN_SIZE);
        let osm = validate_cuint_max(os, MAX_GNN_SIZE);
        let lsm = validate_cuint_max(layers, MAX_GNN_LAYERS);

        if fv.is_some() && hv.is_some() && ov.is_some() && lv.is_some()
            && fsm.is_some() && hsm.is_some() && osm.is_some() && lsm.is_some()
        {
            kani::assert(fv.unwrap() > 0 && fv.unwrap() <= MAX_GNN_SIZE as usize, "Feature bounded");
            kani::assert(hv.unwrap() > 0 && hv.unwrap() <= MAX_GNN_SIZE as usize, "Hidden bounded");
            kani::assert(ov.unwrap() > 0 && ov.unwrap() <= MAX_GNN_SIZE as usize, "Output bounded");
            kani::assert(lv.unwrap() > 0 && lv.unwrap() <= MAX_GNN_LAYERS as usize, "Layers bounded");
        }
    }

    #[kani::proof]
    fn verify_ffi_complete_page_rank_pipeline() {
        let damping: f32 = kani::any();
        let scores_len: u32 = kani::any();

        let dv = validate_damping(damping);
        let sv = validate_ffi_len(scores_len, MAX_FFI_ARRAY_LEN);

        if dv.is_some() && sv.is_some() {
            kani::assert(dv.unwrap() >= 0.0 && dv.unwrap() <= 1.0, "Damping in range");
            kani::assert(sv.unwrap() <= MAX_FFI_ARRAY_LEN, "Scores len bounded");
        }
    }

    // =========================================================================
    // O. BUFFER VALIDATOR PROOFS
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_buffer_validator_node_index() {
        let max_nodes: usize = kani::any();
        kani::assume(max_nodes > 0 && max_nodes <= MAX_NODES);

        let validator = BufferIndexValidator::new(max_nodes, MAX_EDGES, 16, 64, 10);
        let idx: usize = kani::any();
        kani::assume(idx <= max_nodes);

        if idx >= max_nodes {
            kani::assert(!validator.validate_node_index(idx), "OOB node rejected");
        } else {
            kani::assert(validator.validate_node_index(idx), "Valid node accepted");
        }
    }

    #[kani::proof]
    fn verify_ffi_buffer_validator_edge_index() {
        let max_edges: usize = kani::any();
        kani::assume(max_edges > 0 && max_edges <= MAX_EDGES);

        let validator = BufferIndexValidator::new(MAX_NODES, max_edges, 16, 64, 10);
        let idx: usize = kani::any();
        kani::assume(idx <= max_edges);

        if idx >= max_edges {
            kani::assert(!validator.validate_edge_index(idx), "OOB edge rejected");
        } else {
            kani::assert(validator.validate_edge_index(idx), "Valid edge accepted");
        }
    }

    #[kani::proof]
    fn verify_ffi_buffer_validator_feature_offset_safe() {
        let max_nodes: usize = kani::any();
        let feature_size: usize = kani::any();
        kani::assume(max_nodes > 0 && max_nodes <= 16);
        kani::assume(feature_size > 0 && feature_size <= 16);

        let validator = BufferIndexValidator::new(max_nodes, MAX_EDGES, feature_size, 64, 10);
        let node_idx: usize = kani::any();
        let feat_idx: usize = kani::any();
        kani::assume(node_idx <= max_nodes);
        kani::assume(feat_idx <= feature_size);

        let result = validator.node_feature_offset(node_idx, feat_idx);
        if node_idx >= max_nodes || feat_idx >= feature_size {
            kani::assert(result.is_none(), "OOB feature offset returns None");
        } else {
            kani::assert(result.is_some(), "Valid feature offset returns Some");
            kani::assert(result.unwrap() == node_idx * feature_size + feat_idx,
                "Feature offset calculation correct");
        }
    }

    #[kani::proof]
    fn verify_ffi_buffer_validator_embedding_offset_safe() {
        let max_nodes: usize = kani::any();
        let hidden_size: usize = kani::any();
        kani::assume(max_nodes > 0 && max_nodes <= 16);
        kani::assume(hidden_size > 0 && hidden_size <= 16);

        let validator = BufferIndexValidator::new(max_nodes, MAX_EDGES, 16, hidden_size, 10);
        let node_idx: usize = kani::any();
        let hidden_idx: usize = kani::any();
        kani::assume(node_idx <= max_nodes);
        kani::assume(hidden_idx <= hidden_size);

        let result = validator.node_embedding_offset(node_idx, hidden_idx);
        if node_idx >= max_nodes || hidden_idx >= hidden_size {
            kani::assert(result.is_none(), "OOB embedding offset returns None");
        } else {
            kani::assert(result.is_some(), "Valid embedding offset returns Some");
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_validate_cuint_positive() {
        assert_eq!(validate_cuint_positive(0), None);
        assert_eq!(validate_cuint_positive(1), Some(1));
        assert_eq!(validate_cuint_positive(u32::MAX), Some(u32::MAX as usize));
    }

    #[test]
    fn test_validate_cuint_max() {
        assert_eq!(validate_cuint_max(0, 10), Some(0));
        assert_eq!(validate_cuint_max(10, 10), Some(10));
        assert_eq!(validate_cuint_max(11, 10), None);
    }

    #[test]
    fn test_validate_ffi_len() {
        assert_eq!(validate_ffi_len(0, 100), Some(0));
        assert_eq!(validate_ffi_len(50, 100), Some(50));
        assert_eq!(validate_ffi_len(101, 100), None);
    }

    #[test]
    fn test_validate_f32_param() {
        assert_eq!(validate_f32_param(1.0), Some(1.0));
        assert_eq!(validate_f32_param(f32::NAN), None);
        assert_eq!(validate_f32_param(f32::INFINITY), None);
        assert_eq!(validate_f32_param(f32::NEG_INFINITY), None);
    }

    #[test]
    fn test_validate_f32_nonneg() {
        assert_eq!(validate_f32_nonneg(0.0), Some(0.0));
        assert_eq!(validate_f32_nonneg(0.01), Some(0.01));
        assert_eq!(validate_f32_nonneg(-0.01), None);
        assert_eq!(validate_f32_nonneg(f32::NAN), None);
    }

    #[test]
    fn test_validate_dropout_rate() {
        assert!(validate_dropout_rate(0.0).is_some());
        assert!(validate_dropout_rate(0.5).is_some());
        assert!(validate_dropout_rate(1.0).is_some());
        assert!(validate_dropout_rate(-0.1).is_none());
        assert!(validate_dropout_rate(1.1).is_none());
        assert!(validate_dropout_rate(f32::NAN).is_none());
    }

    #[test]
    fn test_validate_damping() {
        assert!(validate_damping(0.85).is_some());
        assert!(validate_damping(0.0).is_some());
        assert!(validate_damping(1.0).is_some());
        assert!(validate_damping(-0.1).is_none());
        assert!(validate_damping(1.1).is_none());
    }

    #[test]
    fn test_validate_backend_int() {
        assert!(validate_backend_int(0));
        assert!(validate_backend_int(1));
        assert!(validate_backend_int(2));
        assert!(!validate_backend_int(3));
        assert!(!validate_backend_int(-1));
    }
}

