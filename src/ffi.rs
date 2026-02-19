//! @file
//! @ingroup GNN_Internal_Logic
/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * C/C++ FFI bindings for GlassBoxAI GNN
 */

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint};
use std::ptr;
use std::slice;

use crate::{GnnFacade, GpuBackendType};

/// Opaque handle to a GNN Facade instance
pub struct GnnHandle {
    inner: GnnFacade,
}

/// Gradient flow information structure (C-compatible)
#[repr(C)]
pub struct GnnGradientFlowInfo {
    pub layer_idx: c_uint,
    pub mean_gradient: c_float,
    pub max_gradient: c_float,
    pub min_gradient: c_float,
    pub gradient_norm: c_float,
}

/// Model header information structure (C-compatible)
#[repr(C)]
pub struct GnnModelHeader {
    pub feature_size: c_uint,
    pub hidden_size: c_uint,
    pub output_size: c_uint,
    pub mp_layers: c_uint,
    pub learning_rate: c_float,
}

/// Error codes
pub const GNN_OK: c_int = 0;
pub const GNN_ERROR_NULL_POINTER: c_int = -1;
pub const GNN_ERROR_INVALID_ARGUMENT: c_int = -2;
pub const GNN_ERROR_CUDA_ERROR: c_int = -3;
pub const GNN_ERROR_IO_ERROR: c_int = -4;
pub const GNN_ERROR_OPENCL_ERROR: c_int = -5;
pub const GNN_ERROR_UNKNOWN: c_int = -99;

/// Backend type constants
pub const GNN_BACKEND_CUDA: c_int = 0;
pub const GNN_BACKEND_OPENCL: c_int = 1;
pub const GNN_BACKEND_AUTO: c_int = 2;

fn backend_from_int(v: c_int) -> GpuBackendType {
    match v {
        0 => GpuBackendType::Cuda,
        1 => GpuBackendType::OpenCL,
        _ => GpuBackendType::Auto,
    }
}

// ============================================================================
// Lifecycle Functions
// ============================================================================

/// Create a new GNN Facade with auto-detected backend
///
/// @param feature_size Size of input node features
/// @param hidden_size Size of hidden layers
/// @param output_size Size of output predictions
/// @param num_mp_layers Number of message passing layers
/// @returns Handle to the GNN, or NULL on error
#[no_mangle]
pub extern "C" fn gnn_create(
    feature_size: c_uint,
    hidden_size: c_uint,
    output_size: c_uint,
    num_mp_layers: c_uint,
) -> *mut GnnHandle {
    gnn_create_with_backend(feature_size, hidden_size, output_size, num_mp_layers, GNN_BACKEND_AUTO)
}

/// Create a new GNN Facade with a specific backend
///
/// @param feature_size Size of input node features
/// @param hidden_size Size of hidden layers
/// @param output_size Size of output predictions
/// @param num_mp_layers Number of message passing layers
/// @param backend Backend type: GNN_BACKEND_CUDA (0), GNN_BACKEND_OPENCL (1), GNN_BACKEND_AUTO (2)
/// @returns Handle to the GNN, or NULL on error
#[no_mangle]
pub extern "C" fn gnn_create_with_backend(
    feature_size: c_uint,
    hidden_size: c_uint,
    output_size: c_uint,
    num_mp_layers: c_uint,
    backend: c_int,
) -> *mut GnnHandle {
    if feature_size == 0 || hidden_size == 0 || output_size == 0 || num_mp_layers == 0 {
        return ptr::null_mut();
    }
    if feature_size > 4096 || hidden_size > 4096 || output_size > 4096 || num_mp_layers > 64 {
        return ptr::null_mut();
    }

    match GnnFacade::with_backend(
        feature_size as usize,
        hidden_size as usize,
        output_size as usize,
        num_mp_layers as usize,
        backend_from_int(backend),
    ) {
        Ok(facade) => Box::into_raw(Box::new(GnnHandle { inner: facade })),
        Err(_) => ptr::null_mut(),
    }
}

/// Load a GNN from a saved model file
///
/// @param filename Path to the model file (null-terminated string)
/// @returns Handle to the GNN, or NULL on error
#[no_mangle]
pub extern "C" fn gnn_load(filename: *const c_char) -> *mut GnnHandle {
    if filename.is_null() {
        return ptr::null_mut();
    }

    let filename = unsafe { CStr::from_ptr(filename) };
    let filename = match filename.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    match GnnFacade::from_model_file(filename) {
        Ok(facade) => Box::into_raw(Box::new(GnnHandle { inner: facade })),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a GNN handle
///
/// @param handle Handle to free (safe to pass NULL)
#[no_mangle]
pub extern "C" fn gnn_free(handle: *mut GnnHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Read model header without loading full model
///
/// @param filename Path to the model file
/// @param header Pointer to header struct to fill
/// @returns GNN_OK on success, error code on failure
#[no_mangle]
pub extern "C" fn gnn_read_model_header(
    filename: *const c_char,
    header: *mut GnnModelHeader,
) -> c_int {
    if filename.is_null() || header.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }

    let filename = unsafe { CStr::from_ptr(filename) };
    let filename = match filename.to_str() {
        Ok(s) => s,
        Err(_) => return GNN_ERROR_INVALID_ARGUMENT,
    };

    match GnnFacade::read_model_header(filename) {
        Ok((feature_size, hidden_size, output_size, mp_layers, learning_rate)) => {
            unsafe {
                (*header).feature_size = feature_size as c_uint;
                (*header).hidden_size = hidden_size as c_uint;
                (*header).output_size = output_size as c_uint;
                (*header).mp_layers = mp_layers as c_uint;
                (*header).learning_rate = learning_rate;
            }
            GNN_OK
        }
        Err(_) => GNN_ERROR_IO_ERROR,
    }
}

// ============================================================================
// Model I/O Functions
// ============================================================================

/// Save model to file
///
/// @param handle GNN handle
/// @param filename Path to save the model
/// @returns GNN_OK on success, error code on failure
#[no_mangle]
pub extern "C" fn gnn_save_model(handle: *const GnnHandle, filename: *const c_char) -> c_int {
    if handle.is_null() || filename.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }

    let filename = unsafe { CStr::from_ptr(filename) };
    let filename = match filename.to_str() {
        Ok(s) => s,
        Err(_) => return GNN_ERROR_INVALID_ARGUMENT,
    };

    let handle = unsafe { &*handle };
    match handle.inner.save_model(filename) {
        Ok(()) => GNN_OK,
        Err(_) => GNN_ERROR_IO_ERROR,
    }
}

/// Load model from file
///
/// @param handle GNN handle
/// @param filename Path to the model file
/// @returns GNN_OK on success, error code on failure
#[no_mangle]
pub extern "C" fn gnn_load_model(handle: *mut GnnHandle, filename: *const c_char) -> c_int {
    if handle.is_null() || filename.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }

    let filename = unsafe { CStr::from_ptr(filename) };
    let filename = match filename.to_str() {
        Ok(s) => s,
        Err(_) => return GNN_ERROR_INVALID_ARGUMENT,
    };

    let handle = unsafe { &mut *handle };
    match handle.inner.load_model(filename) {
        Ok(()) => GNN_OK,
        Err(_) => GNN_ERROR_IO_ERROR,
    }
}

// ============================================================================
// Graph Operations
// ============================================================================

/// Create an empty graph
///
/// @param handle GNN handle
/// @param num_nodes Number of nodes
/// @param feature_size Size of node features
/// @returns GNN_OK on success
#[no_mangle]
pub extern "C" fn gnn_create_empty_graph(
    handle: *mut GnnHandle,
    num_nodes: c_uint,
    feature_size: c_uint,
) -> c_int {
    if handle.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }

    let handle = unsafe { &mut *handle };
    handle.inner.create_empty_graph(num_nodes as usize, feature_size as usize);
    GNN_OK
}

/// Add an edge to the graph
///
/// @param handle GNN handle
/// @param source Source node index
/// @param target Target node index
/// @param features Array of edge features (can be NULL)
/// @param features_len Length of features array
/// @returns Edge index on success, -1 on error
#[no_mangle]
pub extern "C" fn gnn_add_edge(
    handle: *mut GnnHandle,
    source: c_uint,
    target: c_uint,
    features: *const c_float,
    features_len: c_uint,
) -> c_int {
    if handle.is_null() {
        return -1;
    }

    let handle = unsafe { &mut *handle };
    if features_len > 4096 {
        return -1;
    }

    let features_vec = if features.is_null() || features_len == 0 {
        vec![]
    } else {
        unsafe { slice::from_raw_parts(features, features_len as usize).to_vec() }
    };

    handle.inner.add_edge(source as usize, target as usize, features_vec) as c_int
}

/// Remove an edge by index
///
/// @param handle GNN handle
/// @param edge_idx Edge index to remove
/// @returns GNN_OK on success
#[no_mangle]
pub extern "C" fn gnn_remove_edge(handle: *mut GnnHandle, edge_idx: c_uint) -> c_int {
    if handle.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }

    let handle = unsafe { &mut *handle };
    handle.inner.remove_edge(edge_idx as usize);
    GNN_OK
}

/// Check if an edge exists
///
/// @param handle GNN handle
/// @param source Source node index
/// @param target Target node index
/// @returns 1 if edge exists, 0 if not, -1 on error
#[no_mangle]
pub extern "C" fn gnn_has_edge(
    handle: *const GnnHandle,
    source: c_uint,
    target: c_uint,
) -> c_int {
    if handle.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    if handle.inner.has_edge(source as usize, target as usize) {
        1
    } else {
        0
    }
}

/// Find edge index between two nodes
///
/// @param handle GNN handle
/// @param source Source node index
/// @param target Target node index
/// @returns Edge index if found, -1 if not found or error
#[no_mangle]
pub extern "C" fn gnn_find_edge_index(
    handle: *const GnnHandle,
    source: c_uint,
    target: c_uint,
) -> c_int {
    if handle.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    match handle.inner.find_edge_index(source as usize, target as usize) {
        Some(idx) => idx as c_int,
        None => -1,
    }
}

/// Rebuild adjacency list from edges
///
/// @param handle GNN handle
/// @returns GNN_OK on success
#[no_mangle]
pub extern "C" fn gnn_rebuild_adjacency_list(handle: *mut GnnHandle) -> c_int {
    if handle.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }

    let handle = unsafe { &mut *handle };
    handle.inner.rebuild_adjacency_list();
    GNN_OK
}

// ============================================================================
// Node Features
// ============================================================================

/// Set all features for a node
///
/// @param handle GNN handle
/// @param node_idx Node index
/// @param features Array of feature values
/// @param features_len Length of features array
/// @returns GNN_OK on success
#[no_mangle]
pub extern "C" fn gnn_set_node_features(
    handle: *mut GnnHandle,
    node_idx: c_uint,
    features: *const c_float,
    features_len: c_uint,
) -> c_int {
    if handle.is_null() || features.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }
    if features_len > 4096 {
        return GNN_ERROR_INVALID_ARGUMENT;
    }

    let handle = unsafe { &mut *handle };
    let features_vec = unsafe { slice::from_raw_parts(features, features_len as usize).to_vec() };
    handle.inner.set_node_features(node_idx as usize, features_vec);
    GNN_OK
}

/// Get all features for a node
///
/// @param handle GNN handle
/// @param node_idx Node index
/// @param features_out Output buffer for features
/// @param features_len Length of output buffer
/// @returns Number of features copied, -1 on error
#[no_mangle]
pub extern "C" fn gnn_get_node_features(
    handle: *const GnnHandle,
    node_idx: c_uint,
    features_out: *mut c_float,
    features_len: c_uint,
) -> c_int {
    if handle.is_null() || features_out.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    match handle.inner.get_node_features(node_idx as usize) {
        Some(features) => {
            let copy_len = features.len().min(features_len as usize);
            let out_slice = unsafe { slice::from_raw_parts_mut(features_out, copy_len) };
            out_slice.copy_from_slice(&features[..copy_len]);
            copy_len as c_int
        }
        None => -1,
    }
}

/// Set a single feature value for a node
///
/// @param handle GNN handle
/// @param node_idx Node index
/// @param feature_idx Feature index
/// @param value Feature value
/// @returns GNN_OK on success
#[no_mangle]
pub extern "C" fn gnn_set_node_feature(
    handle: *mut GnnHandle,
    node_idx: c_uint,
    feature_idx: c_uint,
    value: c_float,
) -> c_int {
    if handle.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }
    if value.is_nan() || value.is_infinite() {
        return GNN_ERROR_INVALID_ARGUMENT;
    }

    let handle = unsafe { &mut *handle };
    handle.inner.set_node_feature(node_idx as usize, feature_idx as usize, value);
    GNN_OK
}

/// Get a single feature value for a node
///
/// @param handle GNN handle
/// @param node_idx Node index
/// @param feature_idx Feature index
/// @returns Feature value (0.0 if not found)
#[no_mangle]
pub extern "C" fn gnn_get_node_feature(
    handle: *const GnnHandle,
    node_idx: c_uint,
    feature_idx: c_uint,
) -> c_float {
    if handle.is_null() {
        return 0.0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_node_feature(node_idx as usize, feature_idx as usize)
}

// ============================================================================
// Edge Features
// ============================================================================

/// Set features for an edge
///
/// @param handle GNN handle
/// @param edge_idx Edge index
/// @param features Array of feature values
/// @param features_len Length of features array
/// @returns GNN_OK on success
#[no_mangle]
pub extern "C" fn gnn_set_edge_features(
    handle: *mut GnnHandle,
    edge_idx: c_uint,
    features: *const c_float,
    features_len: c_uint,
) -> c_int {
    if handle.is_null() || features.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }
    if features_len > 4096 {
        return GNN_ERROR_INVALID_ARGUMENT;
    }

    let handle = unsafe { &mut *handle };
    let features_vec = unsafe { slice::from_raw_parts(features, features_len as usize).to_vec() };
    handle.inner.set_edge_features(edge_idx as usize, features_vec);
    GNN_OK
}

/// Get features for an edge
///
/// @param handle GNN handle
/// @param edge_idx Edge index
/// @param features_out Output buffer for features
/// @param features_len Length of output buffer
/// @returns Number of features copied, -1 on error
#[no_mangle]
pub extern "C" fn gnn_get_edge_features(
    handle: *const GnnHandle,
    edge_idx: c_uint,
    features_out: *mut c_float,
    features_len: c_uint,
) -> c_int {
    if handle.is_null() || features_out.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    match handle.inner.get_edge_features(edge_idx as usize) {
        Some(features) => {
            let copy_len = features.len().min(features_len as usize);
            let out_slice = unsafe { slice::from_raw_parts_mut(features_out, copy_len) };
            out_slice.copy_from_slice(&features[..copy_len]);
            copy_len as c_int
        }
        None => -1,
    }
}

// ============================================================================
// Training & Inference
// ============================================================================

/// Run prediction on the current graph
///
/// @param handle GNN handle
/// @param output Output buffer for predictions
/// @param output_len Length of output buffer
/// @returns Number of outputs, -1 on error
#[no_mangle]
pub extern "C" fn gnn_predict(
    handle: *mut GnnHandle,
    output: *mut c_float,
    output_len: c_uint,
) -> c_int {
    if handle.is_null() || output.is_null() {
        return -1;
    }
    if output_len > 1048576 {
        return -1;
    }

    let handle = unsafe { &mut *handle };
    match handle.inner.predict() {
        Ok(predictions) => {
            let copy_len = predictions.len().min(output_len as usize);
            let out_slice = unsafe { slice::from_raw_parts_mut(output, copy_len) };
            out_slice.copy_from_slice(&predictions[..copy_len]);
            copy_len as c_int
        }
        Err(_) => -1,
    }
}

/// Train on the current graph with target values
///
/// @param handle GNN handle
/// @param target Array of target values
/// @param target_len Length of target array
/// @param loss_out Pointer to store loss value
/// @returns GNN_OK on success, error code on failure
#[no_mangle]
pub extern "C" fn gnn_train(
    handle: *mut GnnHandle,
    target: *const c_float,
    target_len: c_uint,
    loss_out: *mut c_float,
) -> c_int {
    if handle.is_null() || target.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }
    if target_len > 1048576 {
        return GNN_ERROR_INVALID_ARGUMENT;
    }

    let handle = unsafe { &mut *handle };
    let target_slice = unsafe { slice::from_raw_parts(target, target_len as usize) };

    match handle.inner.train(target_slice) {
        Ok(loss) => {
            if !loss_out.is_null() {
                unsafe { *loss_out = loss; }
            }
            GNN_OK
        }
        Err(_) => GNN_ERROR_CUDA_ERROR,
    }
}

/// Train for multiple iterations
///
/// @param handle GNN handle
/// @param target Array of target values
/// @param target_len Length of target array
/// @param iterations Number of training iterations
/// @returns GNN_OK on success, error code on failure
#[no_mangle]
pub extern "C" fn gnn_train_multiple(
    handle: *mut GnnHandle,
    target: *const c_float,
    target_len: c_uint,
    iterations: c_uint,
) -> c_int {
    if handle.is_null() || target.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }
    if target_len > 1048576 {
        return GNN_ERROR_INVALID_ARGUMENT;
    }

    let handle = unsafe { &mut *handle };
    let target_slice = unsafe { slice::from_raw_parts(target, target_len as usize) };

    match handle.inner.train_multiple(target_slice, iterations as usize) {
        Ok(()) => GNN_OK,
        Err(_) => GNN_ERROR_CUDA_ERROR,
    }
}

// ============================================================================
// Hyperparameters
// ============================================================================

/// Set learning rate
#[no_mangle]
pub extern "C" fn gnn_set_learning_rate(handle: *mut GnnHandle, lr: c_float) -> c_int {
    if handle.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }
    if lr.is_nan() || lr.is_infinite() || lr < 0.0 {
        return GNN_ERROR_INVALID_ARGUMENT;
    }

    let handle = unsafe { &mut *handle };
    handle.inner.set_learning_rate(lr);
    GNN_OK
}

/// Get learning rate
#[no_mangle]
pub extern "C" fn gnn_get_learning_rate(handle: *const GnnHandle) -> c_float {
    if handle.is_null() {
        return 0.0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_learning_rate()
}

// ============================================================================
// Graph Info
// ============================================================================

/// Get number of nodes
#[no_mangle]
pub extern "C" fn gnn_get_num_nodes(handle: *const GnnHandle) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_num_nodes() as c_uint
}

/// Get number of edges
#[no_mangle]
pub extern "C" fn gnn_get_num_edges(handle: *const GnnHandle) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_num_edges() as c_uint
}

/// Check if graph is loaded
#[no_mangle]
pub extern "C" fn gnn_is_graph_loaded(handle: *const GnnHandle) -> c_int {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    if handle.inner.is_graph_loaded() { 1 } else { 0 }
}

/// Get feature size
#[no_mangle]
pub extern "C" fn gnn_get_feature_size(handle: *const GnnHandle) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_feature_size() as c_uint
}

/// Get hidden size
#[no_mangle]
pub extern "C" fn gnn_get_hidden_size(handle: *const GnnHandle) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_hidden_size() as c_uint
}

/// Get output size
#[no_mangle]
pub extern "C" fn gnn_get_output_size(handle: *const GnnHandle) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_output_size() as c_uint
}

/// Get number of message passing layers
#[no_mangle]
pub extern "C" fn gnn_get_num_message_passing_layers(handle: *const GnnHandle) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_num_message_passing_layers() as c_uint
}

/// Get in-degree of a node
#[no_mangle]
pub extern "C" fn gnn_get_in_degree(handle: *const GnnHandle, node_idx: c_uint) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_in_degree(node_idx as usize) as c_uint
}

/// Get out-degree of a node
#[no_mangle]
pub extern "C" fn gnn_get_out_degree(handle: *const GnnHandle, node_idx: c_uint) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_out_degree(node_idx as usize) as c_uint
}

/// Get neighbors of a node
///
/// @param handle GNN handle
/// @param node_idx Node index
/// @param neighbors_out Output buffer for neighbor indices
/// @param neighbors_len Length of output buffer
/// @returns Number of neighbors, -1 on error
#[no_mangle]
pub extern "C" fn gnn_get_neighbors(
    handle: *const GnnHandle,
    node_idx: c_uint,
    neighbors_out: *mut c_uint,
    neighbors_len: c_uint,
) -> c_int {
    if handle.is_null() || neighbors_out.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    match handle.inner.get_neighbors(node_idx as usize) {
        Some(neighbors) => {
            let copy_len = neighbors.len().min(neighbors_len as usize);
            let out_slice = unsafe { slice::from_raw_parts_mut(neighbors_out, copy_len) };
            for (i, &n) in neighbors.iter().take(copy_len).enumerate() {
                out_slice[i] = n as c_uint;
            }
            neighbors.len() as c_int
        }
        None => -1,
    }
}

/// Get graph embedding from last forward pass
///
/// @param handle GNN handle
/// @param embedding_out Output buffer for embedding
/// @param embedding_len Length of output buffer
/// @returns Number of values copied, -1 on error
#[no_mangle]
pub extern "C" fn gnn_get_graph_embedding(
    handle: *const GnnHandle,
    embedding_out: *mut c_float,
    embedding_len: c_uint,
) -> c_int {
    if handle.is_null() || embedding_out.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    let embedding = handle.inner.get_graph_embedding();
    let copy_len = embedding.len().min(embedding_len as usize);
    let out_slice = unsafe { slice::from_raw_parts_mut(embedding_out, copy_len) };
    out_slice.copy_from_slice(&embedding[..copy_len]);
    copy_len as c_int
}

// ============================================================================
// Masking & Dropout
// ============================================================================

/// Set node mask
#[no_mangle]
pub extern "C" fn gnn_set_node_mask(
    handle: *mut GnnHandle,
    node_idx: c_uint,
    value: c_int,
) -> c_int {
    if handle.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }

    let handle = unsafe { &mut *handle };
    handle.inner.set_node_mask(node_idx as usize, value != 0);
    GNN_OK
}

/// Get node mask
#[no_mangle]
pub extern "C" fn gnn_get_node_mask(handle: *const GnnHandle, node_idx: c_uint) -> c_int {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    if handle.inner.get_node_mask(node_idx as usize) { 1 } else { 0 }
}

/// Set edge mask
#[no_mangle]
pub extern "C" fn gnn_set_edge_mask(
    handle: *mut GnnHandle,
    edge_idx: c_uint,
    value: c_int,
) -> c_int {
    if handle.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }

    let handle = unsafe { &mut *handle };
    handle.inner.set_edge_mask(edge_idx as usize, value != 0);
    GNN_OK
}

/// Get edge mask
#[no_mangle]
pub extern "C" fn gnn_get_edge_mask(handle: *const GnnHandle, edge_idx: c_uint) -> c_int {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    if handle.inner.get_edge_mask(edge_idx as usize) { 1 } else { 0 }
}

/// Apply node dropout
#[no_mangle]
pub extern "C" fn gnn_apply_node_dropout(handle: *mut GnnHandle, rate: c_float) -> c_int {
    if handle.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }
    if rate.is_nan() || rate.is_infinite() || rate < 0.0 || rate > 1.0 {
        return GNN_ERROR_INVALID_ARGUMENT;
    }

    let handle = unsafe { &mut *handle };
    handle.inner.apply_node_dropout(rate);
    GNN_OK
}

/// Apply edge dropout
#[no_mangle]
pub extern "C" fn gnn_apply_edge_dropout(handle: *mut GnnHandle, rate: c_float) -> c_int {
    if handle.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }
    if rate.is_nan() || rate.is_infinite() || rate < 0.0 || rate > 1.0 {
        return GNN_ERROR_INVALID_ARGUMENT;
    }

    let handle = unsafe { &mut *handle };
    handle.inner.apply_edge_dropout(rate);
    GNN_OK
}

/// Get count of active (masked) nodes
#[no_mangle]
pub extern "C" fn gnn_get_masked_node_count(handle: *const GnnHandle) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_masked_node_count() as c_uint
}

/// Get count of active (masked) edges
#[no_mangle]
pub extern "C" fn gnn_get_masked_edge_count(handle: *const GnnHandle) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_masked_edge_count() as c_uint
}

// ============================================================================
// Analytics
// ============================================================================

/// Compute PageRank scores
///
/// @param handle GNN handle
/// @param damping Damping factor (typically 0.85)
/// @param iterations Number of iterations
/// @param scores_out Output buffer for scores
/// @param scores_len Length of output buffer
/// @returns Number of scores, -1 on error
#[no_mangle]
pub extern "C" fn gnn_compute_page_rank(
    handle: *const GnnHandle,
    damping: c_float,
    iterations: c_uint,
    scores_out: *mut c_float,
    scores_len: c_uint,
) -> c_int {
    if handle.is_null() || scores_out.is_null() {
        return -1;
    }
    if damping.is_nan() || damping.is_infinite() || damping < 0.0 || damping > 1.0 {
        return -1;
    }
    if scores_len > 1048576 {
        return -1;
    }

    let handle = unsafe { &*handle };
    let scores = handle.inner.compute_page_rank(damping, iterations as usize);
    let copy_len = scores.len().min(scores_len as usize);
    let out_slice = unsafe { slice::from_raw_parts_mut(scores_out, copy_len) };
    out_slice.copy_from_slice(&scores[..copy_len]);
    scores.len() as c_int
}

/// Get gradient flow information
#[no_mangle]
pub extern "C" fn gnn_get_gradient_flow(
    handle: *const GnnHandle,
    layer_idx: c_uint,
    info_out: *mut GnnGradientFlowInfo,
) -> c_int {
    if handle.is_null() || info_out.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }

    let handle = unsafe { &*handle };
    let info = handle.inner.get_gradient_flow(layer_idx as usize);
    unsafe {
        (*info_out).layer_idx = info.layer_idx as c_uint;
        (*info_out).mean_gradient = info.mean_gradient;
        (*info_out).max_gradient = info.max_gradient;
        (*info_out).min_gradient = info.min_gradient;
        (*info_out).gradient_norm = info.gradient_norm;
    }
    GNN_OK
}

/// Get total parameter count
#[no_mangle]
pub extern "C" fn gnn_get_parameter_count(handle: *const GnnHandle) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    handle.inner.get_parameter_count() as c_uint
}

/// Get architecture summary
///
/// @param handle GNN handle
/// @param buffer Output buffer for the string
/// @param buffer_len Length of output buffer
/// @returns Number of bytes written (excluding null terminator), -1 on error
#[no_mangle]
pub extern "C" fn gnn_get_architecture_summary(
    handle: *const GnnHandle,
    buffer: *mut c_char,
    buffer_len: c_uint,
) -> c_int {
    if handle.is_null() || buffer.is_null() || buffer_len == 0 {
        return -1;
    }

    let handle = unsafe { &*handle };
    let summary = handle.inner.get_architecture_summary();

    match CString::new(summary) {
        Ok(c_string) => {
            let bytes = c_string.as_bytes_with_nul();
            let copy_len = bytes.len().min(buffer_len as usize);
            let out_slice = unsafe { slice::from_raw_parts_mut(buffer as *mut u8, copy_len) };
            out_slice.copy_from_slice(&bytes[..copy_len]);
            if copy_len > 0 {
                out_slice[copy_len - 1] = 0; // Ensure null termination
            }
            (copy_len - 1) as c_int // Don't count null terminator
        }
        Err(_) => -1,
    }
}

/// Export graph to JSON
///
/// @param handle GNN handle
/// @param buffer Output buffer for the JSON string
/// @param buffer_len Length of output buffer
/// @returns Number of bytes written (excluding null terminator), -1 on error
#[no_mangle]
pub extern "C" fn gnn_export_graph_to_json(
    handle: *const GnnHandle,
    buffer: *mut c_char,
    buffer_len: c_uint,
) -> c_int {
    if handle.is_null() || buffer.is_null() || buffer_len == 0 {
        return -1;
    }

    let handle = unsafe { &*handle };
    let json = handle.inner.export_graph_to_json();

    match CString::new(json) {
        Ok(c_string) => {
            let bytes = c_string.as_bytes_with_nul();
            let copy_len = bytes.len().min(buffer_len as usize);
            let out_slice = unsafe { slice::from_raw_parts_mut(buffer as *mut u8, copy_len) };
            out_slice.copy_from_slice(&bytes[..copy_len]);
            if copy_len > 0 {
                out_slice[copy_len - 1] = 0; // Ensure null termination
            }
            (copy_len - 1) as c_int
        }
        Err(_) => -1,
    }
}

/// Detect the best available GPU backend
///
/// @returns Backend type: GNN_BACKEND_CUDA (0), GNN_BACKEND_OPENCL (1), or GNN_BACKEND_AUTO (2)
#[no_mangle]
pub extern "C" fn gnn_detect_backend() -> c_int {
    let backend = crate::detect_backend();
    match backend {
        GpuBackendType::Cuda => GNN_BACKEND_CUDA,
        GpuBackendType::OpenCL => GNN_BACKEND_OPENCL,
        GpuBackendType::Auto => GNN_BACKEND_AUTO,
    }
}

/// Get the backend type for a handle
///
/// @param handle GNN handle
/// @returns Backend type: GNN_BACKEND_CUDA (0), GNN_BACKEND_OPENCL (1), or GNN_BACKEND_AUTO (2)
#[no_mangle]
pub extern "C" fn gnn_get_backend_type(handle: *const GnnHandle) -> c_int {
    if handle.is_null() {
        return GNN_BACKEND_AUTO;
    }

    let handle = unsafe { &*handle };
    match handle.inner.get_backend_type() {
        GpuBackendType::Cuda => GNN_BACKEND_CUDA,
        GpuBackendType::OpenCL => GNN_BACKEND_OPENCL,
        GpuBackendType::Auto => GNN_BACKEND_AUTO,
    }
}

/// Get edge endpoints (source and target node indices)
///
/// @param handle GNN handle
/// @param edge_idx Edge index
/// @param source_out Pointer to store source node index
/// @param target_out Pointer to store target node index
/// @returns GNN_OK on success, GNN_ERROR_INVALID_ARGUMENT if edge not found
#[no_mangle]
pub extern "C" fn gnn_get_edge_endpoints(
    handle: *const GnnHandle,
    edge_idx: c_uint,
    source_out: *mut c_uint,
    target_out: *mut c_uint,
) -> c_int {
    if handle.is_null() || source_out.is_null() || target_out.is_null() {
        return GNN_ERROR_NULL_POINTER;
    }

    let handle = unsafe { &*handle };
    match handle.inner.get_edge_endpoints(edge_idx as usize) {
        Some((src, tgt)) => {
            unsafe {
                *source_out = src as c_uint;
                *target_out = tgt as c_uint;
            }
            GNN_OK
        }
        None => GNN_ERROR_INVALID_ARGUMENT,
    }
}

/// Get the active backend name
///
/// @param handle GNN handle
/// @param buffer Output buffer for the name string
/// @param buffer_len Length of output buffer
/// @returns Number of bytes written (excluding null), -1 on error
#[no_mangle]
pub extern "C" fn gnn_get_backend_name(
    handle: *const GnnHandle,
    buffer: *mut c_char,
    buffer_len: c_uint,
) -> c_int {
    if handle.is_null() || buffer.is_null() || buffer_len == 0 {
        return -1;
    }

    let handle = unsafe { &*handle };
    let name = handle.inner.get_backend_name();

    match CString::new(name) {
        Ok(c_string) => {
            let bytes = c_string.as_bytes_with_nul();
            let copy_len = bytes.len().min(buffer_len as usize);
            let out_slice = unsafe { slice::from_raw_parts_mut(buffer as *mut u8, copy_len) };
            out_slice.copy_from_slice(&bytes[..copy_len]);
            if copy_len > 0 {
                out_slice[copy_len - 1] = 0;
            }
            (copy_len - 1) as c_int
        }
        Err(_) => -1,
    }
}

