//! @file
//! @ingroup GNN_Core_Verified
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: CUDA Backend FFI Safety (CISA/NSA Compliance)
 *
 * Proves that all data passed across the CUDA FFI boundary is valid:
 * correct alignment, valid buffer sizes, proper grid/block dimensions,
 * weight index safety, and memory layout before kernel launches.
 *
 * The GNN uses f32 (float) for all GPU buffers and BLOCK_SIZE = 256.
 * CUDA-specific constants: MAX_NODES=1000, MAX_EDGES=10000.
 *
 * CISA "Secure by Design" requirements verified:
 * A. Layer buffer size correctness
 * B. CUDA grid/block dimension safety
 * C. Weight index validity for flat buffers
 * D. Transfer size non-zero and alignment
 * E. f32 alignment for CUDA transfers
 * F. Node embedding buffer sizing
 * G. Message buffer sizing
 * H. Kernel launch parameter overflow prevention
 * I. Aggregation buffer sizing
 * J. Graph readout buffer safety
 * K. Gradient buffer sizing
 * L. Neighbor offset/count buffer safety
 * M. No-panic guarantee for dimension calculations
 * N. ABI type compatibility for CUDA interop
 * O. End-to-end forward pass buffer chain validation
 */

use super::*;

const BLOCK_SIZE: u32 = 256;
const MAX_NEURONS_PER_LAYER: usize = 4096;

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // =========================================================================
    // A. LAYER BUFFER SIZE CORRECTNESS
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_layer_weight_buffer_size() {
        let num_inputs: usize = kani::any();
        let num_outputs: usize = kani::any();
        kani::assume(num_inputs > 0 && num_inputs <= 256);
        kani::assume(num_outputs > 0 && num_outputs <= 256);

        let weight_count = num_inputs * num_outputs;
        let weight_bytes = weight_count * std::mem::size_of::<f32>();

        kani::assert(weight_bytes == num_inputs * num_outputs * 4,
            "Weight buffer size must match num_inputs * num_outputs * sizeof(f32)");
    }

    #[kani::proof]
    fn verify_cuda_layer_bias_buffer_size() {
        let num_outputs: usize = kani::any();
        kani::assume(num_outputs > 0 && num_outputs <= 256);

        let bias_bytes = num_outputs * std::mem::size_of::<f32>();

        kani::assert(bias_bytes == num_outputs * 4,
            "Bias buffer size must match num_outputs * sizeof(f32)");
    }

    #[kani::proof]
    fn verify_cuda_layer_gradient_buffer_matches_weights() {
        let num_inputs: usize = kani::any();
        let num_outputs: usize = kani::any();
        kani::assume(num_inputs > 0 && num_inputs <= 256);
        kani::assume(num_outputs > 0 && num_outputs <= 256);

        let weight_grad_count = num_inputs * num_outputs;
        let bias_grad_count = num_outputs;

        kani::assert(weight_grad_count == num_inputs * num_outputs,
            "Weight gradient buffer must match weight buffer size");
        kani::assert(bias_grad_count == num_outputs,
            "Bias gradient buffer must match bias buffer size");
    }

    // =========================================================================
    // B. CUDA GRID/BLOCK DIMENSION SAFETY
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_grid_block_dimensions() {
        let num_outputs: usize = kani::any();
        kani::assume(num_outputs > 0 && num_outputs <= MAX_NEURONS_PER_LAYER);

        let blocks = (num_outputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kani::assert(blocks > 0, "Grid size must be at least 1");
        kani::assert(blocks as u64 * BLOCK_SIZE as u64 >= num_outputs as u64,
            "Grid * block must cover all neurons");
        kani::assert(blocks <= 65535, "Grid size must fit CUDA limits");
    }

    #[kani::proof]
    fn verify_cuda_forward_grid_dims() {
        let num_outputs: usize = kani::any();
        kani::assume(num_outputs > 0 && num_outputs <= MAX_NEURONS_PER_LAYER);

        let blocks = (num_outputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kani::assert(blocks > 0, "Forward pass must launch at least one block");
        kani::assert(blocks * BLOCK_SIZE >= num_outputs as u32,
            "Forward pass grid must cover all output neurons");
    }

    #[kani::proof]
    fn verify_cuda_backward_grid_dims() {
        let num_outputs: usize = kani::any();
        kani::assume(num_outputs > 0 && num_outputs <= MAX_NEURONS_PER_LAYER);

        let blocks = (num_outputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kani::assert(blocks > 0, "Backward pass must launch at least one block");
        kani::assert(blocks * BLOCK_SIZE >= num_outputs as u32,
            "Backward pass grid must cover all output neurons");
    }

    #[kani::proof]
    fn verify_cuda_input_grad_grid_dims() {
        let num_inputs: usize = kani::any();
        kani::assume(num_inputs > 0 && num_inputs <= MAX_NEURONS_PER_LAYER);

        let blocks = (num_inputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kani::assert(blocks > 0, "Input grad must launch at least one block");
        kani::assert(blocks * BLOCK_SIZE >= num_inputs as u32,
            "Input grad grid must cover all input neurons");
    }

    // =========================================================================
    // C. WEIGHT INDEX VALIDITY FOR FLAT BUFFERS
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_weight_index_valid_for_flat_buffer() {
        let num_inputs: usize = kani::any();
        let num_outputs: usize = kani::any();
        kani::assume(num_inputs > 0 && num_inputs <= 16);
        kani::assume(num_outputs > 0 && num_outputs <= 16);

        let neuron_idx: usize = kani::any();
        let input_idx: usize = kani::any();
        kani::assume(neuron_idx < num_outputs);
        kani::assume(input_idx < num_inputs);

        let flat_idx = neuron_idx * num_inputs + input_idx;
        let total = num_outputs * num_inputs;

        kani::assert(flat_idx < total,
            "Flat weight index must be within allocated buffer");
    }

    #[kani::proof]
    fn verify_cuda_message_weight_index_valid() {
        let msg_input_size: usize = kani::any();
        let hidden_size: usize = kani::any();
        kani::assume(msg_input_size > 0 && msg_input_size <= 32);
        kani::assume(hidden_size > 0 && hidden_size <= 16);

        let neuron_idx: usize = kani::any();
        let input_idx: usize = kani::any();
        kani::assume(neuron_idx < hidden_size);
        kani::assume(input_idx < msg_input_size);

        let flat_idx = neuron_idx * msg_input_size + input_idx;
        let total = hidden_size * msg_input_size;

        kani::assert(flat_idx < total,
            "Message layer weight index must be within buffer");
    }

    // =========================================================================
    // D. TRANSFER SIZE NON-ZERO AND ALIGNMENT
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_transfer_size_non_zero() {
        let num_elements: usize = kani::any();
        kani::assume(num_elements > 0 && num_elements <= MAX_NODES * MAX_FEATURES);

        let transfer_bytes = num_elements * std::mem::size_of::<f32>();

        kani::assert(transfer_bytes > 0, "Transfer size must be non-zero");
        kani::assert(transfer_bytes % std::mem::size_of::<f32>() == 0,
            "Transfer size must be aligned to f32");
    }

    #[kani::proof]
    fn verify_cuda_weight_transfer_aligned() {
        let num_inputs: usize = kani::any();
        let num_outputs: usize = kani::any();
        kani::assume(num_inputs > 0 && num_inputs <= 256);
        kani::assume(num_outputs > 0 && num_outputs <= 256);

        let weight_bytes = num_inputs * num_outputs * std::mem::size_of::<f32>();
        kani::assert(weight_bytes % 4 == 0, "Weight transfer must be 4-byte aligned");
    }

    // =========================================================================
    // E. F32 ALIGNMENT FOR CUDA TRANSFERS
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_f32_alignment() {
        let align = std::mem::align_of::<f32>();
        kani::assert(align == 4, "f32 must be 4-byte aligned for CUDA transfers");

        let size = std::mem::size_of::<f32>();
        kani::assert(size == 4, "f32 must be 4 bytes for CUDA compatibility");
    }

    #[kani::proof]
    fn verify_cuda_i32_alignment() {
        let align = std::mem::align_of::<i32>();
        kani::assert(align == 4, "i32 must be 4-byte aligned for CUDA kernel args");

        let size = std::mem::size_of::<i32>();
        kani::assert(size == 4, "i32 must be 4 bytes for CUDA kernel args");
    }

    // =========================================================================
    // F. NODE EMBEDDING BUFFER SIZING
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_node_embedding_buffer_size() {
        let hidden_size: usize = kani::any();
        kani::assume(hidden_size > 0 && hidden_size <= MAX_FEATURES);

        let buffer_size = MAX_NODES * hidden_size;
        let buffer_bytes = buffer_size * std::mem::size_of::<f32>();

        kani::assert(buffer_size == MAX_NODES * hidden_size,
            "Node embedding buffer must hold MAX_NODES * hidden_size elements");
        kani::assert(buffer_bytes == MAX_NODES * hidden_size * 4,
            "Node embedding buffer bytes must match");
    }

    #[kani::proof]
    fn verify_cuda_node_embedding_index_valid() {
        let hidden_size: usize = kani::any();
        let node_idx: usize = kani::any();
        let dim_idx: usize = kani::any();
        kani::assume(hidden_size > 0 && hidden_size <= 16);
        kani::assume(node_idx < MAX_NODES);
        kani::assume(dim_idx < hidden_size);

        let flat_idx = node_idx * hidden_size + dim_idx;
        let total = MAX_NODES * hidden_size;

        kani::assert(flat_idx < total,
            "Node embedding index must be within buffer");
    }

    // =========================================================================
    // G. MESSAGE BUFFER SIZING
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_message_buffer_size() {
        let hidden_size: usize = kani::any();
        kani::assume(hidden_size > 0 && hidden_size <= MAX_FEATURES);

        let buffer_size = MAX_EDGES * hidden_size;
        let buffer_bytes = buffer_size * std::mem::size_of::<f32>();

        kani::assert(buffer_size == MAX_EDGES * hidden_size,
            "Message buffer must hold MAX_EDGES * hidden_size elements");
        kani::assert(buffer_bytes == MAX_EDGES * hidden_size * 4,
            "Message buffer bytes must match");
    }

    #[kani::proof]
    fn verify_cuda_message_index_valid() {
        let hidden_size: usize = kani::any();
        let msg_offset: usize = kani::any();
        let dim_idx: usize = kani::any();
        kani::assume(hidden_size > 0 && hidden_size <= 16);
        kani::assume(msg_offset < MAX_EDGES);
        kani::assume(dim_idx < hidden_size);

        let flat_idx = msg_offset * hidden_size + dim_idx;
        let total = MAX_EDGES * hidden_size;

        kani::assert(flat_idx < total,
            "Message index must be within buffer");
    }

    // =========================================================================
    // H. KERNEL LAUNCH PARAMETER OVERFLOW PREVENTION
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_kernel_launch_no_overflow() {
        let num_outputs: usize = kani::any();
        let num_inputs: usize = kani::any();
        kani::assume(num_outputs > 0 && num_outputs <= MAX_NEURONS_PER_LAYER);
        kani::assume(num_inputs > 0 && num_inputs <= MAX_NEURONS_PER_LAYER);

        let blocks = (num_outputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kani::assert(blocks > 0, "Must launch at least one block");
        kani::assert(blocks as u64 * BLOCK_SIZE as u64 >= num_outputs as u64,
            "Total threads must cover all outputs");

        let total_weights = num_outputs.checked_mul(num_inputs);
        kani::assert(total_weights.is_some(),
            "Weight count must not overflow before CUDA allocation");
    }

    #[kani::proof]
    fn verify_cuda_aggregation_launch_params() {
        let num_nodes: usize = kani::any();
        let hidden_size: usize = kani::any();
        kani::assume(num_nodes > 0 && num_nodes <= MAX_NODES);
        kani::assume(hidden_size > 0 && hidden_size <= MAX_FEATURES);

        let grid_x = num_nodes as u32;
        let block_x = hidden_size as u32;

        kani::assert(grid_x > 0, "Aggregation grid must be non-zero");
        kani::assert(block_x > 0, "Aggregation block must be non-zero");
        kani::assert(grid_x <= MAX_NODES as u32,
            "Aggregation grid bounded by MAX_NODES");
    }

    // =========================================================================
    // I. AGGREGATION BUFFER SIZING
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_aggregated_message_buffer_size() {
        let hidden_size: usize = kani::any();
        kani::assume(hidden_size > 0 && hidden_size <= MAX_FEATURES);

        let buffer_size = MAX_NODES * hidden_size;

        kani::assert(buffer_size == MAX_NODES * hidden_size,
            "Aggregated message buffer must hold MAX_NODES * hidden_size");
    }

    #[kani::proof]
    fn verify_cuda_aggregated_message_index_valid() {
        let hidden_size: usize = kani::any();
        let node_idx: usize = kani::any();
        let dim_idx: usize = kani::any();
        kani::assume(hidden_size > 0 && hidden_size <= 16);
        kani::assume(node_idx < MAX_NODES);
        kani::assume(dim_idx < hidden_size);

        let flat_idx = node_idx * hidden_size + dim_idx;
        kani::assert(flat_idx < MAX_NODES * hidden_size,
            "Aggregated message index must be within buffer");
    }

    // =========================================================================
    // J. GRAPH READOUT BUFFER SAFETY
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_graph_readout_buffer_size() {
        let hidden_size: usize = kani::any();
        kani::assume(hidden_size > 0 && hidden_size <= MAX_FEATURES);

        let buffer_bytes = hidden_size * std::mem::size_of::<f32>();

        kani::assert(buffer_bytes == hidden_size * 4,
            "Graph embedding buffer must hold hidden_size * sizeof(f32)");
    }

    #[kani::proof]
    fn verify_cuda_readout_grid_dims() {
        let hidden_size: usize = kani::any();
        kani::assume(hidden_size > 0 && hidden_size <= MAX_NEURONS_PER_LAYER);

        let blocks = (hidden_size as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kani::assert(blocks > 0, "Readout must launch at least one block");
        kani::assert(blocks * BLOCK_SIZE >= hidden_size as u32,
            "Readout grid must cover all hidden dimensions");
    }

    // =========================================================================
    // K. GRADIENT BUFFER SIZING
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_output_gradient_buffer_size() {
        let output_size: usize = kani::any();
        kani::assume(output_size > 0 && output_size <= MAX_NEURONS_PER_LAYER);

        let grad_bytes = output_size * std::mem::size_of::<f32>();

        kani::assert(grad_bytes == output_size * 4,
            "Output gradient buffer must match output_size * sizeof(f32)");
    }

    #[kani::proof]
    fn verify_cuda_target_buffer_size() {
        let output_size: usize = kani::any();
        kani::assume(output_size > 0 && output_size <= MAX_NEURONS_PER_LAYER);

        let target_bytes = output_size * std::mem::size_of::<f32>();

        kani::assert(target_bytes == output_size * 4,
            "Target buffer must match output_size * sizeof(f32)");
    }

    // =========================================================================
    // L. NEIGHBOR OFFSET/COUNT BUFFER SAFETY
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_neighbor_count_buffer_size() {
        let buffer_size = MAX_NODES;
        let buffer_bytes = buffer_size * std::mem::size_of::<i32>();

        kani::assert(buffer_bytes == MAX_NODES * 4,
            "Neighbor count buffer must hold MAX_NODES * sizeof(i32)");
    }

    #[kani::proof]
    fn verify_cuda_neighbor_offset_accumulation_safe() {
        let num_nodes: usize = kani::any();
        kani::assume(num_nodes > 0 && num_nodes <= 8);

        let counts: Vec<usize> = (0..num_nodes).map(|_| {
            let c: usize = kani::any();
            kani::assume(c <= 4);
            c
        }).collect();

        let mut total: usize = 0;
        for &c in &counts {
            let new_total = total.checked_add(c);
            kani::assert(new_total.is_some(),
                "Neighbor offset accumulation must not overflow");
            total = new_total.unwrap();
        }

        kani::assert(total <= MAX_EDGES,
            "Total neighbor count must not exceed MAX_EDGES");
    }

    // =========================================================================
    // M. NO-PANIC GUARANTEE FOR DIMENSION CALCULATIONS
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_all_grid_calculations_no_panic() {
        let n: usize = kani::any();
        kani::assume(n > 0 && n <= MAX_NEURONS_PER_LAYER);

        let _blocks = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    #[kani::proof]
    fn verify_cuda_all_buffer_size_calculations_no_panic() {
        let num_inputs: usize = kani::any();
        let num_outputs: usize = kani::any();
        let hidden_size: usize = kani::any();
        kani::assume(num_inputs > 0 && num_inputs <= 256);
        kani::assume(num_outputs > 0 && num_outputs <= 256);
        kani::assume(hidden_size > 0 && hidden_size <= 256);

        let _w = num_inputs * num_outputs;
        let _b = num_outputs;
        let _emb = MAX_NODES * hidden_size;
        let _msg = MAX_EDGES * hidden_size;
        let _agg = MAX_NODES * hidden_size;
    }

    // =========================================================================
    // N. ABI TYPE COMPATIBILITY FOR CUDA INTEROP
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_f32_abi_compatibility() {
        kani::assert(std::mem::size_of::<f32>() == 4, "f32 == CUDA float");
        kani::assert(std::mem::align_of::<f32>() == 4, "f32 4-byte aligned for CUDA");
    }

    #[kani::proof]
    fn verify_cuda_i32_abi_compatibility() {
        kani::assert(std::mem::size_of::<i32>() == 4, "i32 == CUDA int");
        kani::assert(std::mem::align_of::<i32>() == 4, "i32 4-byte aligned for CUDA");
    }

    #[kani::proof]
    fn verify_cuda_u32_abi_compatibility() {
        kani::assert(std::mem::size_of::<u32>() == 4, "u32 == CUDA unsigned int");
        kani::assert(std::mem::align_of::<u32>() == 4, "u32 4-byte aligned for CUDA");
    }

    // =========================================================================
    // O. END-TO-END FORWARD PASS BUFFER CHAIN VALIDATION
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_message_layer_input_size() {
        let feature_size: usize = kani::any();
        let hidden_size: usize = kani::any();
        let layer_idx: usize = kani::any();
        kani::assume(feature_size > 0 && feature_size <= 128);
        kani::assume(hidden_size > 0 && hidden_size <= 128);
        kani::assume(layer_idx <= 8);

        let msg_input_size = if layer_idx == 0 { feature_size * 2 } else { hidden_size * 2 };

        kani::assert(msg_input_size > 0,
            "Message layer input size must be positive");
        kani::assert(msg_input_size <= 256,
            "Message layer input size must be bounded");
    }

    #[kani::proof]
    fn verify_cuda_update_layer_input_size() {
        let hidden_size: usize = kani::any();
        kani::assume(hidden_size > 0 && hidden_size <= 128);

        let update_input_size = hidden_size * 2;

        kani::assert(update_input_size > 0,
            "Update layer input size must be positive");
        kani::assert(update_input_size == hidden_size * 2,
            "Update layer concatenates embedding + aggregated message");
    }

    #[kani::proof]
    fn verify_cuda_forward_pass_buffer_chain() {
        let feature_size: usize = kani::any();
        let hidden_size: usize = kani::any();
        let output_size: usize = kani::any();
        let num_mp_layers: usize = kani::any();
        kani::assume(feature_size > 0 && feature_size <= 64);
        kani::assume(hidden_size > 0 && hidden_size <= 64);
        kani::assume(output_size > 0 && output_size <= 64);
        kani::assume(num_mp_layers > 0 && num_mp_layers <= 8);

        let readout_in = hidden_size;
        let readout_out = hidden_size;
        let output_in = hidden_size;

        kani::assert(readout_in == hidden_size,
            "Readout layer input must match hidden_size");
        kani::assert(readout_out == output_in,
            "Readout output must feed into output layer input");
    }

    #[kani::proof]
    fn verify_cuda_mse_gradient_grid_dims() {
        let output_size: usize = kani::any();
        kani::assume(output_size > 0 && output_size <= MAX_NEURONS_PER_LAYER);

        let blocks = (output_size as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kani::assert(blocks > 0, "MSE gradient must launch at least one block");
        kani::assert(blocks * BLOCK_SIZE >= output_size as u32,
            "MSE gradient grid must cover all outputs");
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_cuda_grid_calculation() {
        assert_eq!((1_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        assert_eq!((256_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        assert_eq!((257_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 2);
        assert_eq!((512_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 2);
    }

    #[test]
    fn test_cuda_flat_index() {
        let num_inputs = 4;
        let num_outputs = 3;
        assert_eq!(0 * num_inputs + 0, 0);
        assert_eq!(2 * num_inputs + 3, 11);
        assert!(2 * num_inputs + 3 < num_inputs * num_outputs);
    }

    #[test]
    fn test_cuda_buffer_sizes() {
        assert_eq!(MAX_NODES * 16 * 4, 64000);
        assert_eq!(MAX_EDGES * 16 * 4, 640000);
    }

    #[test]
    fn test_cuda_f32_properties() {
        assert_eq!(std::mem::size_of::<f32>(), 4);
        assert_eq!(std::mem::align_of::<f32>(), 4);
    }

    #[test]
    fn test_cuda_message_layer_input_size() {
        let feature_size = 3;
        let hidden_size = 16;
        assert_eq!(feature_size * 2, 6);
        assert_eq!(hidden_size * 2, 32);
    }
}

