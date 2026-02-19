//! @file
//! @ingroup GNN_Core_Verified
use std::ptr;

use crate::ffi::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnn_create_with_backend_opencl() {
        let handle = gnn_create_with_backend(3, 16, 2, 2, GNN_BACKEND_OPENCL);
        if !handle.is_null() {
            unsafe { gnn_free(handle); }
        }
    }

    #[test]
    fn test_gnn_free_null() {
        gnn_free(ptr::null_mut());
    }

    #[test]
    fn test_gnn_load_null_filename() {
        let handle = gnn_load(ptr::null());
        assert!(handle.is_null());
    }

    #[test]
    fn test_gnn_read_model_header_null() {
        let result = gnn_read_model_header(ptr::null(), ptr::null_mut());
        assert_eq!(result, GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_save_model_null() {
        assert_eq!(gnn_save_model(ptr::null(), ptr::null()), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_load_model_null() {
        assert_eq!(gnn_load_model(ptr::null_mut(), ptr::null()), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_create_empty_graph_null() {
        assert_eq!(gnn_create_empty_graph(ptr::null_mut(), 5, 3), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_add_edge_null() {
        assert_eq!(gnn_add_edge(ptr::null_mut(), 0, 1, ptr::null(), 0), -1);
    }

    #[test]
    fn test_gnn_remove_edge_null() {
        assert_eq!(gnn_remove_edge(ptr::null_mut(), 0), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_has_edge_null() {
        assert_eq!(gnn_has_edge(ptr::null(), 0, 1), -1);
    }

    #[test]
    fn test_gnn_find_edge_index_null() {
        assert_eq!(gnn_find_edge_index(ptr::null(), 0, 1), -1);
    }

    #[test]
    fn test_gnn_rebuild_adjacency_list_null() {
        assert_eq!(gnn_rebuild_adjacency_list(ptr::null_mut()), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_set_node_features_null() {
        assert_eq!(gnn_set_node_features(ptr::null_mut(), 0, ptr::null(), 0), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_get_node_features_null() {
        assert_eq!(gnn_get_node_features(ptr::null(), 0, ptr::null_mut(), 0), -1);
    }

    #[test]
    fn test_gnn_set_node_feature_null() {
        assert_eq!(gnn_set_node_feature(ptr::null_mut(), 0, 0, 1.0), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_get_node_feature_null() {
        assert_eq!(gnn_get_node_feature(ptr::null(), 0, 0), 0.0);
    }

    #[test]
    fn test_gnn_set_edge_features_null() {
        assert_eq!(gnn_set_edge_features(ptr::null_mut(), 0, ptr::null(), 0), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_get_edge_features_null() {
        assert_eq!(gnn_get_edge_features(ptr::null(), 0, ptr::null_mut(), 0), -1);
    }

    #[test]
    fn test_gnn_predict_null() {
        assert_eq!(gnn_predict(ptr::null_mut(), ptr::null_mut(), 0), -1);
    }

    #[test]
    fn test_gnn_train_null() {
        assert_eq!(gnn_train(ptr::null_mut(), ptr::null(), 0, ptr::null_mut()), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_train_multiple_null() {
        assert_eq!(gnn_train_multiple(ptr::null_mut(), ptr::null(), 0, 10), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_set_learning_rate_null() {
        assert_eq!(gnn_set_learning_rate(ptr::null_mut(), 0.01), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_get_learning_rate_null() {
        assert_eq!(gnn_get_learning_rate(ptr::null()), 0.0);
    }

    #[test]
    fn test_gnn_get_num_nodes_null() {
        assert_eq!(gnn_get_num_nodes(ptr::null()), 0);
    }

    #[test]
    fn test_gnn_get_num_edges_null() {
        assert_eq!(gnn_get_num_edges(ptr::null()), 0);
    }

    #[test]
    fn test_gnn_is_graph_loaded_null() {
        assert_eq!(gnn_is_graph_loaded(ptr::null()), 0);
    }

    #[test]
    fn test_gnn_get_feature_size_null() {
        assert_eq!(gnn_get_feature_size(ptr::null()), 0);
    }

    #[test]
    fn test_gnn_get_hidden_size_null() {
        assert_eq!(gnn_get_hidden_size(ptr::null()), 0);
    }

    #[test]
    fn test_gnn_get_output_size_null() {
        assert_eq!(gnn_get_output_size(ptr::null()), 0);
    }

    #[test]
    fn test_gnn_get_num_message_passing_layers_null() {
        assert_eq!(gnn_get_num_message_passing_layers(ptr::null()), 0);
    }

    #[test]
    fn test_gnn_get_in_degree_null() {
        assert_eq!(gnn_get_in_degree(ptr::null(), 0), 0);
    }

    #[test]
    fn test_gnn_get_out_degree_null() {
        assert_eq!(gnn_get_out_degree(ptr::null(), 0), 0);
    }

    #[test]
    fn test_gnn_get_neighbors_null() {
        assert_eq!(gnn_get_neighbors(ptr::null(), 0, ptr::null_mut(), 0), -1);
    }

    #[test]
    fn test_gnn_get_graph_embedding_null() {
        assert_eq!(gnn_get_graph_embedding(ptr::null(), ptr::null_mut(), 0), -1);
    }

    #[test]
    fn test_gnn_set_node_mask_null() {
        assert_eq!(gnn_set_node_mask(ptr::null_mut(), 0, 1), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_get_node_mask_null() {
        assert_eq!(gnn_get_node_mask(ptr::null(), 0), 0);
    }

    #[test]
    fn test_gnn_set_edge_mask_null() {
        assert_eq!(gnn_set_edge_mask(ptr::null_mut(), 0, 1), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_get_edge_mask_null() {
        assert_eq!(gnn_get_edge_mask(ptr::null(), 0), 0);
    }

    #[test]
    fn test_gnn_apply_node_dropout_null() {
        assert_eq!(gnn_apply_node_dropout(ptr::null_mut(), 0.5), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_apply_edge_dropout_null() {
        assert_eq!(gnn_apply_edge_dropout(ptr::null_mut(), 0.5), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_get_masked_node_count_null() {
        assert_eq!(gnn_get_masked_node_count(ptr::null()), 0);
    }

    #[test]
    fn test_gnn_get_masked_edge_count_null() {
        assert_eq!(gnn_get_masked_edge_count(ptr::null()), 0);
    }

    #[test]
    fn test_gnn_compute_page_rank_null() {
        assert_eq!(gnn_compute_page_rank(ptr::null(), 0.85, 10, ptr::null_mut(), 0), -1);
    }

    #[test]
    fn test_gnn_get_gradient_flow_null() {
        assert_eq!(gnn_get_gradient_flow(ptr::null(), 0, ptr::null_mut()), GNN_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_gnn_get_parameter_count_null() {
        assert_eq!(gnn_get_parameter_count(ptr::null()), 0);
    }

    #[test]
    fn test_gnn_get_architecture_summary_null() {
        assert_eq!(gnn_get_architecture_summary(ptr::null(), ptr::null_mut(), 0), -1);
    }

    #[test]
    fn test_gnn_export_graph_to_json_null() {
        assert_eq!(gnn_export_graph_to_json(ptr::null(), ptr::null_mut(), 0), -1);
    }

    #[test]
    fn test_gnn_get_backend_name_null() {
        assert_eq!(gnn_get_backend_name(ptr::null(), ptr::null_mut(), 0), -1);
    }

    #[test]
    fn test_backend_constants() {
        assert_eq!(GNN_BACKEND_CUDA, 0);
        assert_eq!(GNN_BACKEND_OPENCL, 1);
        assert_eq!(GNN_BACKEND_AUTO, 2);
    }

    #[test]
    fn test_error_constants() {
        assert_eq!(GNN_OK, 0);
        assert_eq!(GNN_ERROR_NULL_POINTER, -1);
        assert_eq!(GNN_ERROR_INVALID_ARGUMENT, -2);
        assert_eq!(GNN_ERROR_CUDA_ERROR, -3);
        assert_eq!(GNN_ERROR_IO_ERROR, -4);
        assert_eq!(GNN_ERROR_OPENCL_ERROR, -5);
        assert_eq!(GNN_ERROR_UNKNOWN, -99);
    }
}

