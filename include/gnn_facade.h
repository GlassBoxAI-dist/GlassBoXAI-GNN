/**
 * @file gnn_facade.h
 * @brief GlassBoxAI GNN - CUDA/OpenCL-accelerated Graph Neural Network C API
 *
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * This header provides C bindings for the GPU-accelerated Graph Neural Network
 * library with Facade pattern. Supports CUDA and OpenCL backends.
 *
 * @example
 * @code
 * #include "gnn_facade.h"
 *
 * int main() {
 *     // Create a GNN with 3 input features, 16 hidden, 2 output, 2 MP layers
 *     GnnHandle* gnn = gnn_create(3, 16, 2, 2);
 *     if (!gnn) {
 *         fprintf(stderr, "Failed to create GNN\n");
 *         return 1;
 *     }
 *
 *     // Create a graph
 *     gnn_create_empty_graph(gnn, 5, 3);
 *
 *     // Add edges
 *     gnn_add_edge(gnn, 0, 1, NULL, 0);
 *     gnn_add_edge(gnn, 1, 2, NULL, 0);
 *
 *     // Set node features
 *     float features[] = {1.0f, 0.5f, 0.2f};
 *     gnn_set_node_features(gnn, 0, features, 3);
 *
 *     // Make predictions
 *     float output[2];
 *     int num_outputs = gnn_predict(gnn, output, 2);
 *     printf("Prediction: [%f, %f]\n", output[0], output[1]);
 *
 *     // Train
 *     float target[] = {0.5f, 0.5f};
 *     float loss;
 *     gnn_train(gnn, target, 2, &loss);
 *     printf("Loss: %f\n", loss);
 *
 *     // Save model
 *     gnn_save_model(gnn, "model.bin");
 *
 *     // Cleanup
 *     gnn_free(gnn);
 *     return 0;
 * }
 * @endcode
 */

#ifndef GNN_FACADE_H
#define GNN_FACADE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ============================================================================
 * Types
 * ============================================================================ */

/**
 * @brief Opaque handle to a GNN Facade instance
 */
typedef struct GnnHandle GnnHandle;

/**
 * @brief Gradient flow information for a layer
 */
typedef struct {
    unsigned int layer_idx;      /**< Layer index */
    float mean_gradient;         /**< Mean gradient value */
    float max_gradient;          /**< Maximum gradient value */
    float min_gradient;          /**< Minimum gradient value */
    float gradient_norm;         /**< L2 norm of gradients */
} GnnGradientFlowInfo;

/**
 * @brief Model header information
 */
typedef struct {
    unsigned int feature_size;   /**< Size of input node features */
    unsigned int hidden_size;    /**< Size of hidden layers */
    unsigned int output_size;    /**< Size of output predictions */
    unsigned int mp_layers;      /**< Number of message passing layers */
    float learning_rate;         /**< Current learning rate */
} GnnModelHeader;

/* ============================================================================
 * Error Codes
 * ============================================================================ */

#define GNN_OK                   0   /**< Success */
#define GNN_ERROR_NULL_POINTER  -1   /**< Null pointer argument */
#define GNN_ERROR_INVALID_ARG   -2   /**< Invalid argument */
#define GNN_ERROR_CUDA          -3   /**< CUDA error */
#define GNN_ERROR_IO            -4   /**< I/O error */
#define GNN_ERROR_OPENCL        -5   /**< OpenCL error */
#define GNN_ERROR_UNKNOWN      -99   /**< Unknown error */

/* ============================================================================
 * Backend Selection
 * ============================================================================ */

#define GNN_BACKEND_CUDA         0   /**< NVIDIA CUDA backend */
#define GNN_BACKEND_OPENCL       1   /**< OpenCL backend (AMD, Intel, NVIDIA) */
#define GNN_BACKEND_AUTO         2   /**< Auto-detect best available backend */

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

/**
 * @brief Create a new GNN Facade with auto-detected backend
 *
 * @param feature_size Size of input node features
 * @param hidden_size Size of hidden layers
 * @param output_size Size of output predictions
 * @param num_mp_layers Number of message passing layers
 * @return Handle to the GNN, or NULL on error
 */
GnnHandle* gnn_create(
    unsigned int feature_size,
    unsigned int hidden_size,
    unsigned int output_size,
    unsigned int num_mp_layers
);

/**
 * @brief Create a new GNN Facade with a specific backend
 *
 * @param feature_size Size of input node features
 * @param hidden_size Size of hidden layers
 * @param output_size Size of output predictions
 * @param num_mp_layers Number of message passing layers
 * @param backend Backend type: GNN_BACKEND_CUDA, GNN_BACKEND_OPENCL, or GNN_BACKEND_AUTO
 * @return Handle to the GNN, or NULL on error
 */
GnnHandle* gnn_create_with_backend(
    unsigned int feature_size,
    unsigned int hidden_size,
    unsigned int output_size,
    unsigned int num_mp_layers,
    int backend
);

/**
 * @brief Load a GNN from a saved model file
 *
 * @param filename Path to the model file (null-terminated)
 * @return Handle to the GNN, or NULL on error
 */
GnnHandle* gnn_load(const char* filename);

/**
 * @brief Free a GNN handle
 *
 * @param handle Handle to free (safe to pass NULL)
 */
void gnn_free(GnnHandle* handle);

/**
 * @brief Read model header without loading full model
 *
 * @param filename Path to the model file
 * @param header Pointer to header struct to fill
 * @return GNN_OK on success, error code on failure
 */
int gnn_read_model_header(const char* filename, GnnModelHeader* header);

/* ============================================================================
 * Model I/O Functions
 * ============================================================================ */

/**
 * @brief Save model to file
 *
 * @param handle GNN handle
 * @param filename Path to save the model
 * @return GNN_OK on success, error code on failure
 */
int gnn_save_model(const GnnHandle* handle, const char* filename);

/**
 * @brief Load model from file
 *
 * @param handle GNN handle
 * @param filename Path to the model file
 * @return GNN_OK on success, error code on failure
 */
int gnn_load_model(GnnHandle* handle, const char* filename);

/* ============================================================================
 * Graph Operations
 * ============================================================================ */

/**
 * @brief Create an empty graph
 *
 * @param handle GNN handle
 * @param num_nodes Number of nodes
 * @param feature_size Size of node features
 * @return GNN_OK on success
 */
int gnn_create_empty_graph(
    GnnHandle* handle,
    unsigned int num_nodes,
    unsigned int feature_size
);

/**
 * @brief Add an edge to the graph
 *
 * @param handle GNN handle
 * @param source Source node index
 * @param target Target node index
 * @param features Array of edge features (can be NULL)
 * @param features_len Length of features array
 * @return Edge index on success, -1 on error
 */
int gnn_add_edge(
    GnnHandle* handle,
    unsigned int source,
    unsigned int target,
    const float* features,
    unsigned int features_len
);

/**
 * @brief Remove an edge by index
 *
 * @param handle GNN handle
 * @param edge_idx Edge index to remove
 * @return GNN_OK on success
 */
int gnn_remove_edge(GnnHandle* handle, unsigned int edge_idx);

/**
 * @brief Check if an edge exists
 *
 * @param handle GNN handle
 * @param source Source node index
 * @param target Target node index
 * @return 1 if edge exists, 0 if not, -1 on error
 */
int gnn_has_edge(const GnnHandle* handle, unsigned int source, unsigned int target);

/**
 * @brief Find edge index between two nodes
 *
 * @param handle GNN handle
 * @param source Source node index
 * @param target Target node index
 * @return Edge index if found, -1 if not found or error
 */
int gnn_find_edge_index(const GnnHandle* handle, unsigned int source, unsigned int target);

/**
 * @brief Rebuild adjacency list from edges
 *
 * @param handle GNN handle
 * @return GNN_OK on success
 */
int gnn_rebuild_adjacency_list(GnnHandle* handle);

/* ============================================================================
 * Node Features
 * ============================================================================ */

/**
 * @brief Set all features for a node
 *
 * @param handle GNN handle
 * @param node_idx Node index
 * @param features Array of feature values
 * @param features_len Length of features array
 * @return GNN_OK on success
 */
int gnn_set_node_features(
    GnnHandle* handle,
    unsigned int node_idx,
    const float* features,
    unsigned int features_len
);

/**
 * @brief Get all features for a node
 *
 * @param handle GNN handle
 * @param node_idx Node index
 * @param features_out Output buffer for features
 * @param features_len Length of output buffer
 * @return Number of features copied, -1 on error
 */
int gnn_get_node_features(
    const GnnHandle* handle,
    unsigned int node_idx,
    float* features_out,
    unsigned int features_len
);

/**
 * @brief Set a single feature value for a node
 *
 * @param handle GNN handle
 * @param node_idx Node index
 * @param feature_idx Feature index
 * @param value Feature value
 * @return GNN_OK on success
 */
int gnn_set_node_feature(
    GnnHandle* handle,
    unsigned int node_idx,
    unsigned int feature_idx,
    float value
);

/**
 * @brief Get a single feature value for a node
 *
 * @param handle GNN handle
 * @param node_idx Node index
 * @param feature_idx Feature index
 * @return Feature value (0.0 if not found)
 */
float gnn_get_node_feature(
    const GnnHandle* handle,
    unsigned int node_idx,
    unsigned int feature_idx
);

/* ============================================================================
 * Edge Features
 * ============================================================================ */

/**
 * @brief Set features for an edge
 *
 * @param handle GNN handle
 * @param edge_idx Edge index
 * @param features Array of feature values
 * @param features_len Length of features array
 * @return GNN_OK on success
 */
int gnn_set_edge_features(
    GnnHandle* handle,
    unsigned int edge_idx,
    const float* features,
    unsigned int features_len
);

/**
 * @brief Get features for an edge
 *
 * @param handle GNN handle
 * @param edge_idx Edge index
 * @param features_out Output buffer for features
 * @param features_len Length of output buffer
 * @return Number of features copied, -1 on error
 */
int gnn_get_edge_features(
    const GnnHandle* handle,
    unsigned int edge_idx,
    float* features_out,
    unsigned int features_len
);

/* ============================================================================
 * Training & Inference
 * ============================================================================ */

/**
 * @brief Run prediction on the current graph
 *
 * @param handle GNN handle
 * @param output Output buffer for predictions
 * @param output_len Length of output buffer
 * @return Number of outputs, -1 on error
 */
int gnn_predict(GnnHandle* handle, float* output, unsigned int output_len);

/**
 * @brief Train on the current graph with target values
 *
 * @param handle GNN handle
 * @param target Array of target values
 * @param target_len Length of target array
 * @param loss_out Pointer to store loss value (can be NULL)
 * @return GNN_OK on success, error code on failure
 */
int gnn_train(
    GnnHandle* handle,
    const float* target,
    unsigned int target_len,
    float* loss_out
);

/**
 * @brief Train for multiple iterations
 *
 * @param handle GNN handle
 * @param target Array of target values
 * @param target_len Length of target array
 * @param iterations Number of training iterations
 * @return GNN_OK on success, error code on failure
 */
int gnn_train_multiple(
    GnnHandle* handle,
    const float* target,
    unsigned int target_len,
    unsigned int iterations
);

/* ============================================================================
 * Hyperparameters
 * ============================================================================ */

/**
 * @brief Set learning rate
 *
 * @param handle GNN handle
 * @param lr Learning rate value
 * @return GNN_OK on success
 */
int gnn_set_learning_rate(GnnHandle* handle, float lr);

/**
 * @brief Get learning rate
 *
 * @param handle GNN handle
 * @return Current learning rate
 */
float gnn_get_learning_rate(const GnnHandle* handle);

/* ============================================================================
 * Graph Info
 * ============================================================================ */

/** @brief Get number of nodes */
unsigned int gnn_get_num_nodes(const GnnHandle* handle);

/** @brief Get number of edges */
unsigned int gnn_get_num_edges(const GnnHandle* handle);

/** @brief Check if graph is loaded (returns 1 if true, 0 if false) */
int gnn_is_graph_loaded(const GnnHandle* handle);

/** @brief Get feature size */
unsigned int gnn_get_feature_size(const GnnHandle* handle);

/** @brief Get hidden size */
unsigned int gnn_get_hidden_size(const GnnHandle* handle);

/** @brief Get output size */
unsigned int gnn_get_output_size(const GnnHandle* handle);

/** @brief Get number of message passing layers */
unsigned int gnn_get_num_message_passing_layers(const GnnHandle* handle);

/** @brief Get in-degree of a node */
unsigned int gnn_get_in_degree(const GnnHandle* handle, unsigned int node_idx);

/** @brief Get out-degree of a node */
unsigned int gnn_get_out_degree(const GnnHandle* handle, unsigned int node_idx);

/**
 * @brief Get neighbors of a node
 *
 * @param handle GNN handle
 * @param node_idx Node index
 * @param neighbors_out Output buffer for neighbor indices
 * @param neighbors_len Length of output buffer
 * @return Number of neighbors (may be larger than buffer), -1 on error
 */
int gnn_get_neighbors(
    const GnnHandle* handle,
    unsigned int node_idx,
    unsigned int* neighbors_out,
    unsigned int neighbors_len
);

/**
 * @brief Get graph embedding from last forward pass
 *
 * @param handle GNN handle
 * @param embedding_out Output buffer for embedding
 * @param embedding_len Length of output buffer
 * @return Number of values copied, -1 on error
 */
int gnn_get_graph_embedding(
    const GnnHandle* handle,
    float* embedding_out,
    unsigned int embedding_len
);

/* ============================================================================
 * Masking & Dropout
 * ============================================================================ */

/**
 * @brief Set node mask
 *
 * @param handle GNN handle
 * @param node_idx Node index
 * @param value Mask value (non-zero = active)
 * @return GNN_OK on success
 */
int gnn_set_node_mask(GnnHandle* handle, unsigned int node_idx, int value);

/**
 * @brief Get node mask
 *
 * @param handle GNN handle
 * @param node_idx Node index
 * @return 1 if active, 0 if not
 */
int gnn_get_node_mask(const GnnHandle* handle, unsigned int node_idx);

/**
 * @brief Set edge mask
 *
 * @param handle GNN handle
 * @param edge_idx Edge index
 * @param value Mask value (non-zero = active)
 * @return GNN_OK on success
 */
int gnn_set_edge_mask(GnnHandle* handle, unsigned int edge_idx, int value);

/**
 * @brief Get edge mask
 *
 * @param handle GNN handle
 * @param edge_idx Edge index
 * @return 1 if active, 0 if not
 */
int gnn_get_edge_mask(const GnnHandle* handle, unsigned int edge_idx);

/**
 * @brief Apply random node dropout
 *
 * @param handle GNN handle
 * @param rate Dropout rate (0.0 to 1.0)
 * @return GNN_OK on success
 */
int gnn_apply_node_dropout(GnnHandle* handle, float rate);

/**
 * @brief Apply random edge dropout
 *
 * @param handle GNN handle
 * @param rate Dropout rate (0.0 to 1.0)
 * @return GNN_OK on success
 */
int gnn_apply_edge_dropout(GnnHandle* handle, float rate);

/** @brief Get count of active (masked) nodes */
unsigned int gnn_get_masked_node_count(const GnnHandle* handle);

/** @brief Get count of active (masked) edges */
unsigned int gnn_get_masked_edge_count(const GnnHandle* handle);

/* ============================================================================
 * Analytics
 * ============================================================================ */

/**
 * @brief Compute PageRank scores
 *
 * @param handle GNN handle
 * @param damping Damping factor (typically 0.85)
 * @param iterations Number of iterations
 * @param scores_out Output buffer for scores
 * @param scores_len Length of output buffer
 * @return Number of scores (may be larger than buffer), -1 on error
 */
int gnn_compute_page_rank(
    const GnnHandle* handle,
    float damping,
    unsigned int iterations,
    float* scores_out,
    unsigned int scores_len
);

/**
 * @brief Get gradient flow information for a layer
 *
 * @param handle GNN handle
 * @param layer_idx Layer index
 * @param info_out Pointer to info struct to fill
 * @return GNN_OK on success
 */
int gnn_get_gradient_flow(
    const GnnHandle* handle,
    unsigned int layer_idx,
    GnnGradientFlowInfo* info_out
);

/**
 * @brief Get total parameter count
 *
 * @param handle GNN handle
 * @return Number of trainable parameters
 */
unsigned int gnn_get_parameter_count(const GnnHandle* handle);

/**
 * @brief Get architecture summary string
 *
 * @param handle GNN handle
 * @param buffer Output buffer for the string
 * @param buffer_len Length of output buffer
 * @return Number of bytes written (excluding null), -1 on error
 */
int gnn_get_architecture_summary(
    const GnnHandle* handle,
    char* buffer,
    unsigned int buffer_len
);

/**
 * @brief Export graph to JSON string
 *
 * @param handle GNN handle
 * @param buffer Output buffer for the JSON string
 * @param buffer_len Length of output buffer
 * @return Number of bytes written (excluding null), -1 on error
 */
int gnn_export_graph_to_json(
    const GnnHandle* handle,
    char* buffer,
    unsigned int buffer_len
);

/**
 * @brief Detect the best available GPU backend
 *
 * @return Backend type: GNN_BACKEND_CUDA (0), GNN_BACKEND_OPENCL (1), or GNN_BACKEND_AUTO (2)
 */
int gnn_detect_backend(void);

/**
 * @brief Get the backend type for a handle
 *
 * @param handle GNN handle
 * @return Backend type: GNN_BACKEND_CUDA (0), GNN_BACKEND_OPENCL (1), or GNN_BACKEND_AUTO (2)
 */
int gnn_get_backend_type(const GnnHandle* handle);

/**
 * @brief Get edge endpoints (source and target node indices)
 *
 * @param handle GNN handle
 * @param edge_idx Edge index
 * @param source_out Pointer to store source node index
 * @param target_out Pointer to store target node index
 * @return GNN_OK on success, error code if edge not found
 */
int gnn_get_edge_endpoints(
    const GnnHandle* handle,
    unsigned int edge_idx,
    unsigned int* source_out,
    unsigned int* target_out
);

/**
 * @brief Get the active backend name
 *
 * @param handle GNN handle
 * @param buffer Output buffer for the name string
 * @param buffer_len Length of output buffer
 * @return Number of bytes written (excluding null), -1 on error
 */
int gnn_get_backend_name(
    const GnnHandle* handle,
    char* buffer,
    unsigned int buffer_len
);

#ifdef __cplusplus
}
#endif

#endif /* GNN_FACADE_H */
