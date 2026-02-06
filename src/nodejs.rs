/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Node.js bindings for GlassBoxAI GNN using napi-rs
 */

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::{GnnFacade as RustGnnFacade, GpuBackendType, GradientFlowInfo as RustGradientFlowInfo};

/// Gradient flow information for a layer
#[napi(object)]
pub struct GradientFlowInfo {
    pub layer_idx: u32,
    pub mean_gradient: f64,
    pub max_gradient: f64,
    pub min_gradient: f64,
    pub gradient_norm: f64,
}

impl From<RustGradientFlowInfo> for GradientFlowInfo {
    fn from(info: RustGradientFlowInfo) -> Self {
        Self {
            layer_idx: info.layer_idx as u32,
            mean_gradient: info.mean_gradient as f64,
            max_gradient: info.max_gradient as f64,
            min_gradient: info.min_gradient as f64,
            gradient_norm: info.gradient_norm as f64,
        }
    }
}

fn parse_backend_js(s: Option<String>) -> GpuBackendType {
    match s.as_deref() {
        Some("cuda") => GpuBackendType::Cuda,
        Some("opencl") => GpuBackendType::OpenCL,
        _ => GpuBackendType::Auto,
    }
}

/// GPU-accelerated Graph Neural Network with Facade interface
///
/// This class provides a high-level interface for creating, training,
/// and using Graph Neural Networks with CUDA or OpenCL acceleration.
///
/// @example
/// ```javascript
/// const { GnnFacade } = require('gnn-facade-cuda');
///
/// // Auto-detect backend
/// const gnn = new GnnFacade(3, 16, 2, 2);
///
/// // Or specify explicitly
/// const gnn2 = new GnnFacade(3, 16, 2, 2, 'opencl');
/// ```
#[napi]
pub struct GnnFacade {
    inner: RustGnnFacade,
}

#[napi]
impl GnnFacade {
    /// Create a new GNN Facade
    ///
    /// @param featureSize - Size of input node features
    /// @param hiddenSize - Size of hidden layers
    /// @param outputSize - Size of output predictions
    /// @param numMpLayers - Number of message passing layers
    /// @param backend - GPU backend: "cuda", "opencl", or "auto" (default: "auto")
    #[napi(constructor)]
    pub fn new(
        feature_size: u32,
        hidden_size: u32,
        output_size: u32,
        num_mp_layers: u32,
        backend: Option<String>,
    ) -> Result<Self> {
        RustGnnFacade::with_backend(
            feature_size as usize,
            hidden_size as usize,
            output_size as usize,
            num_mp_layers as usize,
            parse_backend_js(backend),
        )
        .map(|inner| Self { inner })
        .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Load a GNN from a saved model file
    ///
    /// @param filename - Path to the model file
    /// @param backend - GPU backend: "cuda", "opencl", or "auto" (default: "auto")
    /// @returns A new GnnFacade loaded from the file
    #[napi(factory)]
    pub fn from_model_file(filename: String, backend: Option<String>) -> Result<Self> {
        RustGnnFacade::from_model_file_with_backend(&filename, parse_backend_js(backend))
            .map(|inner| Self { inner })
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Read model header without loading full model
    ///
    /// @param filename - Path to the model file
    /// @returns Object with featureSize, hiddenSize, outputSize, mpLayers, learningRate
    #[napi]
    pub fn read_model_header(filename: String) -> Result<ModelHeader> {
        RustGnnFacade::read_model_header(&filename)
            .map(|(feature_size, hidden_size, output_size, mp_layers, learning_rate)| {
                ModelHeader {
                    feature_size: feature_size as u32,
                    hidden_size: hidden_size as u32,
                    output_size: output_size as u32,
                    mp_layers: mp_layers as u32,
                    learning_rate: learning_rate as f64,
                }
            })
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Create an empty graph with specified number of nodes
    ///
    /// @param numNodes - Number of nodes in the graph
    /// @param featureSize - Size of node feature vectors
    #[napi]
    pub fn create_empty_graph(&mut self, num_nodes: u32, feature_size: u32) {
        self.inner.create_empty_graph(num_nodes as usize, feature_size as usize);
    }

    /// Get a single feature value for a node
    ///
    /// @param nodeIdx - Index of the node
    /// @param featureIdx - Index of the feature
    /// @returns The feature value
    #[napi]
    pub fn get_node_feature(&self, node_idx: u32, feature_idx: u32) -> f64 {
        self.inner.get_node_feature(node_idx as usize, feature_idx as usize) as f64
    }

    /// Set a single feature value for a node
    ///
    /// @param nodeIdx - Index of the node
    /// @param featureIdx - Index of the feature
    /// @param value - The value to set
    #[napi]
    pub fn set_node_feature(&mut self, node_idx: u32, feature_idx: u32, value: f64) {
        self.inner.set_node_feature(node_idx as usize, feature_idx as usize, value as f32);
    }

    /// Set all features for a node
    ///
    /// @param nodeIdx - Index of the node
    /// @param features - Array of feature values
    #[napi]
    pub fn set_node_features(&mut self, node_idx: u32, features: Vec<f64>) {
        let features_f32: Vec<f32> = features.into_iter().map(|f| f as f32).collect();
        self.inner.set_node_features(node_idx as usize, features_f32);
    }

    /// Get all features for a node
    ///
    /// @param nodeIdx - Index of the node
    /// @returns Array of feature values, or null if node doesn't exist
    #[napi]
    pub fn get_node_features(&self, node_idx: u32) -> Option<Vec<f64>> {
        self.inner
            .get_node_features(node_idx as usize)
            .map(|f| f.iter().map(|&v| v as f64).collect())
    }

    /// Add an edge to the graph
    ///
    /// @param source - Source node index
    /// @param target - Target node index
    /// @param features - Optional array of edge features
    /// @returns Index of the new edge
    #[napi]
    pub fn add_edge(&mut self, source: u32, target: u32, features: Option<Vec<f64>>) -> u32 {
        let features_f32: Vec<f32> = features
            .unwrap_or_default()
            .into_iter()
            .map(|f| f as f32)
            .collect();
        self.inner.add_edge(source as usize, target as usize, features_f32) as u32
    }

    /// Remove an edge by index
    ///
    /// @param edgeIdx - Index of the edge to remove
    #[napi]
    pub fn remove_edge(&mut self, edge_idx: u32) {
        self.inner.remove_edge(edge_idx as usize);
    }

    /// Get endpoints of an edge
    ///
    /// @param edgeIdx - Index of the edge
    /// @returns Object with source and target, or null if edge doesn't exist
    #[napi]
    pub fn get_edge_endpoints(&self, edge_idx: u32) -> Option<EdgeEndpoints> {
        self.inner
            .get_edge_endpoints(edge_idx as usize)
            .map(|(source, target)| EdgeEndpoints {
                source: source as u32,
                target: target as u32,
            })
    }

    /// Check if an edge exists between two nodes
    ///
    /// @param source - Source node index
    /// @param target - Target node index
    /// @returns true if edge exists
    #[napi]
    pub fn has_edge(&self, source: u32, target: u32) -> bool {
        self.inner.has_edge(source as usize, target as usize)
    }

    /// Find the index of an edge between two nodes
    ///
    /// @param source - Source node index
    /// @param target - Target node index
    /// @returns Edge index, or null if not found
    #[napi]
    pub fn find_edge_index(&self, source: u32, target: u32) -> Option<u32> {
        self.inner
            .find_edge_index(source as usize, target as usize)
            .map(|i| i as u32)
    }

    /// Get neighbors of a node
    ///
    /// @param nodeIdx - Index of the node
    /// @returns Array of neighbor indices, or null if node doesn't exist
    #[napi]
    pub fn get_neighbors(&self, node_idx: u32) -> Option<Vec<u32>> {
        self.inner
            .get_neighbors(node_idx as usize)
            .map(|n| n.iter().map(|&v| v as u32).collect())
    }

    /// Get in-degree of a node
    ///
    /// @param nodeIdx - Index of the node
    /// @returns Number of incoming edges
    #[napi]
    pub fn get_in_degree(&self, node_idx: u32) -> u32 {
        self.inner.get_in_degree(node_idx as usize) as u32
    }

    /// Get out-degree of a node
    ///
    /// @param nodeIdx - Index of the node
    /// @returns Number of outgoing edges
    #[napi]
    pub fn get_out_degree(&self, node_idx: u32) -> u32 {
        self.inner.get_out_degree(node_idx as usize) as u32
    }

    /// Get features for an edge
    ///
    /// @param edgeIdx - Index of the edge
    /// @returns Array of feature values, or null if edge doesn't exist
    #[napi]
    pub fn get_edge_features(&self, edge_idx: u32) -> Option<Vec<f64>> {
        self.inner
            .get_edge_features(edge_idx as usize)
            .map(|f| f.iter().map(|&v| v as f64).collect())
    }

    /// Set features for an edge
    ///
    /// @param edgeIdx - Index of the edge
    /// @param features - Array of feature values
    #[napi]
    pub fn set_edge_features(&mut self, edge_idx: u32, features: Vec<f64>) {
        let features_f32: Vec<f32> = features.into_iter().map(|f| f as f32).collect();
        self.inner.set_edge_features(edge_idx as usize, features_f32);
    }

    /// Rebuild the adjacency list from edges
    #[napi]
    pub fn rebuild_adjacency_list(&mut self) {
        self.inner.rebuild_adjacency_list();
    }

    /// Run prediction on the current graph
    ///
    /// @returns Array of prediction values
    #[napi]
    pub fn predict(&mut self) -> Result<Vec<f64>> {
        self.inner
            .predict()
            .map(|p| p.into_iter().map(|v| v as f64).collect())
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Train on the current graph with target values
    ///
    /// @param target - Array of target values
    /// @returns Loss value
    #[napi]
    pub fn train(&mut self, target: Vec<f64>) -> Result<f64> {
        let target_f32: Vec<f32> = target.into_iter().map(|v| v as f32).collect();
        self.inner
            .train(&target_f32)
            .map(|loss| loss as f64)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Train for multiple iterations
    ///
    /// @param target - Array of target values
    /// @param iterations - Number of training iterations
    #[napi]
    pub fn train_multiple(&mut self, target: Vec<f64>, iterations: u32) -> Result<()> {
        let target_f32: Vec<f32> = target.into_iter().map(|v| v as f32).collect();
        self.inner
            .train_multiple(&target_f32, iterations as usize)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Save model to file
    ///
    /// @param filename - Path to save the model
    #[napi]
    pub fn save_model(&self, filename: String) -> Result<()> {
        self.inner
            .save_model(&filename)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Load model from file
    ///
    /// @param filename - Path to the model file
    #[napi]
    pub fn load_model(&mut self, filename: String) -> Result<()> {
        self.inner
            .load_model(&filename)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Set learning rate
    ///
    /// @param lr - Learning rate value
    #[napi]
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.inner.set_learning_rate(lr as f32);
    }

    /// Get learning rate
    ///
    /// @returns Current learning rate
    #[napi]
    pub fn get_learning_rate(&self) -> f64 {
        self.inner.get_learning_rate() as f64
    }

    /// Get architecture summary
    ///
    /// @returns Summary of the network architecture
    #[napi]
    pub fn get_architecture_summary(&self) -> String {
        self.inner.get_architecture_summary()
    }

    /// Get number of nodes in the graph
    ///
    /// @returns Number of nodes
    #[napi]
    pub fn get_num_nodes(&self) -> u32 {
        self.inner.get_num_nodes() as u32
    }

    /// Get number of edges in the graph
    ///
    /// @returns Number of edges
    #[napi]
    pub fn get_num_edges(&self) -> u32 {
        self.inner.get_num_edges() as u32
    }

    /// Check if a graph is loaded
    ///
    /// @returns true if a graph is loaded
    #[napi]
    pub fn is_graph_loaded(&self) -> bool {
        self.inner.is_graph_loaded()
    }

    /// Get graph embedding from last forward pass
    ///
    /// @returns Array of embedding values
    #[napi]
    pub fn get_graph_embedding(&self) -> Vec<f64> {
        self.inner
            .get_graph_embedding()
            .iter()
            .map(|&v| v as f64)
            .collect()
    }

    /// Get feature size
    ///
    /// @returns Size of node features
    #[napi]
    pub fn get_feature_size(&self) -> u32 {
        self.inner.get_feature_size() as u32
    }

    /// Get hidden size
    ///
    /// @returns Size of hidden layers
    #[napi]
    pub fn get_hidden_size(&self) -> u32 {
        self.inner.get_hidden_size() as u32
    }

    /// Get output size
    ///
    /// @returns Size of output
    #[napi]
    pub fn get_output_size(&self) -> u32 {
        self.inner.get_output_size() as u32
    }

    /// Get number of message passing layers
    ///
    /// @returns Number of message passing layers
    #[napi]
    pub fn get_num_message_passing_layers(&self) -> u32 {
        self.inner.get_num_message_passing_layers() as u32
    }

    /// Get the active GPU backend name
    ///
    /// @returns "cuda" or "opencl"
    #[napi]
    pub fn get_backend_name(&self) -> String {
        self.inner.get_backend_name().to_string()
    }

    /// Get node mask value
    ///
    /// @param nodeIdx - Index of the node
    /// @returns Mask value (true = active)
    #[napi]
    pub fn get_node_mask(&self, node_idx: u32) -> bool {
        self.inner.get_node_mask(node_idx as usize)
    }

    /// Set node mask value
    ///
    /// @param nodeIdx - Index of the node
    /// @param value - Mask value (true = active)
    #[napi]
    pub fn set_node_mask(&mut self, node_idx: u32, value: bool) {
        self.inner.set_node_mask(node_idx as usize, value);
    }

    /// Get edge mask value
    ///
    /// @param edgeIdx - Index of the edge
    /// @returns Mask value (true = active)
    #[napi]
    pub fn get_edge_mask(&self, edge_idx: u32) -> bool {
        self.inner.get_edge_mask(edge_idx as usize)
    }

    /// Set edge mask value
    ///
    /// @param edgeIdx - Index of the edge
    /// @param value - Mask value (true = active)
    #[napi]
    pub fn set_edge_mask(&mut self, edge_idx: u32, value: bool) {
        self.inner.set_edge_mask(edge_idx as usize, value);
    }

    /// Apply random dropout to nodes
    ///
    /// @param rate - Dropout rate (0.0 to 1.0)
    #[napi]
    pub fn apply_node_dropout(&mut self, rate: f64) {
        self.inner.apply_node_dropout(rate as f32);
    }

    /// Apply random dropout to edges
    ///
    /// @param rate - Dropout rate (0.0 to 1.0)
    #[napi]
    pub fn apply_edge_dropout(&mut self, rate: f64) {
        self.inner.apply_edge_dropout(rate as f32);
    }

    /// Get count of active (masked) nodes
    ///
    /// @returns Number of active nodes
    #[napi]
    pub fn get_masked_node_count(&self) -> u32 {
        self.inner.get_masked_node_count() as u32
    }

    /// Get count of active (masked) edges
    ///
    /// @returns Number of active edges
    #[napi]
    pub fn get_masked_edge_count(&self) -> u32 {
        self.inner.get_masked_edge_count() as u32
    }

    /// Compute PageRank scores for all nodes
    ///
    /// @param damping - Damping factor (default: 0.85)
    /// @param iterations - Number of iterations (default: 20)
    /// @returns Array of PageRank scores
    #[napi]
    pub fn compute_page_rank(&self, damping: Option<f64>, iterations: Option<u32>) -> Vec<f64> {
        self.inner
            .compute_page_rank(
                damping.unwrap_or(0.85) as f32,
                iterations.unwrap_or(20) as usize,
            )
            .into_iter()
            .map(|v| v as f64)
            .collect()
    }

    /// Get gradient flow information for a layer
    ///
    /// @param layerIdx - Index of the layer
    /// @returns Gradient flow statistics
    #[napi]
    pub fn get_gradient_flow(&self, layer_idx: u32) -> GradientFlowInfo {
        self.inner.get_gradient_flow(layer_idx as usize).into()
    }

    /// Get total parameter count
    ///
    /// @returns Total number of trainable parameters
    #[napi]
    pub fn get_parameter_count(&self) -> u32 {
        self.inner.get_parameter_count() as u32
    }

    /// Export graph structure to JSON
    ///
    /// @returns JSON representation of the graph
    #[napi]
    pub fn export_graph_to_json(&self) -> String {
        self.inner.export_graph_to_json()
    }
}

/// Model header information
#[napi(object)]
pub struct ModelHeader {
    pub feature_size: u32,
    pub hidden_size: u32,
    pub output_size: u32,
    pub mp_layers: u32,
    pub learning_rate: f64,
}

/// Edge endpoints
#[napi(object)]
pub struct EdgeEndpoints {
    pub source: u32,
    pub target: u32,
}
