/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Python bindings for GlassBoxAI GNN using PyO3
 */

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use crate::{GnnFacade, GpuBackendType, GradientFlowInfo as RustGradientFlowInfo};

/// Python wrapper for GradientFlowInfo
#[pyclass(name = "GradientFlowInfo")]
#[derive(Clone)]
pub struct PyGradientFlowInfo {
    #[pyo3(get)]
    pub layer_idx: usize,
    #[pyo3(get)]
    pub mean_gradient: f32,
    #[pyo3(get)]
    pub max_gradient: f32,
    #[pyo3(get)]
    pub min_gradient: f32,
    #[pyo3(get)]
    pub gradient_norm: f32,
}

impl From<RustGradientFlowInfo> for PyGradientFlowInfo {
    fn from(info: RustGradientFlowInfo) -> Self {
        Self {
            layer_idx: info.layer_idx,
            mean_gradient: info.mean_gradient,
            max_gradient: info.max_gradient,
            min_gradient: info.min_gradient,
            gradient_norm: info.gradient_norm,
        }
    }
}

/// GPU-accelerated Graph Neural Network with Facade interface
///
/// This class provides a high-level interface for creating, training,
/// and using Graph Neural Networks with CUDA or OpenCL acceleration.
///
/// Example:
///     >>> from gnn_facade_cuda import GnnFacade
///     >>> gnn = GnnFacade(feature_size=3, hidden_size=16, output_size=2, num_mp_layers=2)
///     >>> gnn.create_empty_graph(5, 3)
///     >>> gnn.add_edge(0, 1)
///     >>> gnn.set_node_features(0, [1.0, 0.5, 0.2])
///     >>> prediction = gnn.predict()
///
///     # Specify backend explicitly
///     >>> gnn = GnnFacade(3, 16, 2, 2, backend="opencl")
#[pyclass(name = "GnnFacade")]
pub struct PyGnnFacade {
    inner: GnnFacade,
}

fn parse_backend(s: Option<&str>) -> GpuBackendType {
    match s {
        Some("cuda") => GpuBackendType::Cuda,
        Some("opencl") => GpuBackendType::OpenCL,
        _ => GpuBackendType::Auto,
    }
}

#[pymethods]
impl PyGnnFacade {
    /// Create a new GNN Facade
    ///
    /// Args:
    ///     feature_size: Size of input node features
    ///     hidden_size: Size of hidden layers
    ///     output_size: Size of output predictions
    ///     num_mp_layers: Number of message passing layers
    ///     backend: GPU backend - "cuda", "opencl", or "auto" (default: "auto")
    #[new]
    #[pyo3(signature = (feature_size, hidden_size, output_size, num_mp_layers, backend=None))]
    fn new(feature_size: usize, hidden_size: usize, output_size: usize, num_mp_layers: usize, backend: Option<&str>) -> PyResult<Self> {
        GnnFacade::with_backend(feature_size, hidden_size, output_size, num_mp_layers, parse_backend(backend))
            .map(|inner| Self { inner })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Load a GNN from a saved model file
    ///
    /// Args:
    ///     filename: Path to the model file
    ///     backend: GPU backend - "cuda", "opencl", or "auto" (default: "auto")
    ///
    /// Returns:
    ///     GnnFacade: A new facade loaded from the file
    #[staticmethod]
    #[pyo3(signature = (filename, backend=None))]
    fn from_model_file(filename: &str, backend: Option<&str>) -> PyResult<Self> {
        GnnFacade::from_model_file_with_backend(filename, parse_backend(backend))
            .map(|inner| Self { inner })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Read model header without loading full model
    ///
    /// Args:
    ///     filename: Path to the model file
    ///
    /// Returns:
    ///     tuple: (feature_size, hidden_size, output_size, mp_layers, learning_rate)
    #[staticmethod]
    fn read_model_header(filename: &str) -> PyResult<(usize, usize, usize, usize, f32)> {
        GnnFacade::read_model_header(filename)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Create an empty graph with specified number of nodes
    ///
    /// Args:
    ///     num_nodes: Number of nodes in the graph
    ///     feature_size: Size of node feature vectors
    fn create_empty_graph(&mut self, num_nodes: usize, feature_size: usize) {
        self.inner.create_empty_graph(num_nodes, feature_size);
    }

    /// Get a single feature value for a node
    ///
    /// Args:
    ///     node_idx: Index of the node
    ///     feature_idx: Index of the feature
    ///
    /// Returns:
    ///     float: The feature value
    fn get_node_feature(&self, node_idx: usize, feature_idx: usize) -> f32 {
        self.inner.get_node_feature(node_idx, feature_idx)
    }

    /// Set a single feature value for a node
    ///
    /// Args:
    ///     node_idx: Index of the node
    ///     feature_idx: Index of the feature
    ///     value: The value to set
    fn set_node_feature(&mut self, node_idx: usize, feature_idx: usize, value: f32) {
        self.inner.set_node_feature(node_idx, feature_idx, value);
    }

    /// Set all features for a node
    ///
    /// Args:
    ///     node_idx: Index of the node
    ///     features: List of feature values
    fn set_node_features(&mut self, node_idx: usize, features: Vec<f32>) {
        self.inner.set_node_features(node_idx, features);
    }

    /// Get all features for a node
    ///
    /// Args:
    ///     node_idx: Index of the node
    ///
    /// Returns:
    ///     list or None: List of feature values, or None if node doesn't exist
    fn get_node_features(&self, node_idx: usize) -> Option<Vec<f32>> {
        self.inner.get_node_features(node_idx).cloned()
    }

    /// Add an edge to the graph
    ///
    /// Args:
    ///     source: Source node index
    ///     target: Target node index
    ///     features: Optional list of edge features (default: empty)
    ///
    /// Returns:
    ///     int: Index of the new edge
    #[pyo3(signature = (source, target, features=None))]
    fn add_edge(&mut self, source: usize, target: usize, features: Option<Vec<f32>>) -> usize {
        self.inner.add_edge(source, target, features.unwrap_or_default())
    }

    /// Remove an edge by index
    ///
    /// Args:
    ///     edge_idx: Index of the edge to remove
    fn remove_edge(&mut self, edge_idx: usize) {
        self.inner.remove_edge(edge_idx);
    }

    /// Get endpoints of an edge
    ///
    /// Args:
    ///     edge_idx: Index of the edge
    ///
    /// Returns:
    ///     tuple or None: (source, target) tuple, or None if edge doesn't exist
    fn get_edge_endpoints(&self, edge_idx: usize) -> Option<(usize, usize)> {
        self.inner.get_edge_endpoints(edge_idx)
    }

    /// Check if an edge exists between two nodes
    ///
    /// Args:
    ///     source: Source node index
    ///     target: Target node index
    ///
    /// Returns:
    ///     bool: True if edge exists
    fn has_edge(&self, source: usize, target: usize) -> bool {
        self.inner.has_edge(source, target)
    }

    /// Find the index of an edge between two nodes
    ///
    /// Args:
    ///     source: Source node index
    ///     target: Target node index
    ///
    /// Returns:
    ///     int or None: Edge index, or None if not found
    fn find_edge_index(&self, source: usize, target: usize) -> Option<usize> {
        self.inner.find_edge_index(source, target)
    }

    /// Get neighbors of a node
    ///
    /// Args:
    ///     node_idx: Index of the node
    ///
    /// Returns:
    ///     list or None: List of neighbor indices, or None if node doesn't exist
    fn get_neighbors(&self, node_idx: usize) -> Option<Vec<usize>> {
        self.inner.get_neighbors(node_idx).cloned()
    }

    /// Get in-degree of a node
    ///
    /// Args:
    ///     node_idx: Index of the node
    ///
    /// Returns:
    ///     int: Number of incoming edges
    fn get_in_degree(&self, node_idx: usize) -> usize {
        self.inner.get_in_degree(node_idx)
    }

    /// Get out-degree of a node
    ///
    /// Args:
    ///     node_idx: Index of the node
    ///
    /// Returns:
    ///     int: Number of outgoing edges
    fn get_out_degree(&self, node_idx: usize) -> usize {
        self.inner.get_out_degree(node_idx)
    }

    /// Get features for an edge
    ///
    /// Args:
    ///     edge_idx: Index of the edge
    ///
    /// Returns:
    ///     list or None: List of feature values, or None if edge doesn't exist
    fn get_edge_features(&self, edge_idx: usize) -> Option<Vec<f32>> {
        self.inner.get_edge_features(edge_idx).cloned()
    }

    /// Set features for an edge
    ///
    /// Args:
    ///     edge_idx: Index of the edge
    ///     features: List of feature values
    fn set_edge_features(&mut self, edge_idx: usize, features: Vec<f32>) {
        self.inner.set_edge_features(edge_idx, features);
    }

    /// Rebuild the adjacency list from edges
    fn rebuild_adjacency_list(&mut self) {
        self.inner.rebuild_adjacency_list();
    }

    /// Run prediction on the current graph
    ///
    /// Returns:
    ///     list: Prediction values
    fn predict(&mut self) -> PyResult<Vec<f32>> {
        self.inner.predict()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Train on the current graph with target values
    ///
    /// Args:
    ///     target: List of target values
    ///
    /// Returns:
    ///     float: Loss value
    fn train(&mut self, target: Vec<f32>) -> PyResult<f32> {
        self.inner.train(&target)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Train for multiple iterations
    ///
    /// Args:
    ///     target: List of target values
    ///     iterations: Number of training iterations
    fn train_multiple(&mut self, target: Vec<f32>, iterations: usize) -> PyResult<()> {
        self.inner.train_multiple(&target, iterations)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Save model to file
    ///
    /// Args:
    ///     filename: Path to save the model
    fn save_model(&self, filename: &str) -> PyResult<()> {
        self.inner.save_model(filename)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Load model from file
    ///
    /// Args:
    ///     filename: Path to the model file
    fn load_model(&mut self, filename: &str) -> PyResult<()> {
        self.inner.load_model(filename)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Set learning rate
    ///
    /// Args:
    ///     lr: Learning rate value
    fn set_learning_rate(&mut self, lr: f32) {
        self.inner.set_learning_rate(lr);
    }

    /// Get learning rate
    ///
    /// Returns:
    ///     float: Current learning rate
    fn get_learning_rate(&self) -> f32 {
        self.inner.get_learning_rate()
    }

    /// Get architecture summary
    ///
    /// Returns:
    ///     str: Summary of the network architecture
    fn get_architecture_summary(&self) -> String {
        self.inner.get_architecture_summary()
    }

    /// Get number of nodes in the graph
    ///
    /// Returns:
    ///     int: Number of nodes
    fn get_num_nodes(&self) -> usize {
        self.inner.get_num_nodes()
    }

    /// Get number of edges in the graph
    ///
    /// Returns:
    ///     int: Number of edges
    fn get_num_edges(&self) -> usize {
        self.inner.get_num_edges()
    }

    /// Check if a graph is loaded
    ///
    /// Returns:
    ///     bool: True if a graph is loaded
    fn is_graph_loaded(&self) -> bool {
        self.inner.is_graph_loaded()
    }

    /// Get graph embedding from last forward pass
    ///
    /// Returns:
    ///     list: Graph embedding vector
    fn get_graph_embedding(&self) -> Vec<f32> {
        self.inner.get_graph_embedding().to_vec()
    }

    /// Get feature size
    ///
    /// Returns:
    ///     int: Size of node features
    fn get_feature_size(&self) -> usize {
        self.inner.get_feature_size()
    }

    /// Get hidden size
    ///
    /// Returns:
    ///     int: Size of hidden layers
    fn get_hidden_size(&self) -> usize {
        self.inner.get_hidden_size()
    }

    /// Get output size
    ///
    /// Returns:
    ///     int: Size of output
    fn get_output_size(&self) -> usize {
        self.inner.get_output_size()
    }

    /// Get number of message passing layers
    ///
    /// Returns:
    ///     int: Number of message passing layers
    fn get_num_message_passing_layers(&self) -> usize {
        self.inner.get_num_message_passing_layers()
    }

    /// Get the active GPU backend name
    ///
    /// Returns:
    ///     str: "cuda" or "opencl"
    fn get_backend_name(&self) -> &str {
        self.inner.get_backend_name()
    }

    /// Get node mask value
    ///
    /// Args:
    ///     node_idx: Index of the node
    ///
    /// Returns:
    ///     bool: Mask value (True = active)
    fn get_node_mask(&self, node_idx: usize) -> bool {
        self.inner.get_node_mask(node_idx)
    }

    /// Set node mask value
    ///
    /// Args:
    ///     node_idx: Index of the node
    ///     value: Mask value (True = active)
    fn set_node_mask(&mut self, node_idx: usize, value: bool) {
        self.inner.set_node_mask(node_idx, value);
    }

    /// Get edge mask value
    ///
    /// Args:
    ///     edge_idx: Index of the edge
    ///
    /// Returns:
    ///     bool: Mask value (True = active)
    fn get_edge_mask(&self, edge_idx: usize) -> bool {
        self.inner.get_edge_mask(edge_idx)
    }

    /// Set edge mask value
    ///
    /// Args:
    ///     edge_idx: Index of the edge
    ///     value: Mask value (True = active)
    fn set_edge_mask(&mut self, edge_idx: usize, value: bool) {
        self.inner.set_edge_mask(edge_idx, value);
    }

    /// Apply random dropout to nodes
    ///
    /// Args:
    ///     rate: Dropout rate (0.0 to 1.0)
    fn apply_node_dropout(&mut self, rate: f32) {
        self.inner.apply_node_dropout(rate);
    }

    /// Apply random dropout to edges
    ///
    /// Args:
    ///     rate: Dropout rate (0.0 to 1.0)
    fn apply_edge_dropout(&mut self, rate: f32) {
        self.inner.apply_edge_dropout(rate);
    }

    /// Get count of active (masked) nodes
    ///
    /// Returns:
    ///     int: Number of active nodes
    fn get_masked_node_count(&self) -> usize {
        self.inner.get_masked_node_count()
    }

    /// Get count of active (masked) edges
    ///
    /// Returns:
    ///     int: Number of active edges
    fn get_masked_edge_count(&self) -> usize {
        self.inner.get_masked_edge_count()
    }

    /// Compute PageRank scores for all nodes
    ///
    /// Args:
    ///     damping: Damping factor (default: 0.85)
    ///     iterations: Number of iterations (default: 20)
    ///
    /// Returns:
    ///     list: PageRank score for each node
    #[pyo3(signature = (damping=0.85, iterations=20))]
    fn compute_page_rank(&self, damping: f32, iterations: usize) -> Vec<f32> {
        self.inner.compute_page_rank(damping, iterations)
    }

    /// Get gradient flow information for a layer
    ///
    /// Args:
    ///     layer_idx: Index of the layer
    ///
    /// Returns:
    ///     GradientFlowInfo: Gradient flow statistics
    fn get_gradient_flow(&self, layer_idx: usize) -> PyGradientFlowInfo {
        self.inner.get_gradient_flow(layer_idx).into()
    }

    /// Get total parameter count
    ///
    /// Returns:
    ///     int: Total number of trainable parameters
    fn get_parameter_count(&self) -> usize {
        self.inner.get_parameter_count()
    }

    /// Export graph structure to JSON
    ///
    /// Returns:
    ///     str: JSON representation of the graph
    fn export_graph_to_json(&self) -> String {
        self.inner.export_graph_to_json()
    }

    fn __repr__(&self) -> String {
        format!(
            "GnnFacade(feature_size={}, hidden_size={}, output_size={}, mp_layers={}, nodes={}, edges={})",
            self.inner.get_feature_size(),
            self.inner.get_hidden_size(),
            self.inner.get_output_size(),
            self.inner.get_num_message_passing_layers(),
            self.inner.get_num_nodes(),
            self.inner.get_num_edges()
        )
    }
}

/// GlassBoxAI GNN - CUDA-accelerated Graph Neural Network
///
/// This module provides Python bindings for a high-performance
/// Graph Neural Network implementation with CUDA acceleration.
///
/// Example:
///     >>> from gnn_facade_cuda import GnnFacade
///     >>> 
///     >>> # Create a new GNN
///     >>> gnn = GnnFacade(feature_size=3, hidden_size=16, output_size=2, num_mp_layers=2)
///     >>> 
///     >>> # Create a graph
///     >>> gnn.create_empty_graph(5, 3)
///     >>> 
///     >>> # Add edges
///     >>> gnn.add_edge(0, 1)
///     >>> gnn.add_edge(1, 2)
///     >>> gnn.add_edge(2, 3)
///     >>> 
///     >>> # Set node features
///     >>> gnn.set_node_features(0, [1.0, 0.5, 0.2])
///     >>> gnn.set_node_features(1, [0.8, 0.3, 0.1])
///     >>> 
///     >>> # Make predictions
///     >>> prediction = gnn.predict()
///     >>> print(prediction)
///     >>> 
///     >>> # Train the model
///     >>> target = [0.5, 0.5]
///     >>> loss = gnn.train(target)
///     >>> print(f"Loss: {loss}")
///     >>> 
///     >>> # Save and load models
///     >>> gnn.save_model("model.bin")
///     >>> gnn2 = GnnFacade.from_model_file("model.bin")
#[pymodule]
pub fn gnn_facade_cuda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGnnFacade>()?;
    m.add_class::<PyGradientFlowInfo>()?;
    Ok(())
}
