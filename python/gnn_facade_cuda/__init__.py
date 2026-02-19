## @file
## @ingroup GNN_Wrappers
"""
GlassBoxAI GNN - CUDA-accelerated Graph Neural Network

This module provides Python bindings for a high-performance
Graph Neural Network implementation with CUDA acceleration.

Example:
    >>> from gnn_facade_cuda import GnnFacade
    >>> 
    >>> # Create a new GNN
    >>> gnn = GnnFacade(feature_size=3, hidden_size=16, output_size=2, num_mp_layers=2)
    >>> 
    >>> # Create a graph
    >>> gnn.create_empty_graph(5, 3)
    >>> 
    >>> # Add edges
    >>> gnn.add_edge(0, 1)
    >>> gnn.add_edge(1, 2)
    >>> gnn.add_edge(2, 3)
    >>> 
    >>> # Set node features
    >>> gnn.set_node_features(0, [1.0, 0.5, 0.2])
    >>> gnn.set_node_features(1, [0.8, 0.3, 0.1])
    >>> 
    >>> # Make predictions
    >>> prediction = gnn.predict()
    >>> print(prediction)
    >>> 
    >>> # Train the model
    >>> target = [0.5, 0.5]
    >>> loss = gnn.train(target)
    >>> print(f"Loss: {loss}")
    >>> 
    >>> # Save and load models
    >>> gnn.save_model("model.bin")
    >>> gnn2 = GnnFacade.from_model_file("model.bin")
"""

from .gnn_facade_cuda import GnnFacade, GradientFlowInfo

__all__ = ["GnnFacade", "GradientFlowInfo"]
__version__ = "0.1.0"
