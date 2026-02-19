## @file
## @ingroup GNN_Wrappers
"""Type stubs for gnn_facade_cuda"""

from typing import List, Optional, Tuple

class GradientFlowInfo:
    """Information about gradient flow through a layer"""
    layer_idx: int
    mean_gradient: float
    max_gradient: float
    min_gradient: float
    gradient_norm: float

class GnnFacade:
    """
    CUDA-accelerated Graph Neural Network with Facade interface
    
    This class provides a high-level interface for creating, training,
    and using Graph Neural Networks with CUDA acceleration.
    
    Example:
        >>> gnn = GnnFacade(feature_size=3, hidden_size=16, output_size=2, num_mp_layers=2)
        >>> gnn.create_empty_graph(5, 3)
        >>> gnn.add_edge(0, 1)
        >>> prediction = gnn.predict()
    """
    
    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        output_size: int,
        num_mp_layers: int
    ) -> None:
        """
        Create a new GNN Facade
        
        Args:
            feature_size: Size of input node features
            hidden_size: Size of hidden layers
            output_size: Size of output predictions
            num_mp_layers: Number of message passing layers
        """
        ...
    
    @staticmethod
    def from_model_file(filename: str) -> "GnnFacade":
        """Load a GNN from a saved model file"""
        ...
    
    @staticmethod
    def read_model_header(filename: str) -> Tuple[int, int, int, int, float]:
        """Read model header without loading full model
        
        Returns:
            tuple: (feature_size, hidden_size, output_size, mp_layers, learning_rate)
        """
        ...
    
    def create_empty_graph(self, num_nodes: int, feature_size: int) -> None:
        """Create an empty graph with specified number of nodes"""
        ...
    
    def get_node_feature(self, node_idx: int, feature_idx: int) -> float:
        """Get a single feature value for a node"""
        ...
    
    def set_node_feature(self, node_idx: int, feature_idx: int, value: float) -> None:
        """Set a single feature value for a node"""
        ...
    
    def set_node_features(self, node_idx: int, features: List[float]) -> None:
        """Set all features for a node"""
        ...
    
    def get_node_features(self, node_idx: int) -> Optional[List[float]]:
        """Get all features for a node"""
        ...
    
    def add_edge(
        self,
        source: int,
        target: int,
        features: Optional[List[float]] = None
    ) -> int:
        """Add an edge to the graph. Returns the edge index."""
        ...
    
    def remove_edge(self, edge_idx: int) -> None:
        """Remove an edge by index"""
        ...
    
    def get_edge_endpoints(self, edge_idx: int) -> Optional[Tuple[int, int]]:
        """Get endpoints of an edge"""
        ...
    
    def has_edge(self, source: int, target: int) -> bool:
        """Check if an edge exists between two nodes"""
        ...
    
    def find_edge_index(self, source: int, target: int) -> Optional[int]:
        """Find the index of an edge between two nodes"""
        ...
    
    def get_neighbors(self, node_idx: int) -> Optional[List[int]]:
        """Get neighbors of a node"""
        ...
    
    def get_in_degree(self, node_idx: int) -> int:
        """Get in-degree of a node"""
        ...
    
    def get_out_degree(self, node_idx: int) -> int:
        """Get out-degree of a node"""
        ...
    
    def get_edge_features(self, edge_idx: int) -> Optional[List[float]]:
        """Get features for an edge"""
        ...
    
    def set_edge_features(self, edge_idx: int, features: List[float]) -> None:
        """Set features for an edge"""
        ...
    
    def rebuild_adjacency_list(self) -> None:
        """Rebuild the adjacency list from edges"""
        ...
    
    def predict(self) -> List[float]:
        """Run prediction on the current graph"""
        ...
    
    def train(self, target: List[float]) -> float:
        """Train on the current graph with target values. Returns loss."""
        ...
    
    def train_multiple(self, target: List[float], iterations: int) -> None:
        """Train for multiple iterations"""
        ...
    
    def save_model(self, filename: str) -> None:
        """Save model to file"""
        ...
    
    def load_model(self, filename: str) -> None:
        """Load model from file"""
        ...
    
    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate"""
        ...
    
    def get_learning_rate(self) -> float:
        """Get learning rate"""
        ...
    
    def get_architecture_summary(self) -> str:
        """Get architecture summary"""
        ...
    
    def get_num_nodes(self) -> int:
        """Get number of nodes in the graph"""
        ...
    
    def get_num_edges(self) -> int:
        """Get number of edges in the graph"""
        ...
    
    def is_graph_loaded(self) -> bool:
        """Check if a graph is loaded"""
        ...
    
    def get_graph_embedding(self) -> List[float]:
        """Get graph embedding from last forward pass"""
        ...
    
    def get_feature_size(self) -> int:
        """Get feature size"""
        ...
    
    def get_hidden_size(self) -> int:
        """Get hidden size"""
        ...
    
    def get_output_size(self) -> int:
        """Get output size"""
        ...
    
    def get_num_message_passing_layers(self) -> int:
        """Get number of message passing layers"""
        ...
    
    def get_node_mask(self, node_idx: int) -> bool:
        """Get node mask value"""
        ...
    
    def set_node_mask(self, node_idx: int, value: bool) -> None:
        """Set node mask value"""
        ...
    
    def get_edge_mask(self, edge_idx: int) -> bool:
        """Get edge mask value"""
        ...
    
    def set_edge_mask(self, edge_idx: int, value: bool) -> None:
        """Set edge mask value"""
        ...
    
    def apply_node_dropout(self, rate: float) -> None:
        """Apply random dropout to nodes"""
        ...
    
    def apply_edge_dropout(self, rate: float) -> None:
        """Apply random dropout to edges"""
        ...
    
    def get_masked_node_count(self) -> int:
        """Get count of active (masked) nodes"""
        ...
    
    def get_masked_edge_count(self) -> int:
        """Get count of active (masked) edges"""
        ...
    
    def compute_page_rank(
        self,
        damping: float = 0.85,
        iterations: int = 20
    ) -> List[float]:
        """Compute PageRank scores for all nodes"""
        ...
    
    def get_gradient_flow(self, layer_idx: int) -> GradientFlowInfo:
        """Get gradient flow information for a layer"""
        ...
    
    def get_parameter_count(self) -> int:
        """Get total parameter count"""
        ...
    
    def export_graph_to_json(self) -> str:
        """Export graph structure to JSON"""
        ...
