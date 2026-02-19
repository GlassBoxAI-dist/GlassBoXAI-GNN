//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_edge_returns_some_on_valid() {
        let mut graph = VerifiableGraph::new(5);
        assert!(graph.add_edge(0, 1).is_some());
    }

    #[test]
    fn test_add_edge_returns_none_on_invalid() {
        let mut graph = VerifiableGraph::new(5);
        assert!(graph.add_edge(100, 1).is_none());
    }

    #[test]
    fn test_get_edge_returns_some_on_valid() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(2, 3);
        assert!(graph.get_edge(0).is_some());
    }

    #[test]
    fn test_get_edge_returns_none_on_empty() {
        let graph = VerifiableGraph::new(5);
        assert!(graph.get_edge(0).is_none());
    }

    #[test]
    fn test_find_edge_index_returns_some() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        assert!(graph.find_edge_index(0, 1).is_some());
    }

    #[test]
    fn test_find_edge_index_returns_none() {
        let graph = VerifiableGraph::new(5);
        assert!(graph.find_edge_index(0, 1).is_none());
    }

    #[test]
    fn test_get_node_features_returns_some() {
        let graph = VerifiableGraph::with_capacity(5, 3);
        assert!(graph.get_node_features(0).is_some());
    }

    #[test]
    fn test_get_node_features_returns_none() {
        let graph = VerifiableGraph::with_capacity(5, 3);
        assert!(graph.get_node_features(100).is_none());
    }

    #[test]
    fn test_remove_edge_returns_true() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        assert!(graph.remove_edge(0));
    }

    #[test]
    fn test_remove_edge_returns_false() {
        let mut graph = VerifiableGraph::new(5);
        assert!(!graph.remove_edge(0));
    }

    #[test]
    fn test_node_feature_offset_returns_some() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(v.node_feature_offset(0, 0).is_some());
    }

    #[test]
    fn test_node_feature_offset_returns_none() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(v.node_feature_offset(100, 0).is_none());
    }

    #[test]
    fn test_node_embedding_offset_returns_some() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(v.node_embedding_offset(0, 0).is_some());
    }

    #[test]
    fn test_node_embedding_offset_returns_none() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(v.node_embedding_offset(100, 0).is_none());
    }
}

