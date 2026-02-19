//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_node_feature_valid_index() {
        let graph = VerifiableGraph::with_capacity(5, 3);
        assert_eq!(graph.get_node_feature(0, 0), 0.0);
        assert_eq!(graph.get_node_feature(4, 2), 0.0);
    }

    #[test]
    fn test_get_node_feature_invalid_node() {
        let graph = VerifiableGraph::with_capacity(5, 3);
        assert_eq!(graph.get_node_feature(100, 0), 0.0);
        assert_eq!(graph.get_node_feature(usize::MAX, 0), 0.0);
    }

    #[test]
    fn test_get_node_feature_invalid_feature() {
        let graph = VerifiableGraph::with_capacity(5, 3);
        assert_eq!(graph.get_node_feature(0, 100), 0.0);
        assert_eq!(graph.get_node_feature(0, usize::MAX), 0.0);
    }

    #[test]
    fn test_get_node_feature_both_invalid() {
        let graph = VerifiableGraph::with_capacity(5, 3);
        assert_eq!(graph.get_node_feature(100, 100), 0.0);
    }

    #[test]
    fn test_set_node_feature_valid() {
        let mut graph = VerifiableGraph::with_capacity(5, 3);
        graph.set_node_feature(0, 0, 1.5);
        assert_eq!(graph.get_node_feature(0, 0), 1.5);
    }

    #[test]
    fn test_get_node_features_valid() {
        let graph = VerifiableGraph::with_capacity(5, 3);
        let features = graph.get_node_features(0);
        assert!(features.is_some());
        assert_eq!(features.unwrap().len(), 3);
    }

    #[test]
    fn test_get_node_features_invalid() {
        let graph = VerifiableGraph::with_capacity(5, 3);
        assert!(graph.get_node_features(100).is_none());
    }

    #[test]
    fn test_set_node_features_valid() {
        let mut graph = VerifiableGraph::with_capacity(5, 3);
        graph.set_node_features(0, vec![1.0, 2.0, 3.0]);
        assert_eq!(graph.get_node_feature(0, 0), 1.0);
        assert_eq!(graph.get_node_feature(0, 1), 2.0);
        assert_eq!(graph.get_node_feature(0, 2), 3.0);
    }

    #[test]
    fn test_set_node_features_invalid() {
        let mut graph = VerifiableGraph::with_capacity(5, 3);
        graph.set_node_features(100, vec![1.0, 2.0, 3.0]);
    }
}

