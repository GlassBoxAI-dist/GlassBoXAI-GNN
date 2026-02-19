//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_node_feature_usize_max() {
        let graph = VerifiableGraph::with_capacity(5, 3);
        assert_eq!(graph.get_node_feature(usize::MAX, 0), 0.0);
        assert_eq!(graph.get_node_feature(0, usize::MAX), 0.0);
    }

    #[test]
    fn test_validator_usize_max() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(!v.validate_node_index(usize::MAX));
        assert!(!v.validate_edge_index(usize::MAX));
    }

    #[test]
    fn test_node_mask_usize_max() {
        let mgr = NodeMaskManager::new(5);
        assert!(!mgr.get_mask(usize::MAX));
    }
}

