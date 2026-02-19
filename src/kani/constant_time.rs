//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_node_feature_constant_default() {
        let graph = VerifiableGraph::with_capacity(5, 3);
        assert_eq!(graph.get_node_feature(0, 0), 0.0);
        assert_eq!(graph.get_node_feature(100, 0), 0.0);
        assert_eq!(graph.get_node_feature(0, 100), 0.0);
    }

    #[test]
    fn test_get_mask_constant_default() {
        let mgr = NodeMaskManager::new(5);
        assert!(!mgr.get_mask(100));

        let edge_mgr = EdgeMaskManager::new();
        assert!(!edge_mgr.get_mask(100));
    }

    #[test]
    fn test_get_out_degree_constant_default() {
        let graph = VerifiableGraph::new(5);
        assert_eq!(graph.get_out_degree(0), 0);
        assert_eq!(graph.get_out_degree(100), 0);
    }
}

