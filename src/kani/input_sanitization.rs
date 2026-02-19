//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_edge_invalid_source() {
        let mut graph = VerifiableGraph::new(5);
        assert!(graph.add_edge(100, 1).is_none());
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_add_edge_invalid_target() {
        let mut graph = VerifiableGraph::new(5);
        assert!(graph.add_edge(0, 100).is_none());
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_add_edge_both_invalid() {
        let mut graph = VerifiableGraph::new(5);
        assert!(graph.add_edge(100, 200).is_none());
    }

    #[test]
    fn test_get_edge_invalid() {
        let graph = VerifiableGraph::new(5);
        assert!(graph.get_edge(0).is_none());
        assert!(graph.get_edge(100).is_none());
    }

    #[test]
    fn test_remove_edge_invalid() {
        let mut graph = VerifiableGraph::new(5);
        assert!(!graph.remove_edge(0));
        assert!(!graph.remove_edge(100));
    }

    #[test]
    fn test_get_neighbors_invalid() {
        let graph = VerifiableGraph::new(5);
        assert!(graph.get_neighbors(100).is_none());
    }

    #[test]
    fn test_node_mask_get_invalid() {
        let mgr = NodeMaskManager::new(5);
        assert!(!mgr.get_mask(100));
        assert!(!mgr.get_mask(usize::MAX));
    }

    #[test]
    fn test_edge_mask_get_invalid() {
        let mgr = EdgeMaskManager::new();
        assert!(!mgr.get_mask(0));
        assert!(!mgr.get_mask(100));
    }

    #[test]
    fn test_edge_mask_remove_invalid() {
        let mut mgr = EdgeMaskManager::new();
        assert!(!mgr.remove_edge(0));
        assert!(!mgr.remove_edge(100));
    }
}

