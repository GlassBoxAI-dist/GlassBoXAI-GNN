//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_mask_operations() {
        let mut mgr = NodeMaskManager::new(5);
        for i in 0..5 {
            mgr.set_mask(i, false);
        }
        for i in 0..5 {
            assert!(!mgr.get_mask(i));
        }
        for i in 0..5 {
            mgr.toggle_mask(i);
        }
        for i in 0..5 {
            assert!(mgr.get_mask(i));
        }
    }

    #[test]
    fn test_sequential_edge_mask_operations() {
        let mut mgr = EdgeMaskManager::new();
        for _ in 0..5 {
            mgr.add_edge();
        }
        for i in 0..5 {
            mgr.set_mask(i, false);
        }
        for i in 0..5 {
            assert!(!mgr.get_mask(i));
        }
    }

    #[test]
    fn test_interleaved_add_remove_edges() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.remove_edge(0);
        graph.add_edge(2, 3);
        assert_eq!(graph.edges.len(), 2);
    }
}

