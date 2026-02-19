//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_mask_new() {
        let mgr = NodeMaskManager::new(10);
        assert!(mgr.get_mask(0));
        assert!(mgr.get_mask(9));
    }

    #[test]
    fn test_node_mask_get_valid() {
        let mgr = NodeMaskManager::new(5);
        assert!(mgr.get_mask(0));
    }

    #[test]
    fn test_node_mask_set_valid() {
        let mut mgr = NodeMaskManager::new(5);
        mgr.set_mask(0, false);
        assert!(!mgr.get_mask(0));
        mgr.set_mask(0, true);
        assert!(mgr.get_mask(0));
    }

    #[test]
    fn test_node_mask_toggle_valid() {
        let mut mgr = NodeMaskManager::new(5);
        assert!(mgr.get_mask(0));
        mgr.toggle_mask(0);
        assert!(!mgr.get_mask(0));
        mgr.toggle_mask(0);
        assert!(mgr.get_mask(0));
    }

    #[test]
    fn test_edge_mask_new() {
        let mgr = EdgeMaskManager::new();
        assert!(mgr.masks.is_empty());
    }

    #[test]
    fn test_edge_mask_add() {
        let mut mgr = EdgeMaskManager::new();
        assert!(mgr.add_edge());
        assert!(mgr.get_mask(0));
    }

    #[test]
    fn test_edge_mask_get_valid() {
        let mut mgr = EdgeMaskManager::new();
        mgr.add_edge();
        assert!(mgr.get_mask(0));
    }

    #[test]
    fn test_edge_mask_set_valid() {
        let mut mgr = EdgeMaskManager::new();
        mgr.add_edge();
        mgr.set_mask(0, false);
        assert!(!mgr.get_mask(0));
    }

    #[test]
    fn test_edge_mask_remove_valid() {
        let mut mgr = EdgeMaskManager::new();
        mgr.add_edge();
        mgr.add_edge();
        assert!(mgr.remove_edge(0));
        assert_eq!(mgr.masks.len(), 1);
    }
}

