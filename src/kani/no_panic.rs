//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn proof_get_node_feature_never_panics() {
        let graph = VerifiableGraph::with_capacity(4, 4);
        let node_idx: usize = kani::any();
        let feat_idx: usize = kani::any();
        let _result = graph.get_node_feature(node_idx, feat_idx);
    }

    #[kani::proof]
    fn proof_set_node_feature_never_panics() {
        let mut graph = VerifiableGraph::with_capacity(4, 4);
        let node_idx: usize = kani::any();
        let feat_idx: usize = kani::any();
        let value: f32 = kani::any();
        graph.set_node_feature(node_idx, feat_idx, value);
    }

    #[kani::proof]
    fn proof_get_node_features_never_panics() {
        let graph = VerifiableGraph::with_capacity(4, 4);
        let node_idx: usize = kani::any();
        let _result = graph.get_node_features(node_idx);
    }

    #[kani::proof]
    fn proof_has_edge_never_panics() {
        let graph = VerifiableGraph::new(4);
        let source: usize = kani::any();
        let target: usize = kani::any();
        let _result = graph.has_edge(source, target);
    }

    #[kani::proof]
    fn proof_find_edge_index_never_panics() {
        let graph = VerifiableGraph::new(4);
        let source: usize = kani::any();
        let target: usize = kani::any();
        let _result = graph.find_edge_index(source, target);
    }

    #[kani::proof]
    fn proof_remove_edge_never_panics() {
        let mut graph = VerifiableGraph::new(4);
        let edge_idx: usize = kani::any();
        let _result = graph.remove_edge(edge_idx);
    }

    #[kani::proof]
    fn proof_get_neighbors_never_panics() {
        let graph = VerifiableGraph::new(4);
        let node_idx: usize = kani::any();
        let _result = graph.get_neighbors(node_idx);
    }

    #[kani::proof]
    fn proof_get_in_degree_never_panics() {
        let graph = VerifiableGraph::new(4);
        let node_idx: usize = kani::any();
        let _result = graph.get_in_degree(node_idx);
    }

    #[kani::proof]
    fn proof_get_out_degree_never_panics() {
        let graph = VerifiableGraph::new(4);
        let node_idx: usize = kani::any();
        let _result = graph.get_out_degree(node_idx);
    }

    #[kani::proof]
    fn proof_node_mask_get_set_never_panic() {
        let mut mask_mgr = NodeMaskManager::new(4);
        let node_idx: usize = kani::any();
        let value: bool = kani::any();
        let _get = mask_mgr.get_mask(node_idx);
        mask_mgr.set_mask(node_idx, value);
    }

    #[kani::proof]
    fn proof_node_mask_toggle_never_panics() {
        let mut mask_mgr = NodeMaskManager::new(4);
        let node_idx: usize = kani::any();
        mask_mgr.toggle_mask(node_idx);
    }

    #[kani::proof]
    fn proof_edge_mask_get_set_never_panic() {
        let mut mask_mgr = EdgeMaskManager::new();
        for _ in 0..4 { let _ = mask_mgr.add_edge(); }
        let edge_idx: usize = kani::any();
        let value: bool = kani::any();
        let _get = mask_mgr.get_mask(edge_idx);
        mask_mgr.set_mask(edge_idx, value);
    }

    #[kani::proof]
    fn proof_edge_mask_remove_never_panics() {
        let mut mask_mgr = EdgeMaskManager::new();
        for _ in 0..4 { let _ = mask_mgr.add_edge(); }
        let edge_idx: usize = kani::any();
        let _result = mask_mgr.remove_edge(edge_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_node_feature_invalid_node() {
        let mut graph = VerifiableGraph::with_capacity(5, 3);
        graph.set_node_feature(100, 0, 1.5);
    }

    #[test]
    fn test_set_node_feature_invalid_feature() {
        let mut graph = VerifiableGraph::with_capacity(5, 3);
        graph.set_node_feature(0, 100, 1.5);
    }

    #[test]
    fn test_node_mask_set_invalid() {
        let mut mgr = NodeMaskManager::new(5);
        mgr.set_mask(100, false);
    }

    #[test]
    fn test_node_mask_toggle_invalid() {
        let mut mgr = NodeMaskManager::new(5);
        mgr.toggle_mask(100);
    }

    #[test]
    fn test_edge_mask_set_invalid() {
        let mut mgr = EdgeMaskManager::new();
        mgr.set_mask(100, false);
    }
}

