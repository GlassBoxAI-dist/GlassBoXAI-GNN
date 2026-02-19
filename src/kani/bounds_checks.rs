//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn proof_get_edge_bounds_safe() {
        let graph = VerifiableGraph::new(4);
        let edge_idx: usize = kani::any();
        let _result = graph.get_edge(edge_idx);
    }

    #[kani::proof]
    fn proof_add_edge_bounds_checked() {
        let mut graph = VerifiableGraph::new(4);
        let source: usize = kani::any();
        let target: usize = kani::any();
        let result = graph.add_edge(source, target);
        if source >= 4 || target >= 4 {
            kani::assert(result.is_none(), "add_edge must reject out-of-bounds nodes");
        }
    }

    #[kani::proof]
    fn proof_buffer_validator_node_correctness() {
        let max_nodes: usize = 100;
        let validator = BufferIndexValidator::new(max_nodes, 1000, 16, 64, 10);
        let node_idx: usize = kani::any();
        if node_idx < max_nodes {
            kani::assert(validator.validate_node_index(node_idx), "Valid node must pass");
        } else {
            kani::assert(!validator.validate_node_index(node_idx), "Invalid node must fail");
        }
    }

    #[kani::proof]
    fn proof_buffer_validator_edge_correctness() {
        let max_edges: usize = 1000;
        let validator = BufferIndexValidator::new(100, max_edges, 16, 64, 10);
        let edge_idx: usize = kani::any();
        if edge_idx < max_edges {
            kani::assert(validator.validate_edge_index(edge_idx), "Valid edge must pass");
        } else {
            kani::assert(!validator.validate_edge_index(edge_idx), "Invalid edge must fail");
        }
    }

    #[kani::proof]
    fn proof_node_feature_offset_bounds() {
        let max_nodes: usize = 100;
        let feature_size: usize = 16;
        let validator = BufferIndexValidator::new(max_nodes, 1000, feature_size, 64, 10);
        let node_idx: usize = kani::any();
        let feat_idx: usize = kani::any();
        let result = validator.node_feature_offset(node_idx, feat_idx);
        if node_idx >= max_nodes || feat_idx >= feature_size {
            kani::assert(result.is_none(), "Out-of-bounds must return None");
        }
    }

    #[kani::proof]
    fn proof_node_embedding_offset_bounds() {
        let max_nodes: usize = 100;
        let hidden_size: usize = 64;
        let validator = BufferIndexValidator::new(max_nodes, 1000, 16, hidden_size, 10);
        let node_idx: usize = kani::any();
        let hidden_idx: usize = kani::any();
        let result = validator.node_embedding_offset(node_idx, hidden_idx);
        if node_idx >= max_nodes || hidden_idx >= hidden_size {
            kani::assert(result.is_none(), "Out-of-bounds must return None");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_new() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert_eq!(v.max_nodes, 100);
        assert_eq!(v.max_edges, 1000);
        assert_eq!(v.feature_size, 16);
        assert_eq!(v.hidden_size, 64);
        assert_eq!(v.output_size, 10);
    }

    #[test]
    fn test_validator_node_index_valid() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(v.validate_node_index(0));
        assert!(v.validate_node_index(50));
        assert!(v.validate_node_index(99));
    }

    #[test]
    fn test_validator_node_index_invalid() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(!v.validate_node_index(100));
        assert!(!v.validate_node_index(1000));
        assert!(!v.validate_node_index(usize::MAX));
    }

    #[test]
    fn test_validator_edge_index_valid() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(v.validate_edge_index(0));
        assert!(v.validate_edge_index(500));
        assert!(v.validate_edge_index(999));
    }

    #[test]
    fn test_validator_edge_index_invalid() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(!v.validate_edge_index(1000));
        assert!(!v.validate_edge_index(usize::MAX));
    }

    #[test]
    fn test_validator_feature_index_valid() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(v.validate_feature_index(0));
        assert!(v.validate_feature_index(15));
    }

    #[test]
    fn test_validator_feature_index_invalid() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(!v.validate_feature_index(16));
        assert!(!v.validate_feature_index(100));
    }

    #[test]
    fn test_validator_node_feature_offset_valid() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert_eq!(v.node_feature_offset(0, 0), Some(0));
        assert_eq!(v.node_feature_offset(5, 3), Some(5 * 16 + 3));
        assert_eq!(v.node_feature_offset(99, 15), Some(99 * 16 + 15));
    }

    #[test]
    fn test_validator_node_feature_offset_invalid() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(v.node_feature_offset(100, 0).is_none());
        assert!(v.node_feature_offset(0, 16).is_none());
        assert!(v.node_feature_offset(100, 16).is_none());
    }

    #[test]
    fn test_validator_node_embedding_offset_valid() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert_eq!(v.node_embedding_offset(0, 0), Some(0));
        assert_eq!(v.node_embedding_offset(5, 10), Some(5 * 64 + 10));
    }

    #[test]
    fn test_validator_node_embedding_offset_invalid() {
        let v = BufferIndexValidator::new(100, 1000, 16, 64, 10);
        assert!(v.node_embedding_offset(100, 0).is_none());
        assert!(v.node_embedding_offset(0, 64).is_none());
    }

    #[test]
    fn test_edge_at_max_minus_one() {
        let v = BufferIndexValidator::new(MAX_NODES, MAX_EDGES, 16, 64, 10);
        assert!(v.validate_node_index(MAX_NODES - 1));
        assert!(v.validate_edge_index(MAX_EDGES - 1));
    }

    #[test]
    fn test_edge_at_max() {
        let v = BufferIndexValidator::new(MAX_NODES, MAX_EDGES, 16, 64, 10);
        assert!(!v.validate_node_index(MAX_NODES));
        assert!(!v.validate_edge_index(MAX_EDGES));
    }
}

