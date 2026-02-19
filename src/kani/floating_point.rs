//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_node_feature_special_values() {
        let mut graph = VerifiableGraph::with_capacity(5, 3);
        graph.set_node_feature(0, 0, f32::INFINITY);
        assert_eq!(graph.get_node_feature(0, 0), f32::INFINITY);
        graph.set_node_feature(0, 1, f32::NEG_INFINITY);
        assert_eq!(graph.get_node_feature(0, 1), f32::NEG_INFINITY);
        graph.set_node_feature(0, 2, f32::NAN);
        assert!(graph.get_node_feature(0, 2).is_nan());
    }

    #[test]
    fn test_many_features() {
        let mut graph = VerifiableGraph::with_capacity(10, 100);
        for i in 0..10 {
            for j in 0..100 {
                graph.set_node_feature(i, j, (i * 100 + j) as f32);
            }
        }
        assert_eq!(graph.get_node_feature(5, 50), 550.0);
    }
}

