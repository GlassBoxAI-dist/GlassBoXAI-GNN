//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_nodes_boundary() {
        let graph = VerifiableGraph::new(MAX_NODES);
        assert_eq!(graph.num_nodes, MAX_NODES);
    }

    #[test]
    fn test_many_edges() {
        let mut graph = VerifiableGraph::new(10);
        for i in 0..9 {
            for j in 0..10 {
                if i != j {
                    graph.add_edge(i, j);
                }
            }
        }
        assert!(graph.edges.len() > 50);
    }
}

