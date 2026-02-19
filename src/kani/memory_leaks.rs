//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rebuild_adjacency_list() {
        let mut graph = VerifiableGraph::new(5);
        graph.edges.push((0, 1));
        graph.edges.push((0, 2));
        graph.rebuild_adjacency_list();
        assert_eq!(graph.get_out_degree(0), 2);
    }

    #[test]
    fn test_remove_edge_rebuilds_adjacency() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.remove_edge(0);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.get_out_degree(0), 1);
    }
}

