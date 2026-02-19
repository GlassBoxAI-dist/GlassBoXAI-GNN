//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_new_creates_empty_graph() {
        let graph = VerifiableGraph::new(10);
        assert_eq!(graph.num_nodes, 10);
        assert!(graph.edges.is_empty());
        assert_eq!(graph.adjacency_list.len(), 10);
    }

    #[test]
    fn test_graph_with_capacity_initializes_features() {
        let graph = VerifiableGraph::with_capacity(5, 3);
        assert_eq!(graph.num_nodes, 5);
        assert_eq!(graph.node_features.len(), 5);
        assert_eq!(graph.node_features[0].len(), 3);
    }

    #[test]
    fn test_graph_zero_nodes() {
        let graph = VerifiableGraph::new(0);
        assert_eq!(graph.num_nodes, 0);
        assert!(graph.adjacency_list.is_empty());
    }

    #[test]
    fn test_graph_single_node() {
        let graph = VerifiableGraph::with_capacity(1, 4);
        assert_eq!(graph.num_nodes, 1);
        assert_eq!(graph.node_features[0].len(), 4);
    }

    #[test]
    fn test_add_edge_valid() {
        let mut graph = VerifiableGraph::new(5);
        let idx = graph.add_edge(0, 1);
        assert_eq!(idx, Some(0));
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn test_add_edge_self_loop() {
        let mut graph = VerifiableGraph::new(5);
        let idx = graph.add_edge(2, 2);
        assert!(idx.is_some());
    }

    #[test]
    fn test_add_multiple_edges() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        assert_eq!(graph.edges.len(), 3);
    }

    #[test]
    fn test_add_duplicate_edges() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 1);
        assert_eq!(graph.edges.len(), 2);
    }

    #[test]
    fn test_get_edge_valid() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(2, 3);
        let edge = graph.get_edge(0);
        assert_eq!(edge, Some((2, 3)));
    }

    #[test]
    fn test_has_edge_exists() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        assert!(graph.has_edge(0, 1));
    }

    #[test]
    fn test_has_edge_not_exists() {
        let graph = VerifiableGraph::new(5);
        assert!(!graph.has_edge(0, 1));
    }

    #[test]
    fn test_has_edge_reverse_direction() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        assert!(!graph.has_edge(1, 0));
    }

    #[test]
    fn test_find_edge_index_exists() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        assert_eq!(graph.find_edge_index(0, 1), Some(0));
        assert_eq!(graph.find_edge_index(1, 2), Some(1));
    }

    #[test]
    fn test_find_edge_index_not_exists() {
        let graph = VerifiableGraph::new(5);
        assert!(graph.find_edge_index(0, 1).is_none());
    }

    #[test]
    fn test_remove_edge_valid() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        assert!(graph.remove_edge(0));
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_get_neighbors_valid() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        let neighbors = graph.get_neighbors(0);
        assert!(neighbors.is_some());
        assert_eq!(neighbors.unwrap().len(), 2);
    }

    #[test]
    fn test_get_neighbors_no_edges() {
        let graph = VerifiableGraph::new(5);
        let neighbors = graph.get_neighbors(0);
        assert!(neighbors.is_some());
        assert!(neighbors.unwrap().is_empty());
    }
}

