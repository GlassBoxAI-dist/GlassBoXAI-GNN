//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_in_degree_valid() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 2);
        graph.add_edge(1, 2);
        graph.add_edge(3, 2);
        assert_eq!(graph.get_in_degree(2), 3);
    }

    #[test]
    fn test_get_in_degree_zero() {
        let graph = VerifiableGraph::new(5);
        assert_eq!(graph.get_in_degree(0), 0);
    }

    #[test]
    fn test_get_in_degree_invalid_node() {
        let graph = VerifiableGraph::new(5);
        assert_eq!(graph.get_in_degree(100), 0);
    }

    #[test]
    fn test_get_out_degree_valid() {
        let mut graph = VerifiableGraph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        assert_eq!(graph.get_out_degree(0), 3);
    }

    #[test]
    fn test_get_out_degree_zero() {
        let graph = VerifiableGraph::new(5);
        assert_eq!(graph.get_out_degree(0), 0);
    }

    #[test]
    fn test_get_out_degree_invalid() {
        let graph = VerifiableGraph::new(5);
        assert_eq!(graph.get_out_degree(100), 0);
    }
}

