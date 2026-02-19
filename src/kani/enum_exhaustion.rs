//! @file
//! @ingroup GNN_Core_Verified
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_features_creation() {
        let ef = EdgeFeatures {
            source: 0,
            target: 1,
            features: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(ef.source, 0);
        assert_eq!(ef.target, 1);
        assert_eq!(ef.features.len(), 3);
    }

    #[test]
    fn test_edge_features_empty() {
        let ef = EdgeFeatures {
            source: 0,
            target: 1,
            features: vec![],
        };
        assert!(ef.features.is_empty());
    }

    #[test]
    fn test_edge_features_clone() {
        let ef = EdgeFeatures {
            source: 0,
            target: 1,
            features: vec![1.0, 2.0],
        };
        let ef2 = ef.clone();
        assert_eq!(ef.source, ef2.source);
        assert_eq!(ef.target, ef2.target);
        assert_eq!(ef.features, ef2.features);
    }
}

