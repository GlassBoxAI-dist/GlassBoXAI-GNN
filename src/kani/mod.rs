//! @file
//! @ingroup GNN_Core_Verified
#![allow(dead_code)]

pub const MAX_NODES: usize = 1000;
pub const MAX_EDGES: usize = 10000;
pub const MAX_FEATURES: usize = 256;

#[derive(Clone, Debug)]
pub struct VerifiableGraph {
    pub num_nodes: usize,
    pub node_features: Vec<Vec<f32>>,
    pub edges: Vec<(usize, usize)>,
    pub adjacency_list: Vec<Vec<usize>>,
}

impl VerifiableGraph {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            node_features: Vec::new(),
            edges: Vec::new(),
            adjacency_list: vec![Vec::new(); num_nodes],
        }
    }

    pub fn with_capacity(num_nodes: usize, feature_size: usize) -> Self {
        Self {
            num_nodes,
            node_features: vec![vec![0.0; feature_size]; num_nodes],
            edges: Vec::new(),
            adjacency_list: vec![Vec::new(); num_nodes],
        }
    }

    #[inline]
    pub fn get_node_feature(&self, node_idx: usize, feature_idx: usize) -> f32 {
        self.node_features
            .get(node_idx)
            .and_then(|f| f.get(feature_idx))
            .copied()
            .unwrap_or(0.0)
    }

    #[inline]
    pub fn set_node_feature(&mut self, node_idx: usize, feature_idx: usize, value: f32) {
        if let Some(features) = self.node_features.get_mut(node_idx) {
            if let Some(f) = features.get_mut(feature_idx) {
                *f = value;
            }
        }
    }

    #[inline]
    pub fn get_node_features(&self, node_idx: usize) -> Option<&Vec<f32>> {
        self.node_features.get(node_idx)
    }

    #[inline]
    pub fn set_node_features(&mut self, node_idx: usize, features: Vec<f32>) {
        if node_idx < self.node_features.len() {
            self.node_features[node_idx] = features;
        }
    }

    #[inline]
    pub fn get_edge(&self, edge_idx: usize) -> Option<(usize, usize)> {
        self.edges.get(edge_idx).copied()
    }

    #[inline]
    pub fn add_edge(&mut self, source: usize, target: usize) -> Option<usize> {
        if source >= self.num_nodes || target >= self.num_nodes {
            return None;
        }
        if self.edges.len() >= MAX_EDGES {
            return None;
        }

        self.edges.push((source, target));
        if source < self.adjacency_list.len() {
            self.adjacency_list[source].push(target);
        }
        Some(self.edges.len() - 1)
    }

    pub fn remove_edge(&mut self, edge_idx: usize) -> bool {
        if edge_idx >= self.edges.len() {
            return false;
        }
        self.edges.remove(edge_idx);
        self.rebuild_adjacency_list();
        true
    }

    #[inline]
    pub fn get_neighbors(&self, node_idx: usize) -> Option<&Vec<usize>> {
        self.adjacency_list.get(node_idx)
    }

    #[inline]
    pub fn get_in_degree(&self, node_idx: usize) -> usize {
        self.edges.iter().filter(|&&(_, t)| t == node_idx).count()
    }

    #[inline]
    pub fn get_out_degree(&self, node_idx: usize) -> usize {
        self.adjacency_list.get(node_idx).map_or(0, |adj| adj.len())
    }

    #[inline]
    pub fn has_edge(&self, source: usize, target: usize) -> bool {
        self.edges.iter().any(|&(s, t)| s == source && t == target)
    }

    #[inline]
    pub fn find_edge_index(&self, source: usize, target: usize) -> Option<usize> {
        self.edges.iter().position(|&(s, t)| s == source && t == target)
    }

    pub fn rebuild_adjacency_list(&mut self) {
        self.adjacency_list = vec![Vec::new(); self.num_nodes];
        for &(src, tgt) in &self.edges {
            if src < self.adjacency_list.len() {
                self.adjacency_list[src].push(tgt);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct EdgeFeatures {
    pub source: usize,
    pub target: usize,
    pub features: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct NodeMaskManager {
    masks: Vec<bool>,
}

impl NodeMaskManager {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            masks: vec![true; num_nodes],
        }
    }

    #[inline]
    pub fn get_mask(&self, node_idx: usize) -> bool {
        self.masks.get(node_idx).copied().unwrap_or(false)
    }

    #[inline]
    pub fn set_mask(&mut self, node_idx: usize, value: bool) {
        if let Some(m) = self.masks.get_mut(node_idx) {
            *m = value;
        }
    }

    #[inline]
    pub fn toggle_mask(&mut self, node_idx: usize) {
        if let Some(m) = self.masks.get_mut(node_idx) {
            *m = !*m;
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EdgeMaskManager {
    masks: Vec<bool>,
}

impl EdgeMaskManager {
    pub fn new() -> Self {
        Self { masks: Vec::new() }
    }

    #[inline]
    pub fn get_mask(&self, edge_idx: usize) -> bool {
        self.masks.get(edge_idx).copied().unwrap_or(false)
    }

    #[inline]
    pub fn set_mask(&mut self, edge_idx: usize, value: bool) {
        if let Some(m) = self.masks.get_mut(edge_idx) {
            *m = value;
        }
    }

    pub fn add_edge(&mut self) -> bool {
        if self.masks.len() >= MAX_EDGES {
            return false;
        }
        self.masks.push(true);
        true
    }

    pub fn remove_edge(&mut self, edge_idx: usize) -> bool {
        if edge_idx >= self.masks.len() {
            return false;
        }
        self.masks.remove(edge_idx);
        true
    }
}

pub struct BufferIndexValidator {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub feature_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

impl BufferIndexValidator {
    pub fn new(
        max_nodes: usize,
        max_edges: usize,
        feature_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Self {
        Self { max_nodes, max_edges, feature_size, hidden_size, output_size }
    }

    #[inline]
    pub fn validate_node_index(&self, node_idx: usize) -> bool {
        node_idx < self.max_nodes
    }

    #[inline]
    pub fn validate_edge_index(&self, edge_idx: usize) -> bool {
        edge_idx < self.max_edges
    }

    #[inline]
    pub fn validate_feature_index(&self, feature_idx: usize) -> bool {
        feature_idx < self.feature_size
    }

    #[inline]
    pub fn node_feature_offset(&self, node_idx: usize, feature_idx: usize) -> Option<usize> {
        if node_idx < self.max_nodes && feature_idx < self.feature_size {
            Some(node_idx * self.feature_size + feature_idx)
        } else {
            None
        }
    }

    #[inline]
    pub fn node_embedding_offset(&self, node_idx: usize, hidden_idx: usize) -> Option<usize> {
        if node_idx < self.max_nodes && hidden_idx < self.hidden_size {
            Some(node_idx * self.hidden_size + hidden_idx)
        } else {
            None
        }
    }
}

mod bounds_checks;
mod constant_time;
mod deadlock_free;
mod division_by_zero;
mod enum_exhaustion;
mod ffi_c_boundary;
mod ffi_cuda_boundary;
mod ffi_opencl_boundary;
mod floating_point;
mod input_sanitization;
mod integer_overflow;
mod memory_leaks;
mod no_panic;
mod pointer_validity;
mod resource_limits;
mod result_coverage;
mod state_consistency;
mod state_machine;

#[cfg(all(test, feature = "ffi", feature = "cuda"))]
mod ffi_cuda;
#[cfg(all(test, feature = "ffi", feature = "opencl"))]
mod ffi_opencl;

