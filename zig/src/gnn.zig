/**
 * @file
 * @ingroup GNN_Wrappers
 */
// MIT License
// Copyright (c) 2025 Matthew Abbott
//
// GlassBoxAI GNN - Zig bindings for GPU-accelerated Graph Neural Network.
//
// Example:
//
//   const gnn = @import("gnn");
//
//   var facade = try gnn.GnnFacade.init(3, 16, 2, 2, .auto);
//   defer facade.deinit();
//
//   facade.createEmptyGraph(5, 3);
//   facade.addEdge(0, 1, null) catch {};
//   facade.setNodeFeatures(0, &.{ 1.0, 0.5, 0.2 });
//
//   const prediction = try facade.predict(&output_buf);
//   const loss = try facade.train(&.{ 0.5, 0.5 });

const std = @import("std");
const c = @cImport({
    @cInclude("gnn_facade.h");
});

pub const BackendType = enum(c_int) {
    cuda = c.GNN_BACKEND_CUDA,
    opencl = c.GNN_BACKEND_OPENCL,
    auto = c.GNN_BACKEND_AUTO,
};

pub const GnnError = error{
    NullPointer,
    InvalidArgument,
    CudaError,
    IoError,
    OpenClError,
    PredictionFailed,
    Unknown,
};

fn errorFromCode(code: c_int) GnnError {
    return switch (code) {
        c.GNN_ERROR_NULL_POINTER => GnnError.NullPointer,
        c.GNN_ERROR_INVALID_ARG => GnnError.InvalidArgument,
        c.GNN_ERROR_CUDA => GnnError.CudaError,
        c.GNN_ERROR_IO => GnnError.IoError,
        c.GNN_ERROR_OPENCL => GnnError.OpenClError,
        else => GnnError.Unknown,
    };
}

fn checkResult(code: c_int) GnnError!void {
    if (code != c.GNN_OK) return errorFromCode(code);
}

pub const GradientFlowInfo = struct {
    layer_idx: u32,
    mean_gradient: f32,
    max_gradient: f32,
    min_gradient: f32,
    gradient_norm: f32,
};

pub const ModelHeader = struct {
    feature_size: u32,
    hidden_size: u32,
    output_size: u32,
    mp_layers: u32,
    learning_rate: f32,
};

pub const GnnFacade = struct {
    handle: *c.GnnHandle,

    pub fn init(feature_size: u32, hidden_size: u32, output_size: u32, num_mp_layers: u32, backend: BackendType) GnnError!GnnFacade {
        const h = c.gnn_create_with_backend(
            @as(c_uint, feature_size),
            @as(c_uint, hidden_size),
            @as(c_uint, output_size),
            @as(c_uint, num_mp_layers),
            @intFromEnum(backend),
        );
        if (h == null) return GnnError.Unknown;
        return GnnFacade{ .handle = h.? };
    }

    pub fn load(filename: [*:0]const u8) GnnError!GnnFacade {
        const h = c.gnn_load(filename);
        if (h == null) return GnnError.IoError;
        return GnnFacade{ .handle = h.? };
    }

    pub fn readModelHeader(filename: [*:0]const u8) GnnError!ModelHeader {
        var hdr: c.GnnModelHeader = undefined;
        const res = c.gnn_read_model_header(filename, &hdr);
        if (res != c.GNN_OK) return errorFromCode(res);
        return ModelHeader{
            .feature_size = hdr.feature_size,
            .hidden_size = hdr.hidden_size,
            .output_size = hdr.output_size,
            .mp_layers = hdr.mp_layers,
            .learning_rate = hdr.learning_rate,
        };
    }

    pub fn deinit(self: *GnnFacade) void {
        c.gnn_free(self.handle);
        self.handle = undefined;
    }

    // ====================================================================
    // Model I/O
    // ====================================================================

    pub fn saveModel(self: *const GnnFacade, filename: [*:0]const u8) GnnError!void {
        return checkResult(c.gnn_save_model(self.handle, filename));
    }

    pub fn loadModel(self: *GnnFacade, filename: [*:0]const u8) GnnError!void {
        return checkResult(c.gnn_load_model(self.handle, filename));
    }

    // ====================================================================
    // Graph Operations
    // ====================================================================

    pub fn createEmptyGraph(self: *GnnFacade, num_nodes: u32, feature_size: u32) void {
        _ = c.gnn_create_empty_graph(self.handle, num_nodes, feature_size);
    }

    pub fn addEdge(self: *GnnFacade, source: u32, target: u32, features: ?[]const f32) GnnError!u32 {
        const feat_ptr: ?[*]const f32 = if (features) |f| f.ptr else null;
        const feat_len: c_uint = if (features) |f| @intCast(f.len) else 0;
        const result = c.gnn_add_edge(self.handle, source, target, feat_ptr, feat_len);
        if (result < 0) return GnnError.InvalidArgument;
        return @intCast(result);
    }

    pub fn removeEdge(self: *GnnFacade, edge_idx: u32) void {
        _ = c.gnn_remove_edge(self.handle, edge_idx);
    }

    pub fn hasEdge(self: *const GnnFacade, source: u32, target: u32) bool {
        return c.gnn_has_edge(self.handle, source, target) == 1;
    }

    pub fn findEdgeIndex(self: *const GnnFacade, source: u32, target: u32) ?u32 {
        const result = c.gnn_find_edge_index(self.handle, source, target);
        if (result < 0) return null;
        return @intCast(result);
    }

    pub fn rebuildAdjacencyList(self: *GnnFacade) void {
        _ = c.gnn_rebuild_adjacency_list(self.handle);
    }

    // ====================================================================
    // Node Features
    // ====================================================================

    pub fn setNodeFeatures(self: *GnnFacade, node_idx: u32, features: []const f32) void {
        _ = c.gnn_set_node_features(self.handle, node_idx, features.ptr, @intCast(features.len));
    }

    pub fn getNodeFeatures(self: *const GnnFacade, node_idx: u32, buf: []f32) ?[]f32 {
        const count = c.gnn_get_node_features(self.handle, node_idx, buf.ptr, @intCast(buf.len));
        if (count < 0) return null;
        return buf[0..@intCast(count)];
    }

    pub fn setNodeFeature(self: *GnnFacade, node_idx: u32, feature_idx: u32, value: f32) void {
        _ = c.gnn_set_node_feature(self.handle, node_idx, feature_idx, value);
    }

    pub fn getNodeFeature(self: *const GnnFacade, node_idx: u32, feature_idx: u32) f32 {
        return c.gnn_get_node_feature(self.handle, node_idx, feature_idx);
    }

    // ====================================================================
    // Edge Features
    // ====================================================================

    pub fn setEdgeFeatures(self: *GnnFacade, edge_idx: u32, features: []const f32) void {
        _ = c.gnn_set_edge_features(self.handle, edge_idx, features.ptr, @intCast(features.len));
    }

    pub fn getEdgeFeatures(self: *const GnnFacade, edge_idx: u32, buf: []f32) ?[]f32 {
        const count = c.gnn_get_edge_features(self.handle, edge_idx, buf.ptr, @intCast(buf.len));
        if (count < 0) return null;
        return buf[0..@intCast(count)];
    }

    // ====================================================================
    // Training & Inference
    // ====================================================================

    pub fn predict(self: *GnnFacade, output: []f32) GnnError![]f32 {
        const count = c.gnn_predict(self.handle, output.ptr, @intCast(output.len));
        if (count < 0) return GnnError.PredictionFailed;
        return output[0..@intCast(count)];
    }

    pub fn train(self: *GnnFacade, target: []const f32) GnnError!f32 {
        var loss: f32 = 0.0;
        const result = c.gnn_train(self.handle, target.ptr, @intCast(target.len), &loss);
        try checkResult(result);
        return loss;
    }

    pub fn trainMultiple(self: *GnnFacade, target: []const f32, iterations: u32) GnnError!void {
        return checkResult(c.gnn_train_multiple(self.handle, target.ptr, @intCast(target.len), iterations));
    }

    // ====================================================================
    // Hyperparameters
    // ====================================================================

    pub fn setLearningRate(self: *GnnFacade, lr: f32) void {
        _ = c.gnn_set_learning_rate(self.handle, lr);
    }

    pub fn getLearningRate(self: *const GnnFacade) f32 {
        return c.gnn_get_learning_rate(self.handle);
    }

    // ====================================================================
    // Graph Info
    // ====================================================================

    pub fn getNumNodes(self: *const GnnFacade) u32 {
        return c.gnn_get_num_nodes(self.handle);
    }

    pub fn getNumEdges(self: *const GnnFacade) u32 {
        return c.gnn_get_num_edges(self.handle);
    }

    pub fn isGraphLoaded(self: *const GnnFacade) bool {
        return c.gnn_is_graph_loaded(self.handle) != 0;
    }

    pub fn getFeatureSize(self: *const GnnFacade) u32 {
        return c.gnn_get_feature_size(self.handle);
    }

    pub fn getHiddenSize(self: *const GnnFacade) u32 {
        return c.gnn_get_hidden_size(self.handle);
    }

    pub fn getOutputSize(self: *const GnnFacade) u32 {
        return c.gnn_get_output_size(self.handle);
    }

    pub fn getNumMessagePassingLayers(self: *const GnnFacade) u32 {
        return c.gnn_get_num_message_passing_layers(self.handle);
    }

    pub fn getInDegree(self: *const GnnFacade, node_idx: u32) u32 {
        return c.gnn_get_in_degree(self.handle, node_idx);
    }

    pub fn getOutDegree(self: *const GnnFacade, node_idx: u32) u32 {
        return c.gnn_get_out_degree(self.handle, node_idx);
    }

    pub fn getNeighbors(self: *const GnnFacade, node_idx: u32, buf: []u32) ?[]u32 {
        const count = c.gnn_get_neighbors(self.handle, node_idx, @ptrCast(buf.ptr), @intCast(buf.len));
        if (count < 0) return null;
        return buf[0..@intCast(count)];
    }

    pub fn getGraphEmbedding(self: *const GnnFacade, buf: []f32) []f32 {
        const count = c.gnn_get_graph_embedding(self.handle, buf.ptr, @intCast(buf.len));
        if (count <= 0) return buf[0..0];
        return buf[0..@intCast(count)];
    }

    // ====================================================================
    // Masking & Dropout
    // ====================================================================

    pub fn setNodeMask(self: *GnnFacade, node_idx: u32, value: bool) void {
        _ = c.gnn_set_node_mask(self.handle, node_idx, if (value) @as(c_int, 1) else @as(c_int, 0));
    }

    pub fn getNodeMask(self: *const GnnFacade, node_idx: u32) bool {
        return c.gnn_get_node_mask(self.handle, node_idx) != 0;
    }

    pub fn setEdgeMask(self: *GnnFacade, edge_idx: u32, value: bool) void {
        _ = c.gnn_set_edge_mask(self.handle, edge_idx, if (value) @as(c_int, 1) else @as(c_int, 0));
    }

    pub fn getEdgeMask(self: *const GnnFacade, edge_idx: u32) bool {
        return c.gnn_get_edge_mask(self.handle, edge_idx) != 0;
    }

    pub fn applyNodeDropout(self: *GnnFacade, rate: f32) void {
        _ = c.gnn_apply_node_dropout(self.handle, rate);
    }

    pub fn applyEdgeDropout(self: *GnnFacade, rate: f32) void {
        _ = c.gnn_apply_edge_dropout(self.handle, rate);
    }

    pub fn getMaskedNodeCount(self: *const GnnFacade) u32 {
        return c.gnn_get_masked_node_count(self.handle);
    }

    pub fn getMaskedEdgeCount(self: *const GnnFacade) u32 {
        return c.gnn_get_masked_edge_count(self.handle);
    }

    // ====================================================================
    // Analytics
    // ====================================================================

    pub fn computePageRank(self: *const GnnFacade, damping: f32, iterations: u32, buf: []f32) []f32 {
        const count = c.gnn_compute_page_rank(self.handle, damping, iterations, buf.ptr, @intCast(buf.len));
        if (count <= 0) return buf[0..0];
        return buf[0..@intCast(count)];
    }

    pub fn getGradientFlow(self: *const GnnFacade, layer_idx: u32) GradientFlowInfo {
        var info: c.GnnGradientFlowInfo = undefined;
        _ = c.gnn_get_gradient_flow(self.handle, layer_idx, &info);
        return GradientFlowInfo{
            .layer_idx = info.layer_idx,
            .mean_gradient = info.mean_gradient,
            .max_gradient = info.max_gradient,
            .min_gradient = info.min_gradient,
            .gradient_norm = info.gradient_norm,
        };
    }

    pub fn getParameterCount(self: *const GnnFacade) u32 {
        return c.gnn_get_parameter_count(self.handle);
    }

    pub fn getBackendName(self: *const GnnFacade, buf: []u8) []const u8 {
        const len = c.gnn_get_backend_name(self.handle, @ptrCast(buf.ptr), @intCast(buf.len));
        if (len <= 0) return "unknown";
        return buf[0..@intCast(len)];
    }

    pub fn getArchitectureSummary(self: *const GnnFacade, buf: []u8) []const u8 {
        const len = c.gnn_get_architecture_summary(self.handle, @ptrCast(buf.ptr), @intCast(buf.len));
        if (len <= 0) return "";
        return buf[0..@intCast(len)];
    }

    pub fn exportGraphToJson(self: *const GnnFacade, buf: []u8) []const u8 {
        const len = c.gnn_export_graph_to_json(self.handle, @ptrCast(buf.ptr), @intCast(buf.len));
        if (len <= 0) return "{}";
        return buf[0..@intCast(len)];
    }

    pub fn getBackendType(self: *const GnnFacade) BackendType {
        return @enumFromInt(c.gnn_get_backend_type(self.handle));
    }

    pub fn getEdgeEndpoints(self: *const GnnFacade, edge_idx: u32) ?struct { source: u32, target: u32 } {
        var src: c_uint = 0;
        var tgt: c_uint = 0;
        const result = c.gnn_get_edge_endpoints(self.handle, edge_idx, &src, &tgt);
        if (result != c.GNN_OK) return null;
        return .{ .source = src, .target = tgt };
    }
};

pub fn detectBackend() BackendType {
    return @enumFromInt(c.gnn_detect_backend());
}

