/**
 * @file
 * @ingroup GNN_Wrappers
 */
// MIT License
// Copyright (c) 2025 Matthew Abbott
//
// GlassBoxAI GNN Zig example.

const std = @import("std");
const gnn = @import("gnn");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    var facade = try gnn.GnnFacade.init(3, 16, 2, 2, .auto);
    defer facade.deinit();

    var name_buf: [32]u8 = undefined;
    const backend_name = facade.getBackendName(&name_buf);
    try stdout.print("Backend: {s}\n", .{backend_name});
    try stdout.print("Parameters: {d}\n", .{facade.getParameterCount()});

    facade.createEmptyGraph(5, 3);
    _ = try facade.addEdge(0, 1, null);
    _ = try facade.addEdge(1, 2, null);
    _ = try facade.addEdge(2, 3, null);
    _ = try facade.addEdge(3, 4, null);

    facade.setNodeFeatures(0, &.{ 1.0, 0.5, 0.2 });
    facade.setNodeFeatures(1, &.{ 0.3, 0.8, 0.1 });
    facade.setNodeFeatures(2, &.{ 0.7, 0.2, 0.9 });
    facade.setNodeFeatures(3, &.{ 0.4, 0.6, 0.3 });
    facade.setNodeFeatures(4, &.{ 0.9, 0.1, 0.5 });

    try stdout.print("Nodes: {d}, Edges: {d}\n", .{ facade.getNumNodes(), facade.getNumEdges() });

    var output_buf: [2]f32 = undefined;
    const prediction = try facade.predict(&output_buf);
    try stdout.print("Prediction: [{d}, {d}]\n", .{ prediction[0], prediction[1] });

    const target = [_]f32{ 0.5, 0.5 };
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        const loss = try facade.train(&target);
        if (i % 5 == 0) {
            try stdout.print("Epoch {d}: loss = {d}\n", .{ i, loss });
        }
    }

    var rank_buf: [5]f32 = undefined;
    const scores = facade.computePageRank(0.85, 20, &rank_buf);
    try stdout.print("PageRank: [", .{});
    for (scores, 0..) |score, idx| {
        if (idx > 0) try stdout.print(", ", .{});
        try stdout.print("{d}", .{score});
    }
    try stdout.print("]\n", .{});

    var summary_buf: [4096]u8 = undefined;
    const summary = facade.getArchitectureSummary(&summary_buf);
    try stdout.print("{s}\n", .{summary});
}

