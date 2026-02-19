/**
 * @file
 * @ingroup GNN_Wrappers
 */
// MIT License
// Copyright (c) 2025 Matthew Abbott
//
// Build configuration for GlassBoxAI GNN Zig bindings.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const gnn_mod = b.addModule("gnn", .{
        .root_source_file = b.path("src/gnn.zig"),
        .target = target,
        .optimize = optimize,
    });
    gnn_mod.addIncludePath(b.path("../include"));

    const lib = b.addStaticLibrary(.{
        .name = "gnn-zig",
        .root_source_file = b.path("src/gnn.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.addIncludePath(b.path("../include"));
    lib.linkSystemLibrary("gnn_facade_cuda");
    lib.addLibraryPath(b.path("../target/release"));
    lib.linkLibC();
    b.installArtifact(lib);

    const example = b.addExecutable(.{
        .name = "gnn-example",
        .root_source_file = b.path("example/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    example.root_module.addImport("gnn", gnn_mod);
    example.addIncludePath(b.path("../include"));
    example.linkSystemLibrary("gnn_facade_cuda");
    example.addLibraryPath(b.path("../target/release"));
    example.linkLibC();
    b.installArtifact(example);

    const run_cmd = b.addRunArtifact(example);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the example");
    run_step.dependOn(&run_cmd.step);
}

