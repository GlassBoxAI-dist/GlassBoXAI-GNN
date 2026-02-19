/**
 * @file
 * @ingroup GNN_Wrappers
 */
/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * C# P/Invoke declarations for GlassBoxAI GNN native library.
 */

using System;
using System.Runtime.InteropServices;

namespace GnnFacadeCuda
{
    [StructLayout(LayoutKind.Sequential)]
    public struct GnnGradientFlowInfo
    {
        public uint LayerIdx;
        public float MeanGradient;
        public float MaxGradient;
        public float MinGradient;
        public float GradientNorm;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct GnnModelHeader
    {
        public uint FeatureSize;
        public uint HiddenSize;
        public uint OutputSize;
        public uint MpLayers;
        public float LearningRate;
    }

    internal static class NativeMethods
    {
        private const string LibName = "gnn_facade_cuda";

        // Error codes
        public const int GNN_OK = 0;
        public const int GNN_ERROR_NULL_POINTER = -1;
        public const int GNN_ERROR_INVALID_ARG = -2;
        public const int GNN_ERROR_CUDA = -3;
        public const int GNN_ERROR_IO = -4;
        public const int GNN_ERROR_OPENCL = -5;
        public const int GNN_ERROR_UNKNOWN = -99;

        // Backend constants
        public const int GNN_BACKEND_CUDA = 0;
        public const int GNN_BACKEND_OPENCL = 1;
        public const int GNN_BACKEND_AUTO = 2;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr gnn_create(uint feature_size, uint hidden_size, uint output_size, uint num_mp_layers);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr gnn_create_with_backend(uint feature_size, uint hidden_size, uint output_size, uint num_mp_layers, int backend);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr gnn_load([MarshalAs(UnmanagedType.LPUTF8Str)] string filename);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void gnn_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int gnn_read_model_header([MarshalAs(UnmanagedType.LPUTF8Str)] string filename, out GnnModelHeader header);

        // ====================================================================
        // Model I/O
        // ====================================================================

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int gnn_save_model(IntPtr handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string filename);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int gnn_load_model(IntPtr handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string filename);

        // ====================================================================
        // Graph Operations
        // ====================================================================

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_create_empty_graph(IntPtr handle, uint num_nodes, uint feature_size);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_add_edge(IntPtr handle, uint source, uint target, float[]? features, uint features_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_remove_edge(IntPtr handle, uint edge_idx);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_has_edge(IntPtr handle, uint source, uint target);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_find_edge_index(IntPtr handle, uint source, uint target);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_rebuild_adjacency_list(IntPtr handle);

        // ====================================================================
        // Node Features
        // ====================================================================

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_set_node_features(IntPtr handle, uint node_idx, float[] features, uint features_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_get_node_features(IntPtr handle, uint node_idx, float[] features_out, uint features_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_set_node_feature(IntPtr handle, uint node_idx, uint feature_idx, float value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern float gnn_get_node_feature(IntPtr handle, uint node_idx, uint feature_idx);

        // ====================================================================
        // Edge Features
        // ====================================================================

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_set_edge_features(IntPtr handle, uint edge_idx, float[] features, uint features_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_get_edge_features(IntPtr handle, uint edge_idx, float[] features_out, uint features_len);

        // ====================================================================
        // Training & Inference
        // ====================================================================

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_predict(IntPtr handle, float[] output, uint output_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_train(IntPtr handle, float[] target, uint target_len, out float loss_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_train_multiple(IntPtr handle, float[] target, uint target_len, uint iterations);

        // ====================================================================
        // Hyperparameters
        // ====================================================================

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_set_learning_rate(IntPtr handle, float lr);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern float gnn_get_learning_rate(IntPtr handle);

        // ====================================================================
        // Graph Info
        // ====================================================================

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint gnn_get_num_nodes(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint gnn_get_num_edges(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_is_graph_loaded(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint gnn_get_feature_size(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint gnn_get_hidden_size(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint gnn_get_output_size(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint gnn_get_num_message_passing_layers(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint gnn_get_in_degree(IntPtr handle, uint node_idx);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint gnn_get_out_degree(IntPtr handle, uint node_idx);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_get_neighbors(IntPtr handle, uint node_idx, uint[] neighbors_out, uint neighbors_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_get_graph_embedding(IntPtr handle, float[] embedding_out, uint embedding_len);

        // ====================================================================
        // Masking & Dropout
        // ====================================================================

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_set_node_mask(IntPtr handle, uint node_idx, int value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_get_node_mask(IntPtr handle, uint node_idx);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_set_edge_mask(IntPtr handle, uint edge_idx, int value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_get_edge_mask(IntPtr handle, uint edge_idx);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_apply_node_dropout(IntPtr handle, float rate);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_apply_edge_dropout(IntPtr handle, float rate);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint gnn_get_masked_node_count(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint gnn_get_masked_edge_count(IntPtr handle);

        // ====================================================================
        // Analytics
        // ====================================================================

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_compute_page_rank(IntPtr handle, float damping, uint iterations, float[] scores_out, uint scores_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_get_gradient_flow(IntPtr handle, uint layer_idx, out GnnGradientFlowInfo info_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint gnn_get_parameter_count(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_get_architecture_summary(IntPtr handle, byte[] buffer, uint buffer_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_export_graph_to_json(IntPtr handle, byte[] buffer, uint buffer_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_get_backend_name(IntPtr handle, byte[] buffer, uint buffer_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_detect_backend();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_get_backend_type(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int gnn_get_edge_endpoints(IntPtr handle, uint edge_idx, out uint source_out, out uint target_out);
    }
}
