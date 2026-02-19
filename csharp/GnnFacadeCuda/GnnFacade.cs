/**
 * @file
 * @ingroup GNN_Wrappers
 */
/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * GlassBoxAI GNN - C# bindings for GPU-accelerated Graph Neural Network.
 *
 * Example:
 *
 *   using GnnFacadeCuda;
 *
 *   using var gnn = new GnnFacade(3, 16, 2, 2);
 *   gnn.CreateEmptyGraph(5, 3);
 *   gnn.AddEdge(0, 1);
 *   gnn.AddEdge(1, 2);
 *   gnn.SetNodeFeatures(0, new float[] { 1.0f, 0.5f, 0.2f });
 *
 *   float[] prediction = gnn.Predict();
 *   Console.WriteLine($"Prediction: [{prediction[0]}, {prediction[1]}]");
 *
 *   float loss = gnn.Train(new float[] { 0.5f, 0.5f });
 *   Console.WriteLine($"Loss: {loss}");
 *
 *   gnn.SaveModel("model.bin");
 */

using System;
using System.Text;

namespace GnnFacadeCuda
{
    public enum Backend
    {
        Cuda = 0,
        OpenCL = 1,
        Auto = 2,
    }

    public class GnnException : Exception
    {
        public int ErrorCode { get; }

        public GnnException(string message) : base(message) { }

        public GnnException(int errorCode)
            : base(ErrorCodeToString(errorCode))
        {
            ErrorCode = errorCode;
        }

        public static string ErrorCodeToString(int code) => code switch
        {
            NativeMethods.GNN_OK => "Success",
            NativeMethods.GNN_ERROR_NULL_POINTER => "Null pointer",
            NativeMethods.GNN_ERROR_INVALID_ARG => "Invalid argument",
            NativeMethods.GNN_ERROR_CUDA => "CUDA error",
            NativeMethods.GNN_ERROR_IO => "I/O error",
            NativeMethods.GNN_ERROR_OPENCL => "OpenCL error",
            _ => "Unknown error",
        };
    }

    public sealed class GnnFacade : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public GnnFacade(uint featureSize, uint hiddenSize, uint outputSize, uint numMpLayers)
        {
            _handle = NativeMethods.gnn_create(featureSize, hiddenSize, outputSize, numMpLayers);
            if (_handle == IntPtr.Zero)
                throw new GnnException("Failed to create GNN");
        }

        public GnnFacade(uint featureSize, uint hiddenSize, uint outputSize, uint numMpLayers, Backend backend)
        {
            _handle = NativeMethods.gnn_create_with_backend(featureSize, hiddenSize, outputSize, numMpLayers, (int)backend);
            if (_handle == IntPtr.Zero)
                throw new GnnException("Failed to create GNN with specified backend");
        }

        private GnnFacade(IntPtr handle)
        {
            _handle = handle;
        }

        public static GnnFacade Load(string filename)
        {
            var handle = NativeMethods.gnn_load(filename);
            if (handle == IntPtr.Zero)
                throw new GnnException($"Failed to load model: {filename}");
            return new GnnFacade(handle);
        }

        public static GnnModelHeader ReadModelHeader(string filename)
        {
            int result = NativeMethods.gnn_read_model_header(filename, out var header);
            if (result != NativeMethods.GNN_OK)
                throw new GnnException(result);
            return header;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    NativeMethods.gnn_free(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~GnnFacade()
        {
            Dispose();
        }

        private void CheckDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(GnnFacade));
        }

        private static void CheckResult(int result)
        {
            if (result != NativeMethods.GNN_OK)
                throw new GnnException(result);
        }

        // ====================================================================
        // Model I/O
        // ====================================================================

        public void SaveModel(string filename)
        {
            CheckDisposed();
            CheckResult(NativeMethods.gnn_save_model(_handle, filename));
        }

        public void LoadModel(string filename)
        {
            CheckDisposed();
            CheckResult(NativeMethods.gnn_load_model(_handle, filename));
        }

        // ====================================================================
        // Graph Operations
        // ====================================================================

        public void CreateEmptyGraph(uint numNodes, uint featureSize)
        {
            CheckDisposed();
            NativeMethods.gnn_create_empty_graph(_handle, numNodes, featureSize);
        }

        public int AddEdge(uint source, uint target, float[]? features = null)
        {
            CheckDisposed();
            uint len = features != null ? (uint)features.Length : 0;
            int result = NativeMethods.gnn_add_edge(_handle, source, target, features, len);
            if (result < 0)
                throw new GnnException("Failed to add edge");
            return result;
        }

        public void RemoveEdge(uint edgeIdx)
        {
            CheckDisposed();
            NativeMethods.gnn_remove_edge(_handle, edgeIdx);
        }

        public bool HasEdge(uint source, uint target)
        {
            CheckDisposed();
            return NativeMethods.gnn_has_edge(_handle, source, target) == 1;
        }

        public int? FindEdgeIndex(uint source, uint target)
        {
            CheckDisposed();
            int result = NativeMethods.gnn_find_edge_index(_handle, source, target);
            return result >= 0 ? result : null;
        }

        public void RebuildAdjacencyList()
        {
            CheckDisposed();
            NativeMethods.gnn_rebuild_adjacency_list(_handle);
        }

        // ====================================================================
        // Node Features
        // ====================================================================

        public void SetNodeFeatures(uint nodeIdx, float[] features)
        {
            CheckDisposed();
            NativeMethods.gnn_set_node_features(_handle, nodeIdx, features, (uint)features.Length);
        }

        public float[]? GetNodeFeatures(uint nodeIdx)
        {
            CheckDisposed();
            var buf = new float[FeatureSize];
            int count = NativeMethods.gnn_get_node_features(_handle, nodeIdx, buf, (uint)buf.Length);
            if (count < 0) return null;
            if (count != buf.Length) Array.Resize(ref buf, count);
            return buf;
        }

        public void SetNodeFeature(uint nodeIdx, uint featureIdx, float value)
        {
            CheckDisposed();
            NativeMethods.gnn_set_node_feature(_handle, nodeIdx, featureIdx, value);
        }

        public float GetNodeFeature(uint nodeIdx, uint featureIdx)
        {
            CheckDisposed();
            return NativeMethods.gnn_get_node_feature(_handle, nodeIdx, featureIdx);
        }

        // ====================================================================
        // Edge Features
        // ====================================================================

        public void SetEdgeFeatures(uint edgeIdx, float[] features)
        {
            CheckDisposed();
            NativeMethods.gnn_set_edge_features(_handle, edgeIdx, features, (uint)features.Length);
        }

        public float[]? GetEdgeFeatures(uint edgeIdx)
        {
            CheckDisposed();
            var buf = new float[16];
            int count = NativeMethods.gnn_get_edge_features(_handle, edgeIdx, buf, (uint)buf.Length);
            if (count < 0) return null;
            if (count != buf.Length) Array.Resize(ref buf, count);
            return buf;
        }

        // ====================================================================
        // Training & Inference
        // ====================================================================

        public float[] Predict()
        {
            CheckDisposed();
            var output = new float[OutputSize];
            int count = NativeMethods.gnn_predict(_handle, output, (uint)output.Length);
            if (count < 0)
                throw new GnnException("Prediction failed");
            if (count != output.Length) Array.Resize(ref output, count);
            return output;
        }

        public float Train(float[] target)
        {
            CheckDisposed();
            int result = NativeMethods.gnn_train(_handle, target, (uint)target.Length, out float loss);
            CheckResult(result);
            return loss;
        }

        public void TrainMultiple(float[] target, uint iterations)
        {
            CheckDisposed();
            CheckResult(NativeMethods.gnn_train_multiple(_handle, target, (uint)target.Length, iterations));
        }

        // ====================================================================
        // Hyperparameters
        // ====================================================================

        public float LearningRate
        {
            get { CheckDisposed(); return NativeMethods.gnn_get_learning_rate(_handle); }
            set { CheckDisposed(); NativeMethods.gnn_set_learning_rate(_handle, value); }
        }

        // ====================================================================
        // Graph Info
        // ====================================================================

        public uint NumNodes { get { CheckDisposed(); return NativeMethods.gnn_get_num_nodes(_handle); } }
        public uint NumEdges { get { CheckDisposed(); return NativeMethods.gnn_get_num_edges(_handle); } }
        public bool IsGraphLoaded { get { CheckDisposed(); return NativeMethods.gnn_is_graph_loaded(_handle) != 0; } }
        public uint FeatureSize { get { CheckDisposed(); return NativeMethods.gnn_get_feature_size(_handle); } }
        public uint HiddenSize { get { CheckDisposed(); return NativeMethods.gnn_get_hidden_size(_handle); } }
        public uint OutputSize { get { CheckDisposed(); return NativeMethods.gnn_get_output_size(_handle); } }
        public uint NumMessagePassingLayers { get { CheckDisposed(); return NativeMethods.gnn_get_num_message_passing_layers(_handle); } }

        public uint GetInDegree(uint nodeIdx)
        {
            CheckDisposed();
            return NativeMethods.gnn_get_in_degree(_handle, nodeIdx);
        }

        public uint GetOutDegree(uint nodeIdx)
        {
            CheckDisposed();
            return NativeMethods.gnn_get_out_degree(_handle, nodeIdx);
        }

        public uint[]? GetNeighbors(uint nodeIdx)
        {
            CheckDisposed();
            var buf = new uint[NumNodes];
            int count = NativeMethods.gnn_get_neighbors(_handle, nodeIdx, buf, (uint)buf.Length);
            if (count < 0) return null;
            if (count != buf.Length) Array.Resize(ref buf, count);
            return buf;
        }

        public float[] GetGraphEmbedding()
        {
            CheckDisposed();
            var buf = new float[HiddenSize];
            int count = NativeMethods.gnn_get_graph_embedding(_handle, buf, (uint)buf.Length);
            if (count > 0 && count != buf.Length) Array.Resize(ref buf, count);
            return buf;
        }

        // ====================================================================
        // Masking & Dropout
        // ====================================================================

        public void SetNodeMask(uint nodeIdx, bool value)
        {
            CheckDisposed();
            NativeMethods.gnn_set_node_mask(_handle, nodeIdx, value ? 1 : 0);
        }

        public bool GetNodeMask(uint nodeIdx)
        {
            CheckDisposed();
            return NativeMethods.gnn_get_node_mask(_handle, nodeIdx) != 0;
        }

        public void SetEdgeMask(uint edgeIdx, bool value)
        {
            CheckDisposed();
            NativeMethods.gnn_set_edge_mask(_handle, edgeIdx, value ? 1 : 0);
        }

        public bool GetEdgeMask(uint edgeIdx)
        {
            CheckDisposed();
            return NativeMethods.gnn_get_edge_mask(_handle, edgeIdx) != 0;
        }

        public void ApplyNodeDropout(float rate)
        {
            CheckDisposed();
            NativeMethods.gnn_apply_node_dropout(_handle, rate);
        }

        public void ApplyEdgeDropout(float rate)
        {
            CheckDisposed();
            NativeMethods.gnn_apply_edge_dropout(_handle, rate);
        }

        public uint MaskedNodeCount { get { CheckDisposed(); return NativeMethods.gnn_get_masked_node_count(_handle); } }
        public uint MaskedEdgeCount { get { CheckDisposed(); return NativeMethods.gnn_get_masked_edge_count(_handle); } }

        // ====================================================================
        // Analytics
        // ====================================================================

        public float[] ComputePageRank(float damping = 0.85f, uint iterations = 20)
        {
            CheckDisposed();
            var scores = new float[NumNodes];
            int count = NativeMethods.gnn_compute_page_rank(_handle, damping, iterations, scores, (uint)scores.Length);
            if (count > 0 && count != scores.Length) Array.Resize(ref scores, count);
            return scores;
        }

        public GnnGradientFlowInfo GetGradientFlow(uint layerIdx)
        {
            CheckDisposed();
            NativeMethods.gnn_get_gradient_flow(_handle, layerIdx, out var info);
            return info;
        }

        public uint ParameterCount { get { CheckDisposed(); return NativeMethods.gnn_get_parameter_count(_handle); } }

        public string BackendName
        {
            get
            {
                CheckDisposed();
                var buf = new byte[32];
                int len = NativeMethods.gnn_get_backend_name(_handle, buf, (uint)buf.Length);
                return len > 0 ? Encoding.UTF8.GetString(buf, 0, len) : "unknown";
            }
        }

        public string GetArchitectureSummary()
        {
            CheckDisposed();
            var buf = new byte[4096];
            int len = NativeMethods.gnn_get_architecture_summary(_handle, buf, (uint)buf.Length);
            return len > 0 ? Encoding.UTF8.GetString(buf, 0, len) : "";
        }

        public string ExportGraphToJson()
        {
            CheckDisposed();
            var buf = new byte[65536];
            int len = NativeMethods.gnn_export_graph_to_json(_handle, buf, (uint)buf.Length);
            return len > 0 ? Encoding.UTF8.GetString(buf, 0, len) : "{}";
        }

        public static Backend DetectBackend()
        {
            return (Backend)NativeMethods.gnn_detect_backend();
        }

        public Backend BackendType
        {
            get { CheckDisposed(); return (Backend)NativeMethods.gnn_get_backend_type(_handle); }
        }

        public (uint Source, uint Target)? GetEdgeEndpoints(uint edgeIdx)
        {
            CheckDisposed();
            int result = NativeMethods.gnn_get_edge_endpoints(_handle, edgeIdx, out uint src, out uint tgt);
            if (result != NativeMethods.GNN_OK) return null;
            return (src, tgt);
        }
    }
}
