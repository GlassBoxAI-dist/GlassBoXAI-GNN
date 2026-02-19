/**
 * @file
 * @ingroup GNN_Internal_Logic
 */
/**
 * @file gnn_facade.hpp
 * @brief GlassBoxAI GNN - CUDA/OpenCL-accelerated Graph Neural Network C++ API
 *
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * This header provides C++ bindings with RAII, exceptions, and modern C++ features.
 * Supports CUDA and OpenCL backends with auto-detection.
 *
 * @example
 * @code
 * #include "gnn_facade.hpp"
 * #include <iostream>
 *
 * int main() {
 *     try {
 *         // Create a GNN
 *         gnn::GnnFacade gnn(3, 16, 2, 2);
 *
 *         // Create a graph
 *         gnn.createEmptyGraph(5, 3);
 *
 *         // Add edges
 *         gnn.addEdge(0, 1);
 *         gnn.addEdge(1, 2);
 *
 *         // Set node features
 *         gnn.setNodeFeatures(0, {1.0f, 0.5f, 0.2f});
 *
 *         // Make predictions
 *         auto prediction = gnn.predict();
 *         std::cout << "Prediction: [" << prediction[0] << ", " << prediction[1] << "]" << std::endl;
 *
 *         // Train
 *         float loss = gnn.train({0.5f, 0.5f});
 *         std::cout << "Loss: " << loss << std::endl;
 *
 *         // Save model
 *         gnn.saveModel("model.bin");
 *
 *     } catch (const gnn::GnnException& e) {
 *         std::cerr << "GNN Error: " << e.what() << std::endl;
 *         return 1;
 *     }
 *     return 0;
 * }
 * @endcode
 */

#ifndef GNN_FACADE_HPP
#define GNN_FACADE_HPP

#include "gnn_facade.h"
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <optional>

namespace gnn {

/**
 * @brief GPU backend selection
 */
enum class Backend {
    Cuda = GNN_BACKEND_CUDA,
    OpenCL = GNN_BACKEND_OPENCL,
    Auto = GNN_BACKEND_AUTO
};

/**
 * @brief Exception class for GNN errors
 */
class GnnException : public std::runtime_error {
public:
    explicit GnnException(const std::string& message)
        : std::runtime_error(message) {}
    
    explicit GnnException(int error_code)
        : std::runtime_error(errorCodeToString(error_code)), error_code_(error_code) {}
    
    int errorCode() const { return error_code_; }
    
    static std::string errorCodeToString(int code) {
        switch (code) {
            case GNN_OK: return "Success";
            case GNN_ERROR_NULL_POINTER: return "Null pointer";
            case GNN_ERROR_INVALID_ARG: return "Invalid argument";
            case GNN_ERROR_CUDA: return "CUDA error";
            case GNN_ERROR_IO: return "I/O error";
            default: return "Unknown error";
        }
    }
    
private:
    int error_code_ = GNN_ERROR_UNKNOWN;
};

/**
 * @brief Gradient flow information for a layer
 */
struct GradientFlowInfo {
    unsigned int layerIdx;
    float meanGradient;
    float maxGradient;
    float minGradient;
    float gradientNorm;
    
    GradientFlowInfo() = default;
    GradientFlowInfo(const GnnGradientFlowInfo& info)
        : layerIdx(info.layer_idx)
        , meanGradient(info.mean_gradient)
        , maxGradient(info.max_gradient)
        , minGradient(info.min_gradient)
        , gradientNorm(info.gradient_norm)
    {}
};

/**
 * @brief Model header information
 */
struct ModelHeader {
    unsigned int featureSize;
    unsigned int hiddenSize;
    unsigned int outputSize;
    unsigned int mpLayers;
    float learningRate;
    
    ModelHeader() = default;
    ModelHeader(const GnnModelHeader& header)
        : featureSize(header.feature_size)
        , hiddenSize(header.hidden_size)
        , outputSize(header.output_size)
        , mpLayers(header.mp_layers)
        , learningRate(header.learning_rate)
    {}
};

/**
 * @brief GPU-accelerated Graph Neural Network with Facade interface
 *
 * This class provides RAII-based resource management and a modern C++ API
 * for the GNN library. Supports CUDA and OpenCL backends.
 */
class GnnFacade {
public:
    /**
     * @brief Create a new GNN with auto-detected backend
     *
     * @param featureSize Size of input node features
     * @param hiddenSize Size of hidden layers
     * @param outputSize Size of output predictions
     * @param numMpLayers Number of message passing layers
     * @throws GnnException on failure
     */
    GnnFacade(unsigned int featureSize, unsigned int hiddenSize,
              unsigned int outputSize, unsigned int numMpLayers)
        : handle_(gnn_create(featureSize, hiddenSize, outputSize, numMpLayers))
    {
        if (!handle_) {
            throw GnnException("Failed to create GNN");
        }
    }

    /**
     * @brief Create a new GNN with a specific backend
     *
     * @param featureSize Size of input node features
     * @param hiddenSize Size of hidden layers
     * @param outputSize Size of output predictions
     * @param numMpLayers Number of message passing layers
     * @param backend GPU backend selection
     * @throws GnnException on failure
     */
    GnnFacade(unsigned int featureSize, unsigned int hiddenSize,
              unsigned int outputSize, unsigned int numMpLayers,
              Backend backend)
        : handle_(gnn_create_with_backend(featureSize, hiddenSize, outputSize, numMpLayers, static_cast<int>(backend)))
    {
        if (!handle_) {
            throw GnnException("Failed to create GNN with specified backend");
        }
    }
    
    /**
     * @brief Load a GNN from a model file
     *
     * @param filename Path to the model file
     * @throws GnnException on failure
     */
    explicit GnnFacade(const std::string& filename)
        : handle_(gnn_load(filename.c_str()))
    {
        if (!handle_) {
            throw GnnException("Failed to load model: " + filename);
        }
    }
    
    // Non-copyable
    GnnFacade(const GnnFacade&) = delete;
    GnnFacade& operator=(const GnnFacade&) = delete;
    
    // Movable
    GnnFacade(GnnFacade&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    
    GnnFacade& operator=(GnnFacade&& other) noexcept {
        if (this != &other) {
            if (handle_) gnn_free(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    
    ~GnnFacade() {
        if (handle_) {
            gnn_free(handle_);
        }
    }
    
    /**
     * @brief Read model header without loading full model
     *
     * @param filename Path to the model file
     * @return ModelHeader with model information
     * @throws GnnException on failure
     */
    static ModelHeader readModelHeader(const std::string& filename) {
        GnnModelHeader header;
        int result = gnn_read_model_header(filename.c_str(), &header);
        if (result != GNN_OK) {
            throw GnnException(result);
        }
        return ModelHeader(header);
    }
    
    // ========== Model I/O ==========
    
    /**
     * @brief Save model to file
     * @throws GnnException on failure
     */
    void saveModel(const std::string& filename) const {
        int result = gnn_save_model(handle_, filename.c_str());
        if (result != GNN_OK) {
            throw GnnException(result);
        }
    }
    
    /**
     * @brief Load model from file
     * @throws GnnException on failure
     */
    void loadModel(const std::string& filename) {
        int result = gnn_load_model(handle_, filename.c_str());
        if (result != GNN_OK) {
            throw GnnException(result);
        }
    }
    
    // ========== Graph Operations ==========
    
    /** @brief Create an empty graph */
    void createEmptyGraph(unsigned int numNodes, unsigned int featureSize) {
        gnn_create_empty_graph(handle_, numNodes, featureSize);
    }
    
    /**
     * @brief Add an edge to the graph
     * @return Edge index
     */
    int addEdge(unsigned int source, unsigned int target,
                const std::vector<float>& features = {}) {
        return gnn_add_edge(handle_, source, target,
                           features.empty() ? nullptr : features.data(),
                           static_cast<unsigned int>(features.size()));
    }
    
    /** @brief Remove an edge by index */
    void removeEdge(unsigned int edgeIdx) {
        gnn_remove_edge(handle_, edgeIdx);
    }
    
    /** @brief Check if an edge exists */
    bool hasEdge(unsigned int source, unsigned int target) const {
        return gnn_has_edge(handle_, source, target) == 1;
    }
    
    /** @brief Find edge index between two nodes */
    std::optional<int> findEdgeIndex(unsigned int source, unsigned int target) const {
        int result = gnn_find_edge_index(handle_, source, target);
        if (result < 0) return std::nullopt;
        return result;
    }
    
    /** @brief Rebuild adjacency list */
    void rebuildAdjacencyList() {
        gnn_rebuild_adjacency_list(handle_);
    }
    
    // ========== Node Features ==========
    
    /** @brief Set all features for a node */
    void setNodeFeatures(unsigned int nodeIdx, const std::vector<float>& features) {
        gnn_set_node_features(handle_, nodeIdx, features.data(),
                             static_cast<unsigned int>(features.size()));
    }
    
    /** @brief Get all features for a node */
    std::optional<std::vector<float>> getNodeFeatures(unsigned int nodeIdx) const {
        std::vector<float> features(getFeatureSize());
        int count = gnn_get_node_features(handle_, nodeIdx, features.data(),
                                         static_cast<unsigned int>(features.size()));
        if (count < 0) return std::nullopt;
        features.resize(count);
        return features;
    }
    
    /** @brief Set a single feature value */
    void setNodeFeature(unsigned int nodeIdx, unsigned int featureIdx, float value) {
        gnn_set_node_feature(handle_, nodeIdx, featureIdx, value);
    }
    
    /** @brief Get a single feature value */
    float getNodeFeature(unsigned int nodeIdx, unsigned int featureIdx) const {
        return gnn_get_node_feature(handle_, nodeIdx, featureIdx);
    }
    
    // ========== Edge Features ==========
    
    /** @brief Set features for an edge */
    void setEdgeFeatures(unsigned int edgeIdx, const std::vector<float>& features) {
        gnn_set_edge_features(handle_, edgeIdx, features.data(),
                             static_cast<unsigned int>(features.size()));
    }
    
    /** @brief Get features for an edge */
    std::optional<std::vector<float>> getEdgeFeatures(unsigned int edgeIdx) const {
        std::vector<float> features(16); // Reasonable default
        int count = gnn_get_edge_features(handle_, edgeIdx, features.data(),
                                         static_cast<unsigned int>(features.size()));
        if (count < 0) return std::nullopt;
        features.resize(count);
        return features;
    }
    
    // ========== Training & Inference ==========
    
    /**
     * @brief Run prediction on the current graph
     * @return Vector of predictions
     * @throws GnnException on failure
     */
    std::vector<float> predict() {
        std::vector<float> output(getOutputSize());
        int count = gnn_predict(handle_, output.data(),
                               static_cast<unsigned int>(output.size()));
        if (count < 0) {
            throw GnnException("Prediction failed");
        }
        output.resize(count);
        return output;
    }
    
    /**
     * @brief Train on the current graph
     * @return Loss value
     * @throws GnnException on failure
     */
    float train(const std::vector<float>& target) {
        float loss;
        int result = gnn_train(handle_, target.data(),
                              static_cast<unsigned int>(target.size()), &loss);
        if (result != GNN_OK) {
            throw GnnException(result);
        }
        return loss;
    }
    
    /**
     * @brief Train for multiple iterations
     * @throws GnnException on failure
     */
    void trainMultiple(const std::vector<float>& target, unsigned int iterations) {
        int result = gnn_train_multiple(handle_, target.data(),
                                       static_cast<unsigned int>(target.size()),
                                       iterations);
        if (result != GNN_OK) {
            throw GnnException(result);
        }
    }
    
    // ========== Hyperparameters ==========
    
    void setLearningRate(float lr) { gnn_set_learning_rate(handle_, lr); }
    float getLearningRate() const { return gnn_get_learning_rate(handle_); }
    
    // ========== Graph Info ==========
    
    unsigned int getNumNodes() const { return gnn_get_num_nodes(handle_); }
    unsigned int getNumEdges() const { return gnn_get_num_edges(handle_); }
    bool isGraphLoaded() const { return gnn_is_graph_loaded(handle_) != 0; }
    unsigned int getFeatureSize() const { return gnn_get_feature_size(handle_); }
    unsigned int getHiddenSize() const { return gnn_get_hidden_size(handle_); }
    unsigned int getOutputSize() const { return gnn_get_output_size(handle_); }
    unsigned int getNumMessagePassingLayers() const { 
        return gnn_get_num_message_passing_layers(handle_); 
    }
    
    unsigned int getInDegree(unsigned int nodeIdx) const {
        return gnn_get_in_degree(handle_, nodeIdx);
    }
    
    unsigned int getOutDegree(unsigned int nodeIdx) const {
        return gnn_get_out_degree(handle_, nodeIdx);
    }
    
    /** @brief Get neighbors of a node */
    std::optional<std::vector<unsigned int>> getNeighbors(unsigned int nodeIdx) const {
        std::vector<unsigned int> neighbors(getNumNodes());
        int count = gnn_get_neighbors(handle_, nodeIdx, neighbors.data(),
                                     static_cast<unsigned int>(neighbors.size()));
        if (count < 0) return std::nullopt;
        neighbors.resize(count);
        return neighbors;
    }
    
    /** @brief Get graph embedding from last forward pass */
    std::vector<float> getGraphEmbedding() const {
        std::vector<float> embedding(getHiddenSize());
        int count = gnn_get_graph_embedding(handle_, embedding.data(),
                                           static_cast<unsigned int>(embedding.size()));
        if (count > 0) {
            embedding.resize(count);
        }
        return embedding;
    }
    
    // ========== Masking & Dropout ==========
    
    void setNodeMask(unsigned int nodeIdx, bool value) {
        gnn_set_node_mask(handle_, nodeIdx, value ? 1 : 0);
    }
    
    bool getNodeMask(unsigned int nodeIdx) const {
        return gnn_get_node_mask(handle_, nodeIdx) != 0;
    }
    
    void setEdgeMask(unsigned int edgeIdx, bool value) {
        gnn_set_edge_mask(handle_, edgeIdx, value ? 1 : 0);
    }
    
    bool getEdgeMask(unsigned int edgeIdx) const {
        return gnn_get_edge_mask(handle_, edgeIdx) != 0;
    }
    
    void applyNodeDropout(float rate) { gnn_apply_node_dropout(handle_, rate); }
    void applyEdgeDropout(float rate) { gnn_apply_edge_dropout(handle_, rate); }
    
    unsigned int getMaskedNodeCount() const { return gnn_get_masked_node_count(handle_); }
    unsigned int getMaskedEdgeCount() const { return gnn_get_masked_edge_count(handle_); }
    
    // ========== Analytics ==========
    
    /** @brief Compute PageRank scores */
    std::vector<float> computePageRank(float damping = 0.85f, 
                                        unsigned int iterations = 20) const {
        std::vector<float> scores(getNumNodes());
        int count = gnn_compute_page_rank(handle_, damping, iterations,
                                         scores.data(),
                                         static_cast<unsigned int>(scores.size()));
        if (count > 0) {
            scores.resize(count);
        }
        return scores;
    }
    
    /** @brief Get gradient flow information for a layer */
    GradientFlowInfo getGradientFlow(unsigned int layerIdx) const {
        GnnGradientFlowInfo info;
        gnn_get_gradient_flow(handle_, layerIdx, &info);
        return GradientFlowInfo(info);
    }
    
    unsigned int getParameterCount() const { return gnn_get_parameter_count(handle_); }

    /** @brief Detect best available backend (static) */
    static Backend detectBackend() {
        return static_cast<Backend>(gnn_detect_backend());
    }

    /** @brief Get the backend type */
    Backend getBackendType() const {
        return static_cast<Backend>(gnn_get_backend_type(handle_));
    }

    /** @brief Get edge endpoints */
    std::optional<std::pair<unsigned int, unsigned int>> getEdgeEndpoints(unsigned int edgeIdx) const {
        unsigned int src, tgt;
        int result = gnn_get_edge_endpoints(handle_, edgeIdx, &src, &tgt);
        if (result != GNN_OK) return std::nullopt;
        return std::make_pair(src, tgt);
    }

    /** @brief Get the active backend name ("cuda" or "opencl") */
    std::string getBackendName() const {
        std::vector<char> buffer(32);
        int len = gnn_get_backend_name(handle_, buffer.data(),
                                       static_cast<unsigned int>(buffer.size()));
        if (len > 0) {
            return std::string(buffer.data(), len);
        }
        return "unknown";
    }
    
    /** @brief Get architecture summary */
    std::string getArchitectureSummary() const {
        std::vector<char> buffer(4096);
        int len = gnn_get_architecture_summary(handle_, buffer.data(),
                                              static_cast<unsigned int>(buffer.size()));
        if (len > 0) {
            return std::string(buffer.data(), len);
        }
        return "";
    }
    
    /** @brief Export graph to JSON */
    std::string exportGraphToJson() const {
        std::vector<char> buffer(65536);
        int len = gnn_export_graph_to_json(handle_, buffer.data(),
                                          static_cast<unsigned int>(buffer.size()));
        if (len > 0) {
            return std::string(buffer.data(), len);
        }
        return "{}";
    }
    
    /** @brief Get the raw C handle (for advanced use) */
    GnnHandle* handle() { return handle_; }
    const GnnHandle* handle() const { return handle_; }
    
private:
    GnnHandle* handle_;
};

} // namespace gnn

#endif /* GNN_FACADE_HPP */
