/**
 * @file
 * @ingroup GNN_Wrappers
 */
// Package gnn provides Go bindings for the GlassBoxAI GNN library.
//
// This package wraps the GPU-accelerated Graph Neural Network library
// with Facade pattern, providing a native Go API. Supports CUDA and OpenCL backends.
//
// Example:
//
//	package main
//
//	import (
//		"fmt"
//		"log"
//
//		"github.com/GlassBoxAI/GlassBoxAI-GNN/go/gnn"
//	)
//
//	func main() {
//		// Create a new GNN
//		g, err := gnn.New(3, 16, 2, 2)
//		if err != nil {
//			log.Fatal(err)
//		}
//		defer g.Close()
//
//		// Create a graph
//		g.CreateEmptyGraph(5, 3)
//
//		// Add edges
//		g.AddEdge(0, 1, nil)
//		g.AddEdge(1, 2, nil)
//
//		// Set node features
//		g.SetNodeFeatures(0, []float32{1.0, 0.5, 0.2})
//
//		// Make predictions
//		prediction, err := g.Predict()
//		if err != nil {
//			log.Fatal(err)
//		}
//		fmt.Println("Prediction:", prediction)
//
//		// Train
//		loss, err := g.Train([]float32{0.5, 0.5})
//		if err != nil {
//			log.Fatal(err)
//		}
//		fmt.Println("Loss:", loss)
//
//		// Save model
//		if err := g.SaveModel("model.bin"); err != nil {
//			log.Fatal(err)
//		}
//	}
package gnn

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -lgnn_facade_cuda -lm -ldl -lpthread

#include "gnn_facade.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// Backend represents a GPU backend type.
type Backend int

const (
	// BackendCUDA selects the NVIDIA CUDA backend.
	BackendCUDA Backend = 0
	// BackendOpenCL selects the OpenCL backend (AMD, Intel, NVIDIA).
	BackendOpenCL Backend = 1
	// BackendAuto auto-detects the best available backend.
	BackendAuto Backend = 2
)

// Error codes
var (
	ErrNullPointer    = errors.New("null pointer")
	ErrInvalidArg     = errors.New("invalid argument")
	ErrCUDA           = errors.New("CUDA error")
	ErrOpenCL         = errors.New("OpenCL error")
	ErrIO             = errors.New("I/O error")
	ErrUnknown        = errors.New("unknown error")
	ErrPredictFailed  = errors.New("prediction failed")
	ErrNotFound       = errors.New("not found")
)

func errorFromCode(code C.int) error {
	switch code {
	case C.GNN_OK:
		return nil
	case C.GNN_ERROR_NULL_POINTER:
		return ErrNullPointer
	case C.GNN_ERROR_INVALID_ARG:
		return ErrInvalidArg
	case C.GNN_ERROR_CUDA:
		return ErrCUDA
	case C.GNN_ERROR_IO:
		return ErrIO
	default:
		return ErrUnknown
	}
}

// GradientFlowInfo contains gradient flow information for a layer.
type GradientFlowInfo struct {
	LayerIdx     uint
	MeanGradient float32
	MaxGradient  float32
	MinGradient  float32
	GradientNorm float32
}

// ModelHeader contains model header information.
type ModelHeader struct {
	FeatureSize  uint
	HiddenSize   uint
	OutputSize   uint
	MPLayers     uint
	LearningRate float32
}

// GNN represents a GPU-accelerated Graph Neural Network.
type GNN struct {
	handle *C.GnnHandle
}

// New creates a new GNN with the specified parameters and auto-detected backend.
//
// Parameters:
//   - featureSize: Size of input node features
//   - hiddenSize: Size of hidden layers
//   - outputSize: Size of output predictions
//   - numMPLayers: Number of message passing layers
func New(featureSize, hiddenSize, outputSize, numMPLayers uint) (*GNN, error) {
	return NewWithBackend(featureSize, hiddenSize, outputSize, numMPLayers, BackendAuto)
}

// NewWithBackend creates a new GNN with the specified parameters and backend.
//
// Parameters:
//   - featureSize: Size of input node features
//   - hiddenSize: Size of hidden layers
//   - outputSize: Size of output predictions
//   - numMPLayers: Number of message passing layers
//   - backend: GPU backend selection (BackendCUDA, BackendOpenCL, or BackendAuto)
func NewWithBackend(featureSize, hiddenSize, outputSize, numMPLayers uint, backend Backend) (*GNN, error) {
	handle := C.gnn_create_with_backend(
		C.uint(featureSize),
		C.uint(hiddenSize),
		C.uint(outputSize),
		C.uint(numMPLayers),
		C.int(backend),
	)
	if handle == nil {
		return nil, errors.New("failed to create GNN")
	}

	g := &GNN{handle: handle}
	runtime.SetFinalizer(g, (*GNN).Close)
	return g, nil
}

// Load loads a GNN from a saved model file.
func Load(filename string) (*GNN, error) {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	handle := C.gnn_load(cFilename)
	if handle == nil {
		return nil, fmt.Errorf("failed to load model: %s", filename)
	}

	g := &GNN{handle: handle}
	runtime.SetFinalizer(g, (*GNN).Close)
	return g, nil
}

// ReadModelHeader reads model header without loading the full model.
func ReadModelHeader(filename string) (*ModelHeader, error) {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	var header C.GnnModelHeader
	result := C.gnn_read_model_header(cFilename, &header)
	if result != C.GNN_OK {
		return nil, errorFromCode(result)
	}

	return &ModelHeader{
		FeatureSize:  uint(header.feature_size),
		HiddenSize:   uint(header.hidden_size),
		OutputSize:   uint(header.output_size),
		MPLayers:     uint(header.mp_layers),
		LearningRate: float32(header.learning_rate),
	}, nil
}

// Close frees the GNN resources.
func (g *GNN) Close() error {
	if g.handle != nil {
		C.gnn_free(g.handle)
		g.handle = nil
	}
	return nil
}

// SaveModel saves the model to a file.
func (g *GNN) SaveModel(filename string) error {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	result := C.gnn_save_model(g.handle, cFilename)
	return errorFromCode(result)
}

// LoadModel loads model weights from a file.
func (g *GNN) LoadModel(filename string) error {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	result := C.gnn_load_model(g.handle, cFilename)
	return errorFromCode(result)
}

// CreateEmptyGraph creates an empty graph with the specified number of nodes.
func (g *GNN) CreateEmptyGraph(numNodes, featureSize uint) {
	C.gnn_create_empty_graph(g.handle, C.uint(numNodes), C.uint(featureSize))
}

// AddEdge adds an edge to the graph.
// Returns the edge index or an error.
func (g *GNN) AddEdge(source, target uint, features []float32) (int, error) {
	var featPtr *C.float
	var featLen C.uint
	if len(features) > 0 {
		featPtr = (*C.float)(unsafe.Pointer(&features[0]))
		featLen = C.uint(len(features))
	}

	result := C.gnn_add_edge(g.handle, C.uint(source), C.uint(target), featPtr, featLen)
	if result < 0 {
		return -1, ErrInvalidArg
	}
	return int(result), nil
}

// RemoveEdge removes an edge by index.
func (g *GNN) RemoveEdge(edgeIdx uint) {
	C.gnn_remove_edge(g.handle, C.uint(edgeIdx))
}

// HasEdge checks if an edge exists between two nodes.
func (g *GNN) HasEdge(source, target uint) bool {
	return C.gnn_has_edge(g.handle, C.uint(source), C.uint(target)) == 1
}

// FindEdgeIndex finds the index of an edge between two nodes.
// Returns -1 if not found.
func (g *GNN) FindEdgeIndex(source, target uint) int {
	return int(C.gnn_find_edge_index(g.handle, C.uint(source), C.uint(target)))
}

// RebuildAdjacencyList rebuilds the adjacency list from edges.
func (g *GNN) RebuildAdjacencyList() {
	C.gnn_rebuild_adjacency_list(g.handle)
}

// SetNodeFeatures sets all features for a node.
func (g *GNN) SetNodeFeatures(nodeIdx uint, features []float32) error {
	if len(features) == 0 {
		return ErrInvalidArg
	}

	result := C.gnn_set_node_features(
		g.handle,
		C.uint(nodeIdx),
		(*C.float)(unsafe.Pointer(&features[0])),
		C.uint(len(features)),
	)
	return errorFromCode(result)
}

// GetNodeFeatures gets all features for a node.
func (g *GNN) GetNodeFeatures(nodeIdx uint) ([]float32, error) {
	featureSize := g.GetFeatureSize()
	features := make([]float32, featureSize)

	count := C.gnn_get_node_features(
		g.handle,
		C.uint(nodeIdx),
		(*C.float)(unsafe.Pointer(&features[0])),
		C.uint(len(features)),
	)
	if count < 0 {
		return nil, ErrNotFound
	}
	return features[:count], nil
}

// SetNodeFeature sets a single feature value for a node.
func (g *GNN) SetNodeFeature(nodeIdx, featureIdx uint, value float32) {
	C.gnn_set_node_feature(g.handle, C.uint(nodeIdx), C.uint(featureIdx), C.float(value))
}

// GetNodeFeature gets a single feature value for a node.
func (g *GNN) GetNodeFeature(nodeIdx, featureIdx uint) float32 {
	return float32(C.gnn_get_node_feature(g.handle, C.uint(nodeIdx), C.uint(featureIdx)))
}

// SetEdgeFeatures sets features for an edge.
func (g *GNN) SetEdgeFeatures(edgeIdx uint, features []float32) error {
	if len(features) == 0 {
		return ErrInvalidArg
	}

	result := C.gnn_set_edge_features(
		g.handle,
		C.uint(edgeIdx),
		(*C.float)(unsafe.Pointer(&features[0])),
		C.uint(len(features)),
	)
	return errorFromCode(result)
}

// GetEdgeFeatures gets features for an edge.
func (g *GNN) GetEdgeFeatures(edgeIdx uint) ([]float32, error) {
	features := make([]float32, 16) // Reasonable default

	count := C.gnn_get_edge_features(
		g.handle,
		C.uint(edgeIdx),
		(*C.float)(unsafe.Pointer(&features[0])),
		C.uint(len(features)),
	)
	if count < 0 {
		return nil, ErrNotFound
	}
	return features[:count], nil
}

// Predict runs prediction on the current graph.
func (g *GNN) Predict() ([]float32, error) {
	outputSize := g.GetOutputSize()
	output := make([]float32, outputSize)

	count := C.gnn_predict(
		g.handle,
		(*C.float)(unsafe.Pointer(&output[0])),
		C.uint(len(output)),
	)
	if count < 0 {
		return nil, ErrPredictFailed
	}
	return output[:count], nil
}

// Train trains on the current graph with target values.
// Returns the loss value.
func (g *GNN) Train(target []float32) (float32, error) {
	if len(target) == 0 {
		return 0, ErrInvalidArg
	}

	var loss C.float
	result := C.gnn_train(
		g.handle,
		(*C.float)(unsafe.Pointer(&target[0])),
		C.uint(len(target)),
		&loss,
	)
	if result != C.GNN_OK {
		return 0, errorFromCode(result)
	}
	return float32(loss), nil
}

// TrainMultiple trains for multiple iterations.
func (g *GNN) TrainMultiple(target []float32, iterations uint) error {
	if len(target) == 0 {
		return ErrInvalidArg
	}

	result := C.gnn_train_multiple(
		g.handle,
		(*C.float)(unsafe.Pointer(&target[0])),
		C.uint(len(target)),
		C.uint(iterations),
	)
	return errorFromCode(result)
}

// SetLearningRate sets the learning rate.
func (g *GNN) SetLearningRate(lr float32) {
	C.gnn_set_learning_rate(g.handle, C.float(lr))
}

// GetLearningRate gets the current learning rate.
func (g *GNN) GetLearningRate() float32 {
	return float32(C.gnn_get_learning_rate(g.handle))
}

// GetNumNodes returns the number of nodes in the graph.
func (g *GNN) GetNumNodes() uint {
	return uint(C.gnn_get_num_nodes(g.handle))
}

// GetNumEdges returns the number of edges in the graph.
func (g *GNN) GetNumEdges() uint {
	return uint(C.gnn_get_num_edges(g.handle))
}

// IsGraphLoaded returns true if a graph is loaded.
func (g *GNN) IsGraphLoaded() bool {
	return C.gnn_is_graph_loaded(g.handle) != 0
}

// GetFeatureSize returns the feature size.
func (g *GNN) GetFeatureSize() uint {
	return uint(C.gnn_get_feature_size(g.handle))
}

// GetHiddenSize returns the hidden layer size.
func (g *GNN) GetHiddenSize() uint {
	return uint(C.gnn_get_hidden_size(g.handle))
}

// GetOutputSize returns the output size.
func (g *GNN) GetOutputSize() uint {
	return uint(C.gnn_get_output_size(g.handle))
}

// GetNumMessagePassingLayers returns the number of message passing layers.
func (g *GNN) GetNumMessagePassingLayers() uint {
	return uint(C.gnn_get_num_message_passing_layers(g.handle))
}

// GetInDegree returns the in-degree of a node.
func (g *GNN) GetInDegree(nodeIdx uint) uint {
	return uint(C.gnn_get_in_degree(g.handle, C.uint(nodeIdx)))
}

// GetOutDegree returns the out-degree of a node.
func (g *GNN) GetOutDegree(nodeIdx uint) uint {
	return uint(C.gnn_get_out_degree(g.handle, C.uint(nodeIdx)))
}

// GetNeighbors returns the neighbors of a node.
func (g *GNN) GetNeighbors(nodeIdx uint) ([]uint, error) {
	numNodes := g.GetNumNodes()
	neighbors := make([]C.uint, numNodes)

	count := C.gnn_get_neighbors(
		g.handle,
		C.uint(nodeIdx),
		&neighbors[0],
		C.uint(len(neighbors)),
	)
	if count < 0 {
		return nil, ErrNotFound
	}

	result := make([]uint, count)
	for i := 0; i < int(count); i++ {
		result[i] = uint(neighbors[i])
	}
	return result, nil
}

// GetGraphEmbedding returns the graph embedding from the last forward pass.
func (g *GNN) GetGraphEmbedding() []float32 {
	hiddenSize := g.GetHiddenSize()
	embedding := make([]float32, hiddenSize)

	count := C.gnn_get_graph_embedding(
		g.handle,
		(*C.float)(unsafe.Pointer(&embedding[0])),
		C.uint(len(embedding)),
	)
	if count > 0 {
		return embedding[:count]
	}
	return nil
}

// SetNodeMask sets the node mask value.
func (g *GNN) SetNodeMask(nodeIdx uint, value bool) {
	var v C.int
	if value {
		v = 1
	}
	C.gnn_set_node_mask(g.handle, C.uint(nodeIdx), v)
}

// GetNodeMask gets the node mask value.
func (g *GNN) GetNodeMask(nodeIdx uint) bool {
	return C.gnn_get_node_mask(g.handle, C.uint(nodeIdx)) != 0
}

// SetEdgeMask sets the edge mask value.
func (g *GNN) SetEdgeMask(edgeIdx uint, value bool) {
	var v C.int
	if value {
		v = 1
	}
	C.gnn_set_edge_mask(g.handle, C.uint(edgeIdx), v)
}

// GetEdgeMask gets the edge mask value.
func (g *GNN) GetEdgeMask(edgeIdx uint) bool {
	return C.gnn_get_edge_mask(g.handle, C.uint(edgeIdx)) != 0
}

// ApplyNodeDropout applies random dropout to nodes.
func (g *GNN) ApplyNodeDropout(rate float32) {
	C.gnn_apply_node_dropout(g.handle, C.float(rate))
}

// ApplyEdgeDropout applies random dropout to edges.
func (g *GNN) ApplyEdgeDropout(rate float32) {
	C.gnn_apply_edge_dropout(g.handle, C.float(rate))
}

// GetMaskedNodeCount returns the count of active (masked) nodes.
func (g *GNN) GetMaskedNodeCount() uint {
	return uint(C.gnn_get_masked_node_count(g.handle))
}

// GetMaskedEdgeCount returns the count of active (masked) edges.
func (g *GNN) GetMaskedEdgeCount() uint {
	return uint(C.gnn_get_masked_edge_count(g.handle))
}

// ComputePageRank computes PageRank scores for all nodes.
func (g *GNN) ComputePageRank(damping float32, iterations uint) []float32 {
	numNodes := g.GetNumNodes()
	scores := make([]float32, numNodes)

	count := C.gnn_compute_page_rank(
		g.handle,
		C.float(damping),
		C.uint(iterations),
		(*C.float)(unsafe.Pointer(&scores[0])),
		C.uint(len(scores)),
	)
	if count > 0 {
		return scores[:count]
	}
	return nil
}

// GetGradientFlow returns gradient flow information for a layer.
func (g *GNN) GetGradientFlow(layerIdx uint) *GradientFlowInfo {
	var info C.GnnGradientFlowInfo
	C.gnn_get_gradient_flow(g.handle, C.uint(layerIdx), &info)

	return &GradientFlowInfo{
		LayerIdx:     uint(info.layer_idx),
		MeanGradient: float32(info.mean_gradient),
		MaxGradient:  float32(info.max_gradient),
		MinGradient:  float32(info.min_gradient),
		GradientNorm: float32(info.gradient_norm),
	}
}

// GetParameterCount returns the total number of trainable parameters.
func (g *GNN) GetParameterCount() uint {
	return uint(C.gnn_get_parameter_count(g.handle))
}

// GetBackendName returns the name of the active GPU backend ("cuda" or "opencl").
func (g *GNN) GetBackendName() string {
	buf := make([]byte, 32)
	n := C.gnn_get_backend_name(g.handle, (*C.char)(unsafe.Pointer(&buf[0])), C.uint(len(buf)))
	if n > 0 {
		return string(buf[:n])
	}
	return "unknown"
}

// GetArchitectureSummary returns a summary of the network architecture.
func (g *GNN) GetArchitectureSummary() string {
	buffer := make([]byte, 4096)
	length := C.gnn_get_architecture_summary(
		g.handle,
		(*C.char)(unsafe.Pointer(&buffer[0])),
		C.uint(len(buffer)),
	)
	if length > 0 {
		return string(buffer[:length])
	}
	return ""
}

// ExportGraphToJSON exports the graph structure as JSON.
func (g *GNN) ExportGraphToJSON() string {
	buffer := make([]byte, 65536)
	length := C.gnn_export_graph_to_json(
		g.handle,
		(*C.char)(unsafe.Pointer(&buffer[0])),
		C.uint(len(buffer)),
	)
	if length > 0 {
		return string(buffer[:length])
	}
	return "{}"
}

// DetectBackend detects the best available GPU backend without requiring
// an existing GNN handle. Returns BackendCUDA, BackendOpenCL, or BackendAuto.
func DetectBackend() Backend {
	return Backend(C.gnn_detect_backend())
}

// GetBackendType returns the backend type currently in use by this GNN.
func (g *GNN) GetBackendType() Backend {
	return Backend(C.gnn_get_backend_type(g.handle))
}

// GetEdgeEndpoints returns the source and target node indices of an edge.
// Returns (source, target, true) on success, (0, 0, false) if the edge
// does not exist.
func (g *GNN) GetEdgeEndpoints(edgeIdx uint) (uint, uint, bool) {
	var source, target C.uint
	result := C.gnn_get_edge_endpoints(
		g.handle, C.uint(edgeIdx), &source, &target,
	)
	if result < 0 {
		return 0, 0, false
	}
	return uint(source), uint(target), true
}
