package gnn

import (
	"os"
	"testing"
)

func TestNewGNN(t *testing.T) {
	g, err := New(3, 16, 2, 2)
	if err != nil {
		t.Fatalf("Failed to create GNN: %v", err)
	}
	defer g.Close()

	if g.GetFeatureSize() != 3 {
		t.Errorf("Expected feature size 3, got %d", g.GetFeatureSize())
	}
	if g.GetHiddenSize() != 16 {
		t.Errorf("Expected hidden size 16, got %d", g.GetHiddenSize())
	}
	if g.GetOutputSize() != 2 {
		t.Errorf("Expected output size 2, got %d", g.GetOutputSize())
	}
	if g.GetNumMessagePassingLayers() != 2 {
		t.Errorf("Expected 2 MP layers, got %d", g.GetNumMessagePassingLayers())
	}
}

func TestCreateGraph(t *testing.T) {
	g, err := New(3, 16, 2, 2)
	if err != nil {
		t.Fatalf("Failed to create GNN: %v", err)
	}
	defer g.Close()

	g.CreateEmptyGraph(5, 3)

	if g.GetNumNodes() != 5 {
		t.Errorf("Expected 5 nodes, got %d", g.GetNumNodes())
	}
	if !g.IsGraphLoaded() {
		t.Error("Expected graph to be loaded")
	}
}

func TestAddEdge(t *testing.T) {
	g, err := New(3, 16, 2, 2)
	if err != nil {
		t.Fatalf("Failed to create GNN: %v", err)
	}
	defer g.Close()

	g.CreateEmptyGraph(5, 3)

	idx, err := g.AddEdge(0, 1, nil)
	if err != nil {
		t.Fatalf("Failed to add edge: %v", err)
	}
	if idx != 0 {
		t.Errorf("Expected edge index 0, got %d", idx)
	}

	idx, err = g.AddEdge(1, 2, nil)
	if err != nil {
		t.Fatalf("Failed to add edge: %v", err)
	}
	if idx != 1 {
		t.Errorf("Expected edge index 1, got %d", idx)
	}

	if g.GetNumEdges() != 2 {
		t.Errorf("Expected 2 edges, got %d", g.GetNumEdges())
	}

	if !g.HasEdge(0, 1) {
		t.Error("Expected edge 0->1 to exist")
	}
	if g.HasEdge(1, 0) {
		t.Error("Expected edge 1->0 to not exist (directed graph)")
	}
}

func TestNodeFeatures(t *testing.T) {
	g, err := New(3, 16, 2, 2)
	if err != nil {
		t.Fatalf("Failed to create GNN: %v", err)
	}
	defer g.Close()

	g.CreateEmptyGraph(5, 3)

	features := []float32{1.0, 0.5, 0.2}
	if err := g.SetNodeFeatures(0, features); err != nil {
		t.Fatalf("Failed to set node features: %v", err)
	}

	got, err := g.GetNodeFeatures(0)
	if err != nil {
		t.Fatalf("Failed to get node features: %v", err)
	}

	if len(got) != len(features) {
		t.Fatalf("Expected %d features, got %d", len(features), len(got))
	}

	for i := range features {
		if got[i] != features[i] {
			t.Errorf("Feature %d: expected %f, got %f", i, features[i], got[i])
		}
	}
}

func TestPredict(t *testing.T) {
	g, err := New(3, 16, 2, 2)
	if err != nil {
		t.Fatalf("Failed to create GNN: %v", err)
	}
	defer g.Close()

	g.CreateEmptyGraph(5, 3)
	g.AddEdge(0, 1, nil)
	g.AddEdge(1, 2, nil)

	g.SetNodeFeatures(0, []float32{1.0, 0.5, 0.2})
	g.SetNodeFeatures(1, []float32{0.8, 0.3, 0.1})

	prediction, err := g.Predict()
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}

	if len(prediction) != 2 {
		t.Errorf("Expected 2 outputs, got %d", len(prediction))
	}
}

func TestTrain(t *testing.T) {
	g, err := New(3, 16, 2, 2)
	if err != nil {
		t.Fatalf("Failed to create GNN: %v", err)
	}
	defer g.Close()

	g.CreateEmptyGraph(5, 3)
	g.AddEdge(0, 1, nil)
	g.AddEdge(1, 2, nil)

	g.SetNodeFeatures(0, []float32{1.0, 0.5, 0.2})
	g.SetNodeFeatures(1, []float32{0.8, 0.3, 0.1})

	target := []float32{0.5, 0.5}
	loss, err := g.Train(target)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if loss < 0 {
		t.Errorf("Expected non-negative loss, got %f", loss)
	}
}

func TestSaveLoad(t *testing.T) {
	tmpFile := "test_model.bin"
	defer os.Remove(tmpFile)

	// Create and save
	g1, err := New(3, 16, 2, 2)
	if err != nil {
		t.Fatalf("Failed to create GNN: %v", err)
	}

	g1.SetLearningRate(0.05)
	if err := g1.SaveModel(tmpFile); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}
	g1.Close()

	// Load and verify
	g2, err := Load(tmpFile)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer g2.Close()

	if g2.GetFeatureSize() != 3 {
		t.Errorf("Expected feature size 3, got %d", g2.GetFeatureSize())
	}
	if g2.GetHiddenSize() != 16 {
		t.Errorf("Expected hidden size 16, got %d", g2.GetHiddenSize())
	}
}

func TestPageRank(t *testing.T) {
	g, err := New(3, 16, 2, 2)
	if err != nil {
		t.Fatalf("Failed to create GNN: %v", err)
	}
	defer g.Close()

	g.CreateEmptyGraph(5, 3)
	g.AddEdge(0, 1, nil)
	g.AddEdge(1, 2, nil)
	g.AddEdge(2, 3, nil)
	g.AddEdge(3, 4, nil)
	g.AddEdge(4, 0, nil)

	ranks := g.ComputePageRank(0.85, 20)
	if len(ranks) != 5 {
		t.Errorf("Expected 5 PageRank scores, got %d", len(ranks))
	}

	// Sum should be approximately 1
	var sum float32
	for _, r := range ranks {
		sum += r
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("PageRank sum should be ~1.0, got %f", sum)
	}
}

func TestMasking(t *testing.T) {
	g, err := New(3, 16, 2, 2)
	if err != nil {
		t.Fatalf("Failed to create GNN: %v", err)
	}
	defer g.Close()

	g.CreateEmptyGraph(5, 3)

	// All nodes should be active by default
	if g.GetMaskedNodeCount() != 5 {
		t.Errorf("Expected 5 active nodes, got %d", g.GetMaskedNodeCount())
	}

	// Mask a node
	g.SetNodeMask(0, false)
	if g.GetNodeMask(0) != false {
		t.Error("Expected node 0 to be masked")
	}
	if g.GetMaskedNodeCount() != 4 {
		t.Errorf("Expected 4 active nodes, got %d", g.GetMaskedNodeCount())
	}
}
