// Example usage of the GNN Go package
package main

import (
	"fmt"
	"log"

	"github.com/GlassBoxAI/GlassBoxAI-GNN/go/gnn"
)

func main() {
	// Create a new GNN with:
	// - 3 input features per node
	// - 16 hidden units
	// - 2 output values
	// - 2 message passing layers
	g, err := gnn.New(3, 16, 2, 2)
	if err != nil {
		log.Fatalf("Failed to create GNN: %v", err)
	}
	defer g.Close()

	fmt.Println("Created GNN:")
	fmt.Println(g.GetArchitectureSummary())

	// Create a graph with 5 nodes
	g.CreateEmptyGraph(5, 3)
	fmt.Printf("\nCreated graph with %d nodes\n", g.GetNumNodes())

	// Add edges to form a cycle with a shortcut
	g.AddEdge(0, 1, nil)
	g.AddEdge(1, 2, nil)
	g.AddEdge(2, 3, nil)
	g.AddEdge(3, 4, nil)
	g.AddEdge(4, 0, nil)
	g.AddEdge(1, 3, nil) // Shortcut
	fmt.Printf("Added %d edges\n", g.GetNumEdges())

	// Set node features
	g.SetNodeFeatures(0, []float32{1.0, 0.5, 0.2})
	g.SetNodeFeatures(1, []float32{0.8, 0.3, 0.1})
	g.SetNodeFeatures(2, []float32{0.6, 0.7, 0.4})
	g.SetNodeFeatures(3, []float32{0.4, 0.2, 0.8})
	g.SetNodeFeatures(4, []float32{0.9, 0.1, 0.5})

	// Make a prediction
	prediction, err := g.Predict()
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}
	fmt.Printf("\nPrediction: %v\n", prediction)

	// Train for a few iterations
	target := []float32{0.5, 0.5}
	fmt.Printf("\nTraining with target: %v\n", target)

	for i := 0; i < 10; i++ {
		loss, err := g.Train(target)
		if err != nil {
			log.Fatalf("Training failed: %v", err)
		}
		if i%2 == 0 {
			fmt.Printf("Iteration %d, Loss: %.6f\n", i+1, loss)
		}
	}

	// Make another prediction after training
	prediction, err = g.Predict()
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}
	fmt.Printf("\nPrediction after training: %v\n", prediction)

	// Compute PageRank
	ranks := g.ComputePageRank(0.85, 20)
	fmt.Println("\nPageRank scores:")
	for i, r := range ranks {
		fmt.Printf("  Node %d: %.6f\n", i, r)
	}

	// Get graph info
	fmt.Println("\nGraph statistics:")
	for i := uint(0); i < g.GetNumNodes(); i++ {
		fmt.Printf("  Node %d: in-degree=%d, out-degree=%d\n",
			i, g.GetInDegree(i), g.GetOutDegree(i))
	}

	// Export to JSON
	jsonStr := g.ExportGraphToJSON()
	fmt.Printf("\nGraph JSON (first 200 chars):\n%s...\n", jsonStr[:min(len(jsonStr), 200)])

	// Save the model
	if err := g.SaveModel("example_model.bin"); err != nil {
		log.Fatalf("Failed to save model: %v", err)
	}
	fmt.Println("\nModel saved to example_model.bin")

	// Load and verify
	g2, err := gnn.Load("example_model.bin")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer g2.Close()

	fmt.Printf("\nLoaded model with %d parameters\n", g2.GetParameterCount())
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
