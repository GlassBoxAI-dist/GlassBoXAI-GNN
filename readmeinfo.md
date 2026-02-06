# GlassBoxAI GNN

CUDA-accelerated Graph Neural Network with Facade Pattern - available as a Rust library/CLI, Python package, Node.js module, C/C++ library, Go package, and Julia package.

## Features

- **CUDA Acceleration**: High-performance GPU computation using cudarc
- **Facade Pattern**: Simple, unified API for graph neural network operations
- **Multi-Language Support**: Use as Rust library, CLI tool, Python package, or Node.js module
- **Graph Operations**: Full graph manipulation (nodes, edges, features, masks)
- **Training & Inference**: Complete training loop with backpropagation
- **Graph Analytics**: PageRank, degree calculations, neighbor queries

## Installation

### Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
gnn_facade_cuda = "0.1.0"
```

### Python Package

Build with maturin:

```bash
pip install maturin
maturin develop --features python
```

Or build a wheel:

```bash
maturin build --features python --release
pip install target/wheels/gnn_facade_cuda-*.whl
```

### Node.js Package

Build with napi-rs:

```bash
npm install
npm run build
```

Or use the @napi-rs/cli directly:

```bash
npm install -g @napi-rs/cli
napi build --platform --release --features nodejs
```

### C/C++ Library

Build the shared library:

```bash
cargo build --release --lib --features ffi
```

This produces `libgnn_facade_cuda.so` (Linux), `libgnn_facade_cuda.dylib` (macOS), or `gnn_facade_cuda.dll` (Windows) in `target/release/`.

Link against it:

```bash
# C
gcc -o myapp myapp.c -I include -L target/release -lgnn_facade_cuda

# C++
g++ -o myapp myapp.cpp -I include -L target/release -lgnn_facade_cuda -std=c++17
```

### Go Package

First, build the C library:

```bash
cargo build --release --lib --features ffi
```

Then use the Go package:

```bash
cd go/example
go build
LD_LIBRARY_PATH=../../target/release ./example
```

Or import in your project:

```go
import "github.com/GlassBoxAI/GlassBoxAI-GNN/go/gnn"
```

### Julia Package

First, build the C library:

```bash
cargo build --release --lib --features ffi
```

Then use the Julia package:

```julia
using Pkg
Pkg.develop(path="path/to/GlassBoxAI-GNN/julia/GnnFacadeCuda")
using GnnFacadeCuda
```

Or activate directly:

```julia
cd("path/to/GlassBoxAI-GNN/julia")
using Pkg
Pkg.activate("GnnFacadeCuda")
using GnnFacadeCuda
```

### CLI Tool

```bash
cargo build --release
./target/release/gnn_facade_cuda --help
```

## Usage

### Rust API

```rust
use gnn_facade_cuda::CudaGnnFacade;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new GNN model
    let mut facade = CudaGnnFacade::new(
        3,   // feature_size
        16,  // hidden_size
        2,   // output_size
        2,   // num_mp_layers
    )?;

    // Create a graph with 5 nodes
    facade.create_empty_graph(5, 3);

    // Add edges
    facade.add_edge(0, 1, vec![]);
    facade.add_edge(1, 2, vec![]);
    facade.add_edge(2, 3, vec![]);

    // Set node features
    facade.set_node_features(0, vec![1.0, 0.5, 0.2]);
    facade.set_node_features(1, vec![0.8, 0.3, 0.1]);

    // Make predictions
    let prediction = facade.predict()?;
    println!("Prediction: {:?}", prediction);

    // Train the model
    let target = vec![0.5, 0.5];
    let loss = facade.train(&target)?;
    println!("Loss: {}", loss);

    // Save the model
    facade.save_model("model.bin")?;

    // Load a saved model
    let loaded = CudaGnnFacade::from_model_file("model.bin")?;
    println!("Loaded model with {} parameters", loaded.get_parameter_count());

    Ok(())
}
```

### Python API

```python
from gnn_facade_cuda import GnnFacade

# Create a new GNN model
gnn = GnnFacade(
    feature_size=3,
    hidden_size=16,
    output_size=2,
    num_mp_layers=2
)

# Create a graph with 5 nodes
gnn.create_empty_graph(5, 3)

# Add edges
gnn.add_edge(0, 1)
gnn.add_edge(1, 2)
gnn.add_edge(2, 3)

# Set node features
gnn.set_node_features(0, [1.0, 0.5, 0.2])
gnn.set_node_features(1, [0.8, 0.3, 0.1])

# Make predictions
prediction = gnn.predict()
print(f"Prediction: {prediction}")

# Train the model
target = [0.5, 0.5]
loss = gnn.train(target)
print(f"Loss: {loss}")

# Train for multiple epochs
gnn.train_multiple(target, iterations=100)

# Save the model
gnn.save_model("model.bin")

# Load a saved model
gnn2 = GnnFacade.from_model_file("model.bin")
print(f"Loaded model with {gnn2.get_parameter_count()} parameters")

# Graph analytics
ranks = gnn.compute_page_rank(damping=0.85, iterations=20)
print(f"PageRank: {ranks}")

# Export graph to JSON
json_str = gnn.export_graph_to_json()
print(json_str)
```

### Node.js API

```javascript
const { GnnFacade } = require('gnn-facade-cuda');

// Create a new GNN model
const gnn = new GnnFacade(3, 16, 2, 2);

// Create a graph with 5 nodes
gnn.createEmptyGraph(5, 3);

// Add edges
gnn.addEdge(0, 1);
gnn.addEdge(1, 2);
gnn.addEdge(2, 3);

// Set node features
gnn.setNodeFeatures(0, [1.0, 0.5, 0.2]);
gnn.setNodeFeatures(1, [0.8, 0.3, 0.1]);

// Make predictions
const prediction = gnn.predict();
console.log('Prediction:', prediction);

// Train the model
const target = [0.5, 0.5];
const loss = gnn.train(target);
console.log('Loss:', loss);

// Train for multiple epochs
gnn.trainMultiple(target, 100);

// Save the model
gnn.saveModel('model.bin');

// Load a saved model
const gnn2 = GnnFacade.fromModelFile('model.bin');
console.log('Loaded model with', gnn2.getParameterCount(), 'parameters');

// Graph analytics
const ranks = gnn.computePageRank(0.85, 20);
console.log('PageRank:', ranks);

// Export graph to JSON
const jsonStr = gnn.exportGraphToJson();
console.log(jsonStr);
```

### TypeScript

```typescript
import { GnnFacade, GradientFlowInfo, ModelHeader } from 'gnn-facade-cuda';

// Full TypeScript support with type definitions
const gnn = new GnnFacade(3, 16, 2, 2);
gnn.createEmptyGraph(5, 3);

// Types are inferred
const prediction: number[] = gnn.predict();
const info: GradientFlowInfo = gnn.getGradientFlow(0);
const header: ModelHeader = GnnFacade.readModelHeader('model.bin');
```

### Go API

```go
package main

import (
	"fmt"
	"log"

	"github.com/GlassBoxAI/GlassBoxAI-GNN/go/gnn"
)

func main() {
	// Create a new GNN
	g, err := gnn.New(3, 16, 2, 2)
	if err != nil {
		log.Fatal(err)
	}
	defer g.Close()

	// Create a graph
	g.CreateEmptyGraph(5, 3)

	// Add edges
	g.AddEdge(0, 1, nil)
	g.AddEdge(1, 2, nil)
	g.AddEdge(2, 3, nil)

	// Set node features
	g.SetNodeFeatures(0, []float32{1.0, 0.5, 0.2})
	g.SetNodeFeatures(1, []float32{0.8, 0.3, 0.1})

	// Make predictions
	prediction, err := g.Predict()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Prediction:", prediction)

	// Train
	loss, err := g.Train([]float32{0.5, 0.5})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Loss:", loss)

	// Train for multiple iterations
	if err := g.TrainMultiple([]float32{0.5, 0.5}, 100); err != nil {
		log.Fatal(err)
	}

	// Save model
	if err := g.SaveModel("model.bin"); err != nil {
		log.Fatal(err)
	}

	// Load model
	g2, err := gnn.Load("model.bin")
	if err != nil {
		log.Fatal(err)
	}
	defer g2.Close()
	fmt.Printf("Loaded model with %d parameters\n", g2.GetParameterCount())

	// Graph analytics
	ranks := g.ComputePageRank(0.85, 20)
	fmt.Println("PageRank:", ranks)

	// Export to JSON
	jsonStr := g.ExportGraphToJSON()
	fmt.Println(jsonStr)
}
```

### Julia API

```julia
using GnnFacadeCuda

# Create a new GNN
gnn = GnnFacade(3, 16, 2, 2)

# Create a graph with 5 nodes
create_empty_graph!(gnn, 5, 3)

# Add edges (0-indexed)
add_edge!(gnn, 0, 1)
add_edge!(gnn, 1, 2)
add_edge!(gnn, 2, 3)

# Set node features
set_node_features!(gnn, 0, Float32[1.0, 0.5, 0.2])
set_node_features!(gnn, 1, Float32[0.8, 0.3, 0.1])

# Make predictions
prediction = predict!(gnn)
println("Prediction: ", prediction)

# Train the model
loss = train!(gnn, Float32[0.5, 0.5])
println("Loss: ", loss)

# Train for multiple iterations
train_multiple!(gnn, Float32[0.5, 0.5], 100)

# Save model
save_model(gnn, "model.bin")

# Load from file
gnn2 = load_gnn("model.bin")
println("Loaded model with $(get_parameter_count(gnn2)) parameters")

# Graph analytics
ranks = compute_page_rank(gnn)
println("PageRank: ", ranks)

# Export to JSON
json_str = export_graph_to_json(gnn)
println(json_str)
```

### C API

```c
#include "gnn_facade.h"
#include <stdio.h>

int main() {
    // Create a GNN
    GnnHandle* gnn = gnn_create(3, 16, 2, 2);
    if (!gnn) {
        fprintf(stderr, "Failed to create GNN\n");
        return 1;
    }

    // Create a graph
    gnn_create_empty_graph(gnn, 5, 3);

    // Add edges
    gnn_add_edge(gnn, 0, 1, NULL, 0);
    gnn_add_edge(gnn, 1, 2, NULL, 0);

    // Set node features
    float features[] = {1.0f, 0.5f, 0.2f};
    gnn_set_node_features(gnn, 0, features, 3);

    // Make predictions
    float output[2];
    gnn_predict(gnn, output, 2);
    printf("Prediction: [%f, %f]\n", output[0], output[1]);

    // Train
    float target[] = {0.5f, 0.5f};
    float loss;
    gnn_train(gnn, target, 2, &loss);
    printf("Loss: %f\n", loss);

    // Save model
    gnn_save_model(gnn, "model.bin");

    // Cleanup
    gnn_free(gnn);
    return 0;
}
```

### C++ API

```cpp
#include "gnn_facade.hpp"
#include <iostream>

int main() {
    try {
        // Create a GNN (RAII - automatically freed)
        gnn::GnnFacade gnn(3, 16, 2, 2);

        // Create a graph
        gnn.createEmptyGraph(5, 3);

        // Add edges
        gnn.addEdge(0, 1);
        gnn.addEdge(1, 2);

        // Set node features (using initializer list)
        gnn.setNodeFeatures(0, {1.0f, 0.5f, 0.2f});

        // Make predictions
        auto prediction = gnn.predict();
        std::cout << "Prediction: [" << prediction[0] << ", " << prediction[1] << "]" << std::endl;

        // Train
        float loss = gnn.train({0.5f, 0.5f});
        std::cout << "Loss: " << loss << std::endl;

        // Save model
        gnn.saveModel("model.bin");

        // Load from file
        gnn::GnnFacade gnn2("model.bin");
        std::cout << "Loaded model with " << gnn2.getParameterCount() << " parameters" << std::endl;

    } catch (const gnn::GnnException& e) {
        std::cerr << "GNN Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

### CLI Usage

```bash
# Create a new model
gnn_facade_cuda create --feature=3 --hidden=16 --output=2 --mp-layers=2 --model=model.bin

# Get model info
gnn_facade_cuda info --model=model.bin

# Train the model
gnn_facade_cuda train --model=model.bin --graph=graph.csv --save=trained.bin --epochs=100

# Make predictions
gnn_facade_cuda predict --model=model.bin --graph=graph.csv

# Graph operations
gnn_facade_cuda create-graph --model=model.bin --nodes=10 --features=3
gnn_facade_cuda add-edge --model=model.bin --source=0 --target=1
gnn_facade_cuda set-node-features --model=model.bin --node=0 --features="1.0,0.5,0.2"

# Analytics
gnn_facade_cuda pagerank --model=model.bin --damping=0.85 --iterations=20
gnn_facade_cuda get-parameter-count --model=model.bin

# Export
gnn_facade_cuda export-json --model=model.bin
```

## API Reference

### CudaGnnFacade (Rust) / GnnFacade (Python)

#### Model Creation & IO

| Method | Description |
|--------|-------------|
| `new(feature_size, hidden_size, output_size, num_mp_layers)` | Create a new GNN |
| `from_model_file(filename)` | Load model from file |
| `save_model(filename)` | Save model to file |
| `load_model(filename)` | Load weights into existing model |
| `read_model_header(filename)` | Read model dimensions without loading |

#### Graph Operations

| Method | Description |
|--------|-------------|
| `create_empty_graph(num_nodes, feature_size)` | Create empty graph |
| `add_edge(source, target, features)` | Add edge, returns index |
| `remove_edge(edge_idx)` | Remove edge by index |
| `has_edge(source, target)` | Check if edge exists |
| `find_edge_index(source, target)` | Find edge index |
| `get_neighbors(node_idx)` | Get node neighbors |
| `rebuild_adjacency_list()` | Rebuild adjacency from edges |

#### Node Features

| Method | Description |
|--------|-------------|
| `set_node_features(node_idx, features)` | Set all features for node |
| `get_node_features(node_idx)` | Get all features for node |
| `set_node_feature(node_idx, feat_idx, value)` | Set single feature |
| `get_node_feature(node_idx, feat_idx)` | Get single feature |

#### Edge Features

| Method | Description |
|--------|-------------|
| `set_edge_features(edge_idx, features)` | Set edge features |
| `get_edge_features(edge_idx)` | Get edge features |
| `get_edge_endpoints(edge_idx)` | Get (source, target) tuple |

#### Training & Inference

| Method | Description |
|--------|-------------|
| `predict()` | Run forward pass, return predictions |
| `train(target)` | Single training step, return loss |
| `train_multiple(target, iterations)` | Train for N iterations |
| `set_learning_rate(lr)` | Set learning rate |
| `get_learning_rate()` | Get current learning rate |

#### Masking & Dropout

| Method | Description |
|--------|-------------|
| `set_node_mask(node_idx, value)` | Set node mask (active/inactive) |
| `get_node_mask(node_idx)` | Get node mask |
| `set_edge_mask(edge_idx, value)` | Set edge mask |
| `get_edge_mask(edge_idx)` | Get edge mask |
| `apply_node_dropout(rate)` | Random node dropout |
| `apply_edge_dropout(rate)` | Random edge dropout |
| `get_masked_node_count()` | Count active nodes |
| `get_masked_edge_count()` | Count active edges |

#### Analytics & Info

| Method | Description |
|--------|-------------|
| `compute_page_rank(damping, iterations)` | Compute PageRank scores |
| `get_in_degree(node_idx)` | Get node in-degree |
| `get_out_degree(node_idx)` | Get node out-degree |
| `get_num_nodes()` | Get node count |
| `get_num_edges()` | Get edge count |
| `get_parameter_count()` | Get total parameters |
| `get_architecture_summary()` | Get model summary string |
| `get_gradient_flow(layer_idx)` | Get gradient statistics |
| `export_graph_to_json()` | Export graph as JSON |

## Requirements

- CUDA 12.0+ compatible GPU and drivers
- Rust 1.70+ (for building)
- Python 3.8+ (for Python bindings)

## License

MIT License - Copyright (c) 2025 Matthew Abbott
