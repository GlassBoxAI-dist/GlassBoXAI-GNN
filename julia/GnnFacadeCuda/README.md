# GnnFacadeCuda.jl

Julia bindings for the GlassBoxAI GNN CUDA-accelerated Graph Neural Network library.

## Installation

### Prerequisites

1. Build the Rust library with FFI support:
   ```bash
   cargo build --release --lib --features ffi
   ```

2. The library will be at `target/release/libgnn_facade_cuda.so` (Linux), 
   `.dylib` (macOS), or `.dll` (Windows).

### Using the Package

From the Julia REPL:

```julia
# Navigate to the julia directory
cd("path/to/GlassBoxAI-GNN/julia")

# Activate and use the package
using Pkg
Pkg.activate("GnnFacadeCuda")
using GnnFacadeCuda
```

Or add to your project:

```julia
using Pkg
Pkg.develop(path="path/to/GlassBoxAI-GNN/julia/GnnFacadeCuda")
```

### Custom Library Path

If the library is in a non-standard location, set the environment variable:

```julia
ENV["GNN_LIBRARY_PATH"] = "/path/to/libgnn_facade_cuda"
using GnnFacadeCuda
```

## Usage

```julia
using GnnFacadeCuda

# Create a new GNN
# Parameters: feature_size, hidden_size, output_size, num_mp_layers
gnn = GnnFacade(3, 16, 2, 2)

# Create a graph with 5 nodes, each with 3 features
create_empty_graph!(gnn, 5, 3)

# Add edges (0-indexed)
add_edge!(gnn, 0, 1)
add_edge!(gnn, 1, 2)
add_edge!(gnn, 2, 3)

# Set node features
set_node_features!(gnn, 0, Float32[1.0, 0.5, 0.2])
set_node_features!(gnn, 1, Float32[0.8, 0.3, 0.1])
set_node_features!(gnn, 2, Float32[0.6, 0.4, 0.3])

# Make predictions
prediction = predict!(gnn)
println("Prediction: ", prediction)

# Train the model
target = Float32[0.5, 0.5]
loss = train!(gnn, target)
println("Loss: ", loss)

# Train for multiple iterations
train_multiple!(gnn, target, 100)

# Save the model
save_model(gnn, "model.bin")

# Load from file
gnn2 = load_gnn("model.bin")
```

## API Reference

### Lifecycle

| Function | Description |
|----------|-------------|
| `GnnFacade(feature_size, hidden_size, output_size, num_mp_layers)` | Create a new GNN |
| `load_gnn(filename)` | Load a GNN from file |
| `read_model_header(filename)` | Read model header without loading |

### Model I/O

| Function | Description |
|----------|-------------|
| `save_model(gnn, filename)` | Save model to file |
| `load_model!(gnn, filename)` | Load weights into existing GNN |

### Graph Operations

| Function | Description |
|----------|-------------|
| `create_empty_graph!(gnn, num_nodes, feature_size)` | Create empty graph |
| `add_edge!(gnn, source, target, [features])` | Add edge |
| `remove_edge!(gnn, edge_idx)` | Remove edge by index |
| `has_edge(gnn, source, target)` | Check if edge exists |
| `find_edge_index(gnn, source, target)` | Find edge index |
| `rebuild_adjacency_list!(gnn)` | Rebuild adjacency list |

### Node Features

| Function | Description |
|----------|-------------|
| `set_node_features!(gnn, node_idx, features)` | Set all features for node |
| `get_node_features(gnn, node_idx)` | Get all features for node |
| `set_node_feature!(gnn, node_idx, feature_idx, value)` | Set single feature |
| `get_node_feature(gnn, node_idx, feature_idx)` | Get single feature |

### Edge Features

| Function | Description |
|----------|-------------|
| `set_edge_features!(gnn, edge_idx, features)` | Set edge features |
| `get_edge_features(gnn, edge_idx)` | Get edge features |

### Training & Inference

| Function | Description |
|----------|-------------|
| `predict!(gnn)` | Run prediction |
| `train!(gnn, target)` | Train on target, returns loss |
| `train_multiple!(gnn, target, iterations)` | Train for N iterations |

### Hyperparameters

| Function | Description |
|----------|-------------|
| `set_learning_rate!(gnn, lr)` | Set learning rate |
| `get_learning_rate(gnn)` | Get learning rate |

### Graph Info

| Function | Description |
|----------|-------------|
| `get_num_nodes(gnn)` | Number of nodes |
| `get_num_edges(gnn)` | Number of edges |
| `is_graph_loaded(gnn)` | Check if graph loaded |
| `get_feature_size(gnn)` | Feature size |
| `get_hidden_size(gnn)` | Hidden layer size |
| `get_output_size(gnn)` | Output size |
| `get_num_message_passing_layers(gnn)` | Number of MP layers |
| `get_in_degree(gnn, node_idx)` | Node in-degree |
| `get_out_degree(gnn, node_idx)` | Node out-degree |
| `get_neighbors(gnn, node_idx)` | Get node neighbors |
| `get_graph_embedding(gnn)` | Get graph embedding |

### Masking & Dropout

| Function | Description |
|----------|-------------|
| `set_node_mask!(gnn, node_idx, value)` | Set node mask |
| `get_node_mask(gnn, node_idx)` | Get node mask |
| `set_edge_mask!(gnn, edge_idx, value)` | Set edge mask |
| `get_edge_mask(gnn, edge_idx)` | Get edge mask |
| `apply_node_dropout!(gnn, rate)` | Random node dropout |
| `apply_edge_dropout!(gnn, rate)` | Random edge dropout |
| `get_masked_node_count(gnn)` | Count of active nodes |
| `get_masked_edge_count(gnn)` | Count of active edges |

### Analytics

| Function | Description |
|----------|-------------|
| `compute_page_rank(gnn; damping=0.85, iterations=20)` | Compute PageRank |
| `get_gradient_flow(gnn, layer_idx)` | Get gradient flow info |
| `get_parameter_count(gnn)` | Total parameter count |
| `get_architecture_summary(gnn)` | Architecture summary string |
| `export_graph_to_json(gnn)` | Export graph as JSON |

## Types

### GradientFlowInfo

```julia
struct GradientFlowInfo
    layer_idx::UInt32
    mean_gradient::Float32
    max_gradient::Float32
    min_gradient::Float32
    gradient_norm::Float32
end
```

### ModelHeader

```julia
struct ModelHeader
    feature_size::UInt32
    hidden_size::UInt32
    output_size::UInt32
    mp_layers::UInt32
    learning_rate::Float32
end
```

## Running Tests

```julia
using Pkg
Pkg.activate("GnnFacadeCuda")
Pkg.test()
```

Or directly:

```julia
include("test/runtests.jl")
```

## License

MIT License - Copyright (c) 2025 Matthew Abbott
