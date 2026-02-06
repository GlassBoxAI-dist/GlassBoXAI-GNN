# MIT License
#
# Copyright (c) 2025 Matthew Abbott
#
# Julia bindings for GlassBoxAI GNN

"""
    GnnFacadeCuda

GPU-accelerated Graph Neural Network with Facade interface (CUDA and OpenCL backends).

# Example
```julia
using GnnFacadeCuda

# Create a new GNN
gnn = GnnFacade(3, 16, 2, 2)

# Create a graph
create_empty_graph!(gnn, 5, 3)

# Add edges
add_edge!(gnn, 0, 1)
add_edge!(gnn, 1, 2)

# Set node features
set_node_features!(gnn, 0, Float32[1.0, 0.5, 0.2])
set_node_features!(gnn, 1, Float32[0.8, 0.3, 0.1])

# Make predictions
prediction = predict!(gnn)
println("Prediction: ", prediction)

# Train
loss = train!(gnn, Float32[0.5, 0.5])
println("Loss: ", loss)

# Save model
save_model(gnn, "model.bin")
```
"""
module GnnFacadeCuda

export GnnFacade, GradientFlowInfo, ModelHeader,
       # Lifecycle
       load_gnn, read_model_header,
       # Model I/O
       save_model, load_model!,
       # Graph operations
       create_empty_graph!, add_edge!, remove_edge!, has_edge, find_edge_index,
       rebuild_adjacency_list!,
       # Node features
       set_node_features!, get_node_features, set_node_feature!, get_node_feature,
       # Edge features
       set_edge_features!, get_edge_features,
       # Training & Inference
       predict!, train!, train_multiple!,
       # Hyperparameters
       set_learning_rate!, get_learning_rate,
       # Graph info
       get_num_nodes, get_num_edges, is_graph_loaded, get_feature_size,
       get_hidden_size, get_output_size, get_num_message_passing_layers,
       get_in_degree, get_out_degree, get_neighbors, get_graph_embedding,
       # Masking & Dropout
       set_node_mask!, get_node_mask, set_edge_mask!, get_edge_mask,
       apply_node_dropout!, apply_edge_dropout!,
       get_masked_node_count, get_masked_edge_count,
       # Analytics
       compute_page_rank, get_gradient_flow, get_parameter_count,
       get_architecture_summary, export_graph_to_json,
       # Backend
       get_backend_name,
       GNN_BACKEND_CUDA, GNN_BACKEND_OPENCL, GNN_BACKEND_AUTO

# Library path - can be overridden by setting GNN_LIBRARY_PATH environment variable
const DEFAULT_LIB_PATH = joinpath(@__DIR__, "..", "..", "..", "target", "release", "libgnn_facade_cuda")

function get_lib_path()
    lib = get(ENV, "GNN_LIBRARY_PATH", DEFAULT_LIB_PATH)
    if Sys.iswindows()
        return lib * ".dll"
    elseif Sys.isapple()
        return lib * ".dylib"
    else
        return lib * ".so"
    end
end

const libgnn = Ref{String}()

function __init__()
    libgnn[] = get_lib_path()
end

# Error codes
const GNN_OK = Cint(0)
const GNN_ERROR_NULL_POINTER = Cint(-1)
const GNN_ERROR_INVALID_ARG = Cint(-2)
const GNN_ERROR_CUDA = Cint(-3)
const GNN_ERROR_IO = Cint(-4)
const GNN_ERROR_OPENCL = Cint(-5)
const GNN_ERROR_UNKNOWN = Cint(-99)

# Backend type constants
"""NVIDIA CUDA backend"""
const GNN_BACKEND_CUDA = Cint(0)
"""OpenCL backend (AMD, Intel, NVIDIA)"""
const GNN_BACKEND_OPENCL = Cint(1)
"""Auto-detect best available backend"""
const GNN_BACKEND_AUTO = Cint(2)

"""Exception thrown when GNN operations fail."""
struct GnnError <: Exception
    code::Cint
    msg::String
end

function Base.showerror(io::IO, e::GnnError)
    print(io, "GnnError($(e.code)): $(e.msg)")
end

function check_error(code::Cint)
    if code == GNN_OK
        return
    elseif code == GNN_ERROR_NULL_POINTER
        throw(GnnError(code, "null pointer"))
    elseif code == GNN_ERROR_INVALID_ARG
        throw(GnnError(code, "invalid argument"))
    elseif code == GNN_ERROR_CUDA
        throw(GnnError(code, "CUDA error"))
    elseif code == GNN_ERROR_IO
        throw(GnnError(code, "I/O error"))
    else
        throw(GnnError(code, "unknown error"))
    end
end

"""
    GradientFlowInfo

Gradient flow information for a layer.

# Fields
- `layer_idx::UInt32`: Layer index
- `mean_gradient::Float32`: Mean gradient value
- `max_gradient::Float32`: Maximum gradient value  
- `min_gradient::Float32`: Minimum gradient value
- `gradient_norm::Float32`: L2 norm of gradients
"""
struct GradientFlowInfo
    layer_idx::UInt32
    mean_gradient::Float32
    max_gradient::Float32
    min_gradient::Float32
    gradient_norm::Float32
end

"""
    ModelHeader

Model header information.

# Fields
- `feature_size::UInt32`: Size of input node features
- `hidden_size::UInt32`: Size of hidden layers
- `output_size::UInt32`: Size of output predictions
- `mp_layers::UInt32`: Number of message passing layers
- `learning_rate::Float32`: Current learning rate
"""
struct ModelHeader
    feature_size::UInt32
    hidden_size::UInt32
    output_size::UInt32
    mp_layers::UInt32
    learning_rate::Float32
end

"""
    GnnFacade

CUDA-accelerated Graph Neural Network facade.

Create a new GNN with:
```julia
gnn = GnnFacade(feature_size, hidden_size, output_size, num_mp_layers)
```

Or load from a file:
```julia
gnn = load_gnn("model.bin")
```
"""
mutable struct GnnFacade
    handle::Ptr{Cvoid}
    
    function GnnFacade(handle::Ptr{Cvoid})
        gnn = new(handle)
        finalizer(gnn) do g
            if g.handle != C_NULL
                ccall((:gnn_free, libgnn[]), Cvoid, (Ptr{Cvoid},), g.handle)
                g.handle = C_NULL
            end
        end
        return gnn
    end
end

"""
    GnnFacade(feature_size, hidden_size, output_size, num_mp_layers; backend=GNN_BACKEND_AUTO)

Create a new GNN with the specified architecture.

# Arguments
- `feature_size::Integer`: Size of input node features
- `hidden_size::Integer`: Size of hidden layers
- `output_size::Integer`: Size of output predictions
- `num_mp_layers::Integer`: Number of message passing layers
- `backend::Cint`: GPU backend (GNN_BACKEND_CUDA, GNN_BACKEND_OPENCL, or GNN_BACKEND_AUTO)

# Returns
- `GnnFacade`: A new GNN instance

# Example
```julia
gnn = GnnFacade(3, 16, 2, 2)                         # Auto-detect
gnn = GnnFacade(3, 16, 2, 2, backend=GNN_BACKEND_OPENCL)  # OpenCL
```
"""
function GnnFacade(feature_size::Integer, hidden_size::Integer,
                   output_size::Integer, num_mp_layers::Integer;
                   backend::Cint=GNN_BACKEND_AUTO)
    handle = ccall((:gnn_create_with_backend, libgnn[]), Ptr{Cvoid},
                   (Cuint, Cuint, Cuint, Cuint, Cint),
                   feature_size, hidden_size, output_size, num_mp_layers, backend)
    if handle == C_NULL
        throw(GnnError(GNN_ERROR_UNKNOWN, "failed to create GNN"))
    end
    return GnnFacade(handle)
end

"""
    load_gnn(filename::String) -> GnnFacade

Load a GNN from a saved model file.

# Example
```julia
gnn = load_gnn("model.bin")
```
"""
function load_gnn(filename::String)
    handle = ccall((:gnn_load, libgnn[]), Ptr{Cvoid}, (Cstring,), filename)
    if handle == C_NULL
        throw(GnnError(GNN_ERROR_IO, "failed to load model: $filename"))
    end
    return GnnFacade(handle)
end

"""
    read_model_header(filename::String) -> ModelHeader

Read model header without loading the full model.

# Example
```julia
header = read_model_header("model.bin")
println("Feature size: ", header.feature_size)
```
"""
function read_model_header(filename::String)
    header = Ref{ModelHeader}()
    result = ccall((:gnn_read_model_header, libgnn[]), Cint,
                   (Cstring, Ptr{ModelHeader}),
                   filename, header)
    check_error(result)
    return header[]
end

# ============================================================================
# Model I/O
# ============================================================================

"""
    save_model(gnn::GnnFacade, filename::String)

Save the model to a file.

# Example
```julia
save_model(gnn, "model.bin")
```
"""
function save_model(gnn::GnnFacade, filename::String)
    result = ccall((:gnn_save_model, libgnn[]), Cint,
                   (Ptr{Cvoid}, Cstring),
                   gnn.handle, filename)
    check_error(result)
end

"""
    load_model!(gnn::GnnFacade, filename::String)

Load model weights from a file into an existing GNN.

# Example
```julia
load_model!(gnn, "model.bin")
```
"""
function load_model!(gnn::GnnFacade, filename::String)
    result = ccall((:gnn_load_model, libgnn[]), Cint,
                   (Ptr{Cvoid}, Cstring),
                   gnn.handle, filename)
    check_error(result)
end

# ============================================================================
# Graph Operations
# ============================================================================

"""
    create_empty_graph!(gnn::GnnFacade, num_nodes::Integer, feature_size::Integer)

Create an empty graph with the specified number of nodes.

# Example
```julia
create_empty_graph!(gnn, 5, 3)  # 5 nodes with 3 features each
```
"""
function create_empty_graph!(gnn::GnnFacade, num_nodes::Integer, feature_size::Integer)
    result = ccall((:gnn_create_empty_graph, libgnn[]), Cint,
                   (Ptr{Cvoid}, Cuint, Cuint),
                   gnn.handle, num_nodes, feature_size)
    check_error(result)
end

"""
    add_edge!(gnn::GnnFacade, source::Integer, target::Integer, 
              features::Vector{Float32}=Float32[]) -> Int

Add an edge to the graph. Returns the edge index.

# Example
```julia
idx = add_edge!(gnn, 0, 1)  # Edge from node 0 to node 1
idx = add_edge!(gnn, 1, 2, Float32[0.5, 0.3])  # With edge features
```
"""
function add_edge!(gnn::GnnFacade, source::Integer, target::Integer,
                   features::Vector{Float32}=Float32[])
    feat_ptr = isempty(features) ? C_NULL : pointer(features)
    feat_len = length(features)
    
    result = ccall((:gnn_add_edge, libgnn[]), Cint,
                   (Ptr{Cvoid}, Cuint, Cuint, Ptr{Cfloat}, Cuint),
                   gnn.handle, source, target, feat_ptr, feat_len)
    if result < 0
        throw(GnnError(GNN_ERROR_INVALID_ARG, "failed to add edge"))
    end
    return Int(result)
end

"""
    remove_edge!(gnn::GnnFacade, edge_idx::Integer)

Remove an edge by index.
"""
function remove_edge!(gnn::GnnFacade, edge_idx::Integer)
    ccall((:gnn_remove_edge, libgnn[]), Cint,
          (Ptr{Cvoid}, Cuint),
          gnn.handle, edge_idx)
end

"""
    has_edge(gnn::GnnFacade, source::Integer, target::Integer) -> Bool

Check if an edge exists between two nodes.

# Example
```julia
if has_edge(gnn, 0, 1)
    println("Edge exists from 0 to 1")
end
```
"""
function has_edge(gnn::GnnFacade, source::Integer, target::Integer)
    result = ccall((:gnn_has_edge, libgnn[]), Cint,
                   (Ptr{Cvoid}, Cuint, Cuint),
                   gnn.handle, source, target)
    return result == 1
end

"""
    find_edge_index(gnn::GnnFacade, source::Integer, target::Integer) -> Union{Int, Nothing}

Find the index of an edge between two nodes.
Returns `nothing` if not found.

# Example
```julia
idx = find_edge_index(gnn, 0, 1)
if idx !== nothing
    println("Edge index: ", idx)
end
```
"""
function find_edge_index(gnn::GnnFacade, source::Integer, target::Integer)
    result = ccall((:gnn_find_edge_index, libgnn[]), Cint,
                   (Ptr{Cvoid}, Cuint, Cuint),
                   gnn.handle, source, target)
    return result >= 0 ? Int(result) : nothing
end

"""
    rebuild_adjacency_list!(gnn::GnnFacade)

Rebuild the adjacency list from edges.
"""
function rebuild_adjacency_list!(gnn::GnnFacade)
    ccall((:gnn_rebuild_adjacency_list, libgnn[]), Cint,
          (Ptr{Cvoid},), gnn.handle)
end

# ============================================================================
# Node Features
# ============================================================================

"""
    set_node_features!(gnn::GnnFacade, node_idx::Integer, features::Vector{Float32})

Set all features for a node.

# Example
```julia
set_node_features!(gnn, 0, Float32[1.0, 0.5, 0.2])
```
"""
function set_node_features!(gnn::GnnFacade, node_idx::Integer, features::Vector{Float32})
    result = ccall((:gnn_set_node_features, libgnn[]), Cint,
                   (Ptr{Cvoid}, Cuint, Ptr{Cfloat}, Cuint),
                   gnn.handle, node_idx, features, length(features))
    check_error(result)
end

"""
    get_node_features(gnn::GnnFacade, node_idx::Integer) -> Union{Vector{Float32}, Nothing}

Get all features for a node. Returns `nothing` if node doesn't exist.

# Example
```julia
features = get_node_features(gnn, 0)
if features !== nothing
    println("Features: ", features)
end
```
"""
function get_node_features(gnn::GnnFacade, node_idx::Integer)
    feature_size = get_feature_size(gnn)
    features = Vector{Float32}(undef, feature_size)
    
    count = ccall((:gnn_get_node_features, libgnn[]), Cint,
                  (Ptr{Cvoid}, Cuint, Ptr{Cfloat}, Cuint),
                  gnn.handle, node_idx, features, feature_size)
    
    return count >= 0 ? features[1:count] : nothing
end

"""
    set_node_feature!(gnn::GnnFacade, node_idx::Integer, feature_idx::Integer, value::Float32)

Set a single feature value for a node.

# Example
```julia
set_node_feature!(gnn, 0, 1, 0.75f0)  # Set feature 1 of node 0 to 0.75
```
"""
function set_node_feature!(gnn::GnnFacade, node_idx::Integer, 
                           feature_idx::Integer, value::Real)
    result = ccall((:gnn_set_node_feature, libgnn[]), Cint,
                   (Ptr{Cvoid}, Cuint, Cuint, Cfloat),
                   gnn.handle, node_idx, feature_idx, Float32(value))
    check_error(result)
end

"""
    get_node_feature(gnn::GnnFacade, node_idx::Integer, feature_idx::Integer) -> Float32

Get a single feature value for a node.

# Example
```julia
value = get_node_feature(gnn, 0, 1)
```
"""
function get_node_feature(gnn::GnnFacade, node_idx::Integer, feature_idx::Integer)
    return ccall((:gnn_get_node_feature, libgnn[]), Cfloat,
                 (Ptr{Cvoid}, Cuint, Cuint),
                 gnn.handle, node_idx, feature_idx)
end

# ============================================================================
# Edge Features
# ============================================================================

"""
    set_edge_features!(gnn::GnnFacade, edge_idx::Integer, features::Vector{Float32})

Set features for an edge.

# Example
```julia
set_edge_features!(gnn, 0, Float32[0.5, 0.3])
```
"""
function set_edge_features!(gnn::GnnFacade, edge_idx::Integer, features::Vector{Float32})
    result = ccall((:gnn_set_edge_features, libgnn[]), Cint,
                   (Ptr{Cvoid}, Cuint, Ptr{Cfloat}, Cuint),
                   gnn.handle, edge_idx, features, length(features))
    check_error(result)
end

"""
    get_edge_features(gnn::GnnFacade, edge_idx::Integer) -> Union{Vector{Float32}, Nothing}

Get features for an edge. Returns `nothing` if edge doesn't exist.
"""
function get_edge_features(gnn::GnnFacade, edge_idx::Integer)
    max_features = 64
    features = Vector{Float32}(undef, max_features)
    
    count = ccall((:gnn_get_edge_features, libgnn[]), Cint,
                  (Ptr{Cvoid}, Cuint, Ptr{Cfloat}, Cuint),
                  gnn.handle, edge_idx, features, max_features)
    
    return count >= 0 ? features[1:count] : nothing
end

# ============================================================================
# Training & Inference
# ============================================================================

"""
    predict!(gnn::GnnFacade) -> Vector{Float32}

Run prediction on the current graph.

# Example
```julia
prediction = predict!(gnn)
println("Outputs: ", prediction)
```
"""
function predict!(gnn::GnnFacade)
    output_size = get_output_size(gnn)
    output = Vector{Float32}(undef, output_size)
    
    count = ccall((:gnn_predict, libgnn[]), Cint,
                  (Ptr{Cvoid}, Ptr{Cfloat}, Cuint),
                  gnn.handle, output, output_size)
    
    if count < 0
        throw(GnnError(GNN_ERROR_CUDA, "prediction failed"))
    end
    return output[1:count]
end

"""
    train!(gnn::GnnFacade, target::Vector{Float32}) -> Float32

Train on the current graph with target values. Returns the loss.

# Example
```julia
loss = train!(gnn, Float32[0.5, 0.5])
println("Loss: ", loss)
```
"""
function train!(gnn::GnnFacade, target::Vector{Float32})
    loss = Ref{Cfloat}(0.0f0)
    
    result = ccall((:gnn_train, libgnn[]), Cint,
                   (Ptr{Cvoid}, Ptr{Cfloat}, Cuint, Ptr{Cfloat}),
                   gnn.handle, target, length(target), loss)
    check_error(result)
    return loss[]
end

"""
    train_multiple!(gnn::GnnFacade, target::Vector{Float32}, iterations::Integer)

Train for multiple iterations.

# Example
```julia
train_multiple!(gnn, Float32[0.5, 0.5], 100)
```
"""
function train_multiple!(gnn::GnnFacade, target::Vector{Float32}, iterations::Integer)
    result = ccall((:gnn_train_multiple, libgnn[]), Cint,
                   (Ptr{Cvoid}, Ptr{Cfloat}, Cuint, Cuint),
                   gnn.handle, target, length(target), iterations)
    check_error(result)
end

# ============================================================================
# Hyperparameters
# ============================================================================

"""
    set_learning_rate!(gnn::GnnFacade, lr::Real)

Set the learning rate.

# Example
```julia
set_learning_rate!(gnn, 0.001)
```
"""
function set_learning_rate!(gnn::GnnFacade, lr::Real)
    ccall((:gnn_set_learning_rate, libgnn[]), Cint,
          (Ptr{Cvoid}, Cfloat),
          gnn.handle, Float32(lr))
end

"""
    get_learning_rate(gnn::GnnFacade) -> Float32

Get the current learning rate.
"""
function get_learning_rate(gnn::GnnFacade)
    return ccall((:gnn_get_learning_rate, libgnn[]), Cfloat,
                 (Ptr{Cvoid},), gnn.handle)
end

# ============================================================================
# Graph Info
# ============================================================================

"""
    get_num_nodes(gnn::GnnFacade) -> UInt32

Get the number of nodes in the graph.
"""
function get_num_nodes(gnn::GnnFacade)
    return ccall((:gnn_get_num_nodes, libgnn[]), Cuint,
                 (Ptr{Cvoid},), gnn.handle)
end

"""
    get_num_edges(gnn::GnnFacade) -> UInt32

Get the number of edges in the graph.
"""
function get_num_edges(gnn::GnnFacade)
    return ccall((:gnn_get_num_edges, libgnn[]), Cuint,
                 (Ptr{Cvoid},), gnn.handle)
end

"""
    is_graph_loaded(gnn::GnnFacade) -> Bool

Check if a graph is loaded.
"""
function is_graph_loaded(gnn::GnnFacade)
    result = ccall((:gnn_is_graph_loaded, libgnn[]), Cint,
                   (Ptr{Cvoid},), gnn.handle)
    return result != 0
end

"""
    get_feature_size(gnn::GnnFacade) -> UInt32

Get the feature size.
"""
function get_feature_size(gnn::GnnFacade)
    return ccall((:gnn_get_feature_size, libgnn[]), Cuint,
                 (Ptr{Cvoid},), gnn.handle)
end

"""
    get_hidden_size(gnn::GnnFacade) -> UInt32

Get the hidden layer size.
"""
function get_hidden_size(gnn::GnnFacade)
    return ccall((:gnn_get_hidden_size, libgnn[]), Cuint,
                 (Ptr{Cvoid},), gnn.handle)
end

"""
    get_output_size(gnn::GnnFacade) -> UInt32

Get the output size.
"""
function get_output_size(gnn::GnnFacade)
    return ccall((:gnn_get_output_size, libgnn[]), Cuint,
                 (Ptr{Cvoid},), gnn.handle)
end

"""
    get_num_message_passing_layers(gnn::GnnFacade) -> UInt32

Get the number of message passing layers.
"""
function get_num_message_passing_layers(gnn::GnnFacade)
    return ccall((:gnn_get_num_message_passing_layers, libgnn[]), Cuint,
                 (Ptr{Cvoid},), gnn.handle)
end

"""
    get_in_degree(gnn::GnnFacade, node_idx::Integer) -> UInt32

Get the in-degree of a node.
"""
function get_in_degree(gnn::GnnFacade, node_idx::Integer)
    return ccall((:gnn_get_in_degree, libgnn[]), Cuint,
                 (Ptr{Cvoid}, Cuint),
                 gnn.handle, node_idx)
end

"""
    get_out_degree(gnn::GnnFacade, node_idx::Integer) -> UInt32

Get the out-degree of a node.
"""
function get_out_degree(gnn::GnnFacade, node_idx::Integer)
    return ccall((:gnn_get_out_degree, libgnn[]), Cuint,
                 (Ptr{Cvoid}, Cuint),
                 gnn.handle, node_idx)
end

"""
    get_neighbors(gnn::GnnFacade, node_idx::Integer) -> Union{Vector{UInt32}, Nothing}

Get the neighbors of a node. Returns `nothing` if node doesn't exist.
"""
function get_neighbors(gnn::GnnFacade, node_idx::Integer)
    num_nodes = get_num_nodes(gnn)
    neighbors = Vector{Cuint}(undef, num_nodes)
    
    count = ccall((:gnn_get_neighbors, libgnn[]), Cint,
                  (Ptr{Cvoid}, Cuint, Ptr{Cuint}, Cuint),
                  gnn.handle, node_idx, neighbors, num_nodes)
    
    return count >= 0 ? neighbors[1:count] : nothing
end

"""
    get_graph_embedding(gnn::GnnFacade) -> Vector{Float32}

Get the graph embedding from the last forward pass.
"""
function get_graph_embedding(gnn::GnnFacade)
    hidden_size = get_hidden_size(gnn)
    embedding = Vector{Float32}(undef, hidden_size)
    
    count = ccall((:gnn_get_graph_embedding, libgnn[]), Cint,
                  (Ptr{Cvoid}, Ptr{Cfloat}, Cuint),
                  gnn.handle, embedding, hidden_size)
    
    return count > 0 ? embedding[1:count] : Float32[]
end

# ============================================================================
# Masking & Dropout
# ============================================================================

"""
    set_node_mask!(gnn::GnnFacade, node_idx::Integer, value::Bool)

Set the node mask value. `true` = active, `false` = masked.
"""
function set_node_mask!(gnn::GnnFacade, node_idx::Integer, value::Bool)
    ccall((:gnn_set_node_mask, libgnn[]), Cint,
          (Ptr{Cvoid}, Cuint, Cint),
          gnn.handle, node_idx, value ? 1 : 0)
end

"""
    get_node_mask(gnn::GnnFacade, node_idx::Integer) -> Bool

Get the node mask value.
"""
function get_node_mask(gnn::GnnFacade, node_idx::Integer)
    result = ccall((:gnn_get_node_mask, libgnn[]), Cint,
                   (Ptr{Cvoid}, Cuint),
                   gnn.handle, node_idx)
    return result != 0
end

"""
    set_edge_mask!(gnn::GnnFacade, edge_idx::Integer, value::Bool)

Set the edge mask value. `true` = active, `false` = masked.
"""
function set_edge_mask!(gnn::GnnFacade, edge_idx::Integer, value::Bool)
    ccall((:gnn_set_edge_mask, libgnn[]), Cint,
          (Ptr{Cvoid}, Cuint, Cint),
          gnn.handle, edge_idx, value ? 1 : 0)
end

"""
    get_edge_mask(gnn::GnnFacade, edge_idx::Integer) -> Bool

Get the edge mask value.
"""
function get_edge_mask(gnn::GnnFacade, edge_idx::Integer)
    result = ccall((:gnn_get_edge_mask, libgnn[]), Cint,
                   (Ptr{Cvoid}, Cuint),
                   gnn.handle, edge_idx)
    return result != 0
end

"""
    apply_node_dropout!(gnn::GnnFacade, rate::Real)

Apply random dropout to nodes.

# Arguments
- `rate`: Dropout rate (0.0 to 1.0)

# Example
```julia
apply_node_dropout!(gnn, 0.1)  # 10% dropout
```
"""
function apply_node_dropout!(gnn::GnnFacade, rate::Real)
    ccall((:gnn_apply_node_dropout, libgnn[]), Cint,
          (Ptr{Cvoid}, Cfloat),
          gnn.handle, Float32(rate))
end

"""
    apply_edge_dropout!(gnn::GnnFacade, rate::Real)

Apply random dropout to edges.

# Arguments
- `rate`: Dropout rate (0.0 to 1.0)
"""
function apply_edge_dropout!(gnn::GnnFacade, rate::Real)
    ccall((:gnn_apply_edge_dropout, libgnn[]), Cint,
          (Ptr{Cvoid}, Cfloat),
          gnn.handle, Float32(rate))
end

"""
    get_masked_node_count(gnn::GnnFacade) -> UInt32

Get the count of active (masked) nodes.
"""
function get_masked_node_count(gnn::GnnFacade)
    return ccall((:gnn_get_masked_node_count, libgnn[]), Cuint,
                 (Ptr{Cvoid},), gnn.handle)
end

"""
    get_masked_edge_count(gnn::GnnFacade) -> UInt32

Get the count of active (masked) edges.
"""
function get_masked_edge_count(gnn::GnnFacade)
    return ccall((:gnn_get_masked_edge_count, libgnn[]), Cuint,
                 (Ptr{Cvoid},), gnn.handle)
end

# ============================================================================
# Analytics
# ============================================================================

"""
    compute_page_rank(gnn::GnnFacade; damping::Real=0.85, iterations::Integer=20) -> Vector{Float32}

Compute PageRank scores for all nodes.

# Arguments
- `damping`: Damping factor (default: 0.85)
- `iterations`: Number of iterations (default: 20)

# Example
```julia
scores = compute_page_rank(gnn)
println("PageRank scores: ", scores)
```
"""
function compute_page_rank(gnn::GnnFacade; damping::Real=0.85, iterations::Integer=20)
    num_nodes = get_num_nodes(gnn)
    scores = Vector{Float32}(undef, num_nodes)
    
    count = ccall((:gnn_compute_page_rank, libgnn[]), Cint,
                  (Ptr{Cvoid}, Cfloat, Cuint, Ptr{Cfloat}, Cuint),
                  gnn.handle, Float32(damping), iterations, scores, num_nodes)
    
    return count > 0 ? scores[1:count] : Float32[]
end

"""
    get_gradient_flow(gnn::GnnFacade, layer_idx::Integer) -> GradientFlowInfo

Get gradient flow information for a layer.

# Example
```julia
info = get_gradient_flow(gnn, 0)
println("Mean gradient: ", info.mean_gradient)
```
"""
function get_gradient_flow(gnn::GnnFacade, layer_idx::Integer)
    info = Ref{GradientFlowInfo}()
    
    ccall((:gnn_get_gradient_flow, libgnn[]), Cint,
          (Ptr{Cvoid}, Cuint, Ptr{GradientFlowInfo}),
          gnn.handle, layer_idx, info)
    
    return info[]
end

"""
    get_parameter_count(gnn::GnnFacade) -> UInt32

Get the total number of trainable parameters.
"""
function get_parameter_count(gnn::GnnFacade)
    return ccall((:gnn_get_parameter_count, libgnn[]), Cuint,
                 (Ptr{Cvoid},), gnn.handle)
end

"""
    get_architecture_summary(gnn::GnnFacade) -> String

Get a summary of the network architecture.

# Example
```julia
summary = get_architecture_summary(gnn)
println(summary)
```
"""
function get_architecture_summary(gnn::GnnFacade)
    buffer = Vector{UInt8}(undef, 4096)
    
    length = ccall((:gnn_get_architecture_summary, libgnn[]), Cint,
                   (Ptr{Cvoid}, Ptr{Cchar}, Cuint),
                   gnn.handle, buffer, 4096)
    
    return length > 0 ? unsafe_string(pointer(buffer), length) : ""
end

"""
    export_graph_to_json(gnn::GnnFacade) -> String

Export the graph structure as JSON.

# Example
```julia
json = export_graph_to_json(gnn)
println(json)
```
"""
function export_graph_to_json(gnn::GnnFacade)
    buffer = Vector{UInt8}(undef, 65536)
    
    length = ccall((:gnn_export_graph_to_json, libgnn[]), Cint,
                   (Ptr{Cvoid}, Ptr{Cchar}, Cuint),
                   gnn.handle, buffer, 65536)
    
    return length > 0 ? unsafe_string(pointer(buffer), length) : "{}"
end

# ============================================================================
# Backend
# ============================================================================

"""
    get_backend_name(gnn::GnnFacade) -> String

Get the name of the active GPU backend ("cuda" or "opencl").
"""
function get_backend_name(gnn::GnnFacade)
    buffer = Vector{UInt8}(undef, 32)

    length = ccall((:gnn_get_backend_name, libgnn[]), Cint,
                   (Ptr{Cvoid}, Ptr{Cchar}, Cuint),
                   gnn.handle, buffer, 32)

    return length > 0 ? unsafe_string(pointer(buffer), length) : "unknown"
end

# ============================================================================
# Base overloads for nice printing
# ============================================================================

function Base.show(io::IO, gnn::GnnFacade)
    if gnn.handle == C_NULL
        print(io, "GnnFacade(closed)")
    else
        print(io, "GnnFacade(feature_size=$(get_feature_size(gnn)), ",
              "hidden_size=$(get_hidden_size(gnn)), ",
              "output_size=$(get_output_size(gnn)), ",
              "mp_layers=$(get_num_message_passing_layers(gnn)), ",
              "nodes=$(get_num_nodes(gnn)), ",
              "edges=$(get_num_edges(gnn)))")
    end
end

function Base.show(io::IO, info::GradientFlowInfo)
    print(io, "GradientFlowInfo(layer=$(info.layer_idx), ",
          "mean=$(info.mean_gradient), max=$(info.max_gradient), ",
          "min=$(info.min_gradient), norm=$(info.gradient_norm))")
end

function Base.show(io::IO, header::ModelHeader)
    print(io, "ModelHeader(features=$(header.feature_size), ",
          "hidden=$(header.hidden_size), output=$(header.output_size), ",
          "mp_layers=$(header.mp_layers), lr=$(header.learning_rate))")
end

end # module
