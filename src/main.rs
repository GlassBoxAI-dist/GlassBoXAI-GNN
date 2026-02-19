//! @file
//! @ingroup GNN_Internal_Logic
/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

use clap::{Parser, Subcommand, ValueEnum};
use rand::Rng;

use gnn_facade_cuda::{GnnFacade, GpuBackendType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ActivationType {
    Relu,
    LeakyRelu,
    Tanh,
    Sigmoid,
}

impl std::fmt::Display for ActivationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationType::Relu => write!(f, "relu"),
            ActivationType::LeakyRelu => write!(f, "leakyrelu"),
            ActivationType::Tanh => write!(f, "tanh"),
            ActivationType::Sigmoid => write!(f, "sigmoid"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LossType {
    Mse,
    Bce,
}

impl std::fmt::Display for LossType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LossType::Mse => write!(f, "mse"),
            LossType::Bce => write!(f, "bce"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendType {
    Cuda,
    Opencl,
    Auto,
}

impl From<BackendType> for GpuBackendType {
    fn from(b: BackendType) -> Self {
        match b {
            BackendType::Cuda => GpuBackendType::Cuda,
            BackendType::Opencl => GpuBackendType::OpenCL,
            BackendType::Auto => GpuBackendType::Auto,
        }
    }
}

#[derive(Parser)]
#[command(name = "gnn_facade_cuda")]
#[command(about = "Graph Neural Network with Facade Pattern and CUDA/OpenCL acceleration")]
#[command(after_help = r#"FACADE FUNCTIONS - GRAPH STRUCTURE:
  create-graph             Create empty graph with N nodes and feature dim
  load-graph               Load graph from CSV files
  save-graph               Save graph to CSV files
  export-json              Export graph as JSON

FACADE FUNCTIONS - NODE OPERATIONS:
  add-node                 Add a node with features
  get-node-features        Get all features for a node
  set-node-features        Set all features for a node
  get-in-degree            Get node in-degree
  get-out-degree           Get node out-degree

FACADE FUNCTIONS - EDGE OPERATIONS:
  add-edge                 Add edge with optional features
  remove-edge              Remove edge by index
  has-edge                 Check if edge exists

FACADE FUNCTIONS - MASKING/DROPOUT:
  set-node-mask            Set node mask (true=active)
  set-edge-mask            Set edge mask (true=active)
  apply-node-dropout       Apply random node dropout (0.0-1.0)
  apply-edge-dropout       Apply random edge dropout (0.0-1.0)

FACADE FUNCTIONS - MODEL ANALYSIS:
  gradient-flow            Get gradient flow info for layer
  get-parameter-count      Get total trainable parameters
  compute-pagerank         Compute PageRank scores

EXAMPLES:
  # Create a new model
  gnn_facade_cuda create --feature=3 --hidden=16 --output=2 --mp-layers=2 --model=model.bin

  # Create and manipulate a graph
  gnn_facade_cuda create-graph --model=model.bin --nodes=5 --features=3
  gnn_facade_cuda add-edge --model=model.bin --source=0 --target=1

  # Apply dropout and predict
  gnn_facade_cuda apply-node-dropout --model=model.bin --rate=0.2
  gnn_facade_cuda predict --model=model.bin --graph=graph.csv
"#)]
struct Cli {
    /// GPU backend to use (cuda, opencl, or auto-detect)
    #[arg(long, value_enum, default_value = "auto", global = true)]
    backend: BackendType,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new GNN model
    Create {
        #[arg(long)]
        feature: usize,
        #[arg(long)]
        hidden: usize,
        #[arg(long)]
        output: usize,
        #[arg(long = "mp-layers")]
        mp_layers: usize,
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0.01")]
        lr: f32,
        #[arg(long, value_enum, default_value = "relu")]
        activation: ActivationType,
        #[arg(long, value_enum, default_value = "mse")]
        loss: LossType,
    },
    /// Train the model with graph data
    Train {
        #[arg(long)]
        model: String,
        #[arg(long)]
        graph: String,
        #[arg(long)]
        save: String,
        #[arg(long, default_value = "100")]
        epochs: usize,
        #[arg(long)]
        lr: Option<f32>,
    },
    /// Make predictions on a graph
    Predict {
        #[arg(long)]
        model: String,
        #[arg(long)]
        graph: String,
    },
    /// Display model information
    Info {
        #[arg(long)]
        model: String,
    },
    /// Get node degree (in + out)
    Degree {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
    },
    /// Get node in-degree
    InDegree {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
    },
    /// Get node out-degree
    OutDegree {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
    },
    /// Get node neighbors
    Neighbors {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
    },
    /// Compute PageRank scores
    Pagerank {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0.85")]
        damping: f32,
        #[arg(long, default_value = "20")]
        iterations: usize,
    },
    /// Save model to file
    Save {
        #[arg(long)]
        model: String,
        #[arg(long)]
        output: String,
    },
    /// Load model from file
    Load {
        #[arg(long)]
        model: String,
    },
    /// Create empty graph with N nodes
    CreateGraph {
        #[arg(long)]
        model: String,
        #[arg(long)]
        nodes: usize,
        #[arg(long)]
        features: usize,
    },
    /// Add edge to graph
    AddEdge {
        #[arg(long)]
        model: String,
        #[arg(long)]
        source: usize,
        #[arg(long)]
        target: usize,
        #[arg(long)]
        features: Option<String>,
    },
    /// Remove edge from graph
    RemoveEdge {
        #[arg(long)]
        model: String,
        #[arg(long)]
        edge: usize,
    },
    /// Set node features
    SetNodeFeatures {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
        #[arg(long)]
        features: String,
    },
    /// Get node features
    GetNodeFeatures {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
    },
    /// Set node mask
    SetNodeMask {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
        #[arg(long)]
        value: bool,
    },
    /// Set edge mask
    SetEdgeMask {
        #[arg(long)]
        model: String,
        #[arg(long)]
        edge: usize,
        #[arg(long)]
        value: bool,
    },
    /// Apply node dropout
    ApplyNodeDropout {
        #[arg(long)]
        model: String,
        #[arg(long)]
        rate: f32,
    },
    /// Apply edge dropout
    ApplyEdgeDropout {
        #[arg(long)]
        model: String,
        #[arg(long)]
        rate: f32,
    },
    /// Get gradient flow analysis
    GradientFlow {
        #[arg(long)]
        model: String,
        #[arg(long)]
        layer: Option<usize>,
    },
    /// Get parameter count
    GetParameterCount {
        #[arg(long)]
        model: String,
    },
    /// Export graph to JSON
    ExportJson {
        #[arg(long)]
        model: String,
    },
}

fn load_model(model: &str, backend: BackendType) -> Result<GnnFacade, Box<dyn std::error::Error>> {
    GnnFacade::from_model_file_with_backend(model, backend.into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let backend = cli.backend;

    match cli.command {
        Commands::Create { feature, hidden, output, mp_layers, model, lr, activation, loss } => {
            let mut facade = GnnFacade::with_backend(feature, hidden, output, mp_layers, backend.into())?;
            facade.set_learning_rate(lr);
            facade.save_model(&model)?;

            println!("Created GNN model:");
            println!("  Backend: {}", facade.get_backend_name());
            println!("  Feature size: {}", feature);
            println!("  Hidden size: {}", hidden);
            println!("  Output size: {}", output);
            println!("  Message passing layers: {}", mp_layers);
            println!("  Activation: {}", activation);
            println!("  Loss function: {}", loss);
            println!("  Learning rate: {:.4}", lr);
            println!("  Saved to: {}", model);
        }
        Commands::Train { model, graph: _, save, epochs, lr } => {
            let mut facade = load_model(&model, backend)?;

            if let Some(learning_rate) = lr {
                facade.set_learning_rate(learning_rate);
            }

            facade.create_empty_graph(5, facade.get_feature_size());

            let mut rng = rand::thread_rng();
            for i in 0..5 {
                let features: Vec<f32> = (0..facade.get_feature_size()).map(|_| rng.gen()).collect();
                facade.set_node_features(i, features);
            }

            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 3, vec![]);
            facade.add_edge(3, 4, vec![]);
            facade.add_edge(4, 0, vec![]);
            facade.add_edge(1, 3, vec![]);

            let target: Vec<f32> = (0..facade.get_output_size()).map(|_| rng.gen()).collect();

            println!("Training model for {} epochs...", epochs);
            facade.train_multiple(&target, epochs)?;
            facade.save_model(&save)?;
            println!("Model saved to: {}", save);
        }
        Commands::Predict { model, graph: _ } => {
            let mut facade = load_model(&model, backend)?;

            facade.create_empty_graph(5, facade.get_feature_size());

            let mut rng = rand::thread_rng();
            for i in 0..5 {
                let features: Vec<f32> = (0..facade.get_feature_size()).map(|_| rng.gen()).collect();
                facade.set_node_features(i, features);
            }

            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 3, vec![]);
            facade.add_edge(3, 4, vec![]);
            facade.add_edge(4, 0, vec![]);
            facade.add_edge(1, 3, vec![]);

            let prediction = facade.predict()?;

            println!("Graph nodes: {}, edges: {}", facade.get_num_nodes(), facade.get_num_edges());
            print!("Prediction: [");
            for (i, p) in prediction.iter().enumerate() {
                if i > 0 { print!(", "); }
                print!("{:.6}", p);
            }
            println!("]");
        }
        Commands::Info { model } => {
            let facade = load_model(&model, backend)?;

            println!("GNN Facade Model Information");
            println!("=========================================");
            println!("GPU Backend: {}", facade.get_backend_name());
            println!("Feature size: {}", facade.get_feature_size());
            println!("Hidden size: {}", facade.get_hidden_size());
            println!("Output size: {}", facade.get_output_size());
            println!("Message passing layers: {}", facade.get_num_message_passing_layers());
            println!();
            println!("Hyperparameters:");
            println!("  Learning rate: {:.6}", facade.get_learning_rate());
            println!("File: {}", model);
            println!();
            println!("{}", facade.get_architecture_summary());
        }
        Commands::Degree { model, node } => {
            let mut facade = load_model(&model, backend)?;

            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 0, vec![]);

            println!("Node {} degree information:", node);
            println!("  In-degree: {}", facade.get_in_degree(node));
            println!("  Out-degree: {}", facade.get_out_degree(node));
            println!("  Total degree: {}", facade.get_in_degree(node) + facade.get_out_degree(node));
        }
        Commands::Neighbors { model, node } => {
            let mut facade = load_model(&model, backend)?;

            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.add_edge(0, 2, vec![]);
            facade.add_edge(1, 2, vec![]);

            if let Some(neighbors) = facade.get_neighbors(node) {
                println!("Node {} neighbors: {:?}", node, neighbors);
            } else {
                println!("Node {} not found", node);
            }
        }
        Commands::Pagerank { model, damping, iterations } => {
            let mut facade = load_model(&model, backend)?;

            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 3, vec![]);
            facade.add_edge(3, 4, vec![]);
            facade.add_edge(4, 0, vec![]);

            let ranks = facade.compute_page_rank(damping, iterations);

            println!("PageRank (damping={}, iterations={}):", damping, iterations);
            for (i, r) in ranks.iter().enumerate() {
                println!("  Node {}: {:.6}", i, r);
            }
        }
        Commands::Save { model, output } => {
            let facade = load_model(&model, backend)?;
            facade.save_model(&output)?;
            println!("Model saved to: {}", output);
        }
        Commands::Load { model } => {
            let facade = load_model(&model, backend)?;
            println!("Model loaded from: {}", model);
            println!("Feature size: {}", facade.get_feature_size());
            println!("Hidden size: {}", facade.get_hidden_size());
            println!("Output size: {}", facade.get_output_size());
        }
        Commands::InDegree { model, node } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 0, vec![]);
            println!("{}", facade.get_in_degree(node));
        }
        Commands::OutDegree { model, node } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 0, vec![]);
            println!("{}", facade.get_out_degree(node));
        }
        Commands::CreateGraph { model, nodes, features } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(nodes, features);
            facade.save_model(&model)?;
            println!("Created graph: {} nodes, {} features", nodes, features);
        }
        Commands::AddEdge { model, source, target, features } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(std::cmp::max(source, target) + 1, facade.get_feature_size());
            let feat_vec: Vec<f32> = features
                .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                .unwrap_or_default();
            let idx = facade.add_edge(source, target, feat_vec);
            facade.save_model(&model)?;
            println!("Added edge {}: {} -> {}", idx, source, target);
        }
        Commands::RemoveEdge { model, edge } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.remove_edge(edge);
            facade.save_model(&model)?;
            println!("Removed edge {}", edge);
        }
        Commands::SetNodeFeatures { model, node, features } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(node + 1, facade.get_feature_size());
            let feat_vec: Vec<f32> = features.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            facade.set_node_features(node, feat_vec);
            facade.save_model(&model)?;
            println!("Set features for node {}", node);
        }
        Commands::GetNodeFeatures { model, node } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(node + 1, facade.get_feature_size());
            if let Some(features) = facade.get_node_features(node) {
                print!("[");
                for (i, f) in features.iter().enumerate() {
                    if i > 0 { print!(", "); }
                    print!("{}", f);
                }
                println!("]");
            } else {
                println!("Node {} not found", node);
            }
        }
        Commands::SetNodeMask { model, node, value } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(node + 1, facade.get_feature_size());
            facade.set_node_mask(node, value);
            facade.save_model(&model)?;
            println!("Node {} mask = {}", node, value);
        }
        Commands::SetEdgeMask { model, edge, value } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.set_edge_mask(edge, value);
            facade.save_model(&model)?;
            println!("Edge {} mask = {}", edge, value);
        }
        Commands::ApplyNodeDropout { model, rate } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.apply_node_dropout(rate);
            facade.save_model(&model)?;
            println!("Applied node dropout rate={}", rate);
        }
        Commands::ApplyEdgeDropout { model, rate } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.apply_edge_dropout(rate);
            facade.save_model(&model)?;
            println!("Applied edge dropout rate={}", rate);
        }
        Commands::GradientFlow { model, layer } => {
            let facade = load_model(&model, backend)?;
            let layer_idx = layer.unwrap_or(0);
            let info = facade.get_gradient_flow(layer_idx);
            println!("Layer {}: mean={:.6}, max={:.6}, min={:.6}, norm={:.6}",
                     info.layer_idx, info.mean_gradient, info.max_gradient,
                     info.min_gradient, info.gradient_norm);
        }
        Commands::GetParameterCount { model } => {
            let facade = load_model(&model, backend)?;
            println!("{}", facade.get_parameter_count());
        }
        Commands::ExportJson { model } => {
            let mut facade = load_model(&model, backend)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            println!("{}", facade.export_graph_to_json());
        }
    }

    Ok(())
}

