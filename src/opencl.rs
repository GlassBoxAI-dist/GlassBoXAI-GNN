//! @file
//! @ingroup GNN_GPU_Accelerated
/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * OpenCL backend for GlassBoxAI GNN
 */

//! OpenCL-accelerated Graph Neural Network backend
//!
//! This module provides an OpenCL implementation of the GNN compute backend,
//! functionally equivalent to the CUDA backend. It can run on any GPU or
//! accelerator that supports OpenCL (AMD, Intel, NVIDIA, etc.).

use crate::{ActivationType, Graph, LossType, MAX_NODES, MAX_EDGES};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_float, cl_int, CL_BLOCKING};
use rand::Rng;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::ptr;

const BLOCK_SIZE: usize = 256;

const OPENCL_KERNELS: &str = r#"
// ============================================================================
// Device helper functions
// ============================================================================

float cl_activate(float x, int activationType) {
    switch (activationType) {
        case 0: // ReLU
            return x > 0.0f ? x : 0.0f;
        case 1: // LeakyReLU
            return x > 0.0f ? x : 0.01f * x;
        case 2: // Tanh
            return tanh(x);
        case 3: // Sigmoid
            return 1.0f / (1.0f + exp(-fmax(-500.0f, fmin(500.0f, x))));
        default:
            return x;
    }
}

float cl_activateDerivative(float x, int activationType) {
    switch (activationType) {
        case 0: // ReLU
            return x > 0.0f ? 1.0f : 0.0f;
        case 1: // LeakyReLU
            return x > 0.0f ? 1.0f : 0.01f;
        case 2: // Tanh
            { float t = tanh(x); return 1.0f - t * t; }
        case 3: // Sigmoid
            { float s = 1.0f / (1.0f + exp(-fmax(-500.0f, fmin(500.0f, x)))); return s * (1.0f - s); }
        default:
            return 1.0f;
    }
}

float cl_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-fmax(-500.0f, fmin(500.0f, x))));
}

float cl_sigmoidDerivative(float x) {
    float s = cl_sigmoid(x);
    return s * (1.0f - s);
}

float cl_clipGradient(float g) {
    return fmax(-5.0f, fmin(5.0f, g));
}

// ============================================================================
// Kernels
// ============================================================================

__kernel void k_forwardLayer(
    __global const float* restrict input,
    __global const float* restrict weights,
    __global const float* restrict biases,
    __global float* restrict preActivations,
    __global float* restrict outputs,
    int numInputs,
    int numOutputs,
    int activationType,
    int useOutputActivation
) {
    int neuronIdx = get_global_id(0);
    if (neuronIdx >= numOutputs) return;

    float sum = biases[neuronIdx];
    for (int j = 0; j < numInputs; ++j) {
        sum += weights[neuronIdx * numInputs + j] * input[j];
    }

    preActivations[neuronIdx] = sum;

    if (useOutputActivation) {
        outputs[neuronIdx] = cl_sigmoid(sum);
    } else {
        outputs[neuronIdx] = cl_activate(sum, activationType);
    }
}

__kernel void k_backwardLayer(
    __global const float* restrict lastInput,
    __global const float* restrict preActivations,
    __global const float* restrict upstreamGrad,
    __global float* restrict weights,
    __global float* restrict biases,
    __global float* restrict weightGradients,
    __global float* restrict biasGradients,
    int numInputs,
    int numOutputs,
    int activationType,
    int useOutputActivation,
    float learningRate
) {
    int neuronIdx = get_global_id(0);
    if (neuronIdx >= numOutputs) return;

    float preActGrad;
    if (useOutputActivation) {
        preActGrad = upstreamGrad[neuronIdx] * cl_sigmoidDerivative(preActivations[neuronIdx]);
    } else {
        preActGrad = upstreamGrad[neuronIdx] * cl_activateDerivative(preActivations[neuronIdx], activationType);
    }
    preActGrad = cl_clipGradient(preActGrad);

    for (int j = 0; j < numInputs; ++j) {
        float grad = cl_clipGradient(preActGrad * lastInput[j]);
        weightGradients[neuronIdx * numInputs + j] = grad;
        weights[neuronIdx * numInputs + j] -= learningRate * grad;
    }

    biasGradients[neuronIdx] = preActGrad;
    biases[neuronIdx] -= learningRate * preActGrad;
}

__kernel void k_computeInputGrad(
    __global const float* restrict weights,
    __global const float* restrict preActivations,
    __global const float* restrict upstreamGrad,
    __global float* restrict inputGrad,
    int numInputs,
    int numOutputs,
    int activationType,
    int useOutputActivation
) {
    int inputIdx = get_global_id(0);
    if (inputIdx >= numInputs) return;

    float grad = 0.0f;
    for (int i = 0; i < numOutputs; ++i) {
        float preActGrad;
        if (useOutputActivation) {
            preActGrad = upstreamGrad[i] * cl_sigmoidDerivative(preActivations[i]);
        } else {
            preActGrad = upstreamGrad[i] * cl_activateDerivative(preActivations[i], activationType);
        }
        grad += weights[i * numInputs + inputIdx] * preActGrad;
    }
    inputGrad[inputIdx] = cl_clipGradient(grad);
}

__kernel void k_aggregateMessages(
    __global const float* restrict allMessages,
    __global const int* restrict neighborCounts,
    __global const int* restrict neighborOffsets,
    __global float* restrict aggregatedMessages,
    int numNodes,
    int hiddenSize
) {
    int nodeIdx = get_group_id(0);
    int dimIdx = get_local_id(0);

    if (nodeIdx >= numNodes || dimIdx >= hiddenSize) return;

    int numNeighbors = neighborCounts[nodeIdx];
    int offset = neighborOffsets[nodeIdx];

    float sum = 0.0f;
    for (int n = 0; n < numNeighbors; ++n) {
        sum += allMessages[(offset + n) * hiddenSize + dimIdx];
    }

    if (numNeighbors > 0) {
        aggregatedMessages[nodeIdx * hiddenSize + dimIdx] = sum / numNeighbors;
    } else {
        aggregatedMessages[nodeIdx * hiddenSize + dimIdx] = 0.0f;
    }
}

__kernel void k_graphReadout(
    __global const float* restrict nodeEmbeddings,
    __global float* restrict graphEmbedding,
    int numNodes,
    int hiddenSize
) {
    int dimIdx = get_global_id(0);
    if (dimIdx >= hiddenSize) return;

    float sum = 0.0f;
    for (int n = 0; n < numNodes; ++n) {
        sum += nodeEmbeddings[n * hiddenSize + dimIdx];
    }
    graphEmbedding[dimIdx] = sum / numNodes;
}

__kernel void k_computeMSEGradient(
    __global const float* restrict prediction,
    __global const float* restrict target,
    __global float* restrict gradient,
    int size
) {
    int idx = get_global_id(0);
    if (idx >= size) return;

    gradient[idx] = 2.0f * (prediction[idx] - target[idx]) / size;
}
"#;

/// An OpenCL GPU layer for neural network computations
pub struct OpenCLGpuLayer {
    d_weights: Buffer<cl_float>,
    d_biases: Buffer<cl_float>,
    d_weight_gradients: Buffer<cl_float>,
    d_bias_gradients: Buffer<cl_float>,
    d_pre_activations: Buffer<cl_float>,
    d_outputs: Buffer<cl_float>,
    d_last_input: Buffer<cl_float>,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

impl OpenCLGpuLayer {
    fn new(context: &Context, queue: &CommandQueue, num_inputs: usize, num_outputs: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (num_inputs + num_outputs) as f32).sqrt();

        let weights: Vec<f32> = (0..num_inputs * num_outputs)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let biases = vec![0.0f32; num_outputs];

        let mut d_weights = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, num_inputs * num_outputs, ptr::null_mut())?
        };
        let mut d_biases = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, num_outputs, ptr::null_mut())?
        };
        let mut d_weight_gradients = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, num_inputs * num_outputs, ptr::null_mut())?
        };
        let mut d_bias_gradients = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, num_outputs, ptr::null_mut())?
        };
        let d_pre_activations = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, num_outputs, ptr::null_mut())?
        };
        let d_outputs = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, num_outputs, ptr::null_mut())?
        };
        let d_last_input = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, num_inputs, ptr::null_mut())?
        };

        // Upload initial weights and biases
        unsafe {
            queue.enqueue_write_buffer(&mut d_weights, CL_BLOCKING, 0, &weights, &[])?;
            queue.enqueue_write_buffer(&mut d_biases, CL_BLOCKING, 0, &biases, &[])?;
        }

        // Zero-fill gradient buffers
        let zeros_wg = vec![0.0f32; num_inputs * num_outputs];
        let zeros_bg = vec![0.0f32; num_outputs];
        unsafe {
            queue.enqueue_write_buffer(&mut d_weight_gradients, CL_BLOCKING, 0, &zeros_wg, &[])?;
            queue.enqueue_write_buffer(&mut d_bias_gradients, CL_BLOCKING, 0, &zeros_bg, &[])?;
        }

        Ok(Self {
            d_weights,
            d_biases,
            d_weight_gradients,
            d_bias_gradients,
            d_pre_activations,
            d_outputs,
            d_last_input,
            num_inputs,
            num_outputs,
        })
    }

    fn forward_from_host(
        &mut self,
        queue: &CommandQueue,
        kernel: &Kernel,
        input: &[f32],
        activation: i32,
        use_output_activation: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Upload input
        unsafe {
            queue.enqueue_write_buffer(&mut self.d_last_input, CL_BLOCKING, 0, input, &[])?;
        }

        let global_work_size = ((self.num_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

        let kernel_event = unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(&self.d_last_input)
                .set_arg(&self.d_weights)
                .set_arg(&self.d_biases)
                .set_arg(&self.d_pre_activations)
                .set_arg(&self.d_outputs)
                .set_arg(&(self.num_inputs as cl_int))
                .set_arg(&(self.num_outputs as cl_int))
                .set_arg(&activation)
                .set_arg(&(use_output_activation as cl_int))
                .set_global_work_size(global_work_size)
                .set_local_work_size(BLOCK_SIZE)
                .enqueue_nd_range(queue)?
        };
        kernel_event.wait()?;

        let mut result = vec![0.0f32; self.num_outputs];
        unsafe {
            queue.enqueue_read_buffer(&self.d_outputs, CL_BLOCKING, 0, &mut result, &[])?;
        }
        Ok(result)
    }

    fn backward_from_host(
        &mut self,
        context: &Context,
        queue: &CommandQueue,
        kernel: &Kernel,
        upstream_grad: &[f32],
        activation: i32,
        use_output_activation: bool,
        learning_rate: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut d_upstream_grad = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, self.num_outputs, ptr::null_mut())?
        };
        unsafe {
            queue.enqueue_write_buffer(&mut d_upstream_grad, CL_BLOCKING, 0, upstream_grad, &[])?;
        }

        let global_work_size = ((self.num_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

        let kernel_event = unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(&self.d_last_input)
                .set_arg(&self.d_pre_activations)
                .set_arg(&d_upstream_grad)
                .set_arg(&self.d_weights)
                .set_arg(&self.d_biases)
                .set_arg(&self.d_weight_gradients)
                .set_arg(&self.d_bias_gradients)
                .set_arg(&(self.num_inputs as cl_int))
                .set_arg(&(self.num_outputs as cl_int))
                .set_arg(&activation)
                .set_arg(&(use_output_activation as cl_int))
                .set_arg(&learning_rate)
                .set_global_work_size(global_work_size)
                .set_local_work_size(BLOCK_SIZE)
                .enqueue_nd_range(queue)?
        };
        kernel_event.wait()?;
        Ok(())
    }

    fn compute_input_grad_from_host(
        &self,
        context: &Context,
        queue: &CommandQueue,
        kernel: &Kernel,
        upstream_grad: &[f32],
        activation: i32,
        use_output_activation: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut d_upstream_grad = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, self.num_outputs, ptr::null_mut())?
        };
        unsafe {
            queue.enqueue_write_buffer(&mut d_upstream_grad, CL_BLOCKING, 0, upstream_grad, &[])?;
        }

        let d_input_grad = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, self.num_inputs, ptr::null_mut())?
        };

        let global_work_size = ((self.num_inputs + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

        let kernel_event = unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(&self.d_weights)
                .set_arg(&self.d_pre_activations)
                .set_arg(&d_upstream_grad)
                .set_arg(&d_input_grad)
                .set_arg(&(self.num_inputs as cl_int))
                .set_arg(&(self.num_outputs as cl_int))
                .set_arg(&activation)
                .set_arg(&(use_output_activation as cl_int))
                .set_global_work_size(global_work_size)
                .set_local_work_size(BLOCK_SIZE)
                .enqueue_nd_range(queue)?
        };
        kernel_event.wait()?;

        let mut result = vec![0.0f32; self.num_inputs];
        unsafe {
            queue.enqueue_read_buffer(&d_input_grad, CL_BLOCKING, 0, &mut result, &[])?;
        }
        Ok(result)
    }

    fn copy_weights_to_host(&self, queue: &CommandQueue) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let mut weights = vec![0.0f32; self.num_inputs * self.num_outputs];
        let mut biases = vec![0.0f32; self.num_outputs];
        unsafe {
            queue.enqueue_read_buffer(&self.d_weights, CL_BLOCKING, 0, &mut weights, &[])?;
            queue.enqueue_read_buffer(&self.d_biases, CL_BLOCKING, 0, &mut biases, &[])?;
        }
        Ok((weights, biases))
    }

    fn copy_weights_from_host(&mut self, queue: &CommandQueue, weights: &[f32], biases: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            queue.enqueue_write_buffer(&mut self.d_weights, CL_BLOCKING, 0, weights, &[])?;
            queue.enqueue_write_buffer(&mut self.d_biases, CL_BLOCKING, 0, biases, &[])?;
        }
        Ok(())
    }
}

/// OpenCL-accelerated Graph Neural Network
///
/// This is the core GNN implementation with OpenCL acceleration.
/// It provides the same interface as CudaGraphNeuralNetwork but uses
/// OpenCL for GPU computation, supporting AMD, Intel, and NVIDIA GPUs.
pub struct OpenCLGraphNeuralNetwork {
    learning_rate: f32,
    num_message_passing_layers: usize,
    pub feature_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    activation: ActivationType,
    #[allow(dead_code)]
    loss_type: LossType,

    context: Context,
    queue: CommandQueue,
    program: Program,

    message_layers: Vec<OpenCLGpuLayer>,
    update_layers: Vec<OpenCLGpuLayer>,
    readout_layer: OpenCLGpuLayer,
    output_layer: OpenCLGpuLayer,

    d_node_embeddings: Buffer<cl_float>,
    #[allow(dead_code)]
    d_new_node_embeddings: Buffer<cl_float>,
    d_graph_embedding: Buffer<cl_float>,
    d_messages: Buffer<cl_float>,
    d_aggregated_messages: Buffer<cl_float>,
    d_neighbor_counts: Buffer<cl_int>,
    d_neighbor_offsets: Buffer<cl_int>,

    h_graph_embedding: Vec<f32>,
}

impl OpenCLGraphNeuralNetwork {
    /// Create a new OpenCL Graph Neural Network
    pub fn new(
        feature_size: usize,
        hidden_size: usize,
        output_size: usize,
        num_mp_layers: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Find an OpenCL device (prefer GPU, fall back to any)
        let device_id = get_all_devices(CL_DEVICE_TYPE_GPU)
            .unwrap_or_default()
            .into_iter()
            .next()
            .or_else(|| {
                get_all_devices(CL_DEVICE_TYPE_ALL)
                    .unwrap_or_default()
                    .into_iter()
                    .next()
            })
            .ok_or("No OpenCL devices found")?;

        let device = Device::new(device_id);
        let context = Context::from_device(&device)?;
        let queue = CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)?;

        // Build the OpenCL program
        let program = Program::create_and_build_from_source(&context, OPENCL_KERNELS, "")
            .map_err(|e| format!("OpenCL kernel compilation failed: {:?}", e))?;

        // Verify all kernels can be created
        let _ = Kernel::create(&program, "k_forwardLayer")?;
        let _ = Kernel::create(&program, "k_backwardLayer")?;
        let _ = Kernel::create(&program, "k_computeInputGrad")?;
        let _ = Kernel::create(&program, "k_aggregateMessages")?;
        let _ = Kernel::create(&program, "k_graphReadout")?;
        let _ = Kernel::create(&program, "k_computeMSEGradient")?;

        // Create layers
        let mut message_layers = Vec::with_capacity(num_mp_layers);
        let mut update_layers = Vec::with_capacity(num_mp_layers);

        for i in 0..num_mp_layers {
            let msg_input_size = if i == 0 { feature_size * 2 } else { hidden_size * 2 };
            message_layers.push(OpenCLGpuLayer::new(&context, &queue, msg_input_size, hidden_size)?);
            update_layers.push(OpenCLGpuLayer::new(&context, &queue, hidden_size * 2, hidden_size)?);
        }

        let readout_layer = OpenCLGpuLayer::new(&context, &queue, hidden_size, hidden_size)?;
        let output_layer = OpenCLGpuLayer::new(&context, &queue, hidden_size, output_size)?;

        // Allocate shared buffers
        let d_node_embeddings = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, MAX_NODES * hidden_size, ptr::null_mut())?
        };
        let d_new_node_embeddings = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, MAX_NODES * hidden_size, ptr::null_mut())?
        };
        let d_graph_embedding = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, hidden_size, ptr::null_mut())?
        };
        let d_messages = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, MAX_EDGES * hidden_size, ptr::null_mut())?
        };
        let d_aggregated_messages = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, MAX_NODES * hidden_size, ptr::null_mut())?
        };
        let d_neighbor_counts = unsafe {
            Buffer::<cl_int>::create(&context, CL_MEM_READ_WRITE, MAX_NODES, ptr::null_mut())?
        };
        let d_neighbor_offsets = unsafe {
            Buffer::<cl_int>::create(&context, CL_MEM_READ_WRITE, MAX_NODES, ptr::null_mut())?
        };

        Ok(Self {
            learning_rate: 0.01,
            num_message_passing_layers: num_mp_layers,
            feature_size,
            hidden_size,
            output_size,
            activation: ActivationType::Relu,
            loss_type: LossType::Mse,
            context,
            queue,
            program,
            message_layers,
            update_layers,
            readout_layer,
            output_layer,
            d_node_embeddings,
            d_new_node_embeddings,
            d_graph_embedding,
            d_messages,
            d_aggregated_messages,
            d_neighbor_counts,
            d_neighbor_offsets,
            h_graph_embedding: vec![0.0; hidden_size],
        })
    }

    /// Run forward pass and get predictions for a graph
    pub fn predict(&mut self, graph: &mut Graph) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        graph.build_adjacency_list();
        let num_nodes = graph.num_nodes;

        // Build neighbor count/offset arrays
        let mut h_neighbor_counts = vec![0i32; MAX_NODES];
        for (i, adj) in graph.adjacency_list.iter().enumerate() {
            h_neighbor_counts[i] = adj.len() as i32;
        }
        let mut h_neighbor_offsets = vec![0i32; MAX_NODES];
        let mut total_neighbors = 0i32;
        for i in 0..num_nodes {
            h_neighbor_offsets[i] = total_neighbors;
            total_neighbors += h_neighbor_counts[i];
        }

        unsafe {
            self.queue.enqueue_write_buffer(&mut self.d_neighbor_counts, CL_BLOCKING, 0, &h_neighbor_counts, &[])?;
            self.queue.enqueue_write_buffer(&mut self.d_neighbor_offsets, CL_BLOCKING, 0, &h_neighbor_offsets, &[])?;
        }

        // Initialize node embeddings from features
        let mut h_embeddings = vec![0.0f32; MAX_NODES * self.hidden_size];
        for n in 0..num_nodes {
            let copy_size = self.feature_size.min(graph.node_features.get(n).map_or(0, |f| f.len()));
            for f in 0..copy_size {
                h_embeddings[n * self.hidden_size + f] = graph.node_features[n][f];
            }
        }
        unsafe {
            self.queue.enqueue_write_buffer(&mut self.d_node_embeddings, CL_BLOCKING, 0, &h_embeddings, &[])?;
        }

        let activation = self.activation.to_int();
        let mut h_all_embeddings = vec![0.0f32; MAX_NODES * self.hidden_size];

        // Message passing layers
        for layer in 0..self.num_message_passing_layers {
            unsafe {
                self.queue.enqueue_read_buffer(&self.d_node_embeddings, CL_BLOCKING, 0, &mut h_all_embeddings, &[])?;
            }

            let mut h_all_messages = vec![0.0f32; MAX_EDGES * self.hidden_size];
            let mut msg_offset = 0usize;

            // Create a new kernel for each forward call to avoid aliasing issues
            let k_fwd = Kernel::create(&self.program, "k_forwardLayer")?;

            for node in 0..num_nodes {
                for &neighbor in &graph.adjacency_list[node] {
                    let input_size = self.message_layers[layer].num_inputs;
                    let mut h_temp_input = vec![0.0f32; input_size];

                    let emb_size = if layer == 0 { self.feature_size } else { self.hidden_size };
                    for i in 0..emb_size.min(input_size / 2) {
                        h_temp_input[i] = h_all_embeddings[node * self.hidden_size + i];
                    }
                    for i in 0..emb_size.min(input_size / 2) {
                        h_temp_input[input_size / 2 + i] = h_all_embeddings[neighbor * self.hidden_size + i];
                    }

                    let message = self.message_layers[layer].forward_from_host(
                        &self.queue, &k_fwd, &h_temp_input, activation, false,
                    )?;

                    for (i, &m) in message.iter().enumerate() {
                        h_all_messages[msg_offset * self.hidden_size + i] = m;
                    }
                    msg_offset += 1;
                }
            }

            // Upload messages and aggregate
            unsafe {
                self.queue.enqueue_write_buffer(&mut self.d_messages, CL_BLOCKING, 0, &h_all_messages, &[])?;
            }

            // Launch aggregation kernel
            let k_agg = Kernel::create(&self.program, "k_aggregateMessages")?;
            let agg_event = unsafe {
                ExecuteKernel::new(&k_agg)
                    .set_arg(&self.d_messages)
                    .set_arg(&self.d_neighbor_counts)
                    .set_arg(&self.d_neighbor_offsets)
                    .set_arg(&self.d_aggregated_messages)
                    .set_arg(&(num_nodes as cl_int))
                    .set_arg(&(self.hidden_size as cl_int))
                    .set_global_work_size(num_nodes * self.hidden_size)
                    .set_local_work_size(self.hidden_size)
                    .enqueue_nd_range(&self.queue)?
            };
            agg_event.wait()?;

            let mut h_agg_messages = vec![0.0f32; MAX_NODES * self.hidden_size];
            unsafe {
                self.queue.enqueue_read_buffer(&self.d_aggregated_messages, CL_BLOCKING, 0, &mut h_agg_messages, &[])?;
            }

            // Update embeddings
            let mut h_new_embeddings = vec![0.0f32; MAX_NODES * self.hidden_size];
            let k_fwd2 = Kernel::create(&self.program, "k_forwardLayer")?;

            for node in 0..num_nodes {
                let mut h_temp_input = vec![0.0f32; self.hidden_size * 2];
                for i in 0..self.hidden_size {
                    h_temp_input[i] = h_all_embeddings[node * self.hidden_size + i];
                    h_temp_input[self.hidden_size + i] = h_agg_messages[node * self.hidden_size + i];
                }

                let new_emb = self.update_layers[layer].forward_from_host(
                    &self.queue, &k_fwd2, &h_temp_input, activation, false,
                )?;

                for (i, &e) in new_emb.iter().enumerate() {
                    h_new_embeddings[node * self.hidden_size + i] = e;
                }
            }

            unsafe {
                self.queue.enqueue_write_buffer(&mut self.d_node_embeddings, CL_BLOCKING, 0, &h_new_embeddings, &[])?;
            }
        }

        // Graph readout
        let readout_global = ((self.hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        let k_ro = Kernel::create(&self.program, "k_graphReadout")?;
        let readout_event = unsafe {
            ExecuteKernel::new(&k_ro)
                .set_arg(&self.d_node_embeddings)
                .set_arg(&self.d_graph_embedding)
                .set_arg(&(num_nodes as cl_int))
                .set_arg(&(self.hidden_size as cl_int))
                .set_global_work_size(readout_global)
                .set_local_work_size(BLOCK_SIZE)
                .enqueue_nd_range(&self.queue)?
        };
        readout_event.wait()?;

        let mut h_graph_emb = vec![0.0f32; self.hidden_size];
        unsafe {
            self.queue.enqueue_read_buffer(&self.d_graph_embedding, CL_BLOCKING, 0, &mut h_graph_emb, &[])?;
        }

        let k_fwd3 = Kernel::create(&self.program, "k_forwardLayer")?;
        let readout_out = self.readout_layer.forward_from_host(
            &self.queue, &k_fwd3, &h_graph_emb, activation, false,
        )?;
        let k_fwd4 = Kernel::create(&self.program, "k_forwardLayer")?;
        let result = self.output_layer.forward_from_host(
            &self.queue, &k_fwd4, &readout_out, activation, true,
        )?;

        self.h_graph_embedding = h_graph_emb;

        Ok(result)
    }

    /// Train the network on a graph with target labels
    pub fn train(&mut self, graph: &mut Graph, target: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
        let prediction = self.predict(graph)?;

        let loss: f32 = prediction.iter().zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>() / prediction.len() as f32;

        let loss_grad: Vec<f32> = prediction.iter().zip(target.iter())
            .map(|(p, t)| 2.0 * (p - t) / prediction.len() as f32)
            .collect();

        let activation = self.activation.to_int();
        let lr = self.learning_rate;

        let k_bwd = Kernel::create(&self.program, "k_backwardLayer")?;
        let k_ig = Kernel::create(&self.program, "k_computeInputGrad")?;

        self.output_layer.backward_from_host(
            &self.context, &self.queue, &k_bwd, &loss_grad, activation, true, lr,
        )?;
        let readout_grad = self.output_layer.compute_input_grad_from_host(
            &self.context, &self.queue, &k_ig, &loss_grad, activation, true,
        )?;

        let k_bwd2 = Kernel::create(&self.program, "k_backwardLayer")?;
        self.readout_layer.backward_from_host(
            &self.context, &self.queue, &k_bwd2, &readout_grad, activation, false, lr,
        )?;

        Ok(loss)
    }

    /// Train for multiple iterations
    pub fn train_multiple(&mut self, graph: &mut Graph, target: &[f32], iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
        for i in 0..iterations {
            let loss = self.train(graph, target)?;
            if i % 10 == 0 || i == iterations - 1 {
                println!("Iteration {}/{}, Loss: {:.6}", i + 1, iterations, loss);
            }
        }
        Ok(())
    }

    /// Save model weights to a file
    pub fn save_model(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&(self.feature_size as i32).to_le_bytes())?;
        writer.write_all(&(self.hidden_size as i32).to_le_bytes())?;
        writer.write_all(&(self.output_size as i32).to_le_bytes())?;
        writer.write_all(&(self.num_message_passing_layers as i32).to_le_bytes())?;
        writer.write_all(&self.learning_rate.to_le_bytes())?;

        let save_layer = |writer: &mut BufWriter<File>, layer: &OpenCLGpuLayer, queue: &CommandQueue| -> Result<(), Box<dyn std::error::Error>> {
            let (weights, biases) = layer.copy_weights_to_host(queue)?;
            writer.write_all(&(layer.num_inputs as i32).to_le_bytes())?;
            writer.write_all(&(layer.num_outputs as i32).to_le_bytes())?;
            for w in &weights {
                writer.write_all(&w.to_le_bytes())?;
            }
            for b in &biases {
                writer.write_all(&b.to_le_bytes())?;
            }
            Ok(())
        };

        for layer in &self.message_layers {
            save_layer(&mut writer, layer, &self.queue)?;
        }
        for layer in &self.update_layers {
            save_layer(&mut writer, layer, &self.queue)?;
        }
        save_layer(&mut writer, &self.readout_layer, &self.queue)?;
        save_layer(&mut writer, &self.output_layer, &self.queue)?;

        Ok(())
    }

    /// Load model weights from a file
    pub fn load_model(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        let mut buf_i32 = [0u8; 4];
        let mut buf_f32 = [0u8; 4];

        reader.read_exact(&mut buf_i32)?;
        self.feature_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        self.hidden_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        self.output_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        self.num_message_passing_layers = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_f32)?;
        self.learning_rate = f32::from_le_bytes(buf_f32);

        let load_layer = |reader: &mut BufReader<File>, layer: &mut OpenCLGpuLayer, queue: &CommandQueue| -> Result<(), Box<dyn std::error::Error>> {
            let mut buf_i32 = [0u8; 4];
            let mut buf_f32 = [0u8; 4];

            reader.read_exact(&mut buf_i32)?;
            let num_inputs = i32::from_le_bytes(buf_i32) as usize;
            reader.read_exact(&mut buf_i32)?;
            let num_outputs = i32::from_le_bytes(buf_i32) as usize;

            let mut weights = vec![0.0f32; num_inputs * num_outputs];
            let mut biases = vec![0.0f32; num_outputs];

            for w in &mut weights {
                reader.read_exact(&mut buf_f32)?;
                *w = f32::from_le_bytes(buf_f32);
            }
            for b in &mut biases {
                reader.read_exact(&mut buf_f32)?;
                *b = f32::from_le_bytes(buf_f32);
            }

            layer.copy_weights_from_host(queue, &weights, &biases)?;
            Ok(())
        };

        for layer in &mut self.message_layers {
            load_layer(&mut reader, layer, &self.queue)?;
        }
        for layer in &mut self.update_layers {
            load_layer(&mut reader, layer, &self.queue)?;
        }
        load_layer(&mut reader, &mut self.readout_layer, &self.queue)?;
        load_layer(&mut reader, &mut self.output_layer, &self.queue)?;

        Ok(())
    }

    pub fn get_learning_rate(&self) -> f32 { self.learning_rate }
    pub fn set_learning_rate(&mut self, lr: f32) { self.learning_rate = lr; }
    pub fn get_feature_size(&self) -> usize { self.feature_size }
    pub fn get_hidden_size(&self) -> usize { self.hidden_size }
    pub fn get_output_size(&self) -> usize { self.output_size }
    pub fn get_num_message_passing_layers(&self) -> usize { self.num_message_passing_layers }
    pub fn get_graph_embedding(&self) -> &[f32] { &self.h_graph_embedding }

    pub fn get_architecture_summary(&self) -> String {
        let mut param_count = 0usize;
        for layer in &self.message_layers {
            param_count += layer.num_inputs * layer.num_outputs + layer.num_outputs;
        }
        for layer in &self.update_layers {
            param_count += layer.num_inputs * layer.num_outputs + layer.num_outputs;
        }
        param_count += self.readout_layer.num_inputs * self.readout_layer.num_outputs + self.readout_layer.num_outputs;
        param_count += self.output_layer.num_inputs * self.output_layer.num_outputs + self.output_layer.num_outputs;

        format!(
            "=== OpenCL GNN Architecture Summary ===\n\
             Feature Size: {}\n\
             Hidden Size: {}\n\
             Output Size: {}\n\
             Message Passing Layers: {}\n\
             Learning Rate: {}\n\
             Total Parameters: {}",
            self.feature_size,
            self.hidden_size,
            self.output_size,
            self.num_message_passing_layers,
            self.learning_rate,
            param_count
        )
    }
}

/// Check if an OpenCL device is available
pub fn is_opencl_available() -> bool {
    get_all_devices(CL_DEVICE_TYPE_GPU)
        .map(|d| !d.is_empty())
        .unwrap_or(false)
        || get_all_devices(CL_DEVICE_TYPE_ALL)
            .map(|d| !d.is_empty())
            .unwrap_or(false)
}

