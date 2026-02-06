/**
 * GlassBoxAI GNN - CUDA-accelerated Graph Neural Network
 *
 * This module provides Node.js bindings for a high-performance
 * Graph Neural Network implementation with CUDA acceleration.
 *
 * @example
 * ```javascript
 * const { GnnFacade } = require('gnn-facade-cuda');
 *
 * const gnn = new GnnFacade(3, 16, 2, 2);
 * gnn.createEmptyGraph(5, 3);
 * gnn.addEdge(0, 1);
 * gnn.setNodeFeatures(0, [1.0, 0.5, 0.2]);
 * const prediction = gnn.predict();
 * ```
 */

/** Gradient flow information for a layer */
export interface GradientFlowInfo {
  layerIdx: number;
  meanGradient: number;
  maxGradient: number;
  minGradient: number;
  gradientNorm: number;
}

/** Model header information */
export interface ModelHeader {
  featureSize: number;
  hiddenSize: number;
  outputSize: number;
  mpLayers: number;
  learningRate: number;
}

/** Edge endpoints */
export interface EdgeEndpoints {
  source: number;
  target: number;
}

/**
 * CUDA-accelerated Graph Neural Network with Facade interface
 *
 * This class provides a high-level interface for creating, training,
 * and using Graph Neural Networks with CUDA acceleration.
 */
export class GnnFacade {
  /**
   * Create a new GNN Facade
   *
   * @param featureSize - Size of input node features
   * @param hiddenSize - Size of hidden layers
   * @param outputSize - Size of output predictions
   * @param numMpLayers - Number of message passing layers
   */
  constructor(
    featureSize: number,
    hiddenSize: number,
    outputSize: number,
    numMpLayers: number
  );

  /**
   * Load a GNN from a saved model file
   *
   * @param filename - Path to the model file
   * @returns A new GnnFacade loaded from the file
   */
  static fromModelFile(filename: string): GnnFacade;

  /**
   * Read model header without loading full model
   *
   * @param filename - Path to the model file
   * @returns Object with model dimensions and learning rate
   */
  static readModelHeader(filename: string): ModelHeader;

  /**
   * Create an empty graph with specified number of nodes
   *
   * @param numNodes - Number of nodes in the graph
   * @param featureSize - Size of node feature vectors
   */
  createEmptyGraph(numNodes: number, featureSize: number): void;

  /**
   * Get a single feature value for a node
   *
   * @param nodeIdx - Index of the node
   * @param featureIdx - Index of the feature
   * @returns The feature value
   */
  getNodeFeature(nodeIdx: number, featureIdx: number): number;

  /**
   * Set a single feature value for a node
   *
   * @param nodeIdx - Index of the node
   * @param featureIdx - Index of the feature
   * @param value - The value to set
   */
  setNodeFeature(nodeIdx: number, featureIdx: number, value: number): void;

  /**
   * Set all features for a node
   *
   * @param nodeIdx - Index of the node
   * @param features - Array of feature values
   */
  setNodeFeatures(nodeIdx: number, features: number[]): void;

  /**
   * Get all features for a node
   *
   * @param nodeIdx - Index of the node
   * @returns Array of feature values, or null if node doesn't exist
   */
  getNodeFeatures(nodeIdx: number): number[] | null;

  /**
   * Add an edge to the graph
   *
   * @param source - Source node index
   * @param target - Target node index
   * @param features - Optional array of edge features
   * @returns Index of the new edge
   */
  addEdge(source: number, target: number, features?: number[]): number;

  /**
   * Remove an edge by index
   *
   * @param edgeIdx - Index of the edge to remove
   */
  removeEdge(edgeIdx: number): void;

  /**
   * Get endpoints of an edge
   *
   * @param edgeIdx - Index of the edge
   * @returns Object with source and target, or null if edge doesn't exist
   */
  getEdgeEndpoints(edgeIdx: number): EdgeEndpoints | null;

  /**
   * Check if an edge exists between two nodes
   *
   * @param source - Source node index
   * @param target - Target node index
   * @returns true if edge exists
   */
  hasEdge(source: number, target: number): boolean;

  /**
   * Find the index of an edge between two nodes
   *
   * @param source - Source node index
   * @param target - Target node index
   * @returns Edge index, or null if not found
   */
  findEdgeIndex(source: number, target: number): number | null;

  /**
   * Get neighbors of a node
   *
   * @param nodeIdx - Index of the node
   * @returns Array of neighbor indices, or null if node doesn't exist
   */
  getNeighbors(nodeIdx: number): number[] | null;

  /**
   * Get in-degree of a node
   *
   * @param nodeIdx - Index of the node
   * @returns Number of incoming edges
   */
  getInDegree(nodeIdx: number): number;

  /**
   * Get out-degree of a node
   *
   * @param nodeIdx - Index of the node
   * @returns Number of outgoing edges
   */
  getOutDegree(nodeIdx: number): number;

  /**
   * Get features for an edge
   *
   * @param edgeIdx - Index of the edge
   * @returns Array of feature values, or null if edge doesn't exist
   */
  getEdgeFeatures(edgeIdx: number): number[] | null;

  /**
   * Set features for an edge
   *
   * @param edgeIdx - Index of the edge
   * @param features - Array of feature values
   */
  setEdgeFeatures(edgeIdx: number, features: number[]): void;

  /**
   * Rebuild the adjacency list from edges
   */
  rebuildAdjacencyList(): void;

  /**
   * Run prediction on the current graph
   *
   * @returns Array of prediction values
   */
  predict(): number[];

  /**
   * Train on the current graph with target values
   *
   * @param target - Array of target values
   * @returns Loss value
   */
  train(target: number[]): number;

  /**
   * Train for multiple iterations
   *
   * @param target - Array of target values
   * @param iterations - Number of training iterations
   */
  trainMultiple(target: number[], iterations: number): void;

  /**
   * Save model to file
   *
   * @param filename - Path to save the model
   */
  saveModel(filename: string): void;

  /**
   * Load model from file
   *
   * @param filename - Path to the model file
   */
  loadModel(filename: string): void;

  /**
   * Set learning rate
   *
   * @param lr - Learning rate value
   */
  setLearningRate(lr: number): void;

  /**
   * Get learning rate
   *
   * @returns Current learning rate
   */
  getLearningRate(): number;

  /**
   * Get architecture summary
   *
   * @returns Summary of the network architecture
   */
  getArchitectureSummary(): string;

  /**
   * Get number of nodes in the graph
   *
   * @returns Number of nodes
   */
  getNumNodes(): number;

  /**
   * Get number of edges in the graph
   *
   * @returns Number of edges
   */
  getNumEdges(): number;

  /**
   * Check if a graph is loaded
   *
   * @returns true if a graph is loaded
   */
  isGraphLoaded(): boolean;

  /**
   * Get graph embedding from last forward pass
   *
   * @returns Array of embedding values
   */
  getGraphEmbedding(): number[];

  /**
   * Get feature size
   *
   * @returns Size of node features
   */
  getFeatureSize(): number;

  /**
   * Get hidden size
   *
   * @returns Size of hidden layers
   */
  getHiddenSize(): number;

  /**
   * Get output size
   *
   * @returns Size of output
   */
  getOutputSize(): number;

  /**
   * Get number of message passing layers
   *
   * @returns Number of message passing layers
   */
  getNumMessagePassingLayers(): number;

  /**
   * Get node mask value
   *
   * @param nodeIdx - Index of the node
   * @returns Mask value (true = active)
   */
  getNodeMask(nodeIdx: number): boolean;

  /**
   * Set node mask value
   *
   * @param nodeIdx - Index of the node
   * @param value - Mask value (true = active)
   */
  setNodeMask(nodeIdx: number, value: boolean): void;

  /**
   * Get edge mask value
   *
   * @param edgeIdx - Index of the edge
   * @returns Mask value (true = active)
   */
  getEdgeMask(edgeIdx: number): boolean;

  /**
   * Set edge mask value
   *
   * @param edgeIdx - Index of the edge
   * @param value - Mask value (true = active)
   */
  setEdgeMask(edgeIdx: number, value: boolean): void;

  /**
   * Apply random dropout to nodes
   *
   * @param rate - Dropout rate (0.0 to 1.0)
   */
  applyNodeDropout(rate: number): void;

  /**
   * Apply random dropout to edges
   *
   * @param rate - Dropout rate (0.0 to 1.0)
   */
  applyEdgeDropout(rate: number): void;

  /**
   * Get count of active (masked) nodes
   *
   * @returns Number of active nodes
   */
  getMaskedNodeCount(): number;

  /**
   * Get count of active (masked) edges
   *
   * @returns Number of active edges
   */
  getMaskedEdgeCount(): number;

  /**
   * Compute PageRank scores for all nodes
   *
   * @param damping - Damping factor (default: 0.85)
   * @param iterations - Number of iterations (default: 20)
   * @returns Array of PageRank scores
   */
  computePageRank(damping?: number, iterations?: number): number[];

  /**
   * Get gradient flow information for a layer
   *
   * @param layerIdx - Index of the layer
   * @returns Gradient flow statistics
   */
  getGradientFlow(layerIdx: number): GradientFlowInfo;

  /**
   * Get total parameter count
   *
   * @returns Total number of trainable parameters
   */
  getParameterCount(): number;

  /**
   * Export graph structure to JSON
   *
   * @returns JSON representation of the graph
   */
  exportGraphToJson(): string;
}
