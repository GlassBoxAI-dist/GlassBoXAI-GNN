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
 * // Create a new GNN
 * const gnn = new GnnFacade(3, 16, 2, 2);
 *
 * // Create a graph
 * gnn.createEmptyGraph(5, 3);
 *
 * // Add edges
 * gnn.addEdge(0, 1);
 * gnn.addEdge(1, 2);
 * gnn.addEdge(2, 3);
 *
 * // Set node features
 * gnn.setNodeFeatures(0, [1.0, 0.5, 0.2]);
 * gnn.setNodeFeatures(1, [0.8, 0.3, 0.1]);
 *
 * // Make predictions
 * const prediction = gnn.predict();
 * console.log('Prediction:', prediction);
 *
 * // Train the model
 * const target = [0.5, 0.5];
 * const loss = gnn.train(target);
 * console.log('Loss:', loss);
 *
 * // Save and load models
 * gnn.saveModel('model.bin');
 * const gnn2 = GnnFacade.fromModelFile('model.bin');
 * ```
 */

const { platform, arch } = process;

let nativeBinding = null;
let loadError = null;

// Try to load the native binding
function loadNativeBinding() {
  const platformArch = `${platform}-${arch}`;

  // Platform-specific binding names
  const bindingNames = {
    'linux-x64': 'gnn-facade-cuda.linux-x64-gnu.node',
    'linux-arm64': 'gnn-facade-cuda.linux-arm64-gnu.node',
    'darwin-x64': 'gnn-facade-cuda.darwin-x64.node',
    'darwin-arm64': 'gnn-facade-cuda.darwin-arm64.node',
    'win32-x64': 'gnn-facade-cuda.win32-x64-msvc.node',
  };

  const bindingName = bindingNames[platformArch];

  if (!bindingName) {
    throw new Error(
      `Unsupported platform: ${platformArch}. ` +
      `Supported platforms: ${Object.keys(bindingNames).join(', ')}`
    );
  }

  try {
    // Try loading from the same directory
    nativeBinding = require(`./${bindingName}`);
  } catch (e) {
    try {
      // Try loading without the platform suffix (for development)
      nativeBinding = require('./gnn-facade-cuda.node');
    } catch (e2) {
      loadError = new Error(
        `Failed to load native binding for ${platformArch}. ` +
        `Make sure the package is built for your platform.\n` +
        `Original error: ${e.message}`
      );
    }
  }
}

loadNativeBinding();

if (loadError) {
  throw loadError;
}

module.exports = nativeBinding;
