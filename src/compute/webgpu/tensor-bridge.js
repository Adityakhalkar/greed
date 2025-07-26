/**
 * TensorBridge - Bridge between JavaScript WebGPU tensors and Python PyTorch tensors
 * Enables seamless interoperability between WebGPU acceleration and PyTorch API
 */

import { WebGPUTensor } from './webgpu-tensor.js';

export class TensorBridge {
  constructor(computeEngine) {
    this.computeEngine = computeEngine;
    this.tensorRegistry = new Map(); // Python tensor ID -> JS WebGPU tensor
    this.nextTensorId = 0;
  }

  /**
   * Create JavaScript WebGPU tensor from Python tensor data
   */
  createWebGPUTensor(data, shape, dtype = 'float32', device = 'webgpu') {
    const tensor = new WebGPUTensor(data, {
      shape: shape,
      dtype: dtype,
      device: device,
      computeEngine: this.computeEngine
    });
    
    const tensorId = `webgpu_tensor_${this.nextTensorId++}`;
    this.tensorRegistry.set(tensorId, tensor);
    
    return {
      id: tensorId,
      tensor: tensor,
      shape: tensor.shape,
      dtype: tensor.dtype,
      device: tensor.device
    };
  }

  /**
   * Get WebGPU tensor by ID
   */
  getTensor(tensorId) {
    return this.tensorRegistry.get(tensorId);
  }

  /**
   * Execute WebGPU operation on tensor
   */
  async executeOperation(tensorId, operation, otherTensorId = null, options = {}) {
    const tensor = this.tensorRegistry.get(tensorId);
    if (!tensor) {
      throw new Error(`Tensor ${tensorId} not found`);
    }

    let otherTensor = null;
    if (otherTensorId) {
      otherTensor = this.tensorRegistry.get(otherTensorId);
      if (!otherTensor) {
        throw new Error(`Tensor ${otherTensorId} not found`);
      }
    }

    try {
      let result;
      switch (operation) {
        case 'add':
          result = await tensor.add(otherTensor);
          break;
        case 'sub':
          result = await tensor.sub(otherTensor);
          break;
        case 'mul':
          result = await tensor.mul(otherTensor);
          break;
        case 'div':
          result = await tensor.div(otherTensor);
          break;
        case 'matmul':
          result = await tensor.matmul(otherTensor);
          break;
        case 'relu':
          result = await tensor.relu();
          break;
        case 'sigmoid':
          result = await tensor.sigmoid();
          break;
        case 'tanh':
          result = await tensor.tanh();
          break;
        case 'softmax':
          result = await tensor.softmax(options.dim);
          break;
        case 'sum':
          result = await tensor.sum(options.dim, options.keepdim);
          break;
        case 'mean':
          result = await tensor.mean(options.dim, options.keepdim);
          break;
        case 'transpose':
          result = await tensor.transpose(options.dim0, options.dim1);
          break;
        default:
          throw new Error(`Unsupported operation: ${operation}`);
      }

      // Register result tensor
      const resultInfo = this.createWebGPUTensor(
        result.data, 
        result.shape, 
        result.dtype, 
        result.device
      );

      return {
        success: true,
        result: resultInfo,
        data: Array.from(result.data),
        shape: result.shape,
        dtype: result.dtype
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Convert WebGPU tensor back to Python-compatible data
   */
  tensorToArray(tensorId) {
    const tensor = this.tensorRegistry.get(tensorId);
    if (!tensor) {
      throw new Error(`Tensor ${tensorId} not found`);
    }

    return {
      data: Array.from(tensor.data),
      shape: tensor.shape,
      dtype: tensor.dtype
    };
  }

  /**
   * Release tensor from registry
   */
  releaseTensor(tensorId) {
    return this.tensorRegistry.delete(tensorId);
  }

  /**
   * Get registry statistics
   */
  getStats() {
    return {
      tensorCount: this.tensorRegistry.size,
      totalMemory: Array.from(this.tensorRegistry.values()).reduce(
        (sum, tensor) => sum + tensor.size * 4, 0
      ),
      deviceDistribution: this._getDeviceDistribution()
    };
  }

  /**
   * Cleanup all tensors
   */
  cleanup() {
    this.tensorRegistry.clear();
    this.nextTensorId = 0;
  }

  // Private methods
  _getDeviceDistribution() {
    const distribution = {};
    for (const tensor of this.tensorRegistry.values()) {
      distribution[tensor.device] = (distribution[tensor.device] || 0) + 1;
    }
    return distribution;
  }
}

/**
 * Global tensor bridge instance
 */
let globalTensorBridge = null;

export function createTensorBridge(computeEngine) {
  globalTensorBridge = new TensorBridge(computeEngine);
  
  // Expose bridge to global scope for Python integration
  if (typeof window !== 'undefined') {
    window.greedTensorBridge = globalTensorBridge;
  } else if (typeof global !== 'undefined') {
    global.greedTensorBridge = globalTensorBridge;
  }
  
  return globalTensorBridge;
}

export function getTensorBridge() {
  return globalTensorBridge;
}

export default TensorBridge;