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
   * Execute WebGPU tensor creation operation
   */
  async executeCreationOperation(operation, options = {}) {
    try {
      const { shape, size, device = 'webgpu', dtype = 'float32' } = options;
      
      if (!size && !shape) {
        throw new Error('Either size or shape must be provided for tensor creation');
      }
      
      const tensorSize = size || shape.reduce((a, b) => a * b, 1);
      const tensorShape = shape || [size];
      
      // Execute WebGPU creation operation
      const result = await this.computeEngine.execute(operation, [], {
        outputSize: tensorSize,
        dataType: dtype === 'int64' ? 'i32' : 'f32',
        ...options
      });
      
      // Create WebGPU tensor with result
      const tensor = new WebGPUTensor(result, {
        shape: tensorShape,
        dtype: dtype,
        device: device,
        computeEngine: this.computeEngine
      });
      
      const tensorId = `webgpu_tensor_${this.nextTensorId++}`;
      this.tensorRegistry.set(tensorId, tensor);
      
      // Return just the tensor ID for Python integration
      return tensorId;
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Execute WebGPU binary operation between two tensors
   */
  async executeBinaryOperation(tensor1Id, tensor2Id, operation, options = {}) {
    const tensor1 = this.tensorRegistry.get(tensor1Id);
    const tensor2 = this.tensorRegistry.get(tensor2Id);
    
    if (!tensor1 || !tensor2) {
      throw new Error(`Tensor not found: ${!tensor1 ? tensor1Id : tensor2Id}`);
    }

    try {
      // Execute WebGPU binary operation
      const result = await this.computeEngine.execute(operation, [tensor1, tensor2], options);
      
      // Create result tensor
      const resultTensor = new WebGPUTensor(result, {
        shape: tensor1.shape, // Simplified - actual shape calculation needed
        dtype: tensor1.dtype,
        device: tensor1.device,
        computeEngine: this.computeEngine
      });
      
      const tensorId = `webgpu_tensor_${this.nextTensorId++}`;
      this.tensorRegistry.set(tensorId, resultTensor);
      
      return {
        success: true,
        id: tensorId,
        data: Array.from(result),
        shape: resultTensor.shape,
        dtype: resultTensor.dtype
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Execute WebGPU unary operation on single tensor
   */
  async executeUnaryOperation(tensorId, operation, options = {}) {
    const tensor = this.tensorRegistry.get(tensorId);
    if (!tensor) {
      throw new Error(`Tensor ${tensorId} not found`);
    }

    try {
      // Execute WebGPU unary operation
      const result = await this.computeEngine.execute(operation, [tensor], options);
      
      // Create result tensor
      const resultTensor = new WebGPUTensor(result, {
        shape: tensor.shape, // Simplified - actual shape calculation needed
        dtype: tensor.dtype,
        device: tensor.device,
        computeEngine: this.computeEngine
      });
      
      const tensorId = `webgpu_tensor_${this.nextTensorId++}`;
      this.tensorRegistry.set(tensorId, resultTensor);
      
      return {
        success: true,
        id: tensorId,
        data: Array.from(result),
        shape: resultTensor.shape,
        dtype: resultTensor.dtype
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Execute WebGPU operation on tensor (Python-compatible interface)
   */
  async executeOperation(operation, tensorId, otherTensorId = null, options = {}) {
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
        case 'squeeze':
          result = await tensor.squeeze(options.dim);
          break;
        case 'unsqueeze':
          result = await tensor.unsqueeze(options.dim);
          break;
        case 'reshape':
          result = await tensor.reshape(options.shape);
          break;
        case 'std':
          result = await tensor.std(options.dim, options.keepdim, options.unbiased);
          break;
        case 'var':
          result = await tensor.var(options.dim, options.keepdim, options.unbiased);
          break;
        case 'linear':
          // Neural network linear layer operation
          const biasId = options.bias_id;
          const bias = biasId ? this.tensorRegistry.get(biasId) : null;
          result = await this.computeEngine.executeLinear(tensor, otherTensor, bias, options);
          break;
        case 'cross_entropy':
          // Cross entropy loss computation
          result = await this.computeEngine.executeCrossEntropy(tensor, otherTensor, options);
          break;
        default:
          throw new Error(`Unsupported operation: ${operation}`);
      }

      // Register result tensor and return just the ID for Python
      const resultInfo = this.createWebGPUTensor(
        result.data, 
        result.shape, 
        result.dtype, 
        result.device
      );

      // Return just the tensor ID for Python WebGPU integration
      return resultInfo.id;
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get tensor data for Python integration
   */
  getTensorData(tensorId) {
    const tensor = this.tensorRegistry.get(tensorId);
    if (!tensor) {
      throw new Error(`Tensor ${tensorId} not found`);
    }

    return Array.from(tensor.data);
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
   * Execute backward pass for autograd (placeholder)
   */
  executeBackward(tensorId, gradientId) {
    // Placeholder for autograd backward pass
    // In full implementation, this would compute gradients
    console.log(`Executing backward pass for tensor ${tensorId} with gradient ${gradientId}`);
    return true;
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