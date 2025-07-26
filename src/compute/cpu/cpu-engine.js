/**
 * CPU Engine - NumPy-based CPU computation for tensor operations
 * Provides fallback execution when WebGPU is unavailable
 */
import EventEmitter from '../../core/event-emitter.js';

class CPUEngine extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      enableOptimizations: config.enableOptimizations !== false,
      maxConcurrentOps: config.maxConcurrentOps || 2,
      enableProfiling: config.enableProfiling !== false,
      chunkSize: config.chunkSize || 10000,
      ...config
    };

    // Engine state
    this.isInitialized = false;
    this.runtime = null;
    this.numpy = null;
    
    // Performance tracking
    this.stats = {
      operations: 0,
      totalExecutionTime: 0,
      averageExecutionTime: 0,
      lastOperationTime: 0,
      supportedOperations: new Set([
        'add', 'subtract', 'multiply', 'divide', 'matmul',
        'transpose', 'reshape', 'sum', 'mean', 'max', 'min',
        'relu', 'sigmoid', 'tanh', 'softmax', 'exp', 'log',
        'sqrt', 'power', 'abs', 'sign', 'clip'
      ])
    };
    
    // Operation implementations
    this.operations = this._initializeOperations();
  }

  /**
   * Initialize CPU engine with runtime reference
   */
  async initialize(runtime = null) {
    if (this.isInitialized) {
      return true;
    }

    try {
      this.emit('init:start');
      
      if (runtime) {
        this.runtime = runtime;
        this.numpy = runtime.getGlobal('np');
      }
      
      // Install CPU-optimized operations in Python runtime
      if (this.runtime) {
        await this._installCPUOperations();
      }
      
      this.isInitialized = true;
      this.emit('init:complete', { 
        supportedOperations: Array.from(this.stats.supportedOperations)
      });
      
      return true;
    } catch (error) {
      this.emit('init:error', { error });
      throw error;
    }
  }

  /**
   * Execute tensor operation on CPU
   */
  async execute(operation, tensors, options = {}) {
    if (!this.isInitialized) {
      throw new Error('CPU engine not initialized');
    }

    const startTime = performance.now();
    this.emit('compute:start', { operation, options });

    try {
      // Validate operation support
      if (!this.stats.supportedOperations.has(operation)) {
        throw new Error(`Unsupported CPU operation: ${operation}`);
      }

      // Get operation implementation
      const opImpl = this.operations[operation];
      if (!opImpl) {
        throw new Error(`Operation implementation not found: ${operation}`);
      }

      // Execute operation
      const result = await opImpl(tensors, options);
      
      // Update statistics
      const executionTime = performance.now() - startTime;
      this._updateStats(executionTime);
      
      this.emit('compute:complete', { 
        operation, 
        executionTime,
        resultSize: this._getResultSize(result)
      });

      return result;
    } catch (error) {
      const executionTime = performance.now() - startTime;
      this.emit('compute:error', { operation, error, executionTime });
      throw error;
    }
  }

  /**
   * Execute batch of operations
   */
  async executeBatch(operations, options = {}) {
    const { sequential = false } = options;
    
    if (sequential) {
      const results = [];
      for (const op of operations) {
        const result = await this.execute(op.operation, op.tensors, op.options);
        results.push(result);
      }
      return results;
    } else {
      // Limited parallelism for CPU operations
      const semaphore = new Semaphore(this.config.maxConcurrentOps);
      const promises = operations.map(async (op) => {
        await semaphore.acquire();
        try {
          return await this.execute(op.operation, op.tensors, op.options);
        } finally {
          semaphore.release();
        }
      });
      return Promise.all(promises);
    }
  }

  /**
   * Get engine statistics
   */
  getStats() {
    return {
      ...this.stats,
      isInitialized: this.isInitialized,
      config: this.config,
      type: 'cpu'
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    try {
      this.emit('cleanup:start');
      
      this.isInitialized = false;
      this.runtime = null;
      this.numpy = null;
      
      this.emit('cleanup:complete');
    } catch (error) {
      this.emit('cleanup:error', { error });
      throw error;
    }
  }

  // Private methods
  async _installCPUOperations() {
    const cpuOpsCode = `
import numpy as np
from typing import Union, List, Tuple, Optional

class CPUTensorOps:
    """CPU-optimized tensor operations using NumPy"""
    
    @staticmethod
    def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.add(a, b)
    
    @staticmethod
    def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.subtract(a, b)
    
    @staticmethod
    def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.multiply(a, b)
    
    @staticmethod
    def divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.divide(a, b)
    
    @staticmethod
    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)
    
    @staticmethod
    def transpose(a: np.ndarray, axes: Optional[Tuple] = None) -> np.ndarray:
        return np.transpose(a, axes)
    
    @staticmethod
    def reshape(a: np.ndarray, shape: Tuple) -> np.ndarray:
        return np.reshape(a, shape)
    
    @staticmethod
    def sum(a: np.ndarray, axis: Optional[Union[int, Tuple]] = None, keepdims: bool = False) -> np.ndarray:
        return np.sum(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def mean(a: np.ndarray, axis: Optional[Union[int, Tuple]] = None, keepdims: bool = False) -> np.ndarray:
        return np.mean(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def max(a: np.ndarray, axis: Optional[Union[int, Tuple]] = None, keepdims: bool = False) -> np.ndarray:
        return np.max(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def min(a: np.ndarray, axis: Optional[Union[int, Tuple]] = None, keepdims: bool = False) -> np.ndarray:
        return np.min(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def relu(a: np.ndarray) -> np.ndarray:
        return np.maximum(a, 0)
    
    @staticmethod
    def sigmoid(a: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(a, -250, 250)))  # Prevent overflow
    
    @staticmethod
    def tanh(a: np.ndarray) -> np.ndarray:
        return np.tanh(a)
    
    @staticmethod
    def softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
        # Stable softmax implementation
        x_max = np.max(a, axis=axis, keepdims=True)
        exp_x = np.exp(a - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @staticmethod
    def exp(a: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(a, -250, 250))  # Prevent overflow
    
    @staticmethod
    def log(a: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(a, 1e-12))  # Prevent log(0)
    
    @staticmethod
    def sqrt(a: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(a, 0))  # Prevent sqrt of negative
    
    @staticmethod
    def power(a: np.ndarray, b: Union[np.ndarray, float]) -> np.ndarray:
        return np.power(a, b)
    
    @staticmethod
    def abs(a: np.ndarray) -> np.ndarray:
        return np.abs(a)
    
    @staticmethod
    def sign(a: np.ndarray) -> np.ndarray:
        return np.sign(a)
    
    @staticmethod
    def clip(a: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        return np.clip(a, min_val, max_val)

# Install globally
cpu_ops = CPUTensorOps()
`;

    await this.runtime.runPython(cpuOpsCode, { captureOutput: false });
    this.emit('operations:installed', { count: this.stats.supportedOperations.size });
  }

  _initializeOperations() {
    return {
      // Basic arithmetic
      add: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.add', tensors, options);
      },
      
      subtract: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.subtract', tensors, options);
      },
      
      multiply: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.multiply', tensors, options);
      },
      
      divide: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.divide', tensors, options);
      },
      
      // Linear algebra
      matmul: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.matmul', tensors, options);
      },
      
      transpose: async (tensors, options) => {
        const axes = options.axes ? `axes=${JSON.stringify(options.axes)}` : '';
        return this._executePythonOp('cpu_ops.transpose', tensors, { axes });
      },
      
      reshape: async (tensors, options) => {
        if (!options.shape) {
          throw new Error('Reshape operation requires shape parameter');
        }
        return this._executePythonOp('cpu_ops.reshape', tensors, options);
      },
      
      // Reductions
      sum: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.sum', tensors, options);
      },
      
      mean: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.mean', tensors, options);
      },
      
      max: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.max', tensors, options);
      },
      
      min: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.min', tensors, options);
      },
      
      // Activation functions
      relu: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.relu', tensors, options);
      },
      
      sigmoid: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.sigmoid', tensors, options);
      },
      
      tanh: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.tanh', tensors, options);
      },
      
      softmax: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.softmax', tensors, options);
      },
      
      // Mathematical functions
      exp: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.exp', tensors, options);
      },
      
      log: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.log', tensors, options);
      },
      
      sqrt: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.sqrt', tensors, options);
      },
      
      power: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.power', tensors, options);
      },
      
      abs: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.abs', tensors, options);
      },
      
      sign: async (tensors, options) => {
        return this._executePythonOp('cpu_ops.sign', tensors, options);
      },
      
      clip: async (tensors, options) => {
        if (options.min === undefined || options.max === undefined) {
          throw new Error('Clip operation requires min and max parameters');
        }
        return this._executePythonOp('cpu_ops.clip', tensors, options);
      }
    };
  }

  async _executePythonOp(pythonFunc, tensors, options) {
    if (!this.runtime) {
      throw new Error('Runtime not available for CPU operations');
    }

    try {
      // Convert tensors to Python numpy arrays
      const tensorVars = [];
      const tensorArray = Array.isArray(tensors) ? tensors : [tensors];
      
      for (let i = 0; i < tensorArray.length; i++) {
        const varName = `tensor_${i}`;
        this.runtime.setGlobal(varName, tensorArray[i]);
        tensorVars.push(varName);
      }
      
      // Build function call with parameters
      let funcCall = `${pythonFunc}(${tensorVars.join(', ')})`;
      
      // Add options as keyword arguments
      if (options) {
        const kwargs = [];
        for (const [key, value] of Object.entries(options)) {
          if (key === 'axes') {
            kwargs.push(`${key}=${value}`);
          } else if (typeof value === 'string') {
            kwargs.push(`${key}="${value}"`);
          } else if (Array.isArray(value)) {
            kwargs.push(`${key}=${JSON.stringify(value)}`);
          } else {
            kwargs.push(`${key}=${value}`);
          }
        }
        
        if (kwargs.length > 0) {
          funcCall = `${pythonFunc}(${tensorVars.join(', ')}, ${kwargs.join(', ')})`;
        }
      }
      
      // Execute operation and get result
      const resultCode = `
result = ${funcCall}
result.tolist() if hasattr(result, 'tolist') else result
`;
      
      const result = await this.runtime.runPython(resultCode, {
        captureOutput: true,
        timeout: options.timeout || 10000
      });
      
      // Cleanup tensor variables
      for (const varName of tensorVars) {
        await this.runtime.runPython(`del ${varName}`, { captureOutput: false });
      }
      
      return result;
    } catch (error) {
      throw new Error(`CPU operation failed: ${error.message}`);
    }
  }

  _updateStats(executionTime) {
    this.stats.operations++;
    this.stats.totalExecutionTime += executionTime;
    this.stats.averageExecutionTime = this.stats.totalExecutionTime / this.stats.operations;
    this.stats.lastOperationTime = executionTime;
  }

  _getResultSize(result) {
    if (Array.isArray(result)) {
      return result.length;
    } else if (ArrayBuffer.isView(result)) {
      return result.length;
    } else if (result instanceof ArrayBuffer) {
      return result.byteLength / 4; // Assume 32-bit elements
    }
    return 1; // Scalar result
  }
}

// Simple semaphore for concurrency control
class Semaphore {
  constructor(max) {
    this.max = max;
    this.current = 0;
    this.queue = [];
  }

  async acquire() {
    return new Promise((resolve) => {
      if (this.current < this.max) {
        this.current++;
        resolve();
      } else {
        this.queue.push(resolve);
      }
    });
  }

  release() {
    this.current--;
    if (this.queue.length > 0) {
      const resolve = this.queue.shift();
      this.current++;
      resolve();
    }
  }
}

export default CPUEngine;