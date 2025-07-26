/**
 * WebGPU Compute Engine - High-performance tensor operations with GPU acceleration
 * Refactored from monolithic implementation for better modularity and performance
 */
import EventEmitter from '../../core/event-emitter.js';
import BufferManager from './buffer-manager.js';
import PipelineCache from './pipeline-cache.js';
import logger from '../../utils/logger.js';

class WebGPUComputeEngine extends EventEmitter {
  constructor(options = {}) {
    super();
    this.config = {
      powerPreference: options.powerPreference || 'high-performance',
      enableProfiling: options.enableProfiling !== false,
      maxBufferSize: options.maxBufferSize || 256 * 1024 * 1024, // 256MB
      workgroupSize: options.workgroupSize || [64, 1, 1],
      enableValidation: options.enableValidation !== false,
      ...options
    };

    // Core WebGPU resources
    this.adapter = null;
    this.device = null;
    this.isInitialized = false;
    
    // Modular components
    this.bufferManager = null;
    this.pipelineCache = null;
    
    // Feature support
    this.supportedFeatures = new Set();
    this.limits = null;
    
    // Performance tracking
    this.stats = {
      computeOperations: 0,
      totalExecutionTime: 0,
      averageExecutionTime: 0,
      memoryUsage: 0,
      lastOperationTime: 0
    };
  }

  /**
   * Initialize WebGPU device and components
   */
  async initialize() {
    if (this.isInitialized) {
      return true;
    }

    try {
      this.emit('init:start');
      
      // Check WebGPU support
      if (!navigator.gpu) {
        throw new Error('WebGPU not supported in this browser');
      }

      // Request adapter
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: this.config.powerPreference
      });

      if (!this.adapter) {
        throw new Error('Failed to get WebGPU adapter');
      }

      // Get device features and limits
      this.supportedFeatures = this.adapter.features;
      this.limits = this.adapter.limits;
      
      this.emit('init:adapter', { 
        features: Array.from(this.supportedFeatures),
        limits: this.limits 
      });

      // Request device with required features
      const deviceDescriptor = {
        requiredFeatures: [],
        requiredLimits: {}
      };

      // Add optional features if supported
      if (this.supportedFeatures.has('timestamp-query')) {
        deviceDescriptor.requiredFeatures.push('timestamp-query');
      }

      this.device = await this.adapter.requestDevice(deviceDescriptor);
      
      // Set up comprehensive error handling
      this.device.addEventListener('uncapturederror', (event) => {
        const error = event.error;
        this.emit('device:error', { error, type: 'uncaptured', timestamp: Date.now() });
        logger.error('WebGPU uncaptured error:', {
          type: error.constructor.name,
          message: error.message,
          stack: error.stack
        });
        
        // Attempt recovery for recoverable errors
        this._handleDeviceError(error);
      });

      // Initialize modular components
      this.bufferManager = new BufferManager(this.device, {
        maxBufferSize: this.config.maxBufferSize,
        enablePooling: true,
        maxPoolSize: 100
      });

      this.pipelineCache = new PipelineCache(this.device, {
        maxCacheSize: 50,
        enableWarmup: true,
        shaderOptimization: 'balanced'
      });

      // Set up event forwarding
      this._setupEventForwarding();

      // Warmup common operations
      await this.pipelineCache.warmup();

      this.isInitialized = true;
      this.emit('init:complete', { 
        device: this.device,
        features: Array.from(this.supportedFeatures)
      });

      return true;
    } catch (error) {
      this.emit('init:error', { error, timestamp: Date.now() });
      logger.error('WebGPU initialization failed:', {
        type: error.constructor.name,
        message: error.message,
        stack: error.stack,
        config: this.config
      });
      
      // Set failure state for debugging
      this.isInitialized = false;
      this.initFailureReason = error.message;
      
      return false;
    }
  }

  /**
   * Execute tensor operation on GPU
   */
  async execute(operation, tensors, options = {}) {
    if (!this.isInitialized) {
      throw new Error('WebGPU compute engine not initialized');
    }

    const startTime = performance.now();
    this.emit('compute:start', { operation, options });

    try {
      // Validate inputs
      this._validateOperation(operation, tensors, options);

      // Get optimal workgroup size for this operation
      const tensorArray = Array.isArray(tensors) ? tensors : [tensors];
      const optimalWorkgroupSize = this.pipelineCache.getOptimalWorkgroupSize(
        operation, 
        tensorArray[0].shape || [tensorArray[0].length], 
        this.limits
      );

      // Get compute pipeline
      const pipeline = await this.pipelineCache.get(operation, {
        workgroupSize: options.workgroupSize || optimalWorkgroupSize,
        dataType: options.dataType || 'f32',
        inputCount: tensorArray.length,
        outputCount: options.outputCount || 1,
        ...options
      });

      // Prepare buffers
      const buffers = await this._prepareBuffers(tensors, operation, options);
      
      // Create bind group
      const bindGroup = this._createBindGroup(pipeline, buffers, options);
      
      // Execute compute pass
      const result = await this._executeComputePass(pipeline, bindGroup, buffers, options);
      
      // Update statistics
      const executionTime = performance.now() - startTime;
      this._updateStats(operation, executionTime, buffers);
      
      this.emit('compute:complete', { 
        operation, 
        executionTime, 
        resultSize: result.length 
      });

      return result;
    } catch (error) {
      const executionTime = performance.now() - startTime;
      
      // Enhanced error handling with context
      const errorContext = {
        operation,
        error: {
          type: error.constructor.name,
          message: error.message,
          stack: error.stack
        },
        executionTime,
        tensors: Array.isArray(tensors) ? tensors.length : 1,
        options,
        deviceStable: this.deviceStable ?? true,
        timestamp: Date.now()
      };
      
      this.emit('compute:error', errorContext);
      logger.error('WebGPU compute operation failed:', errorContext);
      
      // Attempt recovery for specific error types
      if (error.message.includes('out of memory') || error.constructor.name === 'GPUOutOfMemoryError') {
        logger.warn('GPU memory exhausted, attempting emergency cleanup');
        await this.bufferManager.emergencyCleanup();
        this.emit('recovery:memory', { operation, timestamp: Date.now() });
      }
      
      throw error;
    }
  }

  /**
   * Execute batch of operations efficiently
   */
  async executeBatch(operations, options = {}) {
    const { parallel = false, maxConcurrency = 4 } = options;
    
    if (parallel) {
      // Execute operations in parallel with concurrency limit
      const semaphore = new Semaphore(maxConcurrency);
      const promises = operations.map(async (op) => {
        await semaphore.acquire();
        try {
          return await this.execute(op.operation, op.tensors, op.options);
        } finally {
          semaphore.release();
        }
      });
      return Promise.all(promises);
    } else {
      // Execute operations sequentially
      const results = [];
      for (const op of operations) {
        const result = await this.execute(op.operation, op.tensors, op.options);
        results.push(result);
      }
      return results;
    }
  }

  /**
   * Copy tensor data to GPU buffer
   */
  async uploadTensor(data, options = {}) {
    const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST } = options;
    return this.bufferManager.createMappedBuffer(data, usage);
  }

  /**
   * Download tensor data from GPU buffer
   */
  async downloadTensor(buffer, size, options = {}) {
    const { format = Float32Array } = options;
    
    // Create staging buffer for readback
    const stagingBuffer = this.bufferManager.allocate(
      size * 4, // Assuming 32-bit floats
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    );

    try {
      // Copy GPU buffer to staging buffer
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size * 4);
      this.device.queue.submit([encoder.finish()]);

      // Map and read data
      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const mappedRange = stagingBuffer.getMappedRange();
      const result = new format(mappedRange.slice());
      stagingBuffer.unmap();

      return result;
    } finally {
      this.bufferManager.release(stagingBuffer, { forceDestroy: true });
    }
  }

  /**
   * Get engine statistics
   */
  getStats() {
    return {
      ...this.stats,
      bufferStats: this.bufferManager?.getStats() || {},
      pipelineStats: this.pipelineCache?.getStats() || {},
      deviceLimits: this.limits,
      supportedFeatures: Array.from(this.supportedFeatures || [])
    };
  }

  /**
   * Cleanup all resources
   */
  async cleanup() {
    this.emit('cleanup:start');

    try {
      if (this.bufferManager) {
        await this.bufferManager.cleanup();
        this.bufferManager = null;
      }

      if (this.pipelineCache) {
        this.pipelineCache.cleanup();
        this.pipelineCache = null;
      }

      if (this.device) {
        this.device.destroy();
        this.device = null;
      }

      this.adapter = null;
      this.isInitialized = false;
      
      this.emit('cleanup:complete');
    } catch (error) {
      this.emit('cleanup:error', { error });
      throw error;
    }
  }

  // Private methods
  _validateOperation(operation, tensors, options) {
    if (!operation || typeof operation !== 'string') {
      throw new Error('Operation must be a non-empty string');
    }

    if (!tensors) {
      throw new Error('Tensors parameter is required');
    }

    const tensorArray = Array.isArray(tensors) ? tensors : [tensors];
    for (const tensor of tensorArray) {
      if (!tensor || (!ArrayBuffer.isView(tensor) && !(tensor instanceof ArrayBuffer))) {
        throw new Error('All tensors must be typed arrays or ArrayBuffers');
      }
    }
  }

  async _prepareBuffers(tensors, operation, options) {
    const tensorArray = Array.isArray(tensors) ? tensors : [tensors];
    const buffers = {
      inputs: [],
      output: null,
      params: null
    };

    // Upload input tensors
    for (const tensor of tensorArray) {
      const buffer = await this.uploadTensor(tensor);
      buffers.inputs.push(buffer);
    }

    // Create output buffer
    const outputSize = this._calculateOutputSize(operation, tensorArray, options);
    buffers.output = this.bufferManager.allocate(
      outputSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    // Create parameter buffer using WebGPU shaders helper
    const params = this.pipelineCache.generateOperationParams(operation, tensorArray, options);
    buffers.params = await this.uploadTensor(params, {
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    return buffers;
  }

  _createBindGroup(pipeline, buffers, options) {
    const entries = [];

    // Add input buffers
    for (let i = 0; i < buffers.inputs.length; i++) {
      entries.push({
        binding: i,
        resource: { buffer: buffers.inputs[i] }
      });
    }

    // Add output buffer
    entries.push({
      binding: buffers.inputs.length,
      resource: { buffer: buffers.output }
    });

    // Add parameter buffer
    entries.push({
      binding: buffers.inputs.length + 1,
      resource: { buffer: buffers.params }
    });

    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries
    });
  }

  async _executeComputePass(pipeline, bindGroup, buffers, options) {
    const encoder = this.device.createCommandEncoder();
    const computePass = encoder.beginComputePass();

    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);

    // Calculate dispatch size
    const workgroupSize = options.workgroupSize || this.config.workgroupSize;
    const outputSize = buffers.output.size / 4; // Assuming 32-bit floats
    const dispatchSize = Math.ceil(outputSize / workgroupSize[0]);

    computePass.dispatchWorkgroups(dispatchSize, 1, 1);
    computePass.end();

    this.device.queue.submit([encoder.finish()]);

    // Wait for completion
    await this.device.queue.onSubmittedWorkDone();

    // Download result
    return this.downloadTensor(buffers.output, outputSize);
  }

  _calculateOutputSize(operation, tensors, options) {
    if (options.outputSize) {
      return options.outputSize;
    }
    
    const firstTensor = tensors[0];
    const getElementCount = (tensor) => {
      return ArrayBuffer.isView(tensor) ? tensor.length : tensor.byteLength / 4;
    };
    
    // Operation-specific output size calculation
    switch (operation) {
      case 'matmul':
        // A(M,K) × B(K,N) = C(M,N)
        const M = tensors[0].shape?.[0] || Math.sqrt(getElementCount(tensors[0]));
        const N = tensors[1].shape?.[1] || Math.sqrt(getElementCount(tensors[1]));
        return M * N;
      
      case 'bmm':
        // Batch matrix multiply: A(B,M,K) × B(B,K,N) = C(B,M,N)
        const B = tensors[0].shape?.[0] || 1;
        const bM = tensors[0].shape?.[1] || Math.sqrt(getElementCount(tensors[0]) / B);
        const bN = tensors[1].shape?.[2] || Math.sqrt(getElementCount(tensors[1]) / B);
        return B * bM * bN;
        
      case 'conv2d':
        // Simplified - assumes same padding and stride=1
        const inHeight = tensors[0].shape?.[2] || 28;
        const inWidth = tensors[0].shape?.[3] || 28;
        const outChannels = tensors[1].shape?.[0] || 32;
        const batchSize = tensors[0].shape?.[0] || 1;
        return batchSize * outChannels * inHeight * inWidth;
        
      case 'transpose':
        return getElementCount(firstTensor);
        
      case 'sum':
      case 'mean':
        // Reduction operations output a single value per batch/dimension
        return options.keepDim ? getElementCount(firstTensor) : 1;
        
      case 'softmax':
        return getElementCount(firstTensor);
        
      case 'maxpool2d':
      case 'avgpool2d':
        // Simplified pooling calculation
        const poolKernel = options.kernelSize || 2;
        const poolStride = options.stride || poolKernel;
        const poolInH = tensors[0].shape?.[2] || 28;
        const poolInW = tensors[0].shape?.[3] || 28;
        const poolOutH = Math.floor((poolInH - poolKernel) / poolStride) + 1;
        const poolOutW = Math.floor((poolInW - poolKernel) / poolStride) + 1;
        const poolChannels = tensors[0].shape?.[1] || 1;
        const poolBatch = tensors[0].shape?.[0] || 1;
        return poolBatch * poolChannels * poolOutH * poolOutW;
        
      default:
        // Element-wise operations preserve input size
        return getElementCount(firstTensor);
    }
  }


  _updateStats(operation, executionTime, buffers) {
    this.stats.computeOperations++;
    this.stats.totalExecutionTime += executionTime;
    this.stats.averageExecutionTime = this.stats.totalExecutionTime / this.stats.computeOperations;
    this.stats.lastOperationTime = executionTime;
    
    const bufferMemory = buffers.inputs.reduce((sum, buf) => sum + buf.size, 0) + buffers.output.size;
    this.stats.memoryUsage = Math.max(this.stats.memoryUsage, bufferMemory);
  }

  _setupEventForwarding() {
    // Forward buffer manager events
    this.bufferManager.on('buffer:created', (data) => this.emit('buffer:created', data));
    this.bufferManager.on('buffer:destroyed', (data) => this.emit('buffer:destroyed', data));
    this.bufferManager.on('gc:complete', (data) => this.emit('buffer:gc', data));
    
    // Forward pipeline cache events
    this.pipelineCache.on('cache:miss', (data) => this.emit('pipeline:miss', data));
    this.pipelineCache.on('pipeline:compiled', (data) => this.emit('pipeline:compiled', data));
    this.pipelineCache.on('warmup:complete', (data) => this.emit('pipeline:warmup', data));
  }

  /**
   * Handle device errors with recovery attempts
   */
  _handleDeviceError(error) {
    const errorType = error.constructor.name;
    
    switch (errorType) {
      case 'GPUOutOfMemoryError':
        logger.warn('GPU out of memory, attempting buffer cleanup');
        this.bufferManager.emergencyCleanup();
        this.emit('recovery:attempt', { type: 'memory-cleanup', timestamp: Date.now() });
        break;
        
      case 'GPUInternalError':
        logger.warn('GPU internal error, marking device as potentially unstable');
        this.deviceStable = false;
        this.emit('device:unstable', { reason: 'internal-error', timestamp: Date.now() });
        break;
        
      case 'GPUValidationError':
        logger.warn('GPU validation error, this may indicate shader or pipeline issues');
        this.emit('validation:error', { error, timestamp: Date.now() });
        break;
        
      default:
        logger.warn('Unknown GPU error type:', errorType);
        this.emit('error:unknown', { error, timestamp: Date.now() });
    }
  }

  /**
   * Get comprehensive error diagnostics
   */
  getErrorDiagnostics() {
    return {
      isInitialized: this.isInitialized,
      deviceStable: this.deviceStable ?? true,
      initFailureReason: this.initFailureReason || null,
      bufferStats: this.bufferManager?.getStats() || null,
      pipelineStats: this.pipelineCache?.getStats() || null,
      supportedFeatures: Array.from(this.supportedFeatures || []),
      timestamp: Date.now()
    };
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

export default WebGPUComputeEngine;