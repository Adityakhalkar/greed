/**
 * ComputeStrategy - Intelligent selection between WebGPU, CPU, and Worker execution
 * Optimizes performance through smart workload distribution and fallback handling
 */
import EventEmitter from '../core/event-emitter.js';
import WebGPUComputeEngine from './webgpu/compute-engine.js';
import CPUEngine from './cpu/cpu-engine.js';
import WorkerEngine from './worker/worker-engine.js';

class ComputeStrategy extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      // Performance thresholds for strategy selection
      webgpuMinElements: config.webgpuMinElements || 1000,
      workerMinElements: config.workerMinElements || 10000,
      
      // Resource limits
      maxConcurrentOperations: config.maxConcurrentOperations || 4,
      memoryThresholdMB: config.memoryThresholdMB || 512,
      
      // Fallback behavior
      enableAutoFallback: config.enableAutoFallback !== false,
      fallbackTimeout: config.fallbackTimeout || 5000,
      
      // Performance tracking
      enableAdaptiveSelection: config.enableAdaptiveSelection !== false,
      performanceHistorySize: config.performanceHistorySize || 100,
      
      ...config
    };

    // Compute engines
    this.webgpu = new WebGPUComputeEngine(config.webgpu || {});
    this.cpu = new CPUEngine(config.cpu || {});
    this.worker = new WorkerEngine(config.worker || {});
    
    // Strategy state
    this.isInitialized = false;
    this.availableStrategies = new Set();
    this.currentOperations = new Map();
    
    // Performance tracking for adaptive selection
    this.performanceHistory = new Map(); // operation -> { webgpu: [], cpu: [], worker: [] }
    this.strategyPreferences = new Map(); // operation -> preferred strategy
    
    // Resource monitoring
    this.resourceMonitor = {
      memoryUsage: 0,
      activeOperations: 0,
      gpuUtilization: 0
    };
  }

  /**
   * Initialize all available compute strategies
   */
  async initialize() {
    if (this.isInitialized) {
      return this.availableStrategies;
    }

    this.emit('init:start');
    
    try {
      // Initialize WebGPU (optional)
      try {
        const webgpuReady = await this.webgpu.initialize();
        if (webgpuReady) {
          this.availableStrategies.add('webgpu');
          this._setupEngineForwarding(this.webgpu, 'webgpu');
          this.emit('init:webgpu-success', { message: 'WebGPU engine initialized successfully' });
        } else {
          // WebGPU initialization returned false - check for failure reason
          const failureReason = this.webgpu.initFailureReason || 'Unknown WebGPU initialization failure';
          this.emit('init:webgpu-failed', { 
            error: new Error(failureReason),
            details: 'WebGPU engine initialization returned false'
          });
        }
      } catch (error) {
        this.emit('init:webgpu-failed', { error });
      }

      // Initialize CPU (always available)
      await this.cpu.initialize();
      this.availableStrategies.add('cpu');
      this._setupEngineForwarding(this.cpu, 'cpu');

      // Initialize Worker pool (optional)
      try {
        await this.worker.initialize();
        this.availableStrategies.add('worker');
        this._setupEngineForwarding(this.worker, 'worker');
      } catch (error) {
        this.emit('init:worker-failed', { error });
      }

      this.isInitialized = true;
      this.emit('init:complete', { 
        strategies: Array.from(this.availableStrategies)
      });

      return this.availableStrategies;
    } catch (error) {
      this.emit('init:error', { error });
      throw error;
    }
  }

  /**
   * Execute operation with optimal strategy selection
   */
  async execute(operation, tensors, options = {}) {
    if (!this.isInitialized) {
      throw new Error('ComputeStrategy not initialized. Call initialize() first.');
    }

    const startTime = performance.now();
    const operationId = this._generateOperationId();
    
    this.emit('execute:start', { operation, operationId, options });

    try {
      // Select optimal strategy
      const strategy = await this._selectStrategy(operation, tensors, options);
      
      // Record operation start
      this.currentOperations.set(operationId, {
        operation,
        strategy: strategy.name,
        startTime,
        tensors: Array.isArray(tensors) ? tensors.length : 1
      });

      this.resourceMonitor.activeOperations++;
      
      // Execute with selected strategy
      const result = await this._executeWithFallback(strategy, operation, tensors, options);
      
      // Record performance metrics
      const executionTime = performance.now() - startTime;
      this._recordPerformance(operation, strategy.name, executionTime, tensors);
      
      // Update resource monitoring
      this.currentOperations.delete(operationId);
      this.resourceMonitor.activeOperations--;
      
      this.emit('execute:complete', { 
        operation, 
        operationId, 
        strategy: strategy.name, 
        executionTime,
        resultSize: result.length 
      });

      return result;
    } catch (error) {
      this.currentOperations.delete(operationId);
      this.resourceMonitor.activeOperations--;
      
      this.emit('execute:error', { operation, operationId, error });
      throw error;
    }
  }

  /**
   * Execute multiple operations with optimal distribution
   */
  async executeBatch(operations, options = {}) {
    const {
      parallel = true,
      maxConcurrency = this.config.maxConcurrentOperations,
      loadBalance = true
    } = options;

    if (!parallel) {
      // Sequential execution
      const results = [];
      for (const op of operations) {
        const result = await this.execute(op.operation, op.tensors, op.options);
        results.push(result);
      }
      return results;
    }

    // Parallel execution with load balancing
    if (loadBalance) {
      return this._executeWithLoadBalancing(operations, maxConcurrency);
    }

    // Simple parallel execution
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
  }

  /**
   * Get performance statistics for all strategies
   */
  getStats() {
    const stats = {
      availableStrategies: Array.from(this.availableStrategies),
      activeOperations: this.resourceMonitor.activeOperations,
      resourceMonitor: { ...this.resourceMonitor },
      strategyPreferences: Object.fromEntries(this.strategyPreferences),
      engines: {}
    };

    // Get stats from each engine
    if (this.availableStrategies.has('webgpu')) {
      stats.engines.webgpu = this.webgpu.getStats();
    }
    if (this.availableStrategies.has('cpu')) {
      stats.engines.cpu = this.cpu.getStats();
    }
    if (this.availableStrategies.has('worker')) {
      stats.engines.worker = this.worker.getStats();
    }

    return stats;
  }

  /**
   * Force strategy for specific operation type
   */
  setStrategyPreference(operation, strategy) {
    if (!this.availableStrategies.has(strategy)) {
      throw new Error(`Strategy '${strategy}' not available`);
    }
    this.strategyPreferences.set(operation, strategy);
    this.emit('strategy:preference-set', { operation, strategy });
  }

  /**
   * Clear strategy preferences and reset adaptive learning
   */
  resetAdaptiveSelection() {
    this.performanceHistory.clear();
    this.strategyPreferences.clear();
    this.emit('strategy:reset');
  }

  /**
   * Cleanup all compute engines
   */
  async cleanup() {
    this.emit('cleanup:start');

    try {
      const cleanupPromises = [];

      if (this.webgpu) {
        cleanupPromises.push(this.webgpu.cleanup());
      }
      if (this.cpu) {
        cleanupPromises.push(this.cpu.cleanup());
      }
      if (this.worker) {
        cleanupPromises.push(this.worker.cleanup());
      }

      await Promise.all(cleanupPromises);

      this.availableStrategies.clear();
      this.currentOperations.clear();
      this.performanceHistory.clear();
      this.isInitialized = false;

      this.emit('cleanup:complete');
    } catch (error) {
      this.emit('cleanup:error', { error });
      throw error;
    }
  }

  // Private methods
  async _selectStrategy(operation, tensors, options) {
    // Check for explicit strategy preference
    if (options.strategy && this.availableStrategies.has(options.strategy)) {
      return this._getEngine(options.strategy);
    }

    // Check for learned preferences
    if (this.config.enableAdaptiveSelection && this.strategyPreferences.has(operation)) {
      const preferred = this.strategyPreferences.get(operation);
      if (this.availableStrategies.has(preferred)) {
        return this._getEngine(preferred);
      }
    }

    // Calculate workload characteristics
    const workloadSize = this._calculateWorkloadSize(tensors);
    const complexity = this._estimateComplexity(operation, tensors);
    const resourcePressure = this._assessResourcePressure();

    // Decision tree for strategy selection
    if (this._shouldUseWebGPU(operation, workloadSize, complexity, resourcePressure)) {
      return this._getEngine('webgpu');
    }

    if (this._shouldUseWorker(operation, workloadSize, complexity, resourcePressure)) {
      return this._getEngine('worker');
    }

    // Default to CPU
    return this._getEngine('cpu');
  }

  _shouldUseWebGPU(operation, workloadSize, complexity, resourcePressure) {
    if (!this.availableStrategies.has('webgpu')) {
      return false;
    }

    // Check minimum workload size
    if (workloadSize < this.config.webgpuMinElements) {
      return false;
    }

    // Check resource pressure
    if (resourcePressure.memory > 0.8 || resourcePressure.activeOperations > 0.9) {
      return false;
    }

    // Check operation suitability for GPU
    const gpuSuitableOps = ['matmul', 'conv2d', 'add', 'multiply', 'relu', 'sigmoid'];
    if (!gpuSuitableOps.includes(operation)) {
      return complexity > 0.7; // Only for complex operations
    }

    return true;
  }

  _shouldUseWorker(operation, workloadSize, complexity, resourcePressure) {
    if (!this.availableStrategies.has('worker')) {
      return false;
    }

    // Use workers for large CPU-bound operations
    if (workloadSize > this.config.workerMinElements && complexity > 0.5) {
      return true;
    }

    // Use workers when main thread is busy
    if (resourcePressure.activeOperations > 0.7) {
      return true;
    }

    return false;
  }

  async _executeWithFallback(engine, operation, tensors, options) {
    if (!this.config.enableAutoFallback) {
      return engine.execute(operation, tensors, options);
    }

    try {
      const executePromise = engine.execute(operation, tensors, options);
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Operation timeout')), this.config.fallbackTimeout);
      });

      return await Promise.race([executePromise, timeoutPromise]);
    } catch (error) {
      this.emit('fallback:triggered', { 
        from: engine.name, 
        operation, 
        error: error.message 
      });

      // Try fallback strategies
      const fallbackOrder = this._getFallbackOrder(engine.name);
      for (const fallbackStrategy of fallbackOrder) {
        if (this.availableStrategies.has(fallbackStrategy)) {
          try {
            const fallbackEngine = this._getEngine(fallbackStrategy);
            this.emit('fallback:executing', { 
              strategy: fallbackStrategy, 
              operation 
            });
            return await fallbackEngine.execute(operation, tensors, options);
          } catch (fallbackError) {
            this.emit('fallback:failed', { 
              strategy: fallbackStrategy, 
              operation, 
              error: fallbackError.message 
            });
          }
        }
      }

      // If all fallbacks fail, throw original error
      throw error;
    }
  }

  async _executeWithLoadBalancing(operations, maxConcurrency) {
    // Group operations by predicted execution time and resource requirements
    const operationGroups = this._groupOperationsByLoad(operations);
    
    // Distribute operations across available strategies
    const distributedOps = this._distributeOperations(operationGroups);
    
    // Execute with controlled concurrency
    const semaphore = new Semaphore(maxConcurrency);
    const promises = distributedOps.map(async (op) => {
      await semaphore.acquire();
      try {
        return await this.execute(op.operation, op.tensors, op.options);
      } finally {
        semaphore.release();
      }
    });

    return Promise.all(promises);
  }

  _getEngine(strategyName) {
    const engines = {
      webgpu: this.webgpu,
      cpu: this.cpu,
      worker: this.worker
    };
    
    const engine = engines[strategyName];
    if (!engine) {
      throw new Error(`Engine '${strategyName}' not found`);
    }
    
    engine.name = strategyName;
    return engine;
  }

  _calculateWorkloadSize(tensors) {
    const tensorArray = Array.isArray(tensors) ? tensors : [tensors];
    return tensorArray.reduce((total, tensor) => {
      if (ArrayBuffer.isView(tensor)) {
        return total + tensor.length;
      } else if (tensor instanceof ArrayBuffer) {
        return total + (tensor.byteLength / 4); // Assume 32-bit elements
      }
      return total;
    }, 0);
  }

  _estimateComplexity(operation, tensors) {
    // Simplified complexity estimation
    const complexityMap = {
      'add': 0.1,
      'multiply': 0.1,
      'matmul': 0.8,
      'conv2d': 0.9,
      'relu': 0.2,
      'sigmoid': 0.3,
      'softmax': 0.6,
      'transpose': 0.3
    };
    
    const baseComplexity = complexityMap[operation] || 0.5;
    const workloadSize = this._calculateWorkloadSize(tensors);
    
    // Scale complexity based on workload size
    const sizeMultiplier = Math.log10(Math.max(workloadSize, 1)) / 6; // Log scale
    return Math.min(baseComplexity * (1 + sizeMultiplier), 1.0);
  }

  _assessResourcePressure() {
    return {
      memory: this.resourceMonitor.memoryUsage / (this.config.memoryThresholdMB * 1024 * 1024),
      activeOperations: this.resourceMonitor.activeOperations / this.config.maxConcurrentOperations,
      gpu: this.resourceMonitor.gpuUtilization
    };
  }

  _recordPerformance(operation, strategy, executionTime, tensors) {
    if (!this.config.enableAdaptiveSelection) {
      return;
    }

    if (!this.performanceHistory.has(operation)) {
      this.performanceHistory.set(operation, { webgpu: [], cpu: [], worker: [] });
    }

    const history = this.performanceHistory.get(operation);
    const strategyHistory = history[strategy];
    
    if (strategyHistory.length >= this.config.performanceHistorySize) {
      strategyHistory.shift(); // Remove oldest entry
    }
    
    strategyHistory.push({
      executionTime,
      workloadSize: this._calculateWorkloadSize(tensors),
      timestamp: Date.now()
    });

    // Update strategy preference based on performance
    this._updateStrategyPreference(operation);
  }

  _updateStrategyPreference(operation) {
    const history = this.performanceHistory.get(operation);
    if (!history) return;

    // Calculate average performance for each strategy
    const averages = {};
    for (const [strategy, records] of Object.entries(history)) {
      if (records.length > 0) {
        const avgTime = records.reduce((sum, r) => sum + r.executionTime, 0) / records.length;
        averages[strategy] = avgTime;
      }
    }

    // Find best performing strategy
    const bestStrategy = Object.entries(averages)
      .reduce((best, [strategy, avgTime]) => {
        return !best || avgTime < best.avgTime ? { strategy, avgTime } : best;
      }, null);

    if (bestStrategy && this.availableStrategies.has(bestStrategy.strategy)) {
      this.strategyPreferences.set(operation, bestStrategy.strategy);
    }
  }

  _getFallbackOrder(primaryStrategy) {
    const fallbackChains = {
      'webgpu': ['cpu', 'worker'],
      'worker': ['cpu'],
      'cpu': []
    };
    return fallbackChains[primaryStrategy] || [];
  }

  _generateOperationId() {
    return `op_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  _setupEngineForwarding(engine, name) {
    // Forward important events from engines
    engine.on('init:complete', (data) => this.emit(`${name}:ready`, data));
    engine.on('compute:error', (data) => this.emit(`${name}:error`, data));
    engine.on('cleanup:complete', () => this.emit(`${name}:cleanup`));
  }

  _groupOperationsByLoad(operations) {
    // Simplified grouping - real implementation would be more sophisticated
    return operations.map(op => ({
      ...op,
      estimatedLoad: this._estimateComplexity(op.operation, op.tensors),
      workloadSize: this._calculateWorkloadSize(op.tensors)
    }));
  }

  _distributeOperations(operationGroups) {
    // Simple round-robin distribution for now
    return operationGroups;
  }
}

// Note: CPUEngine and WorkerEngine are now imported from separate files
// src/compute/cpu/cpu-engine.js and src/compute/worker/worker-engine.js

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

export default ComputeStrategy;