/**
 * MemoryManager - Advanced memory management and resource cleanup
 * Handles WebGPU buffers, Python objects, and prevents memory leaks
 */
import EventEmitter from '../core/event-emitter.js';
import logger from './logger.js';

class MemoryManager extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      maxMemoryMB: config.maxMemoryMB || 1024, // 1GB default
      gcThreshold: config.gcThreshold || 0.8, // Trigger GC at 80%
      checkInterval: config.checkInterval || 5000, // 5 seconds
      enableAutoGC: config.enableAutoGC !== false,
      ...config
    };

    // Resource tracking
    this.resources = new Map();
    this.bufferPool = new Map();
    this.activeBuffers = new Set();
    
    // Memory tracking
    this.memoryUsage = 0;
    this.peakMemoryUsage = 0;
    this.gcCount = 0;
    
    // Cleanup registry for automatic resource management
    this.registry = null;
    this.cleanupTasks = new Set();
    
    // Monitoring
    this.monitoringInterval = null;
    
    this._initializeFinalizationRegistry();
    
    if (this.config.enableAutoGC) {
      this._startMemoryMonitoring();
    }
  }

  /**
   * Register a resource for automatic cleanup
   */
  register(resource, cleanup, options = {}) {
    const {
      size = 0,
      type = 'generic',
      priority = 'normal',
      autoRelease = true
    } = options;

    const id = this._generateId();
    const entry = {
      id,
      resource,
      cleanup,
      size,
      type,
      priority,
      autoRelease,
      createdAt: Date.now(),
      lastAccessed: Date.now()
    };

    this.resources.set(id, entry);
    this.memoryUsage += size;
    this.peakMemoryUsage = Math.max(this.peakMemoryUsage, this.memoryUsage);

    // Register with FinalizationRegistry if available
    if (this.registry && autoRelease) {
      this.registry.register(resource, { id, cleanup }, resource);
    }

    this.emit('resource:registered', { id, type, size });
    
    // Check if we need to run GC
    if (this._shouldRunGC()) {
      this._scheduleGC();
    }

    return id;
  }

  /**
   * Unregister and cleanup a resource
   */
  async unregister(id) {
    const entry = this.resources.get(id);
    if (!entry) {
      return false;
    }

    try {
      await this._cleanupResource(entry);
      this.resources.delete(id);
      this.memoryUsage -= entry.size;
      
      this.emit('resource:unregistered', { id, type: entry.type, size: entry.size });
      return true;
    } catch (error) {
      this.emit('cleanup:error', { id, error });
      return false;
    }
  }

  /**
   * WebGPU buffer management
   */
  allocateBuffer(device, size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    const poolKey = `${size}-${usage}`;
    const pool = this.bufferPool.get(poolKey);
    
    let buffer;
    if (pool && pool.length > 0) {
      buffer = pool.pop();
      this.emit('buffer:reused', { size, usage });
    } else {
      buffer = device.createBuffer({ size, usage });
      this.emit('buffer:created', { size, usage });
    }

    this.activeBuffers.add(buffer);
    
    // Register for cleanup
    const id = this.register(buffer, () => this._cleanupBuffer(buffer), {
      size: size,
      type: 'webgpu-buffer',
      priority: 'high'
    });

    buffer._greedId = id;
    return buffer;
  }

  /**
   * Release WebGPU buffer back to pool or destroy
   */
  releaseBuffer(buffer, options = {}) {
    const { destroy = false, poolMaxSize = 20 } = options;
    
    if (!this.activeBuffers.has(buffer)) {
      return false;
    }

    this.activeBuffers.delete(buffer);
    
    if (buffer._greedId) {
      this.unregister(buffer._greedId);
    }

    if (destroy) {
      buffer.destroy();
      this.emit('buffer:destroyed', { buffer });
      return true;
    }

    // Return to pool
    const poolKey = `${buffer.size}-${buffer.usage}`;
    if (!this.bufferPool.has(poolKey)) {
      this.bufferPool.set(poolKey, []);
    }

    const pool = this.bufferPool.get(poolKey);
    if (pool.length < poolMaxSize) {
      pool.push(buffer);
      this.emit('buffer:pooled', { buffer, poolSize: pool.length });
    } else {
      buffer.destroy();
      this.emit('buffer:destroyed', { buffer, reason: 'pool-full' });
    }

    return true;
  }

  /**
   * Force garbage collection
   */
  async forceGC(options = {}) {
    const { 
      aggressive = false, 
      targetReduction = 0.3,
      maxAge = 300000 // 5 minutes
    } = options;

    this.gcCount++;
    const startTime = performance.now();
    const initialMemory = this.memoryUsage;
    
    this.emit('gc:start', { aggressive, targetReduction });

    try {
      let cleaned = 0;
      const now = Date.now();
      
      // Sort resources by priority and age
      const resources = Array.from(this.resources.entries())
        .sort(([, a], [, b]) => {
          if (a.priority !== b.priority) {
            const priorityOrder = { low: 0, normal: 1, high: 2 };
            return priorityOrder[a.priority] - priorityOrder[b.priority];
          }
          return (now - a.lastAccessed) - (now - b.lastAccessed);
        });

      // Cleanup old or low-priority resources
      for (const [id, entry] of resources) {
        const age = now - entry.lastAccessed;
        const shouldClean = aggressive || 
          age > maxAge || 
          (entry.priority === 'low' && this._shouldRunGC());

        if (shouldClean && entry.autoRelease) {
          if (await this.unregister(id)) {
            cleaned++;
          }
        }

        // Stop if we've reached target reduction
        const reduction = (initialMemory - this.memoryUsage) / initialMemory;
        if (reduction >= targetReduction) {
          break;
        }
      }

      // Run browser GC if available
      if (window.gc && aggressive) {
        window.gc();
      }

      const duration = performance.now() - startTime;
      const memoryReduction = initialMemory - this.memoryUsage;
      
      this.emit('gc:complete', { 
        cleaned, 
        duration, 
        memoryReduction,
        newMemoryUsage: this.memoryUsage
      });

      return { cleaned, memoryReduction, duration };
    } catch (error) {
      this.emit('gc:error', { error });
      throw error;
    }
  }

  /**
   * Get memory statistics
   */
  getStats() {
    return {
      memoryUsage: this.memoryUsage,
      memoryUsageMB: Math.round(this.memoryUsage / (1024 * 1024) * 100) / 100,
      maxMemoryMB: this.config.maxMemoryMB,
      peakMemoryUsage: this.peakMemoryUsage,
      peakMemoryUsageMB: Math.round(this.peakMemoryUsage / (1024 * 1024) * 100) / 100,
      memoryUtilization: this.memoryUsage / (this.config.maxMemoryMB * 1024 * 1024),
      resourceCount: this.resources.size,
      activeBuffers: this.activeBuffers.size,
      bufferPools: this.bufferPool.size,
      gcCount: this.gcCount,
      resourceTypes: this._getResourceTypeStats()
    };
  }

  /**
   * Cleanup all resources and stop monitoring
   */
  async cleanup() {
    this.emit('cleanup:start');
    
    try {
      // Stop monitoring
      if (this.monitoringInterval) {
        clearInterval(this.monitoringInterval);
        this.monitoringInterval = null;
      }

      // Cleanup all resources
      const resourceIds = Array.from(this.resources.keys());
      for (const id of resourceIds) {
        await this.unregister(id);
      }

      // Destroy buffer pools
      for (const pool of this.bufferPool.values()) {
        for (const buffer of pool) {
          buffer.destroy();
        }
      }
      this.bufferPool.clear();

      // Clear active buffers
      for (const buffer of this.activeBuffers) {
        buffer.destroy();
      }
      this.activeBuffers.clear();

      this.memoryUsage = 0;
      this.emit('cleanup:complete');
    } catch (error) {
      this.emit('cleanup:error', { error });
      throw error;
    }
  }

  // Private methods
  _initializeFinalizationRegistry() {
    if (typeof FinalizationRegistry !== 'undefined') {
      this.registry = new FinalizationRegistry(({ id, cleanup }) => {
        try {
          cleanup();
          if (this.resources.has(id)) {
            const entry = this.resources.get(id);
            this.resources.delete(id);
            this.memoryUsage -= entry.size;
            this.emit('resource:finalized', { id });
          }
        } catch (error) {
          this.emit('finalization:error', { id, error });
        }
      });
    }
  }

  _startMemoryMonitoring() {
    this.monitoringInterval = setInterval(() => {
      if (this._shouldRunGC()) {
        this.forceGC({ aggressive: false });
      }
    }, this.config.checkInterval);
  }

  _shouldRunGC() {
    const maxMemoryBytes = this.config.maxMemoryMB * 1024 * 1024;
    return this.memoryUsage > (maxMemoryBytes * this.config.gcThreshold);
  }

  _scheduleGC() {
    // Use setTimeout to avoid blocking the main thread
    setTimeout(() => this.forceGC(), 0);
  }

  async _cleanupResource(entry) {
    try {
      if (typeof entry.cleanup === 'function') {
        await entry.cleanup();
      }
    } catch (error) {
      throw new Error(`Failed to cleanup resource ${entry.id}: ${error.message}`);
    }
  }

  _cleanupBuffer(buffer) {
    return new Promise((resolve) => {
      try {
        if (buffer && typeof buffer.destroy === 'function') {
          buffer.destroy();
        }
        resolve();
      } catch (error) {
        logger.warn('Buffer cleanup error:', {
          bufferId: buffer._greedId,
          error: error.message,
          stack: error.stack
        });
        resolve(); // Don't fail the cleanup process
      }
    });
  }

  _generateId() {
    return `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  _getResourceTypeStats() {
    const stats = {};
    for (const entry of this.resources.values()) {
      if (!stats[entry.type]) {
        stats[entry.type] = { count: 0, totalSize: 0 };
      }
      stats[entry.type].count++;
      stats[entry.type].totalSize += entry.size;
    }
    return stats;
  }
}

export default MemoryManager;