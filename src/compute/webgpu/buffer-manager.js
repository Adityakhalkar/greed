/**
 * BufferManager - WebGPU buffer allocation and management
 * Optimizes memory usage with buffer pooling and automatic cleanup
 */
import EventEmitter from '../../core/event-emitter.js';

class BufferManager extends EventEmitter {
  constructor(device, options = {}) {
    super();
    this.device = device;
    this.config = {
      maxPoolSize: options.maxPoolSize || 100,
      maxBufferSize: options.maxBufferSize || 256 * 1024 * 1024, // 256MB
      gcThreshold: options.gcThreshold || 0.8,
      enablePooling: options.enablePooling !== false,
      ...options
    };

    // Buffer pools organized by size and usage
    this.pools = new Map(); // key: `${size}-${usage}` -> buffer[]
    this.activeBuffers = new Map(); // buffer -> metadata
    this.totalMemoryUsage = 0;
    this.peakMemoryUsage = 0;
    
    // Statistics
    this.stats = {
      allocations: 0,
      poolHits: 0,
      poolMisses: 0,
      releases: 0,
      destroyed: 0,
      currentActive: 0,
      totalPooled: 0
    };
  }

  /**
   * Allocate a buffer with automatic pooling
   */
  allocate(size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    this._validateAllocation(size, usage);
    
    const poolKey = this._getPoolKey(size, usage);
    let buffer = null;
    
    // Try to get from pool first
    if (this.config.enablePooling) {
      buffer = this._getFromPool(poolKey);
      if (buffer) {
        this.stats.poolHits++;
        this.emit('buffer:reused', { size, usage, poolKey });
      }
    }
    
    // Create new buffer if not found in pool
    if (!buffer) {
      buffer = this.device.createBuffer({ size, usage });
      this.stats.poolMisses++;
      this.emit('buffer:created', { size, usage, poolKey });
    }
    
    // Track the buffer
    const metadata = {
      size,
      usage,
      poolKey,
      allocatedAt: performance.now(),
      lastAccessed: performance.now()
    };
    
    this.activeBuffers.set(buffer, metadata);
    this.totalMemoryUsage += size;
    this.peakMemoryUsage = Math.max(this.peakMemoryUsage, this.totalMemoryUsage);
    
    this.stats.allocations++;
    this.stats.currentActive = this.activeBuffers.size;
    
    this.emit('buffer:allocated', { buffer, metadata });
    
    // Check memory pressure and trigger appropriate cleanup
    this._checkMemoryPressure();
    
    return buffer;
  }

  /**
   * Release a buffer back to pool or destroy it
   */
  release(buffer, options = {}) {
    const { forceDestroy = false } = options;
    
    const metadata = this.activeBuffers.get(buffer);
    if (!metadata) {
      this.emit('buffer:release-error', { error: 'Buffer not found in active buffers' });
      return false;
    }
    
    // Remove from active tracking
    this.activeBuffers.delete(buffer);
    this.totalMemoryUsage -= metadata.size;
    this.stats.releases++;
    this.stats.currentActive = this.activeBuffers.size;
    
    // Decide whether to pool or destroy
    if (forceDestroy || !this.config.enablePooling || this._shouldDestroyBuffer(buffer, metadata)) {
      this._destroyBuffer(buffer, metadata);
      return true;
    }
    
    // Add to pool
    if (this._addToPool(buffer, metadata)) {
      this.emit('buffer:pooled', { buffer, poolKey: metadata.poolKey });
    } else {
      this._destroyBuffer(buffer, metadata);
    }
    
    return true;
  }

  /**
   * Release multiple buffers
   */
  releaseAll(buffers, options = {}) {
    const results = [];
    for (const buffer of buffers) {
      results.push(this.release(buffer, options));
    }
    return results;
  }

  /**
   * Create a mapped buffer for data transfer
   */
  async createMappedBuffer(data, usage = GPUBufferUsage.COPY_SRC) {
    const size = this._calculateBufferSize(data);
    const buffer = this.allocate(size, usage | GPUBufferUsage.MAP_WRITE);
    
    try {
      await buffer.mapAsync(GPUMapMode.WRITE);
      const mappedRange = buffer.getMappedRange();
      
      if (data instanceof ArrayBuffer) {
        new Uint8Array(mappedRange).set(new Uint8Array(data));
      } else if (ArrayBuffer.isView(data)) {
        new Uint8Array(mappedRange).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      } else {
        throw new Error('Unsupported data type for mapped buffer');
      }
      
      buffer.unmap();
      this.emit('buffer:mapped', { buffer, size, dataType: data.constructor.name });
      
      return buffer;
    } catch (error) {
      this.release(buffer, { forceDestroy: true });
      throw error;
    }
  }

  /**
   * Copy data between buffers
   */
  copyBuffer(source, destination, size, options = {}) {
    const {
      sourceOffset = 0,
      destinationOffset = 0,
      commandEncoder = null
    } = options;
    
    if (!this.activeBuffers.has(source) || !this.activeBuffers.has(destination)) {
      throw new Error('Source or destination buffer not managed by this BufferManager');
    }
    
    const encoder = commandEncoder || this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size);
    
    if (!commandEncoder) {
      const commands = encoder.finish();
      this.device.queue.submit([commands]);
    }
    
    this.emit('buffer:copied', { source, destination, size });
  }

  /**
   * Get buffer statistics
   */
  getStats() {
    return {
      ...this.stats,
      totalMemoryUsageMB: Math.round(this.totalMemoryUsage / (1024 * 1024) * 100) / 100,
      peakMemoryUsageMB: Math.round(this.peakMemoryUsage / (1024 * 1024) * 100) / 100,
      poolCount: this.pools.size,
      totalPooled: Array.from(this.pools.values()).reduce((sum, pool) => sum + pool.length, 0),
      poolEfficiency: this.stats.allocations > 0 ? (this.stats.poolHits / this.stats.allocations) : 0
    };
  }

  /**
   * Force garbage collection of unused pooled buffers
   */
  async gc(options = {}) {
    const { 
      aggressive = false, 
      maxAge = 60000, // 1 minute
      targetReduction = 0.5 
    } = options;
    
    this.emit('gc:start', { aggressive, maxAge, targetReduction });
    
    let destroyed = 0;
    const now = performance.now();
    const initialPooled = this._getTotalPooledBuffers();
    
    for (const [poolKey, pool] of this.pools.entries()) {
      const buffers = pool.slice(); // Copy to avoid modification during iteration
      
      for (let i = buffers.length - 1; i >= 0; i--) {
        const buffer = buffers[i];
        const shouldDestroy = aggressive || 
          (buffer._pooledAt && (now - buffer._pooledAt) > maxAge);
        
        if (shouldDestroy) {
          pool.splice(i, 1);
          buffer.destroy();
          destroyed++;
          this.stats.destroyed++;
        }
        
        // Check if we've reached target reduction
        const currentReduction = destroyed / initialPooled;
        if (currentReduction >= targetReduction) {
          break;
        }
      }
      
      // Remove empty pools
      if (pool.length === 0) {
        this.pools.delete(poolKey);
      }
    }
    
    this.emit('gc:complete', { destroyed, remaining: this._getTotalPooledBuffers() });
    return destroyed;
  }

  /**
   * Emergency cleanup when GPU memory is exhausted
   * More aggressive than forceGC - immediately destroys unused buffers
   */
  async emergencyCleanup() {
    this.emit('emergency:start');
    
    try {
      let destroyed = 0;
      
      // Immediately destroy all pooled buffers
      for (const [poolKey, buffers] of this.pools.entries()) {
        while (buffers.length > 0) {
          const buffer = buffers.pop();
          try {
            buffer.destroy();
            destroyed++;
            this.stats.destroyed++;
          } catch (error) {
            // Continue cleanup even if individual destroy fails
            this.emit('buffer:destroy-error', { buffer, error });
          }
        }
      }
      
      // Clear pools
      this.pools.clear();
      
      // Force browser garbage collection if available
      if (window.gc) {
        window.gc();
      }
      
      this.emit('emergency:complete', { destroyed });
      return destroyed;
    } catch (error) {
      this.emit('emergency:error', { error });
      throw error;
    }
  }

  /**
   * Cleanup all resources
   */
  async cleanup() {
    this.emit('cleanup:start');
    
    try {
      // Destroy all active buffers
      for (const [buffer, metadata] of this.activeBuffers.entries()) {
        this._destroyBuffer(buffer, metadata);
      }
      this.activeBuffers.clear();
      
      // Destroy all pooled buffers
      for (const pool of this.pools.values()) {
        for (const buffer of pool) {
          buffer.destroy();
        }
      }
      this.pools.clear();
      
      // Reset statistics
      this.totalMemoryUsage = 0;
      this.stats.currentActive = 0;
      this.stats.totalPooled = 0;
      
      this.emit('cleanup:complete');
    } catch (error) {
      this.emit('cleanup:error', { error });
      throw error;
    }
  }

  // Private methods
  _validateAllocation(size, usage) {
    if (size <= 0 || size > this.config.maxBufferSize) {
      throw new Error(`Invalid buffer size: ${size}. Must be between 1 and ${this.config.maxBufferSize}`);
    }
    
    if (typeof usage !== 'number') {
      throw new Error('Buffer usage must be a number');
    }
  }

  _getPoolKey(size, usage) {
    return `${size}-${usage}`;
  }

  _getFromPool(poolKey) {
    const pool = this.pools.get(poolKey);
    return pool && pool.length > 0 ? pool.pop() : null;
  }

  _addToPool(buffer, metadata) {
    const poolKey = metadata.poolKey;
    
    if (!this.pools.has(poolKey)) {
      this.pools.set(poolKey, []);
    }
    
    const pool = this.pools.get(poolKey);
    
    if (pool.length >= this.config.maxPoolSize) {
      return false; // Pool is full
    }
    
    buffer._pooledAt = performance.now();
    pool.push(buffer);
    this.stats.totalPooled++;
    
    return true;
  }

  _destroyBuffer(buffer, metadata) {
    try {
      buffer.destroy();
      this.stats.destroyed++;
      this.emit('buffer:destroyed', { buffer, metadata });
    } catch (error) {
      this.emit('buffer:destroy-error', { buffer, error });
    }
  }

  _shouldDestroyBuffer(buffer, metadata) {
    // Destroy if buffer is too large for pooling
    return metadata.size > this.config.maxBufferSize / 4;
  }

  _shouldRunGC() {
    const memoryUsageRatio = this.totalMemoryUsage / this.config.maxBufferSize;
    return memoryUsageRatio > this.config.gcThreshold;
  }

  async _runGC() {
    try {
      await this.gc({ aggressive: false });
    } catch (error) {
      this.emit('gc:error', { error });
    }
  }

  _calculateBufferSize(data) {
    if (data instanceof ArrayBuffer) {
      return data.byteLength;
    } else if (ArrayBuffer.isView(data)) {
      return data.byteLength;
    } else if (Array.isArray(data)) {
      return data.length * 4; // Assume 32-bit numbers
    } else {
      throw new Error('Cannot calculate buffer size for data type');
    }
  }

  _getTotalPooledBuffers() {
    return Array.from(this.pools.values()).reduce((sum, pool) => sum + pool.length, 0);
  }

  /**
   * Check memory pressure and trigger appropriate cleanup
   */
  _checkMemoryPressure() {
    const memoryRatio = this.totalMemoryUsage / this.config.maxBufferSize;
    
    // Emergency cleanup at 95% memory usage
    if (memoryRatio >= 0.95) {
      this.emit('memory:critical', { 
        memoryRatio, 
        totalUsage: this.totalMemoryUsage,
        maxSize: this.config.maxBufferSize
      });
      
      // Trigger emergency cleanup asynchronously
      setTimeout(() => this.emergencyCleanup(), 0);
    }
    // Aggressive GC at 80% memory usage  
    else if (memoryRatio >= this.config.gcThreshold) {
      this.emit('memory:pressure', {
        memoryRatio,
        totalUsage: this.totalMemoryUsage,
        maxSize: this.config.maxBufferSize
      });
      
      // Trigger aggressive GC asynchronously
      setTimeout(() => this.forceGC(), 0);
    }
    // Regular cleanup at 60% memory usage
    else if (memoryRatio >= 0.6) {
      this.emit('memory:warning', {
        memoryRatio,
        totalUsage: this.totalMemoryUsage,
        maxSize: this.config.maxBufferSize
      });
      
      // Trigger regular GC asynchronously
      setTimeout(() => this._runGC(), 0);
    }
  }

  /**
   * Internal GC method (less aggressive than forceGC)
   */
  _runGC() {
    const pooledBuffers = this._getTotalPooledBuffers();
    if (pooledBuffers > 0) {
      // Clean oldest 20% of pooled buffers
      const targetDestruction = Math.ceil(pooledBuffers * 0.2);
      let destroyed = 0;
      
      for (const [poolKey, buffers] of this.pools.entries()) {
        while (buffers.length > 0 && destroyed < targetDestruction) {
          const buffer = buffers.shift();
          try {
            buffer.destroy();
            destroyed++;
            this.stats.destroyed++;
          } catch (error) {
            this.emit('buffer:destroy-error', { buffer, error });
          }
        }
        
        if (buffers.length === 0) {
          this.pools.delete(poolKey);
        }
        
        if (destroyed >= targetDestruction) {
          break;
        }
      }
      
      this.emit('gc:automatic', { destroyed, remaining: this._getTotalPooledBuffers() });
    }
  }
}

export default BufferManager;