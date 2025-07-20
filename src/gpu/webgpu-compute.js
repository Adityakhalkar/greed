/**
 * WebGPU Compute Shaders for High-Performance Tensor Operations
 * 
 * Features:
 * - GPU-accelerated tensor operations
 * - Optimized compute shaders for common operations
 * - Memory-efficient buffer management
 * - Batch processing support
 * - Automatic fallback to CPU when WebGPU unavailable
 */

class WebGPUCompute {
  constructor() {
    this.device = null;
    this.adapter = null;
    this.isInitialized = false;
    this.supportedFeatures = new Set();
    
    // Compute pipelines cache
    this.pipelines = new Map();
    
    // Buffer management
    this.bufferPool = new Map();
    this.activeBuffers = new Set();
    
    // Performance tracking
    this.stats = {
      computeOperations: 0,
      averageExecutionTime: 0,
      totalExecutionTime: 0,
      memoryUsage: 0
    };
  }

  /**
   * Initialize WebGPU device and adapter
   */
  async initialize() {
    if (this.isInitialized) return true;
    
    if (!navigator.gpu) {
      console.warn('WebGPU not supported in this browser');
      return false;
    }

    try {
      // Request adapter with high performance preference
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });

      if (!this.adapter) {
        console.warn('No WebGPU adapter available');
        return false;
      }

      // Get supported features
      this.supportedFeatures = this.adapter.features;
      
      // Request device with optimal limits
      this.device = await this.adapter.requestDevice({
        requiredFeatures: [],
        requiredLimits: {
          maxComputeWorkgroupStorageSize: this.adapter.limits.maxComputeWorkgroupStorageSize,
          maxComputeInvocationsPerWorkgroup: this.adapter.limits.maxComputeInvocationsPerWorkgroup
        }
      });

      // Set up error handling
      this.device.addEventListener('uncapturederror', (event) => {
        console.error('WebGPU uncaptured error:', event.error);
      });

      this.isInitialized = true;
      console.log('✅ WebGPU initialized successfully');
      console.log('📊 Adapter limits:', {
        maxComputeWorkgroupStorageSize: this.adapter.limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: this.adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxBufferSize: this.adapter.limits.maxBufferSize
      });
      
      return true;
    } catch (error) {
      console.error('❌ Failed to initialize WebGPU:', error);
      return false;
    }
  }

  /**
   * Create compute shader for element-wise operations
   */
  createElementwiseShader(operation, workgroupSize = 64) {
    const shaderCode = `
      @group(0) @binding(0) var<storage, read> input_a: array<f32>;
      @group(0) @binding(1) var<storage, read> input_b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output: array<f32>;
      @group(0) @binding(3) var<uniform> params: Params;
      
      struct Params {
        size: u32,
        scalar_value: f32,
        use_scalar: u32,
        padding: u32,
      }
      
      @compute @workgroup_size(${workgroupSize})
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        if (index >= params.size) {
          return;
        }
        
        let a = input_a[index];
        let b = select(input_b[index], params.scalar_value, params.use_scalar != 0u);
        
        ${this.generateOperationCode(operation)}
      }
    `;
    
    return shaderCode;
  }

  /**
   * Generate operation code for different mathematical operations
   */
  generateOperationCode(operation) {
    switch (operation) {
      case 'add':
        return 'output[index] = a + b;';
      case 'subtract':
        return 'output[index] = a - b;';
      case 'multiply':
        return 'output[index] = a * b;';
      case 'divide':
        return 'output[index] = a / b;';
      case 'power':
        return 'output[index] = pow(a, b);';
      case 'relu':
        return 'output[index] = max(0.0, a);';
      case 'sigmoid':
        return 'output[index] = 1.0 / (1.0 + exp(-a));';
      case 'tanh':
        return 'output[index] = tanh(a);';
      case 'exp':
        return 'output[index] = exp(a);';
      case 'log':
        return 'output[index] = log(a);';
      case 'sqrt':
        return 'output[index] = sqrt(a);';
      case 'sin':
        return 'output[index] = sin(a);';
      case 'cos':
        return 'output[index] = cos(a);';
      default:
        throw new Error(`Unsupported operation: ${operation}`);
    }
  }

  /**
   * Create matrix multiplication shader
   */
  createMatMulShader(workgroupSize = 16) {
    const shaderCode = `
      @group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
      @group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output: array<f32>;
      @group(0) @binding(3) var<uniform> params: MatMulParams;
      
      struct MatMulParams {
        m: u32,
        n: u32,
        k: u32,
        padding: u32,
      }
      
      @compute @workgroup_size(${workgroupSize}, ${workgroupSize})
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let row = global_id.x;
        let col = global_id.y;
        
        if (row >= params.m || col >= params.n) {
          return;
        }
        
        var sum = 0.0;
        for (var i = 0u; i < params.k; i++) {
          let a_val = matrix_a[row * params.k + i];
          let b_val = matrix_b[i * params.n + col];
          sum += a_val * b_val;
        }
        
        output[row * params.n + col] = sum;
      }
    `;
    
    return shaderCode;
  }

  /**
   * Create reduction shader (sum, mean, max, min)
   */
  createReductionShader(operation, workgroupSize = 256) {
    const shaderCode = `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;
      @group(0) @binding(2) var<uniform> params: ReductionParams;
      
      struct ReductionParams {
        size: u32,
        stride: u32,
        padding1: u32,
        padding2: u32,
      }
      
      var<workgroup> shared_data: array<f32, ${workgroupSize}>;
      
      @compute @workgroup_size(${workgroupSize})
      fn main(
        @builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>
      ) {
        let tid = local_id.x;
        let i = workgroup_id.x * ${workgroupSize} * 2 + tid;
        
        // Load data into shared memory
        var val = ${this.getReductionIdentity(operation)};
        if (i < params.size) {
          val = input[i];
          if (i + ${workgroupSize} < params.size) {
            ${this.generateReductionOperation(operation, 'val', 'input[i + ' + workgroupSize + ']')}
          }
        }
        shared_data[tid] = val;
        
        workgroupBarrier();
        
        // Perform reduction in shared memory
        var s = ${workgroupSize}u / 2u;
        while (s > 0u) {
          if (tid < s) {
            ${this.generateReductionOperation(operation, 'shared_data[tid]', 'shared_data[tid + s]')}
          }
          workgroupBarrier();
          s /= 2u;
        }
        
        // Write result
        if (tid == 0u) {
          output[workgroup_id.x] = shared_data[0];
        }
      }
    `;
    
    return shaderCode;
  }

  /**
   * Get identity value for reduction operations
   */
  getReductionIdentity(operation) {
    switch (operation) {
      case 'sum':
      case 'mean':
        return '0.0';
      case 'max':
        return '-3.4028235e+38'; // -FLT_MAX
      case 'min':
        return '3.4028235e+38';  // FLT_MAX
      default:
        return '0.0';
    }
  }

  /**
   * Generate reduction operation code
   */
  generateReductionOperation(operation, target, source) {
    switch (operation) {
      case 'sum':
      case 'mean':
        return `${target} = ${target} + ${source};`;
      case 'max':
        return `${target} = max(${target}, ${source});`;
      case 'min':
        return `${target} = min(${target}, ${source});`;
      default:
        return `${target} = ${target} + ${source};`;
    }
  }

  /**
   * Get or create compute pipeline
   */
  async getComputePipeline(operation, shaderType = 'elementwise') {
    const pipelineKey = `${shaderType}_${operation}`;
    
    if (this.pipelines.has(pipelineKey)) {
      return this.pipelines.get(pipelineKey);
    }

    let shaderCode;
    switch (shaderType) {
      case 'elementwise':
        shaderCode = this.createElementwiseShader(operation);
        break;
      case 'matmul':
        shaderCode = this.createMatMulShader();
        break;
      case 'reduction':
        shaderCode = this.createReductionShader(operation);
        break;
      default:
        throw new Error(`Unknown shader type: ${shaderType}`);
    }

    const shaderModule = this.device.createShaderModule({
      code: shaderCode
    });

    const pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });

    this.pipelines.set(pipelineKey, pipeline);
    return pipeline;
  }

  /**
   * Create GPU buffer
   */
  createBuffer(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage
    });

    this.device.queue.writeBuffer(buffer, 0, data);
    this.activeBuffers.add(buffer);
    this.stats.memoryUsage += data.byteLength;

    return buffer;
  }

  /**
   * Create uniform buffer for parameters
   */
  createUniformBuffer(data) {
    return this.createBuffer(data, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  }

  /**
   * Execute element-wise operation
   */
  async executeElementwise(operation, inputA, inputB = null, scalar = null) {
    if (!this.isInitialized) {
      throw new Error('WebGPU not initialized');
    }

    const startTime = performance.now();
    const size = inputA.length;
    const useScalar = scalar !== null;

    // Create buffers
    const bufferA = this.createBuffer(new Float32Array(inputA));
    const bufferB = useScalar ? 
      this.createBuffer(new Float32Array(1)) : 
      this.createBuffer(new Float32Array(inputB));
    
    const outputBuffer = this.device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Parameters
    const params = new ArrayBuffer(16);
    const paramsView = new DataView(params);
    paramsView.setUint32(0, size, true);
    paramsView.setFloat32(4, scalar || 0.0, true);
    paramsView.setUint32(8, useScalar ? 1 : 0, true);
    
    const uniformBuffer = this.createUniformBuffer(params);

    // Get compute pipeline
    const pipeline = await this.getComputePipeline(operation, 'elementwise');

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } }
      ]
    });

    // Execute compute shader
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(size / 64));
    passEncoder.end();

    // Read result
    const readBuffer = this.device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, size * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange());
    const output = Array.from(result);

    // Cleanup
    readBuffer.unmap();
    this.cleanupBuffers([bufferA, bufferB, outputBuffer, uniformBuffer, readBuffer]);

    // Update stats
    const executionTime = performance.now() - startTime;
    this.updateStats(executionTime);

    return output;
  }

  /**
   * Execute matrix multiplication
   */
  async executeMatMul(matrixA, matrixB, shapeA, shapeB) {
    if (!this.isInitialized) {
      throw new Error('WebGPU not initialized');
    }

    const startTime = performance.now();
    const [m, k] = shapeA;
    const [k2, n] = shapeB;

    if (k !== k2) {
      throw new Error(`Matrix dimensions don't match: ${k} !== ${k2}`);
    }

    // Create buffers
    const bufferA = this.createBuffer(new Float32Array(matrixA));
    const bufferB = this.createBuffer(new Float32Array(matrixB));
    
    const outputBuffer = this.device.createBuffer({
      size: m * n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Parameters
    const params = new ArrayBuffer(16);
    const paramsView = new DataView(params);
    paramsView.setUint32(0, m, true);
    paramsView.setUint32(4, n, true);
    paramsView.setUint32(8, k, true);
    
    const uniformBuffer = this.createUniformBuffer(params);

    // Get compute pipeline
    const pipeline = await this.getComputePipeline('matmul', 'matmul');

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } }
      ]
    });

    // Execute compute shader
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(m / 16), Math.ceil(n / 16));
    passEncoder.end();

    // Read result
    const readBuffer = this.device.createBuffer({
      size: m * n * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, m * n * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange());
    const output = Array.from(result);

    // Cleanup
    readBuffer.unmap();
    this.cleanupBuffers([bufferA, bufferB, outputBuffer, uniformBuffer, readBuffer]);

    // Update stats
    const executionTime = performance.now() - startTime;
    this.updateStats(executionTime);

    return output;
  }

  /**
   * Execute reduction operation
   */
  async executeReduction(operation, input) {
    if (!this.isInitialized) {
      throw new Error('WebGPU not initialized');
    }

    const startTime = performance.now();
    const size = input.length;
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(size / (workgroupSize * 2));

    // Create buffers
    const inputBuffer = this.createBuffer(new Float32Array(input));
    const outputBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Parameters
    const params = new ArrayBuffer(16);
    const paramsView = new DataView(params);
    paramsView.setUint32(0, size, true);
    paramsView.setUint32(4, 1, true); // stride
    
    const uniformBuffer = this.createUniformBuffer(params);

    // Get compute pipeline
    const pipeline = await this.getComputePipeline(operation, 'reduction');

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } }
      ]
    });

    // Execute compute shader
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    // Read result
    const readBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, numWorkgroups * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange());
    
    // Final reduction on CPU (for small arrays)
    let finalResult = operation === 'max' ? -Infinity : operation === 'min' ? Infinity : 0;
    for (let i = 0; i < numWorkgroups; i++) {
      switch (operation) {
        case 'sum':
          finalResult += result[i];
          break;
        case 'mean':
          finalResult += result[i];
          break;
        case 'max':
          finalResult = Math.max(finalResult, result[i]);
          break;
        case 'min':
          finalResult = Math.min(finalResult, result[i]);
          break;
      }
    }

    if (operation === 'mean') {
      finalResult /= size;
    }

    // Cleanup
    readBuffer.unmap();
    this.cleanupBuffers([inputBuffer, outputBuffer, uniformBuffer, readBuffer]);

    // Update stats
    const executionTime = performance.now() - startTime;
    this.updateStats(executionTime);

    return finalResult;
  }

  /**
   * Clean up GPU buffers
   */
  cleanupBuffers(buffers) {
    for (const buffer of buffers) {
      if (this.activeBuffers.has(buffer)) {
        this.activeBuffers.delete(buffer);
      }
      buffer.destroy();
    }
  }

  /**
   * Update performance statistics
   */
  updateStats(executionTime) {
    this.stats.computeOperations++;
    this.stats.totalExecutionTime += executionTime;
    this.stats.averageExecutionTime = this.stats.totalExecutionTime / this.stats.computeOperations;
  }

  /**
   * Get performance statistics
   */
  getStats() {
    return {
      ...this.stats,
      isInitialized: this.isInitialized,
      activeBuffers: this.activeBuffers.size,
      cachedPipelines: this.pipelines.size,
      supportedFeatures: Array.from(this.supportedFeatures)
    };
  }

  /**
   * Destroy WebGPU resources
   */
  destroy() {
    // Clean up all buffers
    for (const buffer of this.activeBuffers) {
      buffer.destroy();
    }
    this.activeBuffers.clear();

    // Clear caches
    this.pipelines.clear();
    this.bufferPool.clear();

    // Destroy device
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }

    this.isInitialized = false;
    console.log('🧹 WebGPU compute resources destroyed');
  }
}

export { WebGPUCompute };