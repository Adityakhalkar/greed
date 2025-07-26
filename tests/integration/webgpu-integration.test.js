/**
 * WebGPU Integration Tests
 * Tests the complete WebGPU tensor operations and PyTorch compatibility
 */

const { JSDOM } = require('jsdom');

// Mock WebGPU API for testing
global.navigator = {
  gpu: {
    requestAdapter: async () => ({
      features: new Set(['timestamp-query']),
      limits: {
        maxComputeWorkgroupSizeX: 256,
        maxBufferSize: 256 * 1024 * 1024
      },
      requestDevice: async () => ({
        createShaderModule: jest.fn(),
        createBindGroupLayout: jest.fn(),
        createPipelineLayout: jest.fn(),
        createComputePipelineAsync: jest.fn(),
        createBuffer: jest.fn(),
        createCommandEncoder: jest.fn(),
        queue: {
          submit: jest.fn(),
          onSubmittedWorkDone: jest.fn()
        },
        addEventListener: jest.fn(),
        destroy: jest.fn()
      })
    })
  }
};

global.GPUShaderStage = {
  COMPUTE: 4
};

global.GPUBufferUsage = {
  STORAGE: 128,
  UNIFORM: 64,
  COPY_SRC: 4,
  COPY_DST: 8,
  MAP_READ: 1
};

global.GPUMapMode = {
  READ: 1
};

describe('WebGPU Integration Tests', () => {
  let Greed;
  let greedInstance;

  beforeAll(async () => {
    // Set up DOM environment
    const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
      url: 'http://localhost',
      pretendToBeVisual: true,
      resources: 'usable'
    });
    
    global.window = dom.window;
    global.document = dom.window.document;
    global.performance = dom.window.performance;

    // Mock pyodide
    global.loadPyodide = jest.fn().mockResolvedValue({
      runPython: jest.fn(),
      globals: {
        get: jest.fn()
      }
    });

    // Import Greed after setting up mocks
    const GreedModule = await import('../../src/core/greed-v2.js');
    Greed = GreedModule.default;
  });

  beforeEach(async () => {
    greedInstance = new Greed({
      enableWebGPU: true,
      enableWorkers: false,
      pyodideIndexURL: 'mock://pyodide',
      maxMemoryMB: 256
    });
  });

  afterEach(async () => {
    if (greedInstance) {
      await greedInstance.destroy();
    }
  });

  describe('WebGPU Tensor Operations', () => {
    test('should initialize WebGPU compute engine', async () => {
      await greedInstance.initialize();
      
      expect(greedInstance.isInitialized).toBe(true);
      expect(greedInstance.compute.availableStrategies.has('webgpu')).toBe(true);
      expect(greedInstance.tensorBridge).toBeDefined();
    });

    test('should create WebGPU tensors', async () => {
      await greedInstance.initialize();
      
      const tensorData = [1, 2, 3, 4];
      const tensorInfo = greedInstance.tensorBridge.createWebGPUTensor(
        tensorData, 
        [2, 2], 
        'float32', 
        'webgpu'
      );

      expect(tensorInfo.id).toMatch(/webgpu_tensor_\d+/);
      expect(tensorInfo.shape).toEqual([2, 2]);
      expect(tensorInfo.dtype).toBe('float32');
      expect(tensorInfo.device).toBe('webgpu');
    });

    test('should execute tensor addition', async () => {
      await greedInstance.initialize();
      
      const tensor1Info = greedInstance.tensorBridge.createWebGPUTensor([1, 2, 3, 4], [2, 2]);
      const tensor2Info = greedInstance.tensorBridge.createWebGPUTensor([5, 6, 7, 8], [2, 2]);

      // Mock successful GPU operation
      greedInstance.compute.webgpu.execute = jest.fn().mockResolvedValue(new Float32Array([6, 8, 10, 12]));

      const result = await greedInstance.tensorBridge.executeOperation(
        tensor1Info.id, 
        'add', 
        tensor2Info.id
      );

      expect(result.success).toBe(true);
      expect(result.data).toEqual([6, 8, 10, 12]);
      expect(result.shape).toEqual([2, 2]);
    });

    test('should handle matrix multiplication', async () => {
      await greedInstance.initialize();
      
      const matrixA = greedInstance.tensorBridge.createWebGPUTensor([1, 2, 3, 4], [2, 2]);
      const matrixB = greedInstance.tensorBridge.createWebGPUTensor([5, 6, 7, 8], [2, 2]);

      // Mock matmul result: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
      greedInstance.compute.webgpu.execute = jest.fn().mockResolvedValue(new Float32Array([19, 22, 43, 50]));

      const result = await greedInstance.tensorBridge.executeOperation(
        matrixA.id, 
        'matmul', 
        matrixB.id
      );

      expect(result.success).toBe(true);
      expect(result.data).toEqual([19, 22, 43, 50]);
      expect(result.shape).toEqual([2, 2]);
    });

    test('should execute activation functions', async () => {
      await greedInstance.initialize();
      
      const tensor = greedInstance.tensorBridge.createWebGPUTensor([-1, 0, 1, 2], [4]);

      // Mock ReLU result
      greedInstance.compute.webgpu.execute = jest.fn().mockResolvedValue(new Float32Array([0, 0, 1, 2]));

      const result = await greedInstance.tensorBridge.executeOperation(tensor.id, 'relu');

      expect(result.success).toBe(true);
      expect(result.data).toEqual([0, 0, 1, 2]);
    });

    test('should handle GPU operation failures gracefully', async () => {
      await greedInstance.initialize();
      
      const tensor = greedInstance.tensorBridge.createWebGPUTensor([1, 2, 3, 4], [2, 2]);

      // Mock GPU failure
      greedInstance.compute.webgpu.execute = jest.fn().mockRejectedValue(new Error('GPU operation failed'));

      const result = await greedInstance.tensorBridge.executeOperation(tensor.id, 'relu');

      expect(result.success).toBe(false);
      expect(result.error).toBe('GPU operation failed');
    });
  });

  describe('PyTorch API Integration', () => {
    test('should run PyTorch-like operations', async () => {
      await greedInstance.initialize();
      
      // Mock Python execution
      greedInstance.runtime.runPython = jest.fn().mockResolvedValue({
        output: 'tensor([[6, 8], [10, 12]])',
        success: true
      });

      const result = await greedInstance.run(`
import torch
a = torch.tensor([[1, 2], [3, 4]], device='webgpu')
b = torch.tensor([[5, 6], [7, 8]], device='webgpu')
c = a + b
print(c)
      `);

      expect(result.success).toBe(true);
      expect(greedInstance.runtime.runPython).toHaveBeenCalled();
    });

    test('should handle tensor device transfers', async () => {
      await greedInstance.initialize();
      
      greedInstance.runtime.runPython = jest.fn().mockResolvedValue({
        output: 'tensor([1, 2, 3, 4], device=\'webgpu\')',
        success: true
      });

      const result = await greedInstance.run(`
import torch
tensor_cpu = torch.tensor([1, 2, 3, 4])
tensor_gpu = tensor_cpu.cuda()  # Maps to WebGPU
print(tensor_gpu)
      `);

      expect(result.success).toBe(true);
    });
  });

  describe('Performance and Memory Management', () => {
    test('should track tensor memory usage', async () => {
      await greedInstance.initialize();
      
      const tensor1 = greedInstance.tensorBridge.createWebGPUTensor(new Float32Array(1000), [1000]);
      const tensor2 = greedInstance.tensorBridge.createWebGPUTensor(new Float32Array(2000), [2000]);

      const stats = greedInstance.tensorBridge.getStats();
      
      expect(stats.tensorCount).toBe(2);
      expect(stats.totalMemory).toBe(3000 * 4); // 3000 floats * 4 bytes each
      expect(stats.deviceDistribution.webgpu).toBe(2);
    });

    test('should cleanup tensors properly', async () => {
      await greedInstance.initialize();
      
      const tensor = greedInstance.tensorBridge.createWebGPUTensor([1, 2, 3, 4], [4]);
      
      expect(greedInstance.tensorBridge.getStats().tensorCount).toBe(1);
      
      greedInstance.tensorBridge.releaseTensor(tensor.id);
      
      expect(greedInstance.tensorBridge.getStats().tensorCount).toBe(0);
    });

    test('should handle memory pressure gracefully', async () => {
      // Create instance with low memory limit
      const lowMemoryInstance = new Greed({
        enableWebGPU: true,
        maxMemoryMB: 16 // Very low limit
      });

      await lowMemoryInstance.initialize();
      
      // Try to create large tensors
      const largeTensorCreation = () => {
        return lowMemoryInstance.tensorBridge.createWebGPUTensor(
          new Float32Array(10000000), // ~40MB tensor
          [10000000]
        );
      };

      // Should either succeed with compression or fail gracefully
      expect(() => largeTensorCreation()).not.toThrow();
      
      await lowMemoryInstance.destroy();
    });
  });

  describe('WebGPU Shader Operations', () => {
    test('should compile shaders for supported operations', async () => {
      await greedInstance.initialize();
      
      const supportedOps = ['add', 'mul', 'matmul', 'relu', 'sigmoid', 'softmax'];
      
      for (const op of supportedOps) {
        const pipelineCache = greedInstance.compute.webgpu.pipelineCache;
        
        // Mock shader compilation
        const shader = pipelineCache.generateShader(op, {
          workgroupSize: [64, 1, 1],
          dataType: 'f32'
        });
        
        expect(shader).toContain('@compute');
        expect(shader).toContain('main');
        expect(shader).toContain(op === 'add' ? '+' : op === 'mul' ? '*' : 'fn main');
      }
    });

    test('should optimize workgroup sizes for different operations', async () => {
      await greedInstance.initialize();
      
      const pipelineCache = greedInstance.compute.webgpu.pipelineCache;
      
      const matmulSize = pipelineCache.getOptimalWorkgroupSize('matmul', [100, 100], { maxComputeWorkgroupSizeX: 256 });
      const elemwiseSize = pipelineCache.getOptimalWorkgroupSize('add', [1000], { maxComputeWorkgroupSizeX: 256 });
      
      expect(matmulSize).toEqual([16, 16, 1]); // 2D workgroup for matrix ops
      expect(elemwiseSize).toEqual([64, 1, 1]); // 1D workgroup for element-wise ops
    });
  });

  describe('Error Handling and Fallbacks', () => {
    test('should fallback to CPU when WebGPU fails', async () => {
      // Mock WebGPU initialization failure
      global.navigator.gpu.requestAdapter = jest.fn().mockResolvedValue(null);
      
      const fallbackInstance = new Greed({
        enableWebGPU: true,
        enableAutoFallback: true
      });

      await fallbackInstance.initialize();
      
      expect(fallbackInstance.compute.availableStrategies.has('webgpu')).toBe(false);
      expect(fallbackInstance.compute.availableStrategies.has('cpu')).toBe(true);
      
      await fallbackInstance.destroy();
    });

    test('should handle shader compilation errors', async () => {
      await greedInstance.initialize();
      
      // Mock shader compilation failure
      greedInstance.compute.webgpu.device.createShaderModule = jest.fn().mockImplementation(() => {
        throw new Error('Shader compilation failed');
      });

      const tensor = greedInstance.tensorBridge.createWebGPUTensor([1, 2, 3, 4], [4]);
      
      const result = await greedInstance.tensorBridge.executeOperation(tensor.id, 'relu');
      
      // Should fallback to CPU
      expect(result.success).toBe(false);
      expect(result.error).toContain('failed');
    });
  });
});

module.exports = {};