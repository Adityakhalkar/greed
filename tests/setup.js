/**
 * Test setup for Greed.js v2.0
 * Configures testing environment and mocks
 */

// Mock browser globals for Node.js testing environment
global.performance = {
  now: jest.fn(() => Date.now())
};

global.console = {
  ...console,
  warn: jest.fn(),
  error: jest.fn(),
  log: jest.fn()
};

// Mock Pyodide for all tests
global.loadPyodide = jest.fn().mockResolvedValue({
  loadPackage: jest.fn().mockResolvedValue(),
  runPython: jest.fn().mockResolvedValue('test result'),
  runPythonAsync: jest.fn().mockResolvedValue('test async result'),
  globals: {
    get: jest.fn().mockReturnValue('test global'),
    set: jest.fn(),
    clear: jest.fn()
  },
  version: '0.24.1'
});

// Mock WebGPU for all tests
global.navigator = {
  gpu: {
    requestAdapter: jest.fn().mockResolvedValue({
      requestDevice: jest.fn().mockResolvedValue({
        createBuffer: jest.fn().mockReturnValue({
          size: 1024,
          usage: 1,
          destroy: jest.fn(),
          mapAsync: jest.fn().mockResolvedValue(),
          getMappedRange: jest.fn().mockReturnValue(new ArrayBuffer(1024)),
          unmap: jest.fn()
        }),
        createShaderModule: jest.fn().mockReturnValue({}),
        createBindGroupLayout: jest.fn().mockReturnValue({}),
        createPipelineLayout: jest.fn().mockReturnValue({}),
        createComputePipelineAsync: jest.fn().mockResolvedValue({
          getBindGroupLayout: jest.fn().mockReturnValue({})
        }),
        createBindGroup: jest.fn().mockReturnValue({}),
        createCommandEncoder: jest.fn().mockReturnValue({
          beginComputePass: jest.fn().mockReturnValue({
            setPipeline: jest.fn(),
            setBindGroup: jest.fn(),
            dispatchWorkgroups: jest.fn(),
            end: jest.fn()
          }),
          copyBufferToBuffer: jest.fn(),
          finish: jest.fn().mockReturnValue({})
        }),
        queue: {
          submit: jest.fn(),
          onSubmittedWorkDone: jest.fn().mockResolvedValue()
        },
        destroy: jest.fn(),
        addEventListener: jest.fn()
      }),
      features: new Set(['timestamp-query']),
      limits: { maxBufferSize: 268435456 }
    })
  },
  hardwareConcurrency: 4
};

// Mock FinalizationRegistry if not available in test environment
if (typeof FinalizationRegistry === 'undefined') {
  global.FinalizationRegistry = class MockFinalizationRegistry {
    constructor(cleanup) {
      this.cleanup = cleanup;
      this.targets = new Set();
    }
    
    register(target, heldValue, unregisterToken) {
      this.targets.add({ target, heldValue, unregisterToken });
    }
    
    unregister(unregisterToken) {
      for (const entry of this.targets) {
        if (entry.unregisterToken === unregisterToken) {
          this.targets.delete(entry);
          return true;
        }
      }
      return false;
    }
  };
}

// Mock WeakMap and WeakSet if not available
if (typeof WeakMap === 'undefined') {
  global.WeakMap = class MockWeakMap extends Map {};
}

if (typeof WeakSet === 'undefined') {
  global.WeakSet = class MockWeakSet extends Set {};
}

// Setup test timeouts
jest.setTimeout(10000); // 10 second timeout for integration tests

// Suppress console warnings in tests unless explicitly testing them
beforeEach(() => {
  jest.spyOn(console, 'warn').mockImplementation(() => {});
  jest.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  jest.restoreAllMocks();
});