/**
 * Integration Tests for Greed.js v2.0 Modular Architecture
 * Tests all components working together end-to-end
 */

import Greed from '../../src/core/greed-v2.js';
import EventEmitter from '../../src/core/event-emitter.js';
import RuntimeManager from '../../src/core/runtime-manager.js';
import MemoryManager from '../../src/utils/memory-manager.js';
import SecurityValidator from '../../src/utils/security-validator.js';

describe('Greed.js v2.0 Modular Integration Tests', () => {
  let components;
  
  beforeEach(() => {
    components = {};
    jest.clearAllMocks();
  });
  
  afterEach(async () => {
    // Cleanup components
    for (const component of Object.values(components)) {
      if (component && typeof component.cleanup === 'function') {
        try {
          await component.cleanup();
        } catch (error) {
          // Ignore cleanup errors in tests
        }
      }
    }
    components = {};
  });

  describe('EventEmitter Integration', () => {
    test('should create EventEmitter and handle events', () => {
      const emitter = new EventEmitter();
      let eventReceived = false;

      emitter.on('test', () => {
        eventReceived = true;
      });

      emitter.emit('test');
      expect(eventReceived).toBe(true);
      
      components.emitter = emitter;
    });

    test('should handle async events', async () => {
      const emitter = new EventEmitter();
      let results = [];

      emitter.on('asyncTest', async (data) => {
        results.push(data);
      });

      await emitter.emitAsync('asyncTest', 'test-data');
      expect(results).toContain('test-data');
      
      components.emitter = emitter;
    });
  });

  describe('SecurityValidator Integration', () => {
    test('should create and validate Python code', () => {
      const validator = new SecurityValidator({
        strictMode: true,
        allowEval: false
      });

      const safeCode = 'import numpy as np\nx = np.array([1, 2, 3])';
      const result = validator.validatePythonCode(safeCode);
      
      expect(result.allowed).toBe(true);
      expect(result.riskLevel).toBe('low');
      
      components.validator = validator;
    });

    test('should reject dangerous code', () => {
      const validator = new SecurityValidator();
      
      const dangerousCode = 'import os\nos.system("rm -rf /")';
      
      try {
        validator.validatePythonCode(dangerousCode);
        expect(false).toBe(true); // Should not reach here
      } catch (error) {
        expect(error.name).toBe('SecurityError');
        expect(error.message).toContain('critical risk detected');
      }
      
      components.validator = validator;
    });
  });

  describe('MemoryManager Integration', () => {
    test('should create and track resources', async () => {
      const manager = new MemoryManager({
        maxMemoryMB: 100,
        gcThreshold: 0.8
      });

      const resource = { data: new ArrayBuffer(1024) };
      const cleanup = jest.fn();
      
      manager.register(resource, cleanup, { type: 'buffer' });
      
      const stats = manager.getStats();
      expect(stats.memoryUsage).toBeGreaterThanOrEqual(0);
      
      await manager.forceGC();
      expect(cleanup).toHaveBeenCalled();
      
      components.manager = manager;
    });
  });

  describe('RuntimeManager Integration', () => {
    test('should create and initialize runtime', async () => {
      const runtime = new RuntimeManager({
        pyodideIndexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/",
        preloadPackages: ["numpy"],
        timeout: 10000
      });

      expect(runtime.isReady).toBe(false);
      
      await runtime.initialize();
      expect(runtime.isReady).toBe(true);
      
      const status = runtime.getStatus();
      expect(status.isReady).toBe(true);
      expect(status.installedPackages).toContain('numpy');
      
      components.runtime = runtime;
    });

    test('should execute Python code', async () => {
      const runtime = new RuntimeManager();
      await runtime.initialize();
      
      const result = await runtime.runPython('2 + 2', { captureOutput: true });
      expect(result).toBe('test result'); // From mock
      
      components.runtime = runtime;
    });
  });

  describe('Full Greed.js Integration', () => {
    test('should initialize Greed with all components', async () => {
      const greed = new Greed({
        enableWebGPU: true,
        maxMemoryMB: 512,
        strictSecurity: true
      });

      expect(greed.isInitialized).toBe(false);
      
      await greed.initialize();
      expect(greed.isInitialized).toBe(true);
      
      const stats = greed.getStats();
      expect(stats.isInitialized).toBe(true);
      expect(stats.components.runtime).toBe(true);
      expect(stats.components.compute).toBe(true);
      expect(stats.components.memory).toBe(true);
      expect(stats.components.security).toBe(true);
      
      components.greed = greed;
    });

    test('should execute Python code through Greed', async () => {
      const greed = new Greed();
      await greed.initialize();
      
      const result = await greed.run('import numpy as np\nprint(np.array([1, 2, 3]))', {
        captureOutput: true
      });
      
      expect(result).toBe('test result'); // From mock
      
      const stats = greed.getStats();
      expect(stats.operations).toBe(1);
      
      components.greed = greed;
    });

    test('should handle tensor operations', async () => {
      const greed = new Greed();
      await greed.initialize();
      
      // Use Python code instead of direct tensor operations for now
      const result = await greed.run(`
import torch
a = torch.tensor([1, 2, 3])  
b = torch.tensor([4, 5, 6])
result = torch.add(a, b)
result.tolist()
      `);
      
      expect(result).toBe('test result'); // From mock
      
      components.greed = greed;
    });
  });

  describe('Error Handling Integration', () => {
    test('should handle component initialization errors', async () => {
      // Mock a failing runtime
      const originalLoadPyodide = global.loadPyodide;
      global.loadPyodide = jest.fn().mockRejectedValue(new Error('Mock initialization error'));
      
      const greed = new Greed();
      
      await expect(greed.initialize()).rejects.toThrow();
      expect(greed.isInitialized).toBe(false);
      
      // Restore mock
      global.loadPyodide = originalLoadPyodide;
      components.greed = greed;
    });

    test('should handle runtime errors gracefully', async () => {
      const greed = new Greed();
      await greed.initialize();
      
      // Mock Python execution error
      const mockPyodide = await global.loadPyodide();
      mockPyodide.runPython = jest.fn().mockRejectedValue(new Error('Python execution error'));
      
      await expect(greed.run('invalid python code')).rejects.toThrow();
      
      const stats = greed.getStats();
      expect(stats.errorCount).toBe(1);
      
      components.greed = greed;
    });
  });

  describe('Resource Cleanup Integration', () => {
    test('should clean up all resources properly', async () => {
      const greed = new Greed();
      await greed.initialize();
      
      expect(greed.isInitialized).toBe(true);
      
      await greed.destroy();
      expect(greed.isInitialized).toBe(false);
      
      components.greed = greed;
    });
  });
});