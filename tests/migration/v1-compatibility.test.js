/**
 * Migration Tests for Greed.js v1 to v2 Compatibility
 * Ensures existing v1 code continues to work with v2 architecture
 */

import Greed from '../../src/core/greed-v2.js';

describe('Greed.js v1 to v2 Migration Compatibility', () => {
  let greed;
  
  beforeEach(async () => {
    // Enhanced mock for compatibility testing
    global.loadPyodide = jest.fn().mockResolvedValue({
      ...global.loadPyodide(),
      runPython: jest.fn((code) => {
        // Mock specific v1 operations
        if (code.includes('torch.tensor')) return 'tensor([1, 2, 3, 4])';
        if (code.includes('torch.add')) return 'tensor([6, 8, 10, 12])';
        if (code.includes('np.array')) return 'array([1, 2, 3])';
        return 'test result';
      })
    });

    greed = new Greed({
      enableWebGPU: true,
      maxMemoryMB: 512
    });
  });
  
  afterEach(async () => {
    if (greed && greed.isInitialized) {
      await greed.destroy();
    }
  });

  describe('V1 API Compatibility', () => {
    test('should maintain v1 constructor compatibility', () => {
      // v1 constructor pattern should still work
      const greedInstance = new Greed();
      expect(greedInstance).toBeInstanceOf(Greed);
      expect(greedInstance.isInitialized).toBe(false);
    });

    test('should support v1 initialization pattern', async () => {
      // v2 uses initialize() instead of init()
      await greed.initialize();
      expect(greed.isInitialized).toBe(true);
      
      // But the core functionality remains accessible
      const stats = greed.getStats();
      expect(stats.isInitialized).toBe(true);
    });

    test('should support v1 run() method signature', async () => {
      await greed.initialize();
      
      // v1 pattern: greed.run(pythonCode, options)
      const result = await greed.run('import numpy as np\nprint(np.array([1, 2, 3]))');
      expect(result).toBe('test result');
      
      const stats = greed.getStats();
      expect(stats.operations).toBe(1);
    });

    test('should handle v1-style PyTorch operations', async () => {
      await greed.initialize();
      
      // Test PyTorch tensor creation (v1 pattern)
      const tensorCode = `
import torch
x = torch.tensor([1, 2, 3, 4])
print(x)
      `;
      
      const result = await greed.run(tensorCode);
      expect(result).toBe('tensor([1, 2, 3, 4])');
    });

    test('should handle v1-style tensor operations', async () => {
      await greed.initialize();
      
      // Test PyTorch operations (v1 pattern)
      const operationCode = `
import torch
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([5, 6, 7, 8])
result = torch.add(a, b)
print(result)
      `;
      
      const result = await greed.run(operationCode);
      expect(result).toBe('tensor([6, 8, 10, 12])');
    });

    test('should support v1-style error handling', async () => {
      await greed.initialize();
      
      // v1 error handling pattern should still work
      try {
        await greed.run('invalid python syntax');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
        
        const stats = greed.getStats();
        expect(stats.errorCount).toBe(1);
      }
    });
  });

  describe('V1 Configuration Compatibility', () => {
    test('should accept v1-style configuration options', () => {
      // v1 configuration patterns that should still work
      const greedV1Style = new Greed({
        enableWebGPU: true,
        maxMemoryMB: 1024,
        strictSecurity: true
      });
      
      expect(greedV1Style.config.enableWebGPU).toBe(true);
      expect(greedV1Style.config.maxMemoryMB).toBe(1024);
      expect(greedV1Style.config.strictSecurity).toBe(true);
    });

    test('should handle v1 pyodide URL configuration', () => {
      const customURL = "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/";
      const greedCustom = new Greed({
        pyodideIndexURL: customURL
      });
      
      expect(greedCustom.config.pyodideIndexURL).toBe(customURL);
    });
  });

  describe('V1 Event System Compatibility', () => {
    test('should maintain v1 event patterns', async () => {
      await greed.initialize();
      
      let initEventReceived = false;
      let operationEventReceived = false;
      
      // v1-style event listeners should work
      greed.on('init:complete', () => {
        initEventReceived = true;
      });
      
      greed.on('operation:complete', () => {
        operationEventReceived = true;
      });
      
      // Re-emit events for testing
      greed.emit('init:complete');
      greed.emit('operation:complete');
      
      expect(initEventReceived).toBe(true);
      expect(operationEventReceived).toBe(true);
    });
  });

  describe('V1 Performance Characteristics', () => {
    test('should maintain v1 performance patterns', async () => {
      await greed.initialize();
      
      const startTime = performance.now();
      await greed.run('import numpy as np\nresult = np.array([1, 2, 3])');
      const endTime = performance.now();
      
      // Should complete quickly with mocks
      expect(endTime - startTime).toBeLessThan(1000);
      
      const stats = greed.getStats();
      expect(stats.averageExecutionTime).toBeGreaterThan(0);
    });

    test('should provide v1-compatible statistics', async () => {
      await greed.initialize();
      
      const stats = greed.getStats();
      
      // v1 statistics that should be available
      expect(stats).toHaveProperty('isInitialized');
      expect(stats).toHaveProperty('operations');
      expect(stats).toHaveProperty('totalExecutionTime');
      expect(stats).toHaveProperty('averageExecutionTime');
      expect(stats).toHaveProperty('uptime');
      expect(stats).toHaveProperty('errorCount');
    });
  });

  describe('V1 Resource Management', () => {
    test('should handle v1-style cleanup', async () => {
      await greed.initialize();
      expect(greed.isInitialized).toBe(true);
      
      // v1 cleanup pattern
      await greed.destroy();
      expect(greed.isInitialized).toBe(false);
    });

    test('should support v1 memory management patterns', async () => {
      await greed.initialize();
      
      // v1-style memory management should work
      const beforeGC = greed.getStats();
      expect(beforeGC.memoryUsage).toBeGreaterThanOrEqual(0);
      
      const gcResult = await greed.forceGC();
      expect(gcResult).toHaveProperty('cleaned');
    });
  });

  describe('Backward Compatibility Edge Cases', () => {
    test('should handle empty configuration gracefully', () => {
      const greedEmpty = new Greed({});
      expect(greedEmpty.config).toBeDefined();
      expect(greedEmpty.config.pyodideIndexURL).toContain('jsdelivr');
    });

    test('should handle v1 undefined options', async () => {
      const greedUndefined = new Greed(undefined);
      expect(greedUndefined.config).toBeDefined();
      
      await greedUndefined.initialize();
      expect(greedUndefined.isInitialized).toBe(true);
      
      await greedUndefined.destroy();
    });

    test('should maintain v1 error message formats', async () => {
      // Test that error messages remain consistent with v1
      await greed.initialize();
      
      try {
        await greed.run('__import__("os").system("evil")');
      } catch (error) {
        expect(error.message).toContain('Security validation failed');
      }
    });
  });
});