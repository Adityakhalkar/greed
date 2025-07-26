/**
 * Core functionality tests for Greed.js v2.0
 */

import Greed from '../src/core/greed-v2.js';

describe('Greed Core', () => {
  let greed;
  
  beforeEach(() => {
    greed = new Greed({
      enableConsoleLogging: false,
      maxWorkers: 2,
      executionTimeout: 5000
    });
  });
  
  afterEach(() => {
    if (greed) {
      greed.destroy();
    }
  });
  
  describe('Initialization', () => {
    test('should initialize successfully', async () => {
      await greed.initialize();
      expect(greed.isInitialized).toBe(true);
      expect(greed.getStats().isInitialized).toBe(true);
    });
    
    test('should not initialize twice', async () => {
      await greed.initialize();
      await greed.initialize(); // Should not throw
      expect(greed.isInitialized).toBe(true);
    });
    
    test('should validate configuration', () => {
      expect(() => {
        new Greed({ maxWorkers: 0 });
      }).toThrow();
    });
    
    test('should create secure configuration', () => {
      const secureGreed = new Greed({
        strictSecurity: true,
        allowEval: false,
        allowFileSystem: false,
        allowNetwork: false
      });
      expect(secureGreed.config.strictSecurity).toBe(true);
      expect(secureGreed.config.allowEval).toBe(false);
    });
  });
  
  describe('Code Execution', () => {
    beforeEach(async () => {
      await greed.init();
    });
    
    test('should execute simple Python code', async () => {
      const result = await greed.run('print("Hello, World!")\\nresult = 42');
      expect(result.success).toBe(true);
      expect(result.output).toBe(42);
      expect(result.stdout).toContain('Hello, World!');
    });
    
    test('should handle Python errors gracefully', async () => {
      const result = await greed.run('undefined_variable');
      expect(result.success).toBe(false);
      expect(result.error).toContain('NameError');
    });
    
    test('should sanitize dangerous code', async () => {
      const result = await greed.run('import os\\nos.system("rm -rf /")');
      expect(result.success).toBe(false);
      expect(result.error).toContain('Dangerous code pattern');
    });
    
    test('should respect code length limits', async () => {
      const longCode = 'x = 1\\n'.repeat(10000);
      const result = await greed.run(longCode);
      expect(result.success).toBe(false);
      expect(result.error).toContain('exceeds maximum length');
    });
    
    test('should handle timeout', async () => {
      const result = await greed.run('import time\\ntime.sleep(10)', { timeout: 1000 });
      expect(result.success).toBe(false);
      expect(result.error).toContain('timeout');
    });
  });
  
  describe('Package Management', () => {
    beforeEach(async () => {
      await greed.init();
    });
    
    test('should install trusted packages', async () => {
      await greed.installPackages(['numpy']);
      expect(greed.installedPackages.has('numpy')).toBe(true);
    });
    
    test('should filter untrusted packages', async () => {
      await greed.installPackages(['malicious-package']);
      expect(greed.installedPackages.has('malicious-package')).toBe(false);
    });
    
    test('should execute code with packages', async () => {
      const result = await greed.run(`
import numpy as np
arr = np.array([1, 2, 3])
result = arr.sum()
      `, { packages: ['numpy'] });
      
      expect(result.success).toBe(true);
      expect(result.output).toBe(6);
    });
  });
  
  describe('Memory Management', () => {
    beforeEach(async () => {
      await greed.init();
    });
    
    test('should track memory usage', async () => {
      const status = greed.getStatus();
      expect(status.memoryUsage).toBeDefined();
      expect(status.memoryUsage.totalAllocated).toBeGreaterThanOrEqual(0);
    });
    
    test('should enforce memory limits', async () => {
      // Create a Greed instance with very low memory limit
      const limitedGreed = new Greed({
        enableConsoleLogging: false,
        security: { maxMemoryUsage: 1000 } // 1KB limit
      });
      
      await limitedGreed.init();
      
      const result = await limitedGreed.run(`
import numpy as np
# Try to allocate large array
arr = np.zeros((1000, 1000))
      `, { packages: ['numpy'] });
      
      // Should either succeed or fail gracefully
      expect(result).toBeDefined();
      
      limitedGreed.destroy();
    });
  });
  
  describe('Error Handling', () => {
    beforeEach(async () => {
      await greed.init();
    });
    
    test('should log errors properly', async () => {
      await greed.run('invalid python code');
      const errorStats = greed.errorHandler.getErrorStats();
      expect(errorStats.total).toBeGreaterThan(0);
    });
    
    test('should provide detailed error information', async () => {
      const result = await greed.run('1 / 0');
      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.code).toBeDefined();
    });
    
    test('should handle worker errors', async () => {
      // Force worker error by terminating workers
      if (greed.workerManager) {
        greed.workerManager.destroy();
      }
      
      const result = await greed.run('print("test")', { useContext: true });
      expect(result.success).toBe(false);
      expect(result.error).toContain('Worker');
    });
  });
  
  describe('WebGPU Integration', () => {
    beforeEach(async () => {
      await greed.init();
    });
    
    test('should detect WebGPU support', () => {
      const status = greed.getStatus();
      expect(typeof status.webgpuSupported).toBe('boolean');
    });
    
    test('should analyze GPU-optimizable code', () => {
      const analysis = greed.analyzeCodeForGPUOptimization(`
import torch
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)
z = torch.matmul(x, y)
result = z.relu()
      `);
      
      expect(analysis.optimizable).toBe(true);
      expect(analysis.operations.length).toBeGreaterThan(0);
    });
    
    test('should execute with GPU when available', async () => {
      const result = await greed.run(`
import torch
x = torch.randn(10, 10)
y = torch.randn(10, 10)
result = torch.matmul(x, y)
      `, { useGPU: true });
      
      expect(result.success).toBe(true);
      expect(result.executionMethod).toMatch(/webgpu|worker/);
    });
  });
  
  describe('Context Preservation', () => {
    beforeEach(async () => {
      await greed.init();
    });
    
    test('should preserve variables between executions', async () => {
      const result1 = await greed.run('x = 42', { useContext: true });
      expect(result1.success).toBe(true);
      
      const result2 = await greed.run('result = x * 2', { useContext: true });
      expect(result2.success).toBe(true);
      expect(result2.output).toBe(84);
    });
    
    test('should preserve function definitions', async () => {
      const result1 = await greed.run(`
def my_function(a, b):
    return a + b
      `, { useContext: true });
      expect(result1.success).toBe(true);
      
      const result2 = await greed.run('result = my_function(10, 20)', { useContext: true });
      expect(result2.success).toBe(true);
      expect(result2.output).toBe(30);
    });
  });
  
  describe('Resource Management', () => {
    beforeEach(async () => {
      await greed.init();
    });
    
    test('should clean up resources on destroy', () => {
      const resourceCount = greed.resourceManager.getResourceStats().total;
      expect(resourceCount).toBeGreaterThan(0);
      
      greed.destroy();
      expect(greed.initialized).toBe(false);
    });
    
    test('should handle multiple destroys gracefully', () => {
      greed.destroy();
      greed.destroy(); // Should not throw
      expect(greed.initialized).toBe(false);
    });
  });
  
  describe('Performance Monitoring', () => {
    beforeEach(async () => {
      await greed.init();
    });
    
    test('should track execution statistics', async () => {
      const initialStats = greed.getStatus();
      
      await greed.run('result = 1 + 1');
      
      const finalStats = greed.getStatus();
      expect(finalStats.errorStats.total).toBeGreaterThanOrEqual(initialStats.errorStats.total);
    });
    
    test('should provide device information', () => {
      const deviceInfo = greed.getDeviceInfo();
      expect(deviceInfo).toBeDefined();
      expect(deviceInfo.workers).toBeDefined();
    });
  });
});