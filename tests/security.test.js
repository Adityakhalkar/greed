/**
 * Security tests for Greed.js v2.0
 */

import SecurityValidator from '../src/utils/security-validator.js';
import Greed from '../src/core/greed-v2.js';

describe('Security', () => {
  describe('SecurityValidator', () => {
    let validator;
    
    beforeEach(() => {
      validator = new SecurityValidator();
    });
    
    test('should validate Python code correctly', () => {
      const safeCode = 'import numpy as np\nx = np.array([1, 2, 3])';
      const result = validator.validatePythonCode(safeCode);
      expect(result.allowed).toBe(true);
      
      const dangerousCode = 'import os\nos.system("rm -rf /")';
      const dangerousResult = validator.validatePythonCode(dangerousCode);
      expect(dangerousResult.allowed).toBe(false);
    });
    
    test('should validate Pyodide versions', () => {
      expect(SecurityConfig.validatePyodideVersion('0.26.2')).toBe('0.26.2');
      
      expect(() => {
        SecurityConfig.validatePyodideVersion('invalid-version');
      }).toThrow('Invalid Pyodide version format');
    });
    
    test('should sanitize code input', () => {
      const safeCode = 'print("Hello, World!")';
      expect(SecurityConfig.sanitizeCode(safeCode)).toBe(safeCode);
      
      expect(() => {
        SecurityConfig.sanitizeCode(123);
      }).toThrow('Code must be a string');
      
      expect(() => {
        SecurityConfig.sanitizeCode('x' * 100001);
      }).toThrow('Code exceeds maximum length limit');
    });
    
    test('should detect dangerous code patterns', () => {
      const dangerousCode = 'import os\\nos.system("rm -rf /")';
      
      // Should not throw but should warn
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      SecurityConfig.sanitizeCode(dangerousCode);
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Potentially dangerous code pattern detected')
      );
      consoleSpy.mockRestore();
    });
    
    test('should create secure worker blobs', () => {
      const workerScript = 'console.log("test");';
      const blob = SecurityConfig.createSecureWorkerBlob(workerScript);
      expect(blob).toBeInstanceOf(Blob);
      expect(blob.type).toBe('application/javascript');
    });
    
    test('should generate CSP headers', () => {
      const csp = SecurityConfig.generateCSPHeader();
      expect(csp).toContain("default-src 'self'");
      expect(csp).toContain("script-src 'self' 'unsafe-eval'");
      expect(csp).toContain("worker-src 'self' blob:");
    });
  });
  
  describe('Secure Execution', () => {
    let greed;
    
    beforeEach(async () => {
      greed = Greed.createSecure({
        enableConsoleLogging: false
      });
      await greed.init();
    });
    
    afterEach(() => {
      if (greed) {
        greed.destroy();
      }
    });
    
    test('should block dangerous imports', async () => {
      const result = await greed.run('import os\\nos.system("echo hacked")');
      expect(result.success).toBe(false);
      expect(result.error).toContain('Dangerous code pattern detected');
    });
    
    test('should block eval usage', async () => {
      try {
        await greed.run('eval("print(\'hacked\')")');
        expect(false).toBe(true); // Should not reach here
      } catch (error) {
        expect(error.message).toContain('Security validation failed');
      }
    });
    
    test('should block exec usage', async () => {
      const result = await greed.run('exec("print(\\'hacked\\')")');
      expect(result.success).toBe(false);
      expect(result.error).toContain('Dangerous code pattern detected');
    });
    
    test('should block __import__ usage', async () => {
      const result = await greed.run('__import__("os").system("echo hacked")');
      expect(result.success).toBe(false);
      expect(result.error).toContain('Dangerous code pattern detected');
    });
    
    test('should block subprocess import', async () => {
      const result = await greed.run('import subprocess\\nsubprocess.run(["echo", "hacked"])');
      expect(result.success).toBe(false);
      expect(result.error).toContain('Dangerous code pattern detected');
    });
    
    test('should allow safe numpy operations', async () => {
      const result = await greed.run(`
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
result = arr.sum()
      `, { packages: ['numpy'] });
      
      expect(result.success).toBe(true);
      expect(result.output).toBe(15);
    });
    
    test('should enforce package whitelist', async () => {
      await greed.installPackages(['numpy', 'malicious-package']);
      expect(greed.installedPackages.has('numpy')).toBe(true);
      expect(greed.installedPackages.has('malicious-package')).toBe(false);
    });
    
    test('should validate input parameters', () => {
      expect(() => {
        greed.errorHandler.validateInput({}, {
          required_field: { type: 'string' }
        });
      }).toThrow('Missing required field');
      
      expect(() => {
        greed.errorHandler.validateInput({ field: 123 }, {
          field: { type: 'string' }
        });
      }).toThrow('Invalid type');
    });
  });
  
  describe('Memory Protection', () => {
    let greed;
    
    beforeEach(async () => {
      greed = new Greed({
        enableConsoleLogging: false,
        security: {
          maxMemoryUsage: 10 * 1024 * 1024 // 10MB limit
        }
      });
      await greed.init();
    });
    
    afterEach(() => {
      if (greed) {
        greed.destroy();
      }
    });
    
    test('should enforce memory limits', () => {
      const memoryManager = greed.memoryManager;
      
      // Should succeed within limit
      expect(() => {
        memoryManager.allocate('test1', 1024, 'buffer');
      }).not.toThrow();
      
      // Should fail when exceeding limit
      expect(() => {
        memoryManager.allocate('test2', 20 * 1024 * 1024, 'buffer');
      }).toThrow('Memory limit exceeded');
    });
    
    test('should perform garbage collection', async () => {
      const memoryManager = greed.memoryManager;
      
      // Allocate some memory
      memoryManager.allocate('temp1', 1024, 'buffer');
      memoryManager.allocate('temp2', 1024, 'buffer');
      
      expect(memoryManager.getMemoryUsage().allocations).toBe(2);
      
      // Simulate time passing
      const allocation = memoryManager.allocations.get('temp1');
      allocation.lastAccessed = Date.now() - 400000; // 6+ minutes ago
      
      // Force garbage collection
      memoryManager.performGarbageCollection();
      
      expect(memoryManager.getMemoryUsage().allocations).toBe(1);
    });
  });
  
  describe('Worker Security', () => {
    let greed;
    
    beforeEach(async () => {
      greed = Greed.createSecure({
        enableConsoleLogging: false,
        maxWorkers: 2
      });
      await greed.init();
    });
    
    afterEach(() => {
      if (greed) {
        greed.destroy();
      }
    });
    
    test('should create secure worker blobs', () => {
      const workerManager = greed.workerManager;
      expect(workerManager.workerBlobURLs.size).toBeGreaterThan(0);
      
      // All blob URLs should be properly tracked
      for (const url of workerManager.workerBlobURLs) {
        expect(url).toMatch(/^blob:/);
      }
    });
    
    test('should clean up blob URLs on destroy', () => {
      const workerManager = greed.workerManager;
      const blobCount = workerManager.workerBlobURLs.size;
      expect(blobCount).toBeGreaterThan(0);
      
      greed.destroy();
      expect(workerManager.workerBlobURLs.size).toBe(0);
    });
    
    test('should handle worker errors securely', async () => {
      // Test worker error handling
      const result = await greed.run('invalid python syntax', { useContext: false });
      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      
      // Error should not leak sensitive information
      expect(result.error).not.toContain('file://');
      expect(result.error).not.toContain('localhost');
    });
  });
  
  describe('Input Validation', () => {
    let greed;
    
    beforeEach(async () => {
      greed = Greed.createSecure({
        enableConsoleLogging: false
      });
      await greed.init();
    });
    
    afterEach(() => {
      if (greed) {
        greed.destroy();
      }
    });
    
    test('should validate code input', async () => {
      // Null/undefined code
      let result = await greed.run(null);
      expect(result.success).toBe(false);
      
      result = await greed.run(undefined);
      expect(result.success).toBe(false);
      
      // Non-string code
      result = await greed.run(123);
      expect(result.success).toBe(false);
      
      // Empty string should be valid
      result = await greed.run('');
      expect(result.success).toBe(true);
    });
    
    test('should validate options', async () => {
      // Invalid package format
      const result = await greed.run('print("test")', { packages: 'invalid' });
      expect(result.success).toBe(true); // Should handle gracefully
    });
    
    test('should handle malformed execution options', async () => {
      const result = await greed.run('print("test")', {
        useGPU: 'invalid',
        timeout: 'invalid',
        packages: { invalid: 'format' }
      });
      
      expect(result.success).toBe(true); // Should handle gracefully
    });
  });
  
  describe('Error Information Security', () => {
    let greed;
    
    beforeEach(async () => {
      greed = Greed.createSecure({
        enableConsoleLogging: false
      });
      await greed.init();
    });
    
    afterEach(() => {
      if (greed) {
        greed.destroy();
      }
    });
    
    test('should not leak sensitive paths in errors', async () => {
      const result = await greed.run('import nonexistent_module');
      expect(result.success).toBe(false);
      expect(result.error).not.toContain('/home/');
      expect(result.error).not.toContain('/Users/');
      expect(result.error).not.toContain('C:\\\\');
    });
    
    test('should sanitize stack traces', async () => {
      const result = await greed.run('raise Exception("test error")');
      expect(result.success).toBe(false);
      expect(result.error).toContain('test error');
      // Should not contain full file paths
      expect(result.error).not.toMatch(/\\/.*\\/.*\\//);
    });
  });
});