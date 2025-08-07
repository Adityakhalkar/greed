/**
 * RuntimeManager - Handles Pyodide initialization and Python package management
 * Extracted from monolithic Greed class for better separation of concerns
 */
import EventEmitter from './event-emitter.js';

class RuntimeManager extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      pyodideIndexURL: config.pyodideIndexURL || 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
      preloadPackages: config.preloadPackages || ['numpy'],
      timeout: config.initTimeout || 30000,
      ...config
    };

    this.pyodide = null;
    this.isReady = false;
    this.installedPackages = new Set();
    this.initPromise = null;
  }

  /**
   * Initialize Pyodide runtime with error handling and progress tracking
   */
  async initialize() {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = this._initializeInternal();
    return this.initPromise;
  }

  async _initializeInternal() {
    try {
      this.emit('init:start', { stage: 'pyodide' });
      
      // Validate Pyodide availability
      if (typeof loadPyodide === 'undefined') {
        throw new Error('Pyodide not loaded. Please include pyodide.js in your HTML.');
      }

      // Initialize with timeout
      const pyodidePromise = loadPyodide({
        indexURL: this.config.pyodideIndexURL
      });

      this.pyodide = await Promise.race([
        pyodidePromise,
        this._createTimeoutPromise(this.config.timeout, 'Pyodide initialization timeout')
      ]);

      this.emit('init:progress', { stage: 'pyodide', status: 'loaded' });

      // Pre-load essential packages
      if (this.config.preloadPackages.length > 0) {
        this.emit('init:progress', { stage: 'packages', packages: this.config.preloadPackages });
        await this._loadPackages(this.config.preloadPackages);
      }

      this.isReady = true;
      this.emit('init:complete', { installedPackages: Array.from(this.installedPackages) });
      
      return true;
    } catch (error) {
      this.emit('init:error', { error, stage: 'initialization' });
      throw error;
    }
  }

  /**
   * Load Python packages with progress tracking
   */
  async loadPackages(packages) {
    if (!this.isReady) {
      throw new Error('Runtime not initialized. Call initialize() first.');
    }

    return this._loadPackages(packages);
  }

  async _loadPackages(packages) {
    const packagesToLoad = packages.filter(pkg => !this.installedPackages.has(pkg));
    
    if (packagesToLoad.length === 0) {
      return Array.from(this.installedPackages);
    }

    try {
      this.emit('packages:loading', { packages: packagesToLoad });
      
      await this.pyodide.loadPackage(packagesToLoad);
      
      packagesToLoad.forEach(pkg => this.installedPackages.add(pkg));
      
      this.emit('packages:loaded', { 
        loaded: packagesToLoad, 
        total: Array.from(this.installedPackages) 
      });

      return Array.from(this.installedPackages);
    } catch (error) {
      this.emit('packages:error', { error, packages: packagesToLoad });
      throw new Error(`Failed to load packages [${packagesToLoad.join(', ')}]: ${error.message}`);
    }
  }

  /**
   * Execute Python code with error handling and context isolation
   */
  async runPython(code, options = {}) {
    if (!this.isReady) {
      throw new Error('Runtime not initialized. Call initialize() first.');
    }

    const {
      captureOutput = false,
      timeout = 10000,
      globals = {},
      validateInput = true,
      bypassValidation = false,
      isInternal = false
    } = options;

    if (validateInput && !bypassValidation && !isInternal && this._containsDangerousPatterns(code)) {
      throw new SecurityError('Potentially dangerous code patterns detected');
    }

    try {
      // Set globals if provided
      for (const [key, value] of Object.entries(globals)) {
        this.pyodide.globals.set(key, value);
      }

      let result;
      if (captureOutput) {
        // Capture stdout for print statements
        const outputCode = `
import sys
from io import StringIO
_output_buffer = StringIO()
_original_stdout = sys.stdout
sys.stdout = _output_buffer

try:
${code.split('\n').map(line => '    ' + line).join('\n')}
finally:
    sys.stdout = _original_stdout
    _captured_output = _output_buffer.getvalue()
`;
        
        const executePromise = this.pyodide.runPython(outputCode);
        await Promise.race([
          executePromise,
          this._createTimeoutPromise(timeout, 'Python execution timeout')
        ]);
        
        // Get captured output
        const capturedOutput = this.pyodide.globals.get('_captured_output');
        result = { output: capturedOutput };
      } else {
        const executePromise = this.pyodide.runPythonAsync(code);
        result = await Promise.race([
          executePromise,
          this._createTimeoutPromise(timeout, 'Python execution timeout')
        ]);
      }

      return result;
    } catch (error) {
      this.emit('execution:error', { error, code: code.substring(0, 100) });
      throw error;
    }
  }

  /**
   * Get Python global variable
   */
  getGlobal(name) {
    if (!this.isReady) {
      throw new Error('Runtime not initialized');
    }
    return this.pyodide.globals.get(name);
  }

  /**
   * Set Python global variable
   */
  setGlobal(name, value) {
    if (!this.isReady) {
      throw new Error('Runtime not initialized');
    }
    this.pyodide.globals.set(name, value);
  }

  /**
   * Check if package is installed
   */
  hasPackage(packageName) {
    return this.installedPackages.has(packageName);
  }

  /**
   * Get runtime status
   */
  getStatus() {
    return {
      isReady: this.isReady,
      installedPackages: Array.from(this.installedPackages),
      pyodideVersion: this.pyodide?.version || null,
      config: this.config
    };
  }

  /**
   * Cleanup runtime resources
   */
  cleanup() {
    try {
      if (this.pyodide) {
        this.pyodide.globals.clear();
        this.pyodide = null;
      }
      
      this.isReady = false;
      this.installedPackages.clear();
      this.initPromise = null;
      
      this.emit('cleanup:complete');
    } catch (error) {
      this.emit('cleanup:error', { error });
      // Error already emitted for handling by parent components
    }
  }

  // Private helper methods
  _createTimeoutPromise(timeout, message) {
    return new Promise((_, reject) => {
      setTimeout(() => reject(new Error(message)), timeout);
    });
  }

  _containsDangerousPatterns(code) {
    const dangerousPatterns = [
      /\beval\s*\(/,
      /\bexec\s*\(/,
      /\b__import__\s*\(/,
      /\bsubprocess\./,
      /\bos\.system\s*\(/,
      /\bopen\s*\(/,
      /\bfile\s*\(/
    ];

    return dangerousPatterns.some(pattern => pattern.test(code));
  }
}

// Custom error for security violations
class SecurityError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SecurityError';
  }
}

export default RuntimeManager;
export { SecurityError };