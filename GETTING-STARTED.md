# Greed.js - Getting Started

WebGPU-accelerated PyTorch runtime for browsers with zero dependencies.

## Quick Install

### CDN (Recommended)
```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
    <!-- For production use -->
    <script src="https://unpkg.com/greed.js@latest/dist/greed.min.js"></script>
    <!-- For development/testing -->
    <!-- <script src="https://unpkg.com/greed.js@latest/dist/greed-standalone.js"></script> -->
</head>
<body>
    <script>
        async function main() {
            const greed = new Greed({ enableWebGPU: true });
            await greed.initialize();
            
            const result = await greed.run(`
import torch
x = torch.tensor([1, 2, 3, 4, 5])
y = x * 2 + 1
print(f"Result: {y}")
            `);
            
            console.log(result.output);
        }
        main();
    </script>
</body>
</html>
```

### NPM
```bash
npm install greed.js
```

```javascript
import Greed from 'greed.js';

const greed = new Greed({ enableWebGPU: true });
await greed.initialize();

const result = await greed.run(`
import torch
tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
result = tensor @ tensor.T
print(result)
`);

console.log(result.output);
```

### ES6 Module Import
```javascript
// For modern bundlers (Webpack, Vite, etc.)
import Greed from 'greed.js';

// For Node.js (with experimental ES modules)
import Greed from 'greed.js/dist/greed.js';
```

### CommonJS (Node.js)
```javascript
const { Greed } = require('greed.js');
```

## WebGPU Setup

### Chrome/Edge
1. Go to: `chrome://flags/#enable-unsafe-webgpu`
2. Set: **Enabled**
3. Restart browser

### Brave Browser
1. Go to: `brave://flags/#enable-unsafe-webgpu`
2. Set: **Enabled**
3. Restart browser

## Basic Configuration

```javascript
const greed = new Greed({
    // Core settings
    enableWebGPU: true,
    enableWorkers: true,
    maxWorkers: navigator.hardwareConcurrency || 4,
    
    // Security settings
    strictSecurity: true,
    allowEval: false,
    allowFileSystem: false,
    allowNetwork: false,
    
    // Performance settings
    maxMemoryMB: 1024,
    gcThreshold: 0.8,
    enableProfiling: true,
    
    // Runtime settings
    pyodideIndexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
    preloadPackages: ['numpy'],
    initTimeout: 30000
});
```

### Development vs Production Configuration

```javascript
// Development configuration
const devConfig = {
    enableWebGPU: true,
    enableProfiling: true,
    strictSecurity: false,  // More permissive for testing
    maxMemoryMB: 512,
    preloadPackages: ['numpy', 'torch']  // Preload for faster development
};

// Production configuration
const prodConfig = {
    enableWebGPU: true,
    enableProfiling: false,  // Disable profiling in production
    strictSecurity: true,    // Strict security in production
    maxMemoryMB: 1024,
    gcThreshold: 0.8,        // Aggressive garbage collection
    preloadPackages: ['numpy']  // Only essential packages
};

const greed = new Greed(process.env.NODE_ENV === 'development' ? devConfig : prodConfig);
```

## Example: Neural Network

```python
import torch
import torch.nn as nn

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
    
    def forward(self, x):
        return self.linear(x)

# Create and use model
model = SimpleNet()
input_data = torch.randn(1, 3)
output = model(input_data)
print(f"Output: {output}")
```

## Real-World Project Implementation

### Setting Up Greed.js in Your Project

#### 1. Project Structure
```
my-ml-app/
├── src/
│   ├── greed/
│   │   ├── greed-worker.js
│   │   ├── hooks.js
│   │   └── models/
│   ├── components/
│   └── utils/
├── public/
└── package.json
```

#### 2. Creating a Greed.js Hook (React Example)

```javascript
// src/greed/hooks.js
import { useState, useCallback, useRef, useEffect } from 'react';
import Greed from 'greed.js';

export function useGreed(config = {}) {
    const [isInitialized, setIsInitialized] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const greedRef = useRef(null);

    const initialize = useCallback(async () => {
        if (greedRef.current) return greedRef.current;
        
        setIsLoading(true);
        setError(null);
        
        try {
            const greed = new Greed({
                enableWebGPU: true,
                enableProfiling: process.env.NODE_ENV === 'development',
                strictSecurity: process.env.NODE_ENV === 'production',
                maxMemoryMB: 1024,
                ...config
            });
            
            await greed.initialize();
            greedRef.current = greed;
            setIsInitialized(true);
            
            return greed;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setIsLoading(false);
        }
    }, [config]);

    const runCode = useCallback(async (pythonCode, options = {}) => {
        if (!greedRef.current) {
            throw new Error('Greed not initialized. Call initialize() first.');
        }
        
        setIsLoading(true);
        setError(null);
        
        try {
            const result = await greedRef.current.run(pythonCode, options);
            return result;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setIsLoading(false);
        }
    }, []);

    const runTensorOperation = useCallback(async (operation, tensors, options = {}) => {
        if (!greedRef.current) {
            throw new Error('Greed not initialized. Call initialize() first.');
        }
        
        setIsLoading(true);
        setError(null);
        
        try {
            const result = await greedRef.current.tensor(operation, tensors, options);
            return result;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setIsLoading(false);
        }
    }, []);

    const loadPackages = useCallback(async (packages) => {
        if (!greedRef.current) {
            throw new Error('Greed not initialized. Call initialize() first.');
        }
        
        try {
            return await greedRef.current.loadPackages(packages);
        } catch (err) {
            setError(err.message);
            throw err;
        }
    }, []);

    const getStats = useCallback(() => {
        return greedRef.current?.getStats() || null;
    }, []);

    const forceGC = useCallback(async () => {
        if (!greedRef.current) return { cleaned: 0 };
        try {
            return await greedRef.current.forceGC();
        } catch (err) {
            setError(err.message);
            throw err;
        }
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (greedRef.current) {
                greedRef.current.destroy().catch(console.error);
            }
        };
    }, []);

    return {
        initialize,
        runCode,
        runTensorOperation,
        loadPackages,
        getStats,
        forceGC,
        isInitialized,
        isLoading,
        error,
        greed: greedRef.current
    };
}

// Hook for tensor operations
export function useTensor() {
    const { runCode, isInitialized } = useGreed();

    const createTensor = useCallback(async (data, dtype = 'float32') => {
        const code = `
import torch
import numpy as np

data = ${JSON.stringify(data)}
tensor = torch.tensor(data, dtype=torch.${dtype})
print(f"Tensor shape: {tensor.shape}")
print(f"Tensor device: {tensor.device}")
print(f"Tensor: {tensor}")
tensor
        `;
        return await runCode(code);
    }, [runCode]);

    const matrixMultiply = useCallback(async (a, b) => {
        const code = `
import torch

a = torch.tensor(${JSON.stringify(a)}, dtype=torch.float32)
b = torch.tensor(${JSON.stringify(b)}, dtype=torch.float32)
result = a @ b
print(f"Result shape: {result.shape}")
print(f"Result: {result}")
result
        `;
        return await runCode(code);
    }, [runCode]);

    return {
        createTensor,
        matrixMultiply,
        isInitialized
    };
}
```

#### 3. Using the Hook in a React Component

```javascript
// src/components/MLPlayground.jsx
import React, { useEffect, useState } from 'react';
import { useGreed, useTensor } from '../greed/hooks';

export default function MLPlayground() {
    const { initialize, runCode, getStats, isInitialized, isLoading, error, forceGC } = useGreed();
    const [output, setOutput] = useState('');
    const [stats, setStats] = useState(null);

    useEffect(() => {
        initialize().then(() => {
            setStats(getStats());
        });
    }, [initialize, getStats]);

    const runExample = async () => {
        try {
            const result = await runCode(`
import torch
import torch.nn as nn

# Create a simple neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model and sample data
model = Net()
input_data = torch.randn(5, 4)  # Batch of 5, features of 4

# Forward pass
output = model(input_data)
print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output.shape}")
print(f"WebGPU Device: {input_data.device}")
print(f"Model output: {output}")
            `);
            setOutput(result.output);
        } catch (err) {
            setOutput(`Error: ${err.message}`);
        }
    };

    const runMatrixExample = async () => {
        try {
            const result = await runCode(`
import torch
import numpy as np

# Create matrices
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
b = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float32)

# Matrix multiplication
result = torch.matmul(a, b)
print(f"Matrix A shape: {a.shape}")
print(f"Matrix B shape: {b.shape}")
print(f"Result shape: {result.shape}")
print(f"Result: {result}")
print(f"Device: {a.device}")
            `);
            setOutput(result.output);
        } catch (err) {
            setOutput(`Error: ${err.message}`);
        }
    };

    const runPerformanceTest = async () => {
        try {
            const result = await runCode(`
import torch
import time

# Performance test with WebGPU
start_time = time.time()

# Create large tensors for GPU utilization
x = torch.randn(1000, 1000, dtype=torch.float32)
y = torch.randn(1000, 1000, dtype=torch.float32)

# Matrix operations
z = torch.matmul(x, y)
result = torch.sum(z)

end_time = time.time()
print(f"Operation completed in: {end_time - start_time:.4f} seconds")
print(f"Result: {result}")
print(f"Device: {x.device}")
print(f"Tensor shape: {z.shape}")
            `);
            setOutput(result.output);
        } catch (err) {
            setOutput(`Error: ${err.message}`);
        }
    };

    const runMemoryTest = async () => {
        try {
            // Get memory stats before
            const statsBefore = getStats();
            
            // Force garbage collection
            await forceGC();
            
            // Get memory stats after
            const statsAfter = getStats();
            
            setOutput(`Memory Management Test:
Before GC: ${statsBefore?.memory?.memoryUsageMB || 'N/A'} MB
After GC: ${statsAfter?.memory?.memoryUsageMB || 'N/A'} MB
Operations: ${statsAfter?.operations || 0}
Average execution time: ${statsAfter?.averageExecutionTime?.toFixed(2) || 'N/A'} ms`);
        } catch (err) {
            setOutput(`Error: ${err.message}`);
        }
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'monospace' }}>
            <h2>Greed.js ML Playground</h2>
            
            {error && (
                <div style={{ color: 'red', marginBottom: '10px' }}>
                    Error: {error}
                </div>
            )}
            
            <div style={{ marginBottom: '20px' }}>
                <strong>Status:</strong> {isInitialized ? '✅ Ready' : '⏳ Initializing...'}
                {isLoading && ' (Running...)'}
            </div>

            {stats && (
                <div style={{ marginBottom: '20px', background: '#f5f5f5', padding: '10px' }}>
                    <strong>WebGPU Status:</strong> {stats.compute?.availableStrategies?.includes('webgpu') ? '✅ Enabled' : '❌ Disabled'}
                    <br />
                    <strong>Backend:</strong> {stats.compute?.backend || 'N/A'}
                    <br />
                    <strong>Memory Usage:</strong> {stats.memory?.memoryUsageMB?.toFixed(2) || 'N/A'} MB
                    <br />
                    <strong>Operations:</strong> {stats.operations || 0}
                    <br />
                    <strong>Avg Execution Time:</strong> {stats.averageExecutionTime?.toFixed(2) || 'N/A'} ms
                </div>
            )}

            <div style={{ marginBottom: '20px' }}>
                <button onClick={runExample} disabled={!isInitialized || isLoading}>
                    Run Neural Network Example
                </button>
                <button onClick={runMatrixExample} disabled={!isInitialized || isLoading} style={{ marginLeft: '10px' }}>
                    Run Matrix Multiplication
                </button>
                <button onClick={runPerformanceTest} disabled={!isInitialized || isLoading} style={{ marginLeft: '10px' }}>
                    Performance Test
                </button>
                <button onClick={runMemoryTest} disabled={!isInitialized || isLoading} style={{ marginLeft: '10px' }}>
                    Memory Test
                </button>
            </div>

            <div style={{ background: '#000', color: '#0f0', padding: '15px', borderRadius: '5px' }}>
                <pre>{output || 'Click a button to run an example...'}</pre>
            </div>
        </div>
    );
}
```

#### 4. Vue.js Implementation

```javascript
// src/composables/useGreed.js
import { ref, onMounted, onUnmounted } from 'vue';
import Greed from 'greed.js';

export function useGreed(config = {}) {
    const greed = ref(null);
    const isInitialized = ref(false);
    const isLoading = ref(false);
    const error = ref(null);

    const initialize = async () => {
        if (greed.value) return greed.value;
        
        isLoading.value = true;
        error.value = null;
        
        try {
            const instance = new Greed({
                enableWebGPU: true,
                enableProfiling: process.env.NODE_ENV === 'development',
                strictSecurity: process.env.NODE_ENV === 'production',
                maxMemoryMB: 1024,
                ...config
            });
            
            await instance.initialize();
            greed.value = instance;
            isInitialized.value = true;
            
            return instance;
        } catch (err) {
            error.value = err.message;
            throw err;
        } finally {
            isLoading.value = false;
        }
    };

    const runCode = async (pythonCode, options = {}) => {
        if (!greed.value) throw new Error('Greed not initialized');
        
        isLoading.value = true;
        error.value = null;
        
        try {
            return await greed.value.run(pythonCode, options);
        } catch (err) {
            error.value = err.message;
            throw err;
        } finally {
            isLoading.value = false;
        }
    };

    const runTensorOperation = async (operation, tensors, options = {}) => {
        if (!greed.value) throw new Error('Greed not initialized');
        
        isLoading.value = true;
        error.value = null;
        
        try {
            return await greed.value.tensor(operation, tensors, options);
        } catch (err) {
            error.value = err.message;
            throw err;
        } finally {
            isLoading.value = false;
        }
    };

    const loadPackages = async (packages) => {
        if (!greed.value) throw new Error('Greed not initialized');
        
        try {
            return await greed.value.loadPackages(packages);
        } catch (err) {
            error.value = err.message;
            throw err;
        }
    };

    const getStats = () => {
        return greed.value?.getStats() || null;
    };

    const forceGC = async () => {
        if (!greed.value) return { cleaned: 0 };
        try {
            return await greed.value.forceGC();
        } catch (err) {
            error.value = err.message;
            throw err;
        }
    };

    onMounted(initialize);

    onUnmounted(() => {
        if (greed.value) {
            greed.value.destroy().catch(console.error);
        }
    });

    return {
        greed,
        isInitialized,
        isLoading,
        error,
        initialize,
        runCode,
        runTensorOperation,
        loadPackages,
        getStats,
        forceGC
    };
}
```

#### 5. Worker-Based Implementation for Heavy Computations

```javascript
// src/greed/greed-worker.js
import Greed from 'greed.js';

let greed = null;

self.onmessage = async function(e) {
    const { type, payload, id } = e.data;
    
    try {
        switch (type) {
            case 'INITIALIZE':
                greed = new Greed(payload.config);
                await greed.initialize();
                self.postMessage({ type: 'INITIALIZED', id, success: true });
                break;
                
            case 'RUN_CODE':
                if (!greed) throw new Error('Greed not initialized');
                const result = await greed.run(payload.code);
                self.postMessage({ 
                    type: 'CODE_RESULT', 
                    id, 
                    success: true, 
                    data: result 
                });
                break;
                
            case 'GET_STATS':
                const stats = greed ? greed.getStats() : null;
                self.postMessage({ 
                    type: 'STATS_RESULT', 
                    id, 
                    success: true, 
                    data: stats 
                });
                break;
                
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        self.postMessage({ 
            type: 'ERROR', 
            id, 
            success: false, 
            error: error.message 
        });
    }
};

// Hook for worker-based Greed.js
export function useGreedWorker() {
    const [worker, setWorker] = useState(null);
    const [isReady, setIsReady] = useState(false);
    const messageIdRef = useRef(0);
    const pendingMessages = useRef(new Map());

    useEffect(() => {
        const workerInstance = new Worker(new URL('./greed-worker.js', import.meta.url));
        
        workerInstance.onmessage = (e) => {
            const { id, success, data, error } = e.data;
            const { resolve, reject } = pendingMessages.current.get(id) || {};
            
            if (resolve && reject) {
                if (success) resolve(data);
                else reject(new Error(error));
                pendingMessages.current.delete(id);
            }
        };
        
        setWorker(workerInstance);
        
        return () => workerInstance.terminate();
    }, []);

    const sendMessage = (type, payload = {}) => {
        return new Promise((resolve, reject) => {
            const id = ++messageIdRef.current;
            pendingMessages.current.set(id, { resolve, reject });
            worker.postMessage({ type, payload, id });
        });
    };

    const initialize = async (config = {}) => {
        await sendMessage('INITIALIZE', { config });
        setIsReady(true);
    };

    const runCode = async (code) => {
        return await sendMessage('RUN_CODE', { code });
    };

    return { initialize, runCode, isReady };
}
```

#### 6. Testing Your Implementation

```javascript
// tests/greed.test.js
import { renderHook, act } from '@testing-library/react';
import { useGreed } from '../src/greed/hooks';

describe('useGreed Hook', () => {
    test('initializes correctly', async () => {
        const { result } = renderHook(() => useGreed());
        
        await act(async () => {
            await result.current.initialize();
        });
        
        expect(result.current.isInitialized).toBe(true);
        expect(result.current.error).toBe(null);
    });

    test('runs simple tensor operations', async () => {
        const { result } = renderHook(() => useGreed());
        
        await act(async () => {
            await result.current.initialize();
        });
        
        const output = await act(async () => {
            return await result.current.runCode(`
import torch
x = torch.tensor([1, 2, 3])
print(f"Tensor: {x}")
            `);
        });
        
        expect(output.output).toContain('Tensor:');
    });
});
```

#### 7. Performance Monitoring

```javascript
// src/utils/greedMonitor.js
export class GreedMonitor {
    constructor(greed) {
        this.greed = greed;
        this.metrics = {
            executionTimes: [],
            memoryUsage: [],
            webgpuUtilization: []
        };
    }

    async measureExecution(code) {
        const start = performance.now();
        const result = await this.greed.run(code);
        const end = performance.now();
        
        const executionTime = end - start;
        this.metrics.executionTimes.push(executionTime);
        
        const stats = this.greed.getStats();
        this.metrics.webgpuUtilization.push(
            stats.compute.availableStrategies.includes('webgpu')
        );
        
        return {
            result,
            executionTime,
            stats
        };
    }

    getAverageExecutionTime() {
        const times = this.metrics.executionTimes;
        return times.reduce((a, b) => a + b, 0) / times.length;
    }

    isWebGPUActive() {
        const recent = this.metrics.webgpuUtilization.slice(-5);
        return recent.every(active => active);
    }
}
```

## Check WebGPU Status

```javascript
const stats = greed.getStats();
console.log('WebGPU enabled:', stats.compute?.availableStrategies?.includes('webgpu'));

// Detailed status check
function checkGreedStatus(greed) {
    const stats = greed.getStats();
    
    return {
        system: {
            isInitialized: stats.isInitialized,
            uptime: stats.uptime,
            errorCount: stats.errorCount
        },
        webgpu: {
            available: stats.compute?.availableStrategies?.includes('webgpu') || false,
            backend: stats.compute?.backend || 'unknown',
            strategies: stats.compute?.availableStrategies || []
        },
        performance: {
            operations: stats.operations || 0,
            averageExecutionTime: stats.averageExecutionTime || 0,
            totalExecutionTime: stats.totalExecutionTime || 0
        },
        memory: {
            currentUsage: stats.memory?.memoryUsageMB || 0,
            maxMemory: stats.memory?.maxMemoryMB || 0,
            gcCount: stats.memory?.gcCount || 0
        },
        runtime: {
            isReady: stats.runtime?.isReady || false,
            backend: stats.runtime?.backend || 'unknown'
        },
        security: {
            totalValidations: stats.security?.totalValidations || 0,
            riskLevel: stats.security?.riskLevel || 'unknown'
        }
    };
}

// Example usage
async function runDiagnostics() {
    const greed = new Greed({ enableWebGPU: true });
    await greed.initialize();
    
    const status = checkGreedStatus(greed);
    console.log('System Status:', status);
    
    // Test basic operation
    const result = await greed.run(`
import torch
x = torch.tensor([1, 2, 3])
print(f"Device: {x.device}")
print(f"Tensor: {x}")
    `);
    
    console.log('Test Result:', result.output);
    
    // Final stats after operation
    const finalStats = greed.getStats();
    console.log('Final Stats:', finalStats);
}
```

## Matrix Operations

```python
import torch

# Create matrices
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Matrix multiplication
c = a @ b
print(f"Result: {c}")

# Element-wise operations
d = a * b + 1
print(f"Element-wise: {d}")
```

## Deployment & Production

### 8. Environment Configuration

```javascript
// src/config/greed.config.js
export const getGreedConfig = () => {
    const isDevelopment = process.env.NODE_ENV === 'development';
    const isProduction = process.env.NODE_ENV === 'production';
    
    return {
        enableWebGPU: true,
        webgpu: {
            webgpuMinElements: isDevelopment ? 1 : 100,  // Lower threshold in dev
            enableProfiling: isDevelopment,
            fallbackToJS: true  // Always allow fallback in production
        },
        security: {
            strictMode: isProduction,  // Stricter in production
            allowDynamicImports: !isProduction
        },
        performance: {
            enableCaching: isProduction,
            maxCacheSize: isProduction ? 100 : 10
        }
    };
};
```

### 9. Error Handling & Fallbacks

```javascript
// src/greed/errorHandling.js
export class GreedErrorHandler {
    constructor(greed) {
        this.greed = greed;
        this.fallbackMode = false;
    }

    async safeExecute(code, options = {}) {
        const { retries = 2, fallbackToJS = true } = options;
        
        for (let attempt = 0; attempt <= retries; attempt++) {
            try {
                // Check WebGPU availability before execution
                const stats = this.greed.getStats();
                const hasWebGPU = stats.compute.availableStrategies.includes('webgpu');
                
                if (!hasWebGPU && attempt === 0) {
                    console.warn('WebGPU not available, using CPU fallback');
                    this.fallbackMode = true;
                }
                
                const result = await this.greed.run(code);
                
                // Validate result
                if (!result || !result.output) {
                    throw new Error('Invalid execution result');
                }
                
                return {
                    success: true,
                    result,
                    fallbackMode: this.fallbackMode,
                    attempt: attempt + 1
                };
                
            } catch (error) {
                console.error(`Execution attempt ${attempt + 1} failed:`, error);
                
                if (attempt === retries) {
                    return {
                        success: false,
                        error: error.message,
                        fallbackMode: this.fallbackMode,
                        attempts: attempt + 1
                    };
                }
                
                // Wait before retry
                await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
            }
        }
    }
}

// Enhanced hook with error handling
export function useGreedWithErrorHandling(config = {}) {
    const greedHook = useGreed(config);
    const [errorHandler, setErrorHandler] = useState(null);

    useEffect(() => {
        if (greedHook.greed) {
            setErrorHandler(new GreedErrorHandler(greedHook.greed));
        }
    }, [greedHook.greed]);

    const safeRunCode = useCallback(async (code, options = {}) => {
        if (!errorHandler) throw new Error('Error handler not initialized');
        return await errorHandler.safeExecute(code, options);
    }, [errorHandler]);

    return {
        ...greedHook,
        safeRunCode,
        errorHandler
    };
}
```

### 10. Production Deployment Checklist

```markdown
## Pre-Deployment Checklist

### Browser Requirements
- [ ] WebGPU flags enabled in target browsers
- [ ] HTTPS deployment (required for WebGPU)
- [ ] Cross-origin headers configured for CDN assets
- [ ] Browser compatibility testing across Chrome, Edge, Brave

### Performance Optimization
- [ ] Bundle size optimization (<2MB total)
- [ ] Lazy loading for Greed.js initialization
- [ ] Web Workers for heavy computations
- [ ] Caching strategy for models and tensors
- [ ] Memory usage monitoring and cleanup

### Error Handling
- [ ] Graceful fallback to CPU when WebGPU unavailable
- [ ] User-friendly error messages
- [ ] Automatic retry mechanisms
- [ ] Performance degradation alerts

### Security
- [ ] Content Security Policy (CSP) headers
- [ ] Input validation for Python code execution
- [ ] Resource usage limits and monitoring
- [ ] Audit of allowed Python operations

### Monitoring
- [ ] WebGPU availability tracking
- [ ] Execution time monitoring
- [ ] Error rate tracking
- [ ] User engagement metrics
```

## Available Build Files

Greed.js provides different build files for different use cases:

### Production Files
- **`greed.min.js`**: Minified production build with code splitting
- **`greed.js`**: Unminified production build with code splitting
- **`core.js`, `compute.js`, `utils.js`**: Individual modules for selective imports

### Development/Testing Files
- **`greed-standalone.js`**: Single file build for easy testing and development
- No code splitting, easier debugging, source maps included

### CDN Usage Examples
```html
<!-- Production (recommended) -->
<script src="https://unpkg.com/greed.js@latest/dist/greed.min.js"></script>

<!-- Development/Testing -->
<script src="https://unpkg.com/greed.js@latest/dist/greed-standalone.js"></script>

<!-- Module imports (with bundler) -->
<script type="module">
  import Greed from 'https://unpkg.com/greed.js@latest/dist/greed.js';
</script>
```

## Performance Tips

1. **Enable WebGPU**: Always use `enableWebGPU: true`
2. **Use float32**: Default dtype for GPU operations
3. **Batch operations**: Larger tensors = better GPU utilization
4. **Check device**: Tensors should show `device: webgpu`
5. **Monitor execution**: Use `getStats()` for performance tracking
6. **Error handling**: Implement fallbacks for production reliability
7. **Lazy loading**: Initialize Greed.js only when needed
8. **Memory management**: Use `forceGC()` and proper cleanup with `destroy()`
9. **Build selection**: Use `greed.min.js` for production, `greed-standalone.js` for development

## Browser Support

- ✅ Chrome 113+ (with flags)
- ✅ Edge 113+ (with flags)  
- ✅ Brave (with flags)
- ⚠️ Firefox (experimental)
- ❌ Safari (limited)

## Troubleshooting

**Device shows CPU instead of WebGPU:**
- Enable browser WebGPU flags
- Update graphics drivers
- Use `webgpuMinElements: 1` in config

**Import errors:**
- Ensure Pyodide loads before Greed.js
- Use HTTPS or localhost (required for WebGPU)

**Performance issues:**
- Check `greed.getStats()` for WebGPU status
- Use larger tensors (>100 elements)
- Verify GPU hardware support