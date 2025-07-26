![logo](greed.png)

# Greed.js

[![npm version](https://badge.fury.io/js/greed.js.svg)](https://badge.fury.io/js/greed.js)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/adityakhalkar/greed/workflows/CI/badge.svg)](https://github.com/adityakhalkar/greed/actions)

A powerful JavaScript library that enables seamless execution of Python code in web browsers with **actual WebGPU-accelerated PyTorch support** using compute shaders and intelligent worker-based parallel execution.

## 📦 Installation

```bash
npm install greed.js
```

```bash
yarn add greed.js
```

```html
<!-- CDN -->
<script src="https://unpkg.com/greed.js@2.0.1/dist/greed.min.js"></script>
```

## ✨ Features

- **🏗️ Modular Architecture**: Clean separation of concerns with EventEmitter-based communication
- **PyTorch in Browser**: Full PyTorch polyfill with neural networks, tensors, and deep learning operations
- **⚡ WebGPU Compute Shaders**: True GPU acceleration with 50+ optimized WGSL compute shaders for tensor operations
- **🎯 Intelligent Fallback**: WebGPU → CPU → Worker execution strategy with automatic optimization
- **Complete Neural Networks**: Support for `torch.nn.Module`, layers, loss functions, and training
- **Python in Browser**: Execute Python code directly using Pyodide/WebAssembly
- **🛡️ Enhanced Security**: Advanced input validation and threat detection system
- **🧠 Smart Compute Strategy**: Intelligent fallback between WebGPU → CPU → Worker execution
- **📊 Memory Management**: Automatic resource cleanup and memory pressure monitoring
- **Dynamic Package Installation**: Automatically install Python packages on-demand
- **Simple API**: Easy-to-use interface with comprehensive PyTorch compatibility
- **Context Preservation**: Maintain variables and state across multiple executions
- **📈 Production Ready**: Comprehensive testing, security validation, and performance optimization

## Quick Start

```html
<!DOCTYPE html>
<html>
<head>
    <title>Greed.js PyTorch Demo</title>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
    <script type="module" src="dist/greed.js"></script>
</head>
<body>
    <script>
        async function runPyTorch() {
            const greed = new Greed({ enableWebGPU: true });
            await greed.initialize();
            
            // WebGPU-accelerated tensor operations
            const result = await greed.run(`
import torch

# Create tensors with WebGPU acceleration  
x = torch.randn(1000, 1000, device='webgpu')
y = torch.randn(1000, 1000, device='webgpu')

# Matrix multiplication using WebGPU compute shaders
result = torch.matmul(x, y)

# Neural network operations on GPU
model = torch.nn.Sequential(
    torch.nn.Linear(1000, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10)
).to('webgpu')

output = model(x)
print(f"Model output shape: {output.shape}")
print(f"WebGPU acceleration: {torch.cuda.is_available()}")

output.mean().item()
            `);
            
            console.log('WebGPU PyTorch result:', result);
        }
        
        runPyTorch();
    </script>
</body>
</html>
```

## 🏗️ Architecture

Greed.js features a modular architecture designed for performance, maintainability, and extensibility:

### Core Components

- **`RuntimeManager`**: Handles Pyodide initialization and Python package management
- **`ComputeStrategy`**: Intelligent compute orchestration with WebGPU/CPU/Worker fallback
- **`WebGPUComputeEngine`**: Hardware-accelerated tensor operations using WebGPU compute shaders
- **`WebGPUTensor`**: PyTorch-compatible tensor implementation with GPU acceleration
- **`TensorBridge`**: Seamless interoperability between JavaScript and Python tensors
- **`PipelineCache`**: Optimized shader compilation and caching system
- **`MemoryManager`**: Advanced resource cleanup with automatic garbage collection  
- **`SecurityValidator`**: Comprehensive input validation and threat detection
- **`EventEmitter`**: Base class providing event-driven inter-component communication

### API Usage

```javascript
// Basic usage
const greed = new Greed();
await greed.initialize();
const result = await greed.run('import torch; print(torch.tensor([1,2,3]))');

// Advanced configuration
const greed = new Greed({
  enableWebGPU: true,
  maxMemoryMB: 1024,
  strictSecurity: true,
  enableWorkers: true,
  maxWorkers: 4
});

// Component lifecycle events
greed.on('init:complete', () => console.log('Initialization complete'));
greed.on('memory:warning', (data) => console.log('Memory pressure:', data));

// Advanced tensor operations
await greed.tensor('matmul', [tensorA, tensorB], { device: 'webgpu' });

// Comprehensive statistics
const stats = greed.getStats();
console.log('Memory usage:', stats.memory.memoryUsageMB);
console.log('Operations:', stats.operations);
```

## PyTorch Support

### Tensor Operations
```python
import torch

# Tensor creation
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
y = torch.randn(2, 2)

# GPU acceleration
x_gpu = x.cuda()  # Move to WebGPU
result = torch.mm(x_gpu, y.cuda())  # Matrix multiplication on GPU

# All standard operations supported
z = x + y * 2.0 - torch.ones_like(x)
```

### Neural Networks
```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Create and use model
model = SimpleNet()
x = torch.randn(32, 784)  # Batch of 32 samples
output = model(x)

# Training with loss functions
criterion = nn.CrossEntropyLoss()
target = torch.randint(0, 10, (32,))
loss = criterion(output, target)
```

### GPU Acceleration Features
- **Element-wise operations**: `+`, `-`, `*`, `/` with smart GPU thresholds
- **Matrix operations**: `torch.mm()`, `torch.matmul()`, `@` operator
- **Reduction operations**: `torch.sum()`, `torch.mean()`, `torch.max()`
- **Neural network layers**: `nn.Linear`, `nn.ReLU`, `nn.CrossEntropyLoss`
- **Automatic fallback**: Seamless CPU fallback for small tensors or when WebGPU unavailable

## API Reference

### Constructor
```javascript
const greed = new Greed({
    maxWorkers: 2,           // Number of worker threads
    enableWebGPU: true       // Enable WebGPU acceleration
});
```

### Main Execution Method
```javascript
const result = await greed.executeCode(pythonCode, options);
```

**Options:**
- `packages`: Array of Python packages to install
- `useGPU`: Force GPU acceleration (default: auto-detect)
- `preserveContext`: Maintain variables between executions

**Returns:**
```javascript
{
    success: true,
    output: result,              // Python execution result
    stdout: "print output",      // Console output
    executionTime: 1234,         // Execution time in ms
    usedGPU: true               // Whether GPU was used
}
```

## ⚡ WebGPU Implementation

Greed.js features a **complete WebGPU implementation** that replaces numpy operations with actual GPU compute shaders for **true hardware acceleration**.

### WebGPU Architecture

```
PyTorch Tensor Operation
        ↓
WebGPU Compute Engine
        ↓
WGSL Compute Shader → GPU Execution → Result
        ↑
Pipeline Cache & Optimization
```

### Supported Operations

#### **Arithmetic Operations**
- `add`, `sub`, `mul`, `div`, `pow`
- Element-wise operations with broadcasting support

#### **Matrix Operations** 
- `matmul` - Optimized matrix multiplication with tiled algorithms
- `bmm` - Batch matrix multiplication for neural networks
- `transpose` - Efficient dimension swapping

#### **Activation Functions**
- `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu`
- `softmax` - Numerically stable with workgroup reduction

#### **Neural Network Operations**
- `conv2d` - 2D convolution with optimized memory access
- `maxpool2d`, `avgpool2d` - Pooling operations
- `batch_norm` - Batch normalization with running statistics

#### **Reduction Operations**
- `sum`, `mean` - Parallel reduction with shared memory
- `max`, `min` - Index-preserving reductions

#### **Loss Functions**
- `cross_entropy` - Numerically stable cross-entropy loss
- `mse_loss` - Mean squared error with broadcasting

### Performance Features

#### **Intelligent Workgroup Sizing**
```javascript
// Automatically optimizes workgroup sizes based on operation type
matmul: [16, 16, 1]    // 2D tiled matrix multiplication  
conv2d: [8, 8, 8]      // 3D spatial convolution
elementwise: [64, 1, 1] // 1D parallel processing
reduction: [256, 1, 1]  // Maximize parallel reduction
```

#### **Pipeline Caching**
- Automatic shader compilation and caching
- LRU eviction for memory efficiency  
- Warmup for common operations
- Adaptive optimization based on usage patterns

#### **Memory Management**
- Buffer pooling and reuse
- Automatic garbage collection
- Memory pressure monitoring
- Fallback to CPU when GPU memory exhausted

### Usage Examples

#### **Basic Tensor Operations**
```javascript
const greed = new Greed({ enableWebGPU: true });
await greed.initialize();

// WebGPU-accelerated tensor operations
await greed.run(`
import torch

# Tensors automatically use WebGPU when available
x = torch.randn(1000, 1000, device='webgpu')
y = torch.randn(1000, 1000, device='webgpu')

# GPU-accelerated matrix multiplication
result = torch.matmul(x, y)
print(f"Computed on: {result.device}")
`);
```

#### **Neural Network Training**
```javascript
await greed.run(`
import torch
import torch.nn as nn

# Neural network on GPU
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 10)
    
    def forward(self, x):
        return self.output(self.relu(self.linear(x)))

# Move model to WebGPU
model = SimpleNet().to('webgpu')
optimizer = torch.optim.Adam(model.parameters())

# Training step with GPU acceleration
def train_step(data, target):
    output = model(data)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()
`);
```

### Browser Compatibility

| Browser | WebGPU Support | Status |
|---------|----------------|--------|
| Chrome 113+ | ✅ Full | Production Ready |
| Edge 113+ | ✅ Full | Production Ready |  
| Firefox | 🚧 Flag Required | `dom.webgpu.enabled=true` |
| Safari | 🚧 Technology Preview | Limited Support |

### Fallback Strategy

When WebGPU is unavailable, Greed.js automatically falls back to:

1. **CPU Numpy Operations** - Full compatibility maintained
2. **Web Workers** - Parallel processing for large operations  
3. **Graceful Degradation** - Same API, different execution engine

## Architecture

```
Python Code → Greed.js → WebGPU Tensor Bridge → WGSL Compute Shaders → GPU
                ↓                                    ↓
         Pyodide Runtime                    Pipeline Cache & Optimization
                ↓                                    ↓
    Context Preservation ← Memory Manager ← Buffer Management & GC
```

### Execution Contexts
1. **WebGPU Engine**: Hardware-accelerated tensor operations using compute shaders  
2. **CPU Engine**: NumPy-based operations for compatibility and fallback
3. **Worker Engine**: Multi-threaded parallel processing for large computations
4. **Tensor Bridge**: Seamless interoperability between JavaScript and Python tensors

## Framework Integration

### React Usage

```jsx
import React, { useState, useEffect } from 'react';

function PyTorchComponent() {
  const [greed, setGreed] = useState(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadGreed = async () => {
      try {
        // Import Pyodide first
        await import('https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js');
        
        // Then import Greed
        const { Greed } = await import('greed.js');
        
        const greedInstance = new Greed({ enableWebGPU: true });
        setGreed(greedInstance);
        setLoading(false);
      } catch (error) {
        console.error('Failed to load Greed:', error);
        setLoading(false);
      }
    };

    loadGreed();
  }, []);

  const runPython = async () => {
    if (!greed) return;

    const code = `
import torch
x = torch.randn(100, 100).cuda()
y = torch.randn(100, 100).cuda()
result = torch.mm(x, y)
result.mean().item()
    `;

    const output = await greed.executeCode(code);
    setResult(JSON.stringify(output, null, 2));
  };

  if (loading) return <div>Loading PyTorch...</div>;

  return (
    <div>
      <h2>PyTorch in React</h2>
      <button onClick={runPython}>Run PyTorch Code</button>
      <pre>{result}</pre>
    </div>
  );
}
```

### Next.js Usage

```jsx
import dynamic from 'next/dynamic';

// Dynamically import to avoid SSR issues
const PyTorchRunner = dynamic(() => import('../components/PyTorchRunner'), {
  ssr: false, // Important: Disable server-side rendering
  loading: () => <p>Loading PyTorch...</p>
});

export default function HomePage() {
  return (
    <div>
      <h1>Next.js with PyTorch</h1>
      <PyTorchRunner />
    </div>
  );
}
```

### Custom React Hook

```jsx
import { useState, useEffect } from 'react';

export function useGreed(options = {}) {
  const [greed, setGreed] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const initGreed = async () => {
      try {
        if (!window.loadPyodide) {
          await import('https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js');
        }
        const { Greed } = await import('greed.js');
        const instance = new Greed(options);
        setGreed(instance);
        setLoading(false);
      } catch (err) {
        setError(err);
        setLoading(false);
      }
    };

    initGreed();
    return () => greed?.destroy();
  }, []);

  return { greed, loading, error };
}
```

**⚠️ Important Notes for React/Next.js:**
- **Client-Side Only**: Greed.js only works in browsers, not server-side
- **Next.js**: Use dynamic imports with `ssr: false`
- **Memory**: Call `greed.destroy()` in cleanup
- **WebGPU**: Requires modern browsers (Chrome 113+, Edge 113+)

## Examples

### Image Classification with PyTorch
```python
import torch
import torch.nn as nn

# Define a simple CNN
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.mean(dim=[2, 3])  # Global average pooling
        return self.fc(x)

# Create model and process batch
model = ImageClassifier()
images = torch.randn(8, 3, 32, 32).cuda()  # GPU acceleration
predictions = model(images)
```

### Performance Benchmarking
```python
import torch
import time

def benchmark_operation(name, func, *args):
    torch.cuda.is_available()  # Ensure GPU ready
    
    start = time.time()
    result = func(*args)
    end = time.time()
    
    print(f"{name}: {(end-start)*1000:.2f}ms")
    return result

# Benchmark GPU vs CPU
a_gpu = torch.randn(1000, 1000).cuda()
b_gpu = torch.randn(1000, 1000).cuda()

gpu_result = benchmark_operation("GPU MatMul", torch.mm, a_gpu, b_gpu)
cpu_result = benchmark_operation("CPU MatMul", torch.mm, a_gpu.cpu(), b_gpu.cpu())
```

## Browser Support

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| **Pyodide/WebAssembly** | ✅ 57+ | ✅ 52+ | ✅ 11+ | ✅ 16+ |
| **WebGPU Acceleration** | ✅ 113+ | 🔄 Exp | 🔄 Exp | ✅ 113+ |
| **Web Workers** | ✅ | ✅ | ✅ | ✅ |

## Development

```bash
# Clone repository
git clone https://github.com/your-username/greed.git
cd greed

# Install dependencies  
npm install

# Start development server with live examples
npm run dev

# Build for production
npm run build

# Run test suite
npm test
```

## 📁 Project Structure

```
greed/
├── src/
│   ├── greed.js              # Main library file
│   └── gpu/
│       └── webgpu-compute.js # WebGPU compute engine
├── sandbox.html              # Interactive examples
├── examples/                 # Usage examples
├── tests/                    # Test suite
└── dist/                     # Built files
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. **Bug Reports**: Use GitHub Issues with detailed reproduction steps
2. **Feature Requests**: Propose new PyTorch operations or WebGPU optimizations
3. **Pull Requests**: Include tests and ensure all examples still work

## License

This software is dual-licensed under AGPL v3.0 and commercial licenses.

### Open Source License (AGPL v3.0)
- Free for open source projects and personal use
- Requires your application to be open-sourced under AGPL v3.0
- Suitable for academic research and community contributions
- Must make complete source code available to users

### Commercial License
- Permits use in proprietary commercial applications
- Allows keeping application source code confidential
- No AGPL obligations for end users
- Includes technical support and maintenance services

For commercial licensing inquiries, contact khalkaraditya8@gmail.com

Complete licensing terms are available in the [LICENSE](LICENSE) file.

## Acknowledgments

- **[Pyodide](https://pyodide.org/)**: Python-to-WebAssembly runtime
- **[WebGPU](https://gpuweb.github.io/gpuweb/)**: GPU acceleration standard
- **[PyTorch](https://pytorch.org/)**: Deep learning framework inspiration
- **Python Community**: For the incredible ecosystem

---

**Greed.js** - Bringing the power of PyTorch and GPU acceleration to every web browser! 