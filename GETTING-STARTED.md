# Greed.js - Getting Started

WebGPU-accelerated PyTorch runtime for browsers with zero dependencies.

## Quick Install

### CDN (Recommended)
```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
    <script src="https://unpkg.com/greed.js@latest/dist/greed-standalone.js"></script>
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
    enableWebGPU: true,
    webgpu: {
        webgpuMinElements: 1,  // Use GPU for all operations
        enableProfiling: true
    },
    security: {
        strictMode: false  // Allow broader PyTorch operations
    }
});
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

## Check WebGPU Status

```javascript
const stats = greed.getStats();
console.log('WebGPU enabled:', stats.compute.availableStrategies.includes('webgpu'));
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

## Performance Tips

1. **Enable WebGPU**: Always use `enableWebGPU: true`
2. **Use float32**: Default dtype for GPU operations
3. **Batch operations**: Larger tensors = better GPU utilization
4. **Check device**: Tensors should show `device: webgpu`

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