# GreedJS Usage Guide

## Security Configuration for React and Frameworks

If you encounter security validation errors during initialization, you can configure the security mode:

### Basic Usage in React

```javascript
import React, { useEffect, useState } from 'react';

function App() {
  const [greed, setGreed] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function initializeGreed() {
      try {
        // Configure security for development/testing
        const greedInstance = new Greed({
          securityMode: 'permissive',  // or 'disabled' for development
          allowInternalCode: true,     // allows library internals
          strictSecurity: false        // less strict validation
        });
        
        await greedInstance.initialize();
        setGreed(greedInstance);
        
        // Test Python PyTorch code
        const result = await greedInstance.runPython(`
import torch
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.tensor([[2], [3]], dtype=torch.float32)
result = torch.matmul(x, y)
print(f"Result: {result}")
result.numpy().tolist()
        `);
        
        console.log('PyTorch result:', result);
        
      } catch (err) {
        setError(err.message);
        console.error('GreedJS initialization failed:', err);
      }
    }

    initializeGreed();
  }, []);

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!greed) {
    return <div>Loading GreedJS...</div>;
  }

  return (
    <div>
      <h1>GreedJS PyTorch in React</h1>
      <p>GreedJS initialized successfully!</p>
      {/* Your ML components here */}
    </div>
  );
}

export default App;
```

### Security Configuration Options

```javascript
const greed = new Greed({
  // Security modes
  securityMode: 'strict',      // 'strict', 'permissive', 'disabled'
  strictSecurity: false,       // Enable/disable strict security
  allowInternalCode: true,     // Allow library internals (recommended)
  
  // Optional: More granular control
  allowEval: false,           // Allow eval() in Python code
  allowFileSystem: false,     // Allow file system access
  allowNetwork: false,        // Allow network requests
  
  // Performance settings
  maxMemoryMB: 512,          // Memory limit
  enableWebGPU: true         // WebGPU acceleration
});
```

### Troubleshooting Security Errors

If you encounter `Potentially dangerous code patterns detected`:

1. **Use permissive mode for development:**
   ```javascript
   const greed = new Greed({ securityMode: 'permissive' });
   ```

2. **Disable security for testing:**
   ```javascript
   const greed = new Greed({ securityMode: 'disabled' });
   ```

3. **Check your Python code for flagged patterns:**
   - Avoid `eval()`, `exec()`, `__import__()`
   - Use standard PyTorch operations instead

### Production Configuration

For production environments, use strict security with specific allowances:

```javascript
const greed = new Greed({
  securityMode: 'strict',
  allowInternalCode: true,        // Required for library to work
  strictSecurity: true,           // Strict validation for user code
  maxMemoryMB: 1024,             // Higher memory for production
  allowNetwork: false,           // Block network access
  allowFileSystem: false         // Block file system access
});
```

## Error Resolution

### "PyTorch API setup failed: Potentially dangerous code patterns detected"

This error occurs when the security system flags internal library code. **Fixed in v2.1.1** with:

```javascript
// This now works without errors
const greed = new Greed();
await greed.initialize();
```

If you still encounter issues, use:

```javascript
const greed = new Greed({
  securityMode: 'permissive',
  allowInternalCode: true
});
```

### React Development vs Production

**Development:**
```javascript
const greed = new Greed({
  securityMode: 'permissive',
  strictSecurity: false
});
```

**Production:**
```javascript
const greed = new Greed({
  securityMode: 'strict',
  allowInternalCode: true,
  strictSecurity: true
});
```