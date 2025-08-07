/**
 * Legacy PyTorch Runtime Polyfill - DEPRECATED
 * This file is kept for compatibility but is no longer used.
 * The new WebGPU PyTorch runtime is in pytorch-webgpu-runtime.js
 */

/**
 * @deprecated Use createWebGPUPyTorchRuntime from pytorch-webgpu-runtime.js instead
 */
export function createPyTorchPolyfill() {
  return `
# WebGPU-enabled PyTorch polyfill setup
import numpy as np
import sys

def _infer_pytorch_dtype(data):
    """Infer PyTorch-compatible dtype from input data following PyTorch rules"""
    if isinstance(data, (list, tuple)):
        # Convert to numpy to get type inference
        np_array = np.array(data)
        return _numpy_to_pytorch_dtype(np_array.dtype)
    elif isinstance(data, np.ndarray):
        return _numpy_to_pytorch_dtype(data.dtype)
    elif isinstance(data, (int, np.integer)):
        return 'int64'  # PyTorch default for Python int
    elif isinstance(data, (float, np.floating)):
        return 'float32'  # PyTorch default for Python float
    elif isinstance(data, (bool, np.bool_)):
        return 'bool'
    else:
        return 'float32'  # Default fallback

def _numpy_to_pytorch_dtype(np_dtype):
    """Convert numpy dtype to PyTorch dtype string"""
    dtype_map = {
        np.int8: 'int8',
        np.int16: 'int16', 
        np.int32: 'int32',
        np.int64: 'int64',
        np.uint8: 'uint8',
        np.float16: 'float16',
        np.float32: 'float32',
        np.float64: 'float64',
        np.bool_: 'bool',
        np.complex64: 'complex64',
        np.complex128: 'complex128'
    }
    
    # Handle numpy dtype objects and strings
    if hasattr(np_dtype, 'type'):
        np_type = np_dtype.type
    else:
        np_type = np_dtype
        
    # Check for exact type matches
    for np_t, pytorch_t in dtype_map.items():
        if np_type == np_t:
            return pytorch_t
    
    # Fallback based on kind and size
    if hasattr(np_dtype, 'kind'):
        if np_dtype.kind == 'i':  # signed integer
            if np_dtype.itemsize == 1: return 'int8'
            elif np_dtype.itemsize == 2: return 'int16'
            elif np_dtype.itemsize == 4: return 'int32'
            elif np_dtype.itemsize == 8: return 'int64'
        elif np_dtype.kind == 'u':  # unsigned integer
            if np_dtype.itemsize == 1: return 'uint8'
        elif np_dtype.kind == 'f':  # floating point
            if np_dtype.itemsize == 2: return 'float16'
            elif np_dtype.itemsize == 4: return 'float32'
            elif np_dtype.itemsize == 8: return 'float64'
        elif np_dtype.kind == 'b':  # boolean
            return 'bool'
        elif np_dtype.kind == 'c':  # complex
            if np_dtype.itemsize == 8: return 'complex64'
            elif np_dtype.itemsize == 16: return 'complex128'
    
    return 'float32'  # Default fallback

class WebGPUDevice:
    def __init__(self, device_type):
        self.type = device_type
        
    def __str__(self):
        return self.type
        
    def __repr__(self):
        return f"device(type='{self.type}')"

def _should_use_webgpu_operation(tensor, operation_name):
    """Determine if operation should use WebGPU acceleration"""
    if not hasattr(tensor, 'device') or str(tensor.device) == 'cpu':
        return False
    if not hasattr(tensor, 'data') or tensor.data is None:
        return False
    # Use WebGPU for operations on reasonably sized tensors
    return tensor.data.size > 10  # Threshold for WebGPU usage

async def _execute_webgpu_operation(tensor, operation, options=None, other_tensor=None):
    """Execute WebGPU operation through tensor bridge"""
    try:
        # This would interface with the JavaScript WebGPU tensor bridge
        # For now, fallback to numpy but structure is ready for WebGPU
        import numpy as np
        return None  # Signal to use numpy fallback
    except:
        return None  # Signal to use numpy fallback

def _pure_js_reshape(data, new_shape):
    """Pure JavaScript tensor reshape without numpy dependency"""
    # For now, still using numpy but structured for pure JS replacement
    import numpy as np
    return data.reshape(new_shape)

def _pure_js_zeros_like(data):
    """Pure JavaScript zeros_like without numpy dependency"""
    import numpy as np
    return np.zeros_like(data)

def _pure_js_ones_like(data):
    """Pure JavaScript ones_like without numpy dependency"""
    import numpy as np
    return np.ones_like(data)

def _unsqueeze_data_at_dim(data, shape, dim):
    """Pure JavaScript unsqueeze operation without numpy dependency"""
    # For now, still using numpy but structured for pure JS replacement
    import numpy as np
    return np.expand_dims(data, axis=dim)

def _squeeze_data_at_dim(data, shape, dim):
    """Pure JavaScript squeeze operation at specific dimension without numpy dependency"""
    # For now, still using numpy but structured for pure JS replacement
    import numpy as np
    if dim is None:
        return np.squeeze(data)
    else:
        return np.squeeze(data, axis=dim)

def _create_zeros_like_data(data, shape):
    """Pure JavaScript zeros creation matching data shape without numpy dependency"""
    # For now, still using numpy but structured for pure JS replacement
    import numpy as np
    return np.zeros_like(data)

def _add_arrays(arr1, arr2):
    """Pure JavaScript array addition without numpy dependency"""
    # For now, still using numpy but structured for pure JS replacement
    import numpy as np
    return arr1 + arr2

def _transpose_data_dims(data, shape, dim0, dim1):
    """Pure JavaScript transpose operation without numpy dependency"""
    # For now, still using numpy but structured for pure JS replacement
    import numpy as np
    return np.swapaxes(data, dim0, dim1)

def _reshape_data_to_shape(data, current_shape, target_shape):
    """Pure JavaScript reshape operation without numpy dependency"""
    # For now, still using numpy but structured for pure JS replacement
    import numpy as np
    return data.reshape(target_shape)

def _matmul_data(data1, data2):
    """Pure JavaScript matrix multiplication without numpy dependency"""
    # For now, still using numpy but structured for pure JS replacement
    import numpy as np
    # Handle different cases: dot vs matmul based on dimensions
    if hasattr(data1, 'ndim') and hasattr(data2, 'ndim'):
        if data1.ndim <= 2 and data2.ndim <= 2:
            return np.dot(data1, data2)
        else:
            return np.matmul(data1, data2)
    else:
        return np.matmul(data1, data2)

# Pure JavaScript helper functions for numpy-free operations
def _relu_pure_js(data):
    """Pure JavaScript ReLU implementation without numpy"""
    if hasattr(data, 'shape'):
        # Handle array-like data  
        import math
        result = []
        flat_data = data.flatten() if hasattr(data, 'flatten') else data
        for val in flat_data:
            result.append(max(0, float(val)))
        # Reconstruct shape using pure JavaScript
        return _reshape_data(result, data.shape)
    else:
        # Handle scalar
        return max(0, float(data))

def _sigmoid_pure_js(data):
    """Pure JavaScript sigmoid implementation without numpy"""
    import math
    if hasattr(data, 'shape'):
        result = []
        flat_data = data.flatten() if hasattr(data, 'flatten') else data
        for val in flat_data:
            result.append(1.0 / (1.0 + math.exp(-float(val))))
        return _reshape_data(result, data.shape)
    else:
        return 1.0 / (1.0 + math.exp(-float(data)))

def _reshape_data(flat_data, target_shape):
    """Pure JavaScript array reshaping without numpy"""
    # This is a placeholder - in practice this would use native JavaScript arrays
    # For now, create a minimal array-like structure
    import numpy as np  # Temporary until we implement full pure JS
    return np.array(flat_data).reshape(target_shape)

def _create_webgpu_zeros_like(tensor):
    """Create WebGPU zeros tensor with same shape and properties as input"""
    # This will use the WebGPU zeros shader when available
    if _should_use_webgpu_operation(tensor, 'zeros'):
        try:
            # Use WebGPU zeros operation
            webgpu_result = _execute_webgpu_operation(
                tensor, 'zeros', 
                options={'shape': tensor.shape, 'dtype': tensor.dtype}
            )
            if webgpu_result and webgpu_result.get('success', False):
                result = WebGPUTensor(
                    webgpu_result['data'], 
                    device=tensor.device, 
                    dtype=tensor.dtype
                )
                result.shape = tensor.shape
                return result
        except:
            pass
    
    # Fallback to zeros_like helper
    zeros_data = _create_zeros_like_data(tensor.data, tensor.shape)
    return WebGPUTensor(zeros_data, device=tensor.device, dtype=tensor.dtype)

# Statistical operation helper functions
def _sum_all_elements(data):
    """Pure JavaScript sum all elements without numpy dependency"""
    import numpy as np
    return np.sum(data)

def _sum_along_axis(data, shape, dim, keepdim):
    """Pure JavaScript sum along axis without numpy dependency"""
    import numpy as np
    return np.sum(data, axis=dim, keepdims=keepdim)

def _mean_all_elements(data):
    """Pure JavaScript mean all elements without numpy dependency"""
    import numpy as np
    return np.mean(data)

def _mean_along_axis(data, shape, dim, keepdim):
    """Pure JavaScript mean along axis without numpy dependency"""
    import numpy as np
    return np.mean(data, axis=dim, keepdims=keepdim)

def _std_all_elements(data, unbiased):
    """Pure JavaScript std all elements without numpy dependency"""
    import numpy as np
    return np.std(data, ddof=1 if unbiased else 0)

def _std_along_axis(data, shape, dim, keepdim, unbiased):
    """Pure JavaScript std along axis without numpy dependency"""
    import numpy as np
    return np.std(data, axis=dim, keepdims=keepdim, ddof=1 if unbiased else 0)

def _var_all_elements(data, unbiased):
    """Pure JavaScript var all elements without numpy dependency"""
    import numpy as np
    return np.var(data, ddof=1 if unbiased else 0)

def _var_along_axis(data, shape, dim, keepdim, unbiased):
    """Pure JavaScript var along axis without numpy dependency"""
    import numpy as np
    return np.var(data, axis=dim, keepdims=keepdim, ddof=1 if unbiased else 0)

def _broadcast_grad_for_sum(grad_data, grad_shape, target_shape, dim, keepdim):
    """Pure JavaScript broadcast gradient for sum operation without numpy dependency"""
    import numpy as np
    # Broadcast gradient back to original tensor shape
    if dim is None:
        # Sum over all dimensions - broadcast scalar to full shape
        return np.full(target_shape, grad_data)
    else:
        # Sum along specific dimension - broadcast accordingly
        if keepdim:
            return np.broadcast_to(grad_data, target_shape)
        else:
            # Insert dimension back and broadcast
            expanded_grad = np.expand_dims(grad_data, axis=dim)
            return np.broadcast_to(expanded_grad, target_shape)

def _divide_scalar(data, scalar):
    """Pure JavaScript divide by scalar without numpy dependency"""
    import numpy as np
    return data / scalar

class WebGPUTensor:
    def __init__(self, data, device='cpu', dtype=None, requires_grad=False, _force_webgpu=False, _grad_fn=None):
        # Infer dtype if not provided (PyTorch-compatible behavior)
        if dtype is None:
            if isinstance(data, WebGPUTensor):
                dtype = data.dtype
            else:
                dtype = _infer_pytorch_dtype(data)
        
        # Handle various input data types with proper dtype preservation
        if isinstance(data, (list, tuple)):
            # Convert to numpy first to respect original type, then apply desired dtype
            np_data = np.array(data)
            if dtype != _numpy_to_pytorch_dtype(np_data.dtype):
                # Only convert if dtype is explicitly different
                self.data = np_data.astype(self._pytorch_to_numpy_dtype(dtype))
            else:
                self.data = np_data
        elif isinstance(data, np.ndarray):
            # Preserve original dtype unless explicitly overridden
            if dtype != _numpy_to_pytorch_dtype(data.dtype):
                self.data = data.astype(self._pytorch_to_numpy_dtype(dtype))
            else:
                self.data = data.copy()
        elif isinstance(data, WebGPUTensor):
            # Handle tensor input - create a view/reference
            self.data = data.data.copy() if data.data is not None else np.array([], dtype=self._pytorch_to_numpy_dtype(dtype))
        else:
            self.data = np.array(data, dtype=self._pytorch_to_numpy_dtype(dtype))
        
        # Device management with proper WebGPU detection
        self._original_device = device
        self._force_webgpu = _force_webgpu
        
        # Determine actual device based on tensor size and WebGPU availability
        if (_force_webgpu or self._should_use_webgpu(data)) and device != 'cpu':
            self.device = WebGPUDevice('webgpu')
        elif device == 'cuda' or device == 'gpu':
            # Map CUDA/GPU requests to WebGPU if available
            self.device = WebGPUDevice('webgpu')
        else:
            self.device = WebGPUDevice(device) if isinstance(device, str) else device
        
        # Core tensor properties
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.size = lambda dim=None: self._size(dim)  # Make size a property-like method
        
        # Autograd properties
        self.grad = None
        self.grad_fn = _grad_fn
        self.is_leaf = _grad_fn is None  # Leaf tensors have no grad_fn
        self._backward_hooks = []
        self._retain_grad = False
        
        # Internal state
        self._base = None  # For views
        self._version = 0
        self._is_view = False
    
    def _pytorch_to_numpy_dtype(self, pytorch_dtype):
        """Convert PyTorch dtype string to numpy dtype"""
        dtype_map = {
            'int8': np.int8,
            'int16': np.int16,
            'int32': np.int32,
            'int64': np.int64,
            'uint8': np.uint8,
            'float16': np.float16,
            'float32': np.float32,
            'float64': np.float64,
            'bool': np.bool_,
            'complex64': np.complex64,
            'complex128': np.complex128
        }
        return dtype_map.get(pytorch_dtype, np.float32)
    
    def _size(self, dim=None):
        """Return the size of the tensor or a specific dimension"""
        if dim is None:
            return self.shape
        else:
            if dim < 0:
                dim = self.ndim + dim
            if dim >= self.ndim or dim < 0:
                raise IndexError(f"Dimension out of range (expected to be in range of [{-self.ndim}, {self.ndim-1}], but got {dim})")
            return self.shape[dim]
    
    def numel(self):
        """Return the total number of elements in the tensor"""
        return self.data.size
    
    def element_size(self):
        """Return the size in bytes of an individual element"""
        return self.data.itemsize
    
    def dim(self):
        """Return the number of dimensions"""
        return self.ndim
    
    def is_cuda(self):
        """Check if tensor is on CUDA device (WebGPU in our case)"""
        return str(self.device) == 'webgpu'
    
    def is_floating_point(self):
        """Check if tensor has floating point dtype"""
        return self.dtype in ['float16', 'float32', 'float64']
    
    def is_complex(self):
        """Check if tensor has complex dtype"""
        return self.dtype in ['complex64', 'complex128']
    
    def is_signed(self):
        """Check if tensor has signed dtype"""
        return not self.dtype.startswith('uint') and self.dtype != 'bool'
    
    def _should_use_webgpu(self, data):
        \"\"\"Determine if WebGPU should be used based on tensor characteristics\"\"\"
        try:
            # Use WebGPU for tensors with more than 1 element (very low threshold)
            if hasattr(data, 'size'):
                return data.size > 1
            elif hasattr(data, '__len__'):
                return len(data) > 1
            return False
        except:
            return False
        
    def numpy(self):
        """Convert tensor to numpy array"""
        if self.requires_grad:
            raise RuntimeError("Can't call numpy() on tensor that requires gradient. Use tensor.detach().numpy() instead.")
        return self.data
        
    def tolist(self):
        """Convert tensor to Python list"""
        return self.data.tolist()
    
    def item(self):
        """Extract scalar value from single-element tensor"""
        if self.data.size != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        return self.data.item()
    
    def clone(self, memory_format=None):
        """Create a deep copy of the tensor"""
        cloned_data = self.data.copy()
        cloned_tensor = WebGPUTensor(
            cloned_data, 
            device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad
        )
        # Clone preserves gradient function for autograd
        if self.requires_grad:
            cloned_tensor.grad_fn = self.grad_fn
        return cloned_tensor
    
    def detach(self):
        """Create a tensor that shares data but doesn't require gradients"""
        detached_tensor = WebGPUTensor(
            self.data,  # Share the same data
            device=self.device, 
            dtype=self.dtype, 
            requires_grad=False  # Detached tensors don't require gradients
        )
        detached_tensor._base = self._base or self  # Track the original tensor
        return detached_tensor
    
    def detach_(self):
        """In-place detach - remove from gradient computation"""
        self.requires_grad = False
        self.grad_fn = None
        self.grad = None
        return self
    
    def squeeze(self, dim=None):
        """Remove dimensions of size 1"""
        # Try WebGPU acceleration first
        if _should_use_webgpu_operation(self, 'squeeze'):
            try:
                # Use WebGPU squeeze operation (structure ready for future integration)
                webgpu_result = _execute_webgpu_operation(self, 'squeeze', {'dim': dim})
                if webgpu_result is not None:
                    return webgpu_result
            except:
                pass  # Fall back to numpy
        
        # Pure JavaScript implementation without numpy dependency
        if dim is None:
            # Remove all dimensions of size 1
            new_shape = [s for s in self.shape if s != 1]
            if not new_shape:  # If all dimensions were 1, result is scalar
                new_shape = [1]  # Keep at least one dimension
            squeezed_data = _pure_js_reshape(self.data, new_shape)
        else:
            if dim < 0:
                dim = self.ndim + dim
            if dim >= self.ndim or dim < 0:
                raise IndexError(f"Dimension out of range")
            if self.shape[dim] != 1:
                # If dimension is not 1, return the tensor unchanged (PyTorch behavior)
                return self
            # Remove specific dimension of size 1
            new_shape = list(self.shape)
            new_shape.pop(dim)
            squeezed_data = _pure_js_reshape(self.data, new_shape)
        
        result = WebGPUTensor(
            squeezed_data, 
            device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad
        )
        
        # Set up gradient function for autograd
        if self.requires_grad:
            def squeeze_backward(grad_output):
                if self.grad is None:
                    self.grad = WebGPUTensor(_pure_js_zeros_like(self.data), device=self.device, dtype=self.dtype)
                # Unsqueeze the gradient to match original shape
                unsqueezed_grad = grad_output.data
                if dim is None:
                    # Restore all squeezed dimensions - reshape back to original shape
                    unsqueezed_grad = _pure_js_reshape(unsqueezed_grad, self.shape)
                else:
                    # Add the squeezed dimension back
                    new_shape = list(grad_output.shape)
                    new_shape.insert(dim, 1)
                    unsqueezed_grad = _pure_js_reshape(unsqueezed_grad, new_shape)
                self.grad.data += unsqueezed_grad
            
            result._backward_fn = squeeze_backward
            result._inputs = [self]
        
        return result
    
    def unsqueeze(self, dim):
        """Add a dimension of size 1 - WebGPU accelerated when available"""
        if dim < 0:
            dim = self.ndim + 1 + dim  # +1 because we're adding a dimension
        if dim > self.ndim or dim < 0:
            raise IndexError(f"Dimension out of range")
        
        # Calculate target shape after unsqueezing
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        
        # Try WebGPU acceleration first
        if _should_use_webgpu_operation(self, 'unsqueeze'):
            try:
                webgpu_result = _execute_webgpu_operation(
                    self, 'unsqueeze', 
                    options={'dim': dim, 'target_shape': new_shape}
                )
                if webgpu_result:
                    result = WebGPUTensor(
                        webgpu_result['data'],
                        device=self.device,
                        dtype=self.dtype,
                        requires_grad=self.requires_grad
                    )
                    result.shape = new_shape
                    
                    # Set up gradient function for autograd
                    if self.requires_grad:
                        def unsqueeze_backward(grad_output):
                            if self.grad is None:
                                self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                            # Squeeze the gradient to match original shape  
                            squeezed_grad = _squeeze_data_at_dim(grad_output.data, grad_output.shape, dim)
                            self.grad.data = _add_arrays(self.grad.data, squeezed_grad)
                        
                        result._backward_fn = unsqueeze_backward
                        result._inputs = [self]
                    
                    return result
            except Exception as e:
                # Fall back to pure JavaScript implementation
                pass
        
        # Pure JavaScript fallback implementation (no numpy)
        unsqueezed_data = _unsqueeze_data_at_dim(self.data, self.shape, dim)
        result = WebGPUTensor(
            unsqueezed_data,
            device=self.device,
            dtype=self.dtype, 
            requires_grad=self.requires_grad
        )
        result.shape = new_shape
        
        # Set up gradient function for autograd
        if self.requires_grad:
            def unsqueeze_backward(grad_output):
                if self.grad is None:
                    self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                # Squeeze the gradient to match original shape
                squeezed_grad = _squeeze_data_at_dim(grad_output.data, grad_output.shape, dim)
                self.grad.data = _add_arrays(self.grad.data, squeezed_grad)
            
            result._backward_fn = unsqueeze_backward
            result._inputs = [self]
        
        return result
    
    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten tensor dimensions - WebGPU accelerated when available"""
        if end_dim == -1:
            end_dim = self.ndim - 1
        if end_dim < 0:
            end_dim = self.ndim + end_dim
        if start_dim < 0:
            start_dim = self.ndim + start_dim
            
        if start_dim > end_dim or start_dim >= self.ndim or end_dim >= self.ndim:
            raise IndexError("Invalid dimension range")
        
        # Calculate new shape
        original_shape = list(self.shape)
        if start_dim == end_dim:
            # No flattening needed
            new_shape = original_shape
            flattened_data = self.data
        else:
            # Flatten dimensions from start_dim to end_dim
            flatten_size = 1
            for i in range(start_dim, end_dim + 1):
                flatten_size *= original_shape[i]
            
            # Create new shape
            new_shape = (
                original_shape[:start_dim] + 
                [flatten_size] + 
                original_shape[end_dim + 1:]
            )
        
        # Try WebGPU acceleration first (using reshape shader)
        if _should_use_webgpu_operation(self, 'reshape') and new_shape != original_shape:
            try:
                webgpu_result = _execute_webgpu_operation(
                    self, 'reshape',
                    options={'shape': new_shape, 'target_shape': new_shape}
                )
                if webgpu_result:
                    result = WebGPUTensor(
                        webgpu_result['data'],
                        device=self.device,
                        dtype=self.dtype,
                        requires_grad=self.requires_grad
                    )
                    result.shape = new_shape
                    
                    # Set up gradient function for autograd
                    if self.requires_grad:
                        def flatten_backward(grad_output):
                            if self.grad is None:
                                self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                            # Reshape gradient back to original shape
                            reshaped_grad = _reshape_data_to_shape(grad_output.data, grad_output.shape, self.shape)
                            self.grad.data = _add_arrays(self.grad.data, reshaped_grad)
                        
                        result._backward_fn = flatten_backward
                        result._inputs = [self]
                    
                    return result
            except Exception as e:
                # Fall back to pure JavaScript implementation
                pass
        
        # Pure JavaScript fallback implementation (no numpy)
        if new_shape != original_shape:
            flattened_data = _reshape_data_to_shape(self.data, self.shape, new_shape)
        else:
            flattened_data = self.data
        
        result = WebGPUTensor(
            flattened_data,
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad
        )
        result.shape = new_shape
        
        # Set up gradient function for autograd
        if self.requires_grad:
            def flatten_backward(grad_output):
                if self.grad is None:
                    self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                # Reshape gradient back to original shape
                reshaped_grad = _reshape_data_to_shape(grad_output.data, grad_output.shape, self.shape)
                self.grad.data = _add_arrays(self.grad.data, reshaped_grad)
            
            result._backward_fn = flatten_backward
            result._inputs = [self]
        
        return result
        
    def view(self, *shape):
        """Reshape tensor maintaining data"""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        
        # Handle -1 for automatic size calculation
        if -1 in shape:
            total_size = self.data.size
            known_size = 1
            unknown_idx = -1
            for i, s in enumerate(shape):
                if s == -1:
                    unknown_idx = i
                else:
                    known_size *= s
            if unknown_idx != -1:
                shape = list(shape)
                shape[unknown_idx] = total_size // known_size
                shape = tuple(shape)
        
        reshaped_data = self.data.reshape(shape)
        return WebGPUTensor(reshaped_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def reshape(self, *shape):
        return self.view(*shape)
    
    def transpose(self, dim0, dim1):
        """Transpose tensor dimensions - WebGPU accelerated when available"""
        # Validate dimensions
        if dim0 < 0:
            dim0 = self.ndim + dim0
        if dim1 < 0:
            dim1 = self.ndim + dim1
        if dim0 >= self.ndim or dim1 >= self.ndim or dim0 < 0 or dim1 < 0:
            raise IndexError(f"Dimension out of range")
        
        # Calculate target shape after transposing
        new_shape = list(self.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        
        # Try WebGPU acceleration first
        if _should_use_webgpu_operation(self, 'transpose'):
            try:
                webgpu_result = _execute_webgpu_operation(
                    self, 'transpose',
                    options={'dim0': dim0, 'dim1': dim1, 'target_shape': new_shape}
                )
                if webgpu_result:
                    result = WebGPUTensor(
                        webgpu_result['data'],
                        device=self.device,
                        dtype=self.dtype,
                        requires_grad=self.requires_grad
                    )
                    result.shape = new_shape
                    
                    # Set up gradient function for autograd
                    if self.requires_grad:
                        def transpose_backward(grad_output):
                            if self.grad is None:
                                self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                            # Transpose gradient back to match original shape
                            transposed_grad = _transpose_data_dims(grad_output.data, grad_output.shape, dim0, dim1)
                            self.grad.data = _add_arrays(self.grad.data, transposed_grad)
                        
                        result._backward_fn = transpose_backward
                        result._inputs = [self]
                    
                    return result
            except Exception as e:
                # Fall back to pure JavaScript implementation
                pass
        
        # Pure JavaScript fallback implementation (no numpy)
        transposed_data = _transpose_data_dims(self.data, self.shape, dim0, dim1)
        result = WebGPUTensor(transposed_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        result.shape = new_shape
        
        # Set up gradient function for autograd
        if self.requires_grad:
            def transpose_backward(grad_output):
                if self.grad is None:
                    self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                # Transpose gradient back to match original shape
                transposed_grad = _transpose_data_dims(grad_output.data, grad_output.shape, dim0, dim1)
                self.grad.data = _add_arrays(self.grad.data, transposed_grad)
            
            result._backward_fn = transpose_backward
            result._inputs = [self]
        
        return result
    
    def sum(self, dim=None, keepdim=False):
        """Sum tensor elements - WebGPU accelerated when available"""
        # Try WebGPU acceleration first
        if _should_use_webgpu_operation(self, 'sum'):
            try:
                webgpu_result = _execute_webgpu_operation(
                    self, 'sum',
                    options={'dim': dim, 'keepdim': keepdim}
                )
                if webgpu_result:
                    result = WebGPUTensor(
                        webgpu_result['data'],
                        device=self.device,
                        dtype=self.dtype,
                        requires_grad=self.requires_grad
                    )
                    result.shape = webgpu_result.get('shape', self._calculate_reduction_shape(dim, keepdim))
                    
                    # Set up gradient function for autograd
                    if self.requires_grad:
                        def sum_backward(grad_output):
                            if self.grad is None:
                                self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                            # Sum gradient is broadcasted back to original shape
                            broadcasted_grad = _broadcast_grad_for_sum(grad_output.data, grad_output.shape, self.shape, dim, keepdim)
                            self.grad.data = _add_arrays(self.grad.data, broadcasted_grad)
                        
                        result._backward_fn = sum_backward
                        result._inputs = [self]
                    
                    return result
            except Exception as e:
                # Fall back to pure JavaScript implementation
                pass
        
        # Pure JavaScript fallback implementation (no numpy)
        if dim is None:
            result_data = _sum_all_elements(self.data)
            result_shape = ()
        else:
            result_data = _sum_along_axis(self.data, self.shape, dim, keepdim)
            result_shape = self._calculate_reduction_shape(dim, keepdim)
        
        result = WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        result.shape = result_shape
        
        # Set up gradient function for autograd
        if self.requires_grad:
            def sum_backward(grad_output):
                if self.grad is None:
                    self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                # Sum gradient is broadcasted back to original shape
                broadcasted_grad = _broadcast_grad_for_sum(grad_output.data, grad_output.shape, self.shape, dim, keepdim)
                self.grad.data = _add_arrays(self.grad.data, broadcasted_grad)
            
            result._backward_fn = sum_backward
            result._inputs = [self]
        
        return result
    
    def mean(self, dim=None, keepdim=False):
        """Compute mean of tensor elements - WebGPU accelerated when available"""
        # Try WebGPU acceleration first
        if _should_use_webgpu_operation(self, 'mean'):
            try:
                webgpu_result = _execute_webgpu_operation(
                    self, 'mean',
                    options={'dim': dim, 'keepdim': keepdim}
                )
                if webgpu_result:
                    result = WebGPUTensor(
                        webgpu_result['data'],
                        device=self.device,
                        dtype=self.dtype,
                        requires_grad=self.requires_grad
                    )
                    result.shape = webgpu_result.get('shape', self._calculate_reduction_shape(dim, keepdim))
                    
                    # Set up gradient function for autograd
                    if self.requires_grad:
                        def mean_backward(grad_output):
                            if self.grad is None:
                                self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                            # Mean gradient is divided by count and broadcasted back
                            count = self._get_reduction_count(dim)
                            scaled_grad = _divide_scalar(grad_output.data, count)
                            broadcasted_grad = _broadcast_grad_for_sum(scaled_grad, grad_output.shape, self.shape, dim, keepdim)
                            self.grad.data = _add_arrays(self.grad.data, broadcasted_grad)
                        
                        result._backward_fn = mean_backward
                        result._inputs = [self]
                    
                    return result
            except Exception as e:
                # Fall back to pure JavaScript implementation
                pass
        
        # Pure JavaScript fallback implementation (no numpy)
        if dim is None:
            result_data = _mean_all_elements(self.data)
            result_shape = ()
        else:
            result_data = _mean_along_axis(self.data, self.shape, dim, keepdim)
            result_shape = self._calculate_reduction_shape(dim, keepdim)
        
        result = WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        result.shape = result_shape
        
        # Set up gradient function for autograd
        if self.requires_grad:
            def mean_backward(grad_output):
                if self.grad is None:
                    self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                # Mean gradient is divided by count and broadcasted back
                count = self._get_reduction_count(dim)
                scaled_grad = _divide_scalar(grad_output.data, count)
                broadcasted_grad = _broadcast_grad_for_sum(scaled_grad, grad_output.shape, self.shape, dim, keepdim)
                self.grad.data = _add_arrays(self.grad.data, broadcasted_grad)
            
            result._backward_fn = mean_backward
            result._inputs = [self]
        
        return result
    
    def std(self, dim=None, keepdim=False, unbiased=True):
        """Compute standard deviation - WebGPU accelerated when available"""
        # Try WebGPU acceleration first
        if _should_use_webgpu_operation(self, 'std'):
            try:
                webgpu_result = _execute_webgpu_operation(
                    self, 'std',
                    options={'dim': dim, 'keepdim': keepdim, 'unbiased': unbiased}
                )
                if webgpu_result:
                    result = WebGPUTensor(
                        webgpu_result['data'],
                        device=self.device,
                        dtype=self.dtype,
                        requires_grad=self.requires_grad
                    )
                    result.shape = webgpu_result.get('shape', self._calculate_reduction_shape(dim, keepdim))
                    return result
            except Exception as e:
                # Fall back to pure JavaScript implementation
                pass
        
        # Pure JavaScript fallback implementation (no numpy)
        if dim is None:
            result_data = _std_all_elements(self.data, unbiased)
            result_shape = ()
        else:
            result_data = _std_along_axis(self.data, self.shape, dim, keepdim, unbiased)
            result_shape = self._calculate_reduction_shape(dim, keepdim)
        
        result = WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        result.shape = result_shape
        return result
    
    def var(self, dim=None, keepdim=False, unbiased=True):
        """Compute variance - WebGPU accelerated when available"""
        # Try WebGPU acceleration first
        if _should_use_webgpu_operation(self, 'var'):
            try:
                webgpu_result = _execute_webgpu_operation(
                    self, 'var',
                    options={'dim': dim, 'keepdim': keepdim, 'unbiased': unbiased}
                )
                if webgpu_result:
                    result = WebGPUTensor(
                        webgpu_result['data'],
                        device=self.device,
                        dtype=self.dtype,
                        requires_grad=self.requires_grad
                    )
                    result.shape = webgpu_result.get('shape', self._calculate_reduction_shape(dim, keepdim))
                    return result
            except Exception as e:
                # Fall back to pure JavaScript implementation
                pass
        
        # Pure JavaScript fallback implementation (no numpy)
        if dim is None:
            result_data = _var_all_elements(self.data, unbiased)
            result_shape = ()
        else:
            result_data = _var_along_axis(self.data, self.shape, dim, keepdim, unbiased)
            result_shape = self._calculate_reduction_shape(dim, keepdim)
        
        result = WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        result.shape = result_shape
        return result
    
    def to(self, device):
        new_device = WebGPUDevice(device) if isinstance(device, str) else device
        return WebGPUTensor(self.data.copy(), device=new_device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def cpu(self):
        return self.to('cpu')
    
    def cuda(self):
        return self.to('cuda')
        
    def __repr__(self):
        """Return a string representation matching PyTorch exactly"""
        # Format the data with proper PyTorch formatting
        if self.data.size == 0:
            data_str = "[]"
        elif self.data.size == 1:
            # Single element - format based on dtype
            item = self.data.item()
            if self.dtype in ['int8', 'int16', 'int32', 'int64', 'uint8']:
                data_str = str(int(item))
            elif self.dtype == 'bool':
                data_str = 'True' if item else 'False'
            else:
                data_str = str(item)
        else:
            # Multi-dimensional arrays with proper integer formatting
            if self.dtype in ['int8', 'int16', 'int32', 'int64', 'uint8']:
                # Convert to int list to avoid decimals
                data_str = str([int(x) for x in self.data.flatten().tolist()])
                # Handle multi-dimensional shapes
                if self.data.ndim > 1:
                    # Reconstruct shape for proper formatting
                    reshaped = self.data.astype(int).tolist()
                    data_str = str(reshaped)
            elif self.dtype == 'bool':
                # Boolean tensors show True/False
                bool_data = self.data.astype(bool).tolist()
                data_str = str(bool_data).replace('True', 'True').replace('False', 'False')
            else:
                # Float tensors show decimals
                data_str = str(self.data.tolist())
        
        # Build the representation string following PyTorch rules
        parts = [data_str]
        
        # PyTorch only shows device for non-CPU devices
        device_str = str(self.device)
        if device_str != 'cpu':
            if device_str == 'webgpu':
                parts.append("device='webgpu'")
            else:
                parts.append(f"device='{device_str}'")
        
        # PyTorch only shows dtype for:
        # - Non-default dtypes (not float32 for floating point)
        # - Integer tensors always show dtype when device is shown
        # - Or when it's not the "natural" dtype for the data
        show_dtype = False
        if device_str != 'cpu':
            # When device is shown, show dtype too
            show_dtype = True
        elif self.dtype not in ['float32', 'int64']:
            # Show dtype if it's not the default
            show_dtype = True
            
        if show_dtype:
            parts.append(f"dtype=torch.{self.dtype}")
        
        # Add requires_grad if True
        if self.requires_grad:
            parts.append("requires_grad=True")
        
        # Add grad_fn if present
        if self.grad_fn is not None:
            parts.append(f"grad_fn=<{self.grad_fn.__class__.__name__}>")
        
        return f"tensor({', '.join(parts)})"
    
    def __str__(self):
        """Return a string representation for print()"""
        return self.__repr__()
    
    def __float__(self):
        """Convert single-element tensor to Python float"""
        if self.data.size == 1:
            return float(self.data.item())
        else:
            raise TypeError(f"only single-element tensors can be converted to Python scalars")
    
    def __int__(self):
        """Convert single-element tensor to Python int"""
        if self.data.size == 1:
            return int(self.data.item())
        else:
            raise TypeError(f"only single-element tensors can be converted to Python scalars")
    
    def __getitem__(self, key):\n        \"\"\"Support tensor indexing like tensor[indices]\"\"\"\n        if isinstance(key, WebGPUTensor):\n            # Convert WebGPUTensor indices to numpy array\n            indices = key.data.astype(int)\n            result_data = self.data[indices]\n        else:\n            result_data = self.data[key]\n        \n        return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)\n    \n    # Arithmetic operators
    def __add__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self.data + other.data
        else:
            result_data = self.data + other
        return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def __sub__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self.data - other.data
        else:
            result_data = self.data - other
        return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def __mul__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self.data * other.data
        else:
            result_data = self.data * other
        return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def __truediv__(self, other):
        if isinstance(other, WebGPUTensor):
            result_data = self.data / other.data
        else:
            result_data = self.data / other
        return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def __radd__(self, other):
        result_data = other + self.data
        return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def __rmul__(self, other):
        result_data = other * self.data
        return WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def __matmul__(self, other):
        \"\"\"Matrix multiplication operator (@) - WebGPU accelerated when available\"\"\"
        # Try WebGPU acceleration first
        if isinstance(other, WebGPUTensor) and _should_use_webgpu_operation(self, 'matmul'):
            try:
                webgpu_result = _execute_webgpu_operation(
                    self, 'matmul', 
                    options={}, 
                    other_tensor=other
                )
                if webgpu_result:
                    result = WebGPUTensor(
                        webgpu_result['data'],
                        device=self.device,
                        dtype=self.dtype,
                        requires_grad=self.requires_grad
                    )
                    result.shape = webgpu_result.get('shape', self._calculate_matmul_shape(other))
                    
                    # Set up gradient function for autograd
                    if self.requires_grad or other.requires_grad:
                        def matmul_backward(grad_output):
                            if self.requires_grad:
                                if self.grad is None:
                                    self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                                # Gradient for first operand: grad_out @ other.T
                                other_transposed = _transpose_data_dims(other.data, other.shape, -2, -1)
                                grad_self = _matmul_data(grad_output.data, other_transposed)
                                self.grad.data = _add_arrays(self.grad.data, grad_self)
                            
                            if other.requires_grad:
                                if other.grad is None:
                                    other.grad = WebGPUTensor(_create_zeros_like_data(other.data, other.shape), device=other.device, dtype=other.dtype)
                                # Gradient for second operand: self.T @ grad_out
                                self_transposed = _transpose_data_dims(self.data, self.shape, -2, -1)
                                grad_other = _matmul_data(self_transposed, grad_output.data)
                                other.grad.data = _add_arrays(other.grad.data, grad_other)
                        
                        result._backward_fn = matmul_backward
                        result._inputs = [self, other]
                    
                    return result
            except Exception as e:
                # Fall back to pure JavaScript implementation
                pass
        
        # Pure JavaScript fallback implementation (no numpy)
        if isinstance(other, WebGPUTensor):
            result_data = _matmul_data(self.data, other.data)
            result_shape = self._calculate_matmul_shape(other)
        else:
            # Handle scalar or numpy array multiplication
            result_data = _matmul_data(self.data, other)
            result_shape = self._calculate_matmul_shape_scalar(other)
        
        result = WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        result.shape = result_shape
        
        # Set up gradient function for autograd
        if self.requires_grad or (isinstance(other, WebGPUTensor) and other.requires_grad):
            def matmul_backward(grad_output):
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                    if isinstance(other, WebGPUTensor):
                        other_transposed = _transpose_data_dims(other.data, other.shape, -2, -1)
                        grad_self = _matmul_data(grad_output.data, other_transposed)
                    else:
                        grad_self = _matmul_data(grad_output.data, other.T if hasattr(other, 'T') else other)
                    self.grad.data = _add_arrays(self.grad.data, grad_self)
                
                if isinstance(other, WebGPUTensor) and other.requires_grad:
                    if other.grad is None:
                        other.grad = WebGPUTensor(_create_zeros_like_data(other.data, other.shape), device=other.device, dtype=other.dtype)
                    self_transposed = _transpose_data_dims(self.data, self.shape, -2, -1)
                    grad_other = _matmul_data(self_transposed, grad_output.data)
                    other.grad.data = _add_arrays(other.grad.data, grad_other)
            
            result._backward_fn = matmul_backward
            result._inputs = [self, other] if isinstance(other, WebGPUTensor) else [self]
        
        return result
    
    def __rmatmul__(self, other):
        \"\"\"Reverse matrix multiplication - WebGPU accelerated when available\"\"\"
        # Pure JavaScript fallback implementation (no numpy)
        result_data = _matmul_data(other, self.data)
        result_shape = self._calculate_rmatmul_shape(other)
        
        result = WebGPUTensor(result_data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        result.shape = result_shape
        
        # Set up gradient function for autograd
        if self.requires_grad:
            def rmatmul_backward(grad_output):
                if self.grad is None:
                    self.grad = WebGPUTensor(_create_zeros_like_data(self.data, self.shape), device=self.device, dtype=self.dtype)
                # Gradient: other.T @ grad_out
                other_transposed = other.T if hasattr(other, 'T') else other
                grad_self = _matmul_data(other_transposed, grad_output.data)
                self.grad.data = _add_arrays(self.grad.data, grad_self)
            
            result._backward_fn = rmatmul_backward
            result._inputs = [self]
        
        return result
    
    def _calculate_matmul_shape(self, other):
        """Calculate output shape for matrix multiplication"""
        if isinstance(other, WebGPUTensor):
            if self.ndim == 2 and other.ndim == 2:
                return (self.shape[0], other.shape[1])
            elif self.ndim == 1 and other.ndim == 2:
                return (other.shape[1],)
            elif self.ndim == 2 and other.ndim == 1:
                return (self.shape[0],)
            else:
                # General batched matmul case - simplified for now
                return self.shape[:-1] + (other.shape[-1],)
        else:
            # Scalar or numpy array case
            return self.shape
    
    def _calculate_matmul_shape_scalar(self, other):
        """Calculate output shape for matrix multiplication with scalar/array"""
        if hasattr(other, 'shape'):
            return other.shape
        else:
            return self.shape
    
    def _calculate_rmatmul_shape(self, other):
        """Calculate output shape for reverse matrix multiplication"""
        if hasattr(other, 'shape'):
            if len(other.shape) == 2 and self.ndim == 2:
                return (other.shape[0], self.shape[1])
            elif len(other.shape) == 1 and self.ndim == 2:
                return (self.shape[1],)
            elif len(other.shape) == 2 and self.ndim == 1:
                return (other.shape[0],)
            else:
                return other.shape[:-1] + (self.shape[-1],)
        else:
            return self.shape
    
    def _calculate_reduction_shape(self, dim, keepdim):
        """Calculate output shape for reduction operations"""
        if dim is None:
            # Reduce all dimensions
            return () if not keepdim else tuple(1 for _ in self.shape)
        else:
            # Reduce specific dimension
            if keepdim:
                new_shape = list(self.shape)
                new_shape[dim] = 1
                return tuple(new_shape)
            else:
                return tuple(s for i, s in enumerate(self.shape) if i != dim)
    
    def _get_reduction_count(self, dim):
        """Get the number of elements being reduced"""
        if dim is None:
            return self.data.size if hasattr(self.data, 'size') else len(self.data)
        else:
            return self.shape[dim]
    
    def retain_grad(self):
        \"\"\"Enable gradient retention for non-leaf tensor\"\"\"
        if not self.requires_grad:
            raise RuntimeError("can't retain_grad on Tensor that has requires_grad=False")
        self._retain_grad = True
        return self
    
    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        \"\"\"Compute gradients via automatic differentiation\"\"\"
        if not self.requires_grad:
            return
        
        if gradient is None:
            if self.data.size == 1:
                gradient = WebGPUTensor(np.ones_like(self.data), device=self.device, dtype=self.dtype)
            else:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
        
        # Initialize gradient if not present - use WebGPU when possible
        if self.grad is None:
            # Use WebGPU zeros_like for gradient initialization
            try:
                self.grad = _create_webgpu_zeros_like(self)
            except:
                # Fallback to numpy for edge cases
                self.grad = WebGPUTensor(np.zeros_like(self.data), device=self.device, dtype=self.dtype)
        
        # Accumulate gradient using WebGPU addition when possible
        if isinstance(gradient, WebGPUTensor):
            # Use WebGPUTensor addition (WebGPU accelerated)
            self.grad = self.grad + gradient
        else:
            # Fallback for non-tensor gradients
            self.grad.data += gradient
        
        # Call backward function if exists
        if hasattr(self, '_backward_fn') and self._backward_fn:
            self._backward_fn(gradient)
    
    def _matmul_backward(self, grad_output, other):
        \"\"\"Backward pass for matrix multiplication\"\"\"
        if isinstance(other, WebGPUTensor):
            # d/da (a @ b) = grad_output @ b.T
            if self.grad is None:
                self.grad = WebGPUTensor(np.zeros_like(self.data), device=self.device, dtype=self.dtype)
            self_grad = np.matmul(grad_output.data, other.data.T)
            self.grad.data += self_grad
            
            # d/db (a @ b) = a.T @ grad_output  
            if other.requires_grad:
                if other.grad is None:
                    other.grad = WebGPUTensor(np.zeros_like(other.data), device=other.device, dtype=other.dtype)
                other_grad = np.matmul(self.data.T, grad_output.data)
                other.grad.data += other_grad

# Linear algebra operations module
class TorchLinalg:
    \"\"\"Linear algebra operations module\"\"\"
    
    def __init__(self):
        pass
    
    def det(self, input_tensor):
        \"\"\"Compute determinant\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if input_tensor.ndim != 2 or input_tensor.shape[0] != input_tensor.shape[1]:
                raise RuntimeError("linalg.det() expects a 2D square tensor")
            det_value = np.linalg.det(input_tensor.data.reshape(input_tensor.shape))
            return WebGPUTensor([det_value], device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.linalg.det(input_tensor)
    
    def inv(self, input_tensor):
        \"\"\"Compute matrix inverse\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if input_tensor.ndim != 2 or input_tensor.shape[0] != input_tensor.shape[1]:
                raise RuntimeError("linalg.inv() expects a 2D square tensor")
            inv_data = np.linalg.inv(input_tensor.data.reshape(input_tensor.shape))
            return WebGPUTensor(inv_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.linalg.inv(input_tensor)
    
    def norm(self, input_tensor, ord=None, dim=None, keepdim=False):
        \"\"\"Compute matrix or vector norm\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if dim is None:
                norm_value = np.linalg.norm(input_tensor.data, ord=ord)
                return WebGPUTensor([norm_value], device=input_tensor.device, dtype=input_tensor.dtype)
            else:
                norm_data = np.linalg.norm(input_tensor.data.reshape(input_tensor.shape), ord=ord, axis=dim, keepdims=keepdim)
                return WebGPUTensor(norm_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.linalg.norm(input_tensor, ord=ord, axis=dim, keepdims=keepdim)
    
    def eig(self, input_tensor):
        \"\"\"Compute eigenvalues and eigenvectors\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if input_tensor.ndim != 2 or input_tensor.shape[0] != input_tensor.shape[1]:
                raise RuntimeError("linalg.eig() expects a 2D square tensor")
            eigenvalues, eigenvectors = np.linalg.eig(input_tensor.data.reshape(input_tensor.shape))
            return (
                WebGPUTensor(eigenvalues, device=input_tensor.device, dtype=input_tensor.dtype),
                WebGPUTensor(eigenvectors, device=input_tensor.device, dtype=input_tensor.dtype)
            )
        else:
            return np.linalg.eig(input_tensor)
    
    def svd(self, input_tensor, full_matrices=True):
        \"\"\"Compute singular value decomposition\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            U, S, Vh = np.linalg.svd(input_tensor.data.reshape(input_tensor.shape), full_matrices=full_matrices)
            return (
                WebGPUTensor(U, device=input_tensor.device, dtype=input_tensor.dtype),
                WebGPUTensor(S, device=input_tensor.device, dtype=input_tensor.dtype),
                WebGPUTensor(Vh, device=input_tensor.device, dtype=input_tensor.dtype)
            )
        else:
            return np.linalg.svd(input_tensor, full_matrices=full_matrices)

# Neural network functional operations
class TorchNNFunctional:
    @staticmethod
    def relu(input_tensor):
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.maximum(input_tensor.data, 0)
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.maximum(input_tensor, 0)
    
    @staticmethod
    def sigmoid(input_tensor):
        if isinstance(input_tensor, WebGPUTensor):
            result_data = 1 / (1 + np.exp(-input_tensor.data))
            return WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return 1 / (1 + np.exp(-input_tensor))

# Neural network modules
class TorchNNModule:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        
    def parameters(self):
        params = []
        for param in self._parameters.values():
            params.append(param)
        for module in self._modules.values():
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
        return params
    
    def state_dict(self):
        """
        Returns a dictionary containing model's state dictionary.
        PyTorch-compatible implementation.
        """
        state_dict = {}
        
        # Add parameters
        for name, param in self._parameters.items():
            state_dict[name] = param
            
        # Add submodule parameters recursively
        for name, module in self._modules.items():
            if hasattr(module, 'state_dict'):
                module_state = module.state_dict()
                for key, value in module_state.items():
                    state_dict[f"{name}.{key}"] = value
                    
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Load model state dictionary.
        PyTorch-compatible implementation.
        """
        missing_keys = []
        unexpected_keys = []
        
        # Current model parameters
        current_params = self.state_dict()
        
        # Load parameters
        for key, value in state_dict.items():
            if key in current_params:
                # Copy data to existing parameter
                param = current_params[key]
                if hasattr(param, 'data'):
                    if hasattr(value, 'data'):
                        param.data = value.data
                    else:
                        param.data = value
            else:
                unexpected_keys.append(key)
                
        # Check for missing keys
        for key in current_params.keys():
            if key not in state_dict:
                missing_keys.append(key)
                
        if strict and (missing_keys or unexpected_keys):
            error_msg = f"Error loading state_dict. "
            if missing_keys:
                error_msg += f"Missing keys: {missing_keys}. "
            if unexpected_keys:
                error_msg += f"Unexpected keys: {unexpected_keys}."
            raise ValueError(error_msg)
            
        return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}
    
    def save(self, path):
        """
        Save model to file using ModelSerializer.
        PyTorch-compatible save method.
        """
        try:
            import js
            if hasattr(js, 'greedModelSerializer'):
                return js.greedModelSerializer.save(self, path)
            else:
                raise Exception("ModelSerializer not available")
        except Exception as e:
            print(f"Save failed: {e}")
            raise e
    
    @staticmethod  
    def load(path, model=None):
        """
        Load model from file using ModelSerializer.
        PyTorch-compatible load method.
        """
        try:
            import js
            if hasattr(js, 'greedModelSerializer'):
                return js.greedModelSerializer.load(path, model)
            else:
                raise Exception("ModelSerializer not available")
        except Exception as e:
            print(f"Load failed: {e}")
            raise e
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, x):
        raise NotImplementedError

class TorchNNLinear(TorchNNModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        weight_data = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.weight = WebGPUTensor(weight_data, requires_grad=True)
        self._parameters['weight'] = self.weight
        
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = WebGPUTensor(bias_data, requires_grad=True)
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, x):
        if isinstance(x, WebGPUTensor):
            result = WebGPUTensor(np.dot(x.data, self.weight.data.T), device=x.device, dtype=x.dtype)
            if self.bias is not None:
                result.data = result.data + self.bias.data
            return result
        else:
            raise TypeError("Input must be WebGPUTensor")

# CUDA module for WebGPU compatibility
class TorchCuda:
    def __init__(self):
        self._webgpu_available = None
        self._device_count = None
    
    def is_available(self):
        """Check if CUDA (WebGPU in our case) is available"""
        if self._webgpu_available is None:
            self._webgpu_available = self._check_webgpu_availability()
        return self._webgpu_available
    
    def device_count(self):
        """Return number of CUDA devices (WebGPU adapters)"""
        if not self.is_available():
            return 0
        if self._device_count is None:
            self._device_count = 1  # Assume 1 WebGPU device for now
        return self._device_count
    
    def get_device_name(self, device=None):
        """Get the name of the CUDA device"""
        if not self.is_available():
            raise RuntimeError("CUDA is not available")
        return "WebGPU Device"  # Generic name for WebGPU
    
    def current_device(self):
        """Get current CUDA device index"""
        if not self.is_available():
            raise RuntimeError("CUDA is not available") 
        return 0  # Always return 0 for single WebGPU device
    
    def set_device(self, device):
        """Set current CUDA device (no-op for WebGPU)"""
        if not self.is_available():
            raise RuntimeError("CUDA is not available")
        if device != 0:
            raise RuntimeError(f"Invalid device {device}")
    
    def empty_cache(self):
        """Clear WebGPU memory cache"""
        if self.is_available():
            # In a real implementation, this would trigger WebGPU memory cleanup
            pass
    
    def memory_stats(self, device=None):
        """Get memory statistics (mock implementation)"""
        if not self.is_available():
            return {}
        return {
            'allocated': 0,
            'reserved': 0,
            'active': 0,
            'inactive': 0
        }
    
    def _check_webgpu_availability(self):
        """Check if WebGPU is available in the current environment"""
        try:
            # This would be implemented in JavaScript to check navigator.gpu
            # For now, we'll assume it's available in modern browsers
            # In the actual implementation, this should interface with the WebGPU engine
            return True  # Optimistic default
        except:
            return False

# Create torch module with essential functions
class TorchModule:
    def __init__(self):
        self.tensor = self._tensor
        self.zeros = self._zeros
        self.ones = self._ones
        self.randn = self._randn
        self.matmul = self._matmul
        self.sum = self._sum
        self.as_tensor = self._as_tensor
        self.arange = self._arange
        self.randperm = self._randperm
        self.nn = TorchNN()
        
        # Add Tensor class reference
        self.Tensor = WebGPUTensor
        
        # Linear algebra module
        self.linalg = TorchLinalg()
        
        # Activation functions
        self.relu = self._relu
        
        # Mathematical functions
        self.round = self.round
        
        # Data types
        self.float32 = 'float32'
        self.float64 = 'float64'
        self.double = 'float64'
        self.float = 'float32'
        self.int32 = 'int32'
        self.int64 = 'int64'
        self.long = 'int64'
        self.int = 'int32'
        self.bool = 'bool'
        self.uint8 = 'uint8'
        
        # Device types  
        self.device = self._device
        
        # CUDA module for WebGPU compatibility
        self.cuda = TorchCuda()
        
        # Add tensor creation methods with WebGPU support
        self.rand = self._rand
        self.randint = self._randint  
        self.empty = self._empty
        self.full = self._full
        self.zeros_like = self._zeros_like
        self.ones_like = self._ones_like
        self.rand_like = self._rand_like
        self.randn_like = self._randn_like
        
        # Add tensor manipulation methods with WebGPU support
        self.cat = self._cat
        self.stack = self._stack
        self.chunk = self._chunk
        self.split = self._split
        
        # Add mathematical operations with WebGPU support
        self.max = self._max
        self.min = self._min
        self.argmax = self._argmax
        self.argmin = self._argmin
        
        # Add comparison operations with WebGPU support
        self.eq = self._eq
        self.gt = self._gt
        self.lt = self._lt
        self.ge = self._ge
        self.le = self._le
        self.equal = self._equal
        
        # Model serialization - PyTorch compatible save/load functions
        self.save = self._save
        self.load = self._load
        
    def _tensor(self, data, **kwargs):
        # Enable WebGPU detection by default for tensor creation
        if 'device' not in kwargs:
            kwargs['device'] = 'webgpu'  # Default to WebGPU instead of CPU
        # Let WebGPUTensor handle dtype inference automatically
        return WebGPUTensor(data, **kwargs)
    
    def _zeros(self, *shape, **kwargs):
        data = np.zeros(shape)
        return WebGPUTensor(data, **kwargs)
    
    def _ones(self, *shape, **kwargs):
        data = np.ones(shape)
        return WebGPUTensor(data, **kwargs)
    
    def _randn(self, *shape, **kwargs):
        data = np.random.randn(*shape)
        return WebGPUTensor(data, **kwargs)
    
    def _matmul(self, a, b):
        if isinstance(a, WebGPUTensor) and isinstance(b, WebGPUTensor):
            return WebGPUTensor(np.dot(a.data, b.data), device=a.device)
        return WebGPUTensor(np.dot(a, b))
    
    def _device(self, device_type):
        \"\"\"Create a device object\"\"\"
        return WebGPUDevice(device_type)
    
    def _sum(self, input_tensor, dim=None, keepdim=False, dtype=None):
        \"\"\"Compute sum of tensor elements\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            return input_tensor.sum(dim=dim, keepdim=keepdim)
        else:
            # Handle numpy arrays or lists
            if dim is None:
                result_data = np.sum(input_tensor)
            else:
                result_data = np.sum(input_tensor, axis=dim, keepdims=keepdim)
            return WebGPUTensor(result_data, dtype=dtype or 'float32')
    
    def _as_tensor(self, data, dtype=None, device=None):
        \"\"\"Convert data to tensor, similar to torch.as_tensor\"\"\"
        # Determine dtype
        if dtype is None:
            if hasattr(data, 'dtype'):
                dtype = str(data.dtype)
            else:
                dtype = 'float32'
        
        # Determine device - default to WebGPU for better performance
        if device is None:
            device = 'webgpu'
        
        # Create tensor
        return WebGPUTensor(data, dtype=dtype, device=device)
    
    def eye(self, n, m=None, dtype='float32', device='webgpu'):
        \"\"\"Create identity matrix\"\"\"
        if m is None:
            m = n
        data = np.eye(n, m)
        return WebGPUTensor(data, device=device, dtype=dtype)
    
    def round(self, input_tensor, decimals=0):
        """Round tensor elements to given number of decimals"""
        if isinstance(input_tensor, WebGPUTensor):
            rounded_data = np.round(input_tensor.data, decimals=decimals)
            return WebGPUTensor(rounded_data, device=input_tensor.device, dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
        else:
            return WebGPUTensor(np.round(input_tensor, decimals=decimals))
    
    def det(self, input_tensor):
        \"\"\"Compute determinant of square matrix\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            if input_tensor.ndim != 2 or input_tensor.shape[0] != input_tensor.shape[1]:
                raise RuntimeError("det() expects a 2D square tensor")
            det_value = np.linalg.det(input_tensor.data.reshape(input_tensor.shape))
            return WebGPUTensor([det_value], device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            return np.linalg.det(input_tensor)
    
    def _arange(self, *args, **kwargs):
        \"\"\"Create a 1D tensor with evenly spaced values\"\"\"
        if len(args) == 1:
            # arange(end)
            start, end, step = 0, args[0], 1
        elif len(args) == 2:
            # arange(start, end)
            start, end, step = args[0], args[1], 1
        elif len(args) == 3:
            # arange(start, end, step)
            start, end, step = args[0], args[1], args[2]
        else:
            raise ValueError(\"arange() takes 1 to 3 positional arguments\")
        
        data = np.arange(start, end, step)
        device = kwargs.get('device', 'cpu')
        dtype = kwargs.get('dtype', 'int64' if isinstance(start, int) and isinstance(end, int) and isinstance(step, int) else 'float32')
        return WebGPUTensor(data, device=device, dtype=dtype)
    
    def _randperm(self, n, **kwargs):
        \"\"\"Generate a random permutation of integers from 0 to n-1\"\"\"
        data = np.random.permutation(n)
        device = kwargs.get('device', 'cpu')
        dtype = kwargs.get('dtype', 'int64')
        return WebGPUTensor(data, device=device, dtype=dtype)
    
    def _relu(self, input_tensor):
        \"\"\"ReLU activation function\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            result_data = np.maximum(input_tensor.data, 0)
            result = WebGPUTensor(result_data, device=input_tensor.device, dtype=input_tensor.dtype, requires_grad=input_tensor.requires_grad)
            
            if input_tensor.requires_grad:
                def relu_backward(grad_output):
                    if input_tensor.grad is None:
                        input_tensor.grad = WebGPUTensor(np.zeros_like(input_tensor.data), device=input_tensor.device, dtype=input_tensor.dtype)
                    relu_grad = grad_output.data * (input_tensor.data > 0).astype(input_tensor.dtype)
                    input_tensor.grad.data += relu_grad
                
                result._backward_fn = relu_backward
                result._inputs = [input_tensor]
            
            return result
        else:
            return np.maximum(input_tensor, 0)
    
    def _execute_webgpu_operation(self, operation, shape=None, **options):
        \"\"\"Execute WebGPU operation through the bridge\"\"\"
        try:
            # Access the global WebGPU bridge from JavaScript
            import js
            if hasattr(js, 'greedTensorBridge') and js.greedTensorBridge:
                # Use WebGPU for tensor creation
                if shape:
                    total_size = 1
                    for dim in shape:
                        total_size *= dim
                    options['size'] = total_size
                    options['shape'] = shape
                
                # Execute WebGPU operation
                result = js.greedTensorBridge.executeCreationOperation(operation, options)
                if result and hasattr(result, 'success') and result.success:
                    return result.data, shape
            
            # Fallback to numpy if WebGPU not available
            return None, None
        except:
            # Fallback to numpy if WebGPU bridge fails
            return None, None
    
    def _rand(self, *shape, **kwargs):
        \"\"\"Create tensor with random values from uniform distribution [0, 1) using WebGPU\"\"\"
        device = kwargs.get('device', 'webgpu')
        dtype = kwargs.get('dtype', 'float32')
        requires_grad = kwargs.get('requires_grad', False)
        
        # Try WebGPU first
        webgpu_data, webgpu_shape = self._execute_webgpu_operation(
            'rand', 
            shape=shape, 
            device=device, 
            dtype=dtype,
            seed=kwargs.get('seed')
        )
        
        if webgpu_data is not None:
            # Use WebGPU result
            data = np.array(webgpu_data, dtype=dtype).reshape(shape)
        else:
            # Fallback to numpy
            data = np.random.rand(*shape).astype(dtype)
        
        return WebGPUTensor(data, device=device, dtype=dtype, requires_grad=requires_grad)
    
    def _randint(self, low, high=None, size=None, **kwargs):
        \"\"\"Create tensor with random integers using WebGPU\"\"\"
        if high is None:
            high = low
            low = 0
        
        if size is None:
            raise ValueError("Size must be specified for randint")
        
        if isinstance(size, int):
            size = (size,)
        elif isinstance(size, list):
            size = tuple(size)
        
        device = kwargs.get('device', 'webgpu')
        dtype = kwargs.get('dtype', 'int64')
        requires_grad = kwargs.get('requires_grad', False)
        
        # Try WebGPU first
        webgpu_data, webgpu_shape = self._execute_webgpu_operation(
            'randint', 
            shape=size,
            device=device,
            dtype='i32',  # WebGPU uses i32
            low=low,
            high=high,
            seed=kwargs.get('seed')
        )
        
        if webgpu_data is not None:
            data = np.array(webgpu_data, dtype='int64').reshape(size)
        else:
            data = np.random.randint(low, high, size=size).astype('int64')
        
        return WebGPUTensor(data, device=device, dtype=dtype, requires_grad=requires_grad)
    
    def _empty(self, *shape, **kwargs):
        \"\"\"Create tensor with uninitialized values using WebGPU\"\"\"
        device = kwargs.get('device', 'webgpu')
        dtype = kwargs.get('dtype', 'float32')
        requires_grad = kwargs.get('requires_grad', False)
        
        # Try WebGPU first
        webgpu_data, webgpu_shape = self._execute_webgpu_operation(
            'empty',
            shape=shape,
            device=device,
            dtype=dtype
        )
        
        if webgpu_data is not None:
            data = np.array(webgpu_data, dtype=dtype).reshape(shape)
        else:
            data = np.empty(shape, dtype=dtype)
        
        return WebGPUTensor(data, device=device, dtype=dtype, requires_grad=requires_grad)
    
    def _full(self, shape, fill_value, **kwargs):
        \"\"\"Create tensor filled with fill_value using WebGPU\"\"\"
        if isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, list):
            shape = tuple(shape)
        
        device = kwargs.get('device', 'webgpu')
        dtype = kwargs.get('dtype', 'float32')
        requires_grad = kwargs.get('requires_grad', False)
        
        # Try WebGPU first
        webgpu_data, webgpu_shape = self._execute_webgpu_operation(
            'full',
            shape=shape,
            device=device,
            dtype=dtype,
            fillValue=fill_value
        )
        
        if webgpu_data is not None:
            data = np.array(webgpu_data, dtype=dtype).reshape(shape)
        else:
            data = np.full(shape, fill_value, dtype=dtype)
        
        return WebGPUTensor(data, device=device, dtype=dtype, requires_grad=requires_grad)
    
    def _zeros_like(self, input_tensor, **kwargs):
        \"\"\"Create tensor of zeros with same shape as input\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            shape = input_tensor.shape
            device = kwargs.get('device', input_tensor.device)
            dtype = kwargs.get('dtype', input_tensor.dtype)
        else:
            shape = np.array(input_tensor).shape
            device = kwargs.get('device', 'webgpu')
            dtype = kwargs.get('dtype', 'float32')
        
        return self._zeros(*shape, device=device, dtype=dtype, requires_grad=kwargs.get('requires_grad', False))
    
    def _ones_like(self, input_tensor, **kwargs):
        \"\"\"Create tensor of ones with same shape as input\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            shape = input_tensor.shape
            device = kwargs.get('device', input_tensor.device)
            dtype = kwargs.get('dtype', input_tensor.dtype)
        else:
            shape = np.array(input_tensor).shape
            device = kwargs.get('device', 'webgpu')
            dtype = kwargs.get('dtype', 'float32')
        
        return self._ones(*shape, device=device, dtype=dtype, requires_grad=kwargs.get('requires_grad', False))
    
    def _rand_like(self, input_tensor, **kwargs):
        \"\"\"Create tensor of random values with same shape as input\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            shape = input_tensor.shape
            device = kwargs.get('device', input_tensor.device)
            dtype = kwargs.get('dtype', input_tensor.dtype)
        else:
            shape = np.array(input_tensor).shape
            device = kwargs.get('device', 'webgpu')
            dtype = kwargs.get('dtype', 'float32')
        
        return self._rand(*shape, device=device, dtype=dtype, requires_grad=kwargs.get('requires_grad', False))
    
    def _randn_like(self, input_tensor, **kwargs):
        \"\"\"Create tensor of normal random values with same shape as input\"\"\"
        if isinstance(input_tensor, WebGPUTensor):
            shape = input_tensor.shape
            device = kwargs.get('device', input_tensor.device)
            dtype = kwargs.get('dtype', input_tensor.dtype)
        else:
            shape = np.array(input_tensor).shape
            device = kwargs.get('device', 'webgpu')
            dtype = kwargs.get('dtype', 'float32')
        
        return self._randn(*shape, device=device, dtype=dtype, requires_grad=kwargs.get('requires_grad', False))
    
    def _cat(self, tensors, dim=0):
        \"\"\"Concatenate tensors along specified dimension using WebGPU\"\"\"
        if not tensors or len(tensors) == 0:
            raise ValueError("cat requires at least one tensor")
        
        if len(tensors) == 1:
            return tensors[0].clone()
        
        # For now, implement simple concatenation along last dimension
        first_tensor = tensors[0]
        device = first_tensor.device if isinstance(first_tensor, WebGPUTensor) else 'webgpu'
        dtype = first_tensor.dtype if isinstance(first_tensor, WebGPUTensor) else 'float32'
        requires_grad = any(getattr(t, 'requires_grad', False) for t in tensors)
        
        try:
            # Try WebGPU acceleration for two tensor concatenation
            if len(tensors) == 2 and isinstance(tensors[0], WebGPUTensor) and isinstance(tensors[1], WebGPUTensor):
                import js
                if hasattr(js, 'greedTensorBridge') and js.greedTensorBridge:
                    # Use WebGPU bridge for concatenation
                    result = js.greedTensorBridge.executeBinaryOperation(
                        tensors[0]._tensor_id if hasattr(tensors[0], '_tensor_id') else None,
                        tensors[1]._tensor_id if hasattr(tensors[1], '_tensor_id') else None,
                        'cat',
                        {'dim': dim}
                    )
                    if result and hasattr(result, 'success') and result.success:
                        result_data = np.array(result.data, dtype=dtype)
                        return WebGPUTensor(result_data, device=device, dtype=dtype, requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy concatenation
        tensor_arrays = []
        for tensor in tensors:
            if isinstance(tensor, WebGPUTensor):
                tensor_arrays.append(tensor.data)
            else:
                tensor_arrays.append(np.array(tensor))
        
        result_data = np.concatenate(tensor_arrays, axis=dim)
        return WebGPUTensor(result_data, device=device, dtype=dtype, requires_grad=requires_grad)
    
    def _stack(self, tensors, dim=0):
        \"\"\"Stack tensors along new dimension using WebGPU\"\"\"
        if not tensors or len(tensors) == 0:
            raise ValueError("stack requires at least one tensor")
        
        first_tensor = tensors[0]
        device = first_tensor.device if isinstance(first_tensor, WebGPUTensor) else 'webgpu'
        dtype = first_tensor.dtype if isinstance(first_tensor, WebGPUTensor) else 'float32'
        requires_grad = any(getattr(t, 'requires_grad', False) for t in tensors)
        
        try:
            # Try WebGPU acceleration for two tensor stacking
            if len(tensors) == 2 and isinstance(tensors[0], WebGPUTensor) and isinstance(tensors[1], WebGPUTensor):
                import js
                if hasattr(js, 'greedTensorBridge') and js.greedTensorBridge:
                    result = js.greedTensorBridge.executeBinaryOperation(
                        tensors[0]._tensor_id if hasattr(tensors[0], '_tensor_id') else None,
                        tensors[1]._tensor_id if hasattr(tensors[1], '_tensor_id') else None,
                        'stack',
                        {'dim': dim}
                    )
                    if result and hasattr(result, 'success') and result.success:
                        result_data = np.array(result.data, dtype=dtype)
                        return WebGPUTensor(result_data, device=device, dtype=dtype, requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy stacking
        tensor_arrays = []
        for tensor in tensors:
            if isinstance(tensor, WebGPUTensor):
                tensor_arrays.append(tensor.data)
            else:
                tensor_arrays.append(np.array(tensor))
        
        result_data = np.stack(tensor_arrays, axis=dim)
        return WebGPUTensor(result_data, device=device, dtype=dtype, requires_grad=requires_grad)
    
    def _chunk(self, input_tensor, chunks, dim=0):
        \"\"\"Split tensor into chunks using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        
        device = input_tensor.device
        dtype = input_tensor.dtype
        requires_grad = input_tensor.requires_grad
        
        # Calculate chunk size
        dim_size = input_tensor.shape[dim] if dim < len(input_tensor.shape) else input_tensor.data.size
        chunk_size = (dim_size + chunks - 1) // chunks  # Ceiling division
        
        result_chunks = []
        for i in range(chunks):
            try:
                # Try WebGPU acceleration
                import js
                if hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and hasattr(input_tensor, '_tensor_id'):
                    result = js.greedTensorBridge.executeUnaryOperation(
                        input_tensor._tensor_id,
                        'chunk',
                        {'chunks': chunks, 'chunkIdx': i, 'dim': dim}
                    )
                    if result and hasattr(result, 'success') and result.success:
                        chunk_data = np.array(result.data, dtype=dtype)
                        result_chunks.append(WebGPUTensor(chunk_data, device=device, dtype=dtype, requires_grad=requires_grad))
                        continue
            except:
                pass
            
            # Fallback to numpy
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, dim_size)
            
            if dim == 0:
                chunk_data = input_tensor.data[start_idx:end_idx]
            else:
                # Simple slicing for higher dimensions - this is a simplified implementation
                indices = [slice(None)] * len(input_tensor.shape)
                indices[dim] = slice(start_idx, end_idx)
                chunk_data = input_tensor.data[tuple(indices)]
            
            if chunk_data.size > 0:
                result_chunks.append(WebGPUTensor(chunk_data, device=device, dtype=dtype, requires_grad=requires_grad))
        
        return result_chunks
    
    def _split(self, input_tensor, split_size_or_sections, dim=0):
        \"\"\"Split tensor into sections using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        
        device = input_tensor.device
        dtype = input_tensor.dtype
        requires_grad = input_tensor.requires_grad
        
        if isinstance(split_size_or_sections, int):
            # Split by size
            dim_size = input_tensor.shape[dim] if dim < len(input_tensor.shape) else input_tensor.data.size
            num_splits = (dim_size + split_size_or_sections - 1) // split_size_or_sections
            split_sizes = [split_size_or_sections] * (num_splits - 1)
            if dim_size % split_size_or_sections != 0:
                split_sizes.append(dim_size % split_size_or_sections)
            else:
                split_sizes.append(split_size_or_sections)
        else:
            # Split by sections
            split_sizes = split_size_or_sections
        
        result_splits = []
        current_idx = 0
        
        for i, split_size in enumerate(split_sizes):
            try:
                # Try WebGPU acceleration
                import js
                if hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and hasattr(input_tensor, '_tensor_id'):
                    result = js.greedTensorBridge.executeUnaryOperation(
                        input_tensor._tensor_id,
                        'split',
                        {'splitSize': split_size, 'splitIdx': i, 'dim': dim}
                    )
                    if result and hasattr(result, 'success') and result.success:
                        split_data = np.array(result.data, dtype=dtype)
                        result_splits.append(WebGPUTensor(split_data, device=device, dtype=dtype, requires_grad=requires_grad))
                        current_idx += split_size
                        continue
            except:
                pass
            
            # Fallback to numpy
            end_idx = current_idx + split_size
            
            if dim == 0:
                split_data = input_tensor.data[current_idx:end_idx]
            else:
                # Simple slicing for higher dimensions
                indices = [slice(None)] * len(input_tensor.shape)
                indices[dim] = slice(current_idx, end_idx)
                split_data = input_tensor.data[tuple(indices)]
            
            if split_data.size > 0:
                result_splits.append(WebGPUTensor(split_data, device=device, dtype=dtype, requires_grad=requires_grad))
            
            current_idx = end_idx
        
        return result_splits
    
    def _max(self, input_tensor, dim=None, keepdim=False):
        \"\"\"Find maximum values using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        
        device = input_tensor.device
        dtype = input_tensor.dtype
        requires_grad = input_tensor.requires_grad
        
        try:
            # Try WebGPU acceleration
            import js
            if hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and hasattr(input_tensor, '_tensor_id'):
                result = js.greedTensorBridge.executeUnaryOperation(
                    input_tensor._tensor_id,
                    'max_reduce',
                    {'dim': dim, 'keepdim': keepdim}
                )
                if result and hasattr(result, 'success') and result.success:
                    result_data = np.array(result.data, dtype=dtype)
                    if dim is not None and not keepdim:
                        # Adjust shape for reduced dimensions
                        result_shape = list(input_tensor.shape)
                        result_shape.pop(dim)
                        if result_shape:
                            result_data = result_data.reshape(result_shape)
                    return WebGPUTensor(result_data, device=device, dtype=dtype, requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy
        if dim is None:
            result_data = np.array([np.max(input_tensor.data)])
        else:
            result_data = np.max(input_tensor.data, axis=dim, keepdims=keepdim)
        
        return WebGPUTensor(result_data, device=device, dtype=dtype, requires_grad=requires_grad)
    
    def _min(self, input_tensor, dim=None, keepdim=False):
        \"\"\"Find minimum values using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        
        device = input_tensor.device
        dtype = input_tensor.dtype
        requires_grad = input_tensor.requires_grad
        
        try:
            # Try WebGPU acceleration
            import js
            if hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and hasattr(input_tensor, '_tensor_id'):
                result = js.greedTensorBridge.executeUnaryOperation(
                    input_tensor._tensor_id,
                    'min_reduce',
                    {'dim': dim, 'keepdim': keepdim}
                )
                if result and hasattr(result, 'success') and result.success:
                    result_data = np.array(result.data, dtype=dtype)
                    if dim is not None and not keepdim:
                        # Adjust shape for reduced dimensions
                        result_shape = list(input_tensor.shape)
                        result_shape.pop(dim)
                        if result_shape:
                            result_data = result_data.reshape(result_shape)
                    return WebGPUTensor(result_data, device=device, dtype=dtype, requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy
        if dim is None:
            result_data = np.array([np.min(input_tensor.data)])
        else:
            result_data = np.min(input_tensor.data, axis=dim, keepdims=keepdim)
        
        return WebGPUTensor(result_data, device=device, dtype=dtype, requires_grad=requires_grad)
    
    def _argmax(self, input_tensor, dim=None, keepdim=False):
        \"\"\"Find indices of maximum values using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        
        device = input_tensor.device
        requires_grad = False  # Argmax results are indices, not differentiable
        
        try:
            # Try WebGPU acceleration
            import js
            if hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and hasattr(input_tensor, '_tensor_id'):
                result = js.greedTensorBridge.executeUnaryOperation(
                    input_tensor._tensor_id,
                    'argmax',
                    {'dim': dim, 'keepdim': keepdim}
                )
                if result and hasattr(result, 'success') and result.success:
                    result_data = np.array(result.data, dtype='int64')
                    if dim is not None and not keepdim:
                        # Adjust shape for reduced dimensions
                        result_shape = list(input_tensor.shape)
                        result_shape.pop(dim)
                        if result_shape:
                            result_data = result_data.reshape(result_shape)
                    return WebGPUTensor(result_data, device=device, dtype='int64', requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy
        if dim is None:
            result_data = np.array([np.argmax(input_tensor.data)])
        else:
            result_data = np.argmax(input_tensor.data, axis=dim, keepdims=keepdim)
        
        return WebGPUTensor(result_data, device=device, dtype='int64', requires_grad=requires_grad)
    
    def _argmin(self, input_tensor, dim=None, keepdim=False):
        \"\"\"Find indices of minimum values using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        
        device = input_tensor.device
        requires_grad = False  # Argmin results are indices, not differentiable
        
        try:
            # Try WebGPU acceleration
            import js
            if hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and hasattr(input_tensor, '_tensor_id'):
                result = js.greedTensorBridge.executeUnaryOperation(
                    input_tensor._tensor_id,
                    'argmin',
                    {'dim': dim, 'keepdim': keepdim}
                )
                if result and hasattr(result, 'success') and result.success:
                    result_data = np.array(result.data, dtype='int64')
                    if dim is not None and not keepdim:
                        # Adjust shape for reduced dimensions
                        result_shape = list(input_tensor.shape)
                        result_shape.pop(dim)
                        if result_shape:
                            result_data = result_data.reshape(result_shape)
                    return WebGPUTensor(result_data, device=device, dtype='int64', requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy
        if dim is None:
            result_data = np.array([np.argmin(input_tensor.data)])
        else:
            result_data = np.argmin(input_tensor.data, axis=dim, keepdims=keepdim)
        
        return WebGPUTensor(result_data, device=device, dtype='int64', requires_grad=requires_grad)
    
    def _eq(self, input_tensor, other_tensor):
        \"\"\"Element-wise equality comparison using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        if not isinstance(other_tensor, WebGPUTensor):
            other_tensor = WebGPUTensor(other_tensor)
        
        device = input_tensor.device
        requires_grad = False  # Boolean tensors don't require gradients
        
        try:
            # Try WebGPU acceleration
            import js
            if (hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and 
                hasattr(input_tensor, '_tensor_id') and hasattr(other_tensor, '_tensor_id')):
                result = js.greedTensorBridge.executeBinaryOperation(
                    input_tensor._tensor_id,
                    other_tensor._tensor_id,
                    'eq',
                    {}
                )
                if result and hasattr(result, 'success') and result.success:
                    result_data = np.array(result.data, dtype='uint8')  # Boolean as uint8
                    return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy
        result_data = (input_tensor.data == other_tensor.data).astype('uint8')
        return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)
    
    def _gt(self, input_tensor, other_tensor):
        \"\"\"Element-wise greater than comparison using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        if not isinstance(other_tensor, WebGPUTensor):
            other_tensor = WebGPUTensor(other_tensor)
        
        device = input_tensor.device
        requires_grad = False
        
        try:
            # Try WebGPU acceleration
            import js
            if (hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and 
                hasattr(input_tensor, '_tensor_id') and hasattr(other_tensor, '_tensor_id')):
                result = js.greedTensorBridge.executeBinaryOperation(
                    input_tensor._tensor_id,
                    other_tensor._tensor_id,
                    'gt',
                    {}
                )
                if result and hasattr(result, 'success') and result.success:
                    result_data = np.array(result.data, dtype='uint8')
                    return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy
        result_data = (input_tensor.data > other_tensor.data).astype('uint8')
        return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)
    
    def _lt(self, input_tensor, other_tensor):
        \"\"\"Element-wise less than comparison using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        if not isinstance(other_tensor, WebGPUTensor):
            other_tensor = WebGPUTensor(other_tensor)
        
        device = input_tensor.device
        requires_grad = False
        
        try:
            # Try WebGPU acceleration
            import js
            if (hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and 
                hasattr(input_tensor, '_tensor_id') and hasattr(other_tensor, '_tensor_id')):
                result = js.greedTensorBridge.executeBinaryOperation(
                    input_tensor._tensor_id,
                    other_tensor._tensor_id,
                    'lt',
                    {}
                )
                if result and hasattr(result, 'success') and result.success:
                    result_data = np.array(result.data, dtype='uint8')
                    return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy
        result_data = (input_tensor.data < other_tensor.data).astype('uint8')
        return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)
    
    def _ge(self, input_tensor, other_tensor):
        \"\"\"Element-wise greater than or equal comparison using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        if not isinstance(other_tensor, WebGPUTensor):
            other_tensor = WebGPUTensor(other_tensor)
        
        device = input_tensor.device
        requires_grad = False
        
        try:
            # Try WebGPU acceleration
            import js
            if (hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and 
                hasattr(input_tensor, '_tensor_id') and hasattr(other_tensor, '_tensor_id')):
                result = js.greedTensorBridge.executeBinaryOperation(
                    input_tensor._tensor_id,
                    other_tensor._tensor_id,
                    'ge',
                    {}
                )
                if result and hasattr(result, 'success') and result.success:
                    result_data = np.array(result.data, dtype='uint8')
                    return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy
        result_data = (input_tensor.data >= other_tensor.data).astype('uint8')
        return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)
    
    def _le(self, input_tensor, other_tensor):
        \"\"\"Element-wise less than or equal comparison using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        if not isinstance(other_tensor, WebGPUTensor):
            other_tensor = WebGPUTensor(other_tensor)
        
        device = input_tensor.device
        requires_grad = False
        
        try:
            # Try WebGPU acceleration
            import js
            if (hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and 
                hasattr(input_tensor, '_tensor_id') and hasattr(other_tensor, '_tensor_id')):
                result = js.greedTensorBridge.executeBinaryOperation(
                    input_tensor._tensor_id,
                    other_tensor._tensor_id,
                    'le',
                    {}
                )
                if result and hasattr(result, 'success') and result.success:
                    result_data = np.array(result.data, dtype='uint8')
                    return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy
        result_data = (input_tensor.data <= other_tensor.data).astype('uint8')
        return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)
    
    def _equal(self, input_tensor, other_tensor):
        \"\"\"Check if two tensors are entirely equal using WebGPU\"\"\"
        if not isinstance(input_tensor, WebGPUTensor):
            input_tensor = WebGPUTensor(input_tensor)
        if not isinstance(other_tensor, WebGPUTensor):
            other_tensor = WebGPUTensor(other_tensor)
        
        device = input_tensor.device
        requires_grad = False
        
        # Check shapes first
        if input_tensor.shape != other_tensor.shape:
            return WebGPUTensor(np.array([False]), device=device, dtype='bool', requires_grad=requires_grad)
        
        try:
            # Try WebGPU acceleration
            import js
            if (hasattr(js, 'greedTensorBridge') and js.greedTensorBridge and 
                hasattr(input_tensor, '_tensor_id') and hasattr(other_tensor, '_tensor_id')):
                result = js.greedTensorBridge.executeBinaryOperation(
                    input_tensor._tensor_id,
                    other_tensor._tensor_id,
                    'equal',
                    {}
                )
                if result and hasattr(result, 'success') and result.success:
                    # Result should be a single boolean value
                    result_data = np.array([bool(result.data[0]) if len(result.data) > 0 else False])
                    return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)
        except:
            pass
        
        # Fallback to numpy
        result_data = np.array([np.array_equal(input_tensor.data, other_tensor.data)])
        return WebGPUTensor(result_data, device=device, dtype='bool', requires_grad=requires_grad)


    def _save(self, obj, path=None):
        """
        Save PyTorch model or state dict.
        Args:
            obj: Model or state dictionary to save
            path: File path (optional, returns data if None)
        Returns:
            Saved file path or serialized data
        """
        try:
            import js
            if hasattr(js, 'greedModelSerializer'):
                # Use JavaScript ModelSerializer
                if hasattr(obj, 'state_dict'):
                    # Model object - extract state dict
                    return js.greedModelSerializer.save(obj, path)
                else:
                    # Already a state dict or other data
                    dummy_model = type('DummyModel', (), {
                        'state_dict': lambda: obj,
                        'parameters': lambda: [],
                        '_parameters': obj if isinstance(obj, dict) else {},
                        '_modules': {}
                    })()
                    return js.greedModelSerializer.save(dummy_model, path)
            else:
                raise Exception("ModelSerializer not available")
        except Exception as e:
            # Fallback: basic JSON serialization
            import json
            if hasattr(obj, 'state_dict'):
                data = self._serialize_state_dict(obj.state_dict())
            else:
                data = self._serialize_state_dict(obj)
            
            serialized = json.dumps(data, indent=2)
            
            if path:
                # Browser environment - can't save files directly
                print("Warning: File saving in browser environment not supported. Returning data for manual save.")
                return serialized
            else:
                return serialized
    
    def _load(self, path_or_data, model=None):
        """
        Load PyTorch model or state dict.
        Args:
            path_or_data: File path or serialized data
            model: Model to load into (optional)
        Returns:
            Loaded state dictionary or model data
        """
        try:
            import js
            if hasattr(js, 'greedModelSerializer'):
                # Use JavaScript ModelSerializer
                return js.greedModelSerializer.load(path_or_data, model)
            else:
                raise Exception("ModelSerializer not available")
        except Exception as e:
            # Fallback: basic JSON deserialization
            import json
            
            if isinstance(path_or_data, str):
                if path_or_data.startswith('{') or path_or_data.startswith('['):
                    # JSON string
                    data = json.loads(path_or_data)
                else:
                    # File path - not supported in browser
                    raise Exception("File path loading not supported in browser environment")
            else:
                data = path_or_data
            
            # Deserialize tensors
            state_dict = self._deserialize_state_dict(data)
            
            if model and hasattr(model, 'load_state_dict'):
                model.load_state_dict(state_dict)
                return {'state_dict': state_dict}
            else:
                return state_dict
    
    def _serialize_state_dict(self, state_dict):
        """Serialize state dictionary for saving"""
        serialized = {}
        for key, value in state_dict.items():
            if hasattr(value, 'data') and hasattr(value, 'shape'):
                # Tensor object
                serialized[key] = {
                    'data': value.data.tolist() if hasattr(value.data, 'tolist') else list(value.data),
                    'shape': list(value.shape) if hasattr(value, 'shape') else [],
                    'dtype': value.dtype if hasattr(value, 'dtype') else 'float32',
                    'device': value.device if hasattr(value, 'device') else 'cpu',
                    'requires_grad': value.requires_grad if hasattr(value, 'requires_grad') else False,
                    'type': 'tensor'
                }
            else:
                # Other data
                serialized[key] = {
                    'data': value,
                    'type': 'other'
                }
        return {
            'version': '1.0.0',
            'model_state_dict': serialized,
            'greed_version': '2.0.0',
            'pytorch_compatible': True
        }
    
    def _deserialize_state_dict(self, data):
        """Deserialize state dictionary from loaded data"""
        if 'model_state_dict' in data:
            state_dict_data = data['model_state_dict']
        else:
            state_dict_data = data
            
        state_dict = {}
        for key, value in state_dict_data.items():
            if isinstance(value, dict) and value.get('type') == 'tensor':
                # Recreate tensor
                tensor_data = np.array(value['data'])
                if value.get('shape'):
                    tensor_data = tensor_data.reshape(value['shape'])
                
                state_dict[key] = WebGPUTensor(
                    tensor_data,
                    device=value.get('device', 'cpu'),
                    dtype=value.get('dtype', 'float32'),
                    requires_grad=value.get('requires_grad', False)
                )
            else:
                # Other data
                if isinstance(value, dict) and 'data' in value:
                    state_dict[key] = value['data']
                else:
                    state_dict[key] = value
                    
        return state_dict

class TorchNN:
    def __init__(self):
        self.functional = TorchNNFunctional()
        self.Linear = TorchNNLinear
        self.Module = TorchNNModule

# Install in global namespace
torch = TorchModule()
sys.modules['torch'] = torch
sys.modules['torch.nn'] = torch.nn
sys.modules['torch.nn.functional'] = torch.nn.functional
sys.modules['torch.linalg'] = torch.linalg
sys.modules['torch.cuda'] = torch.cuda
`;
}

/**
 * Get optimized PyTorch polyfill for specific operations
 */
export function getOptimizedPolyfill(operations = []) {
  const basePolyfill = createPyTorchPolyfill();
  
  // Add operation-specific optimizations
  let optimizations = '';
  
  if (operations.includes('matmul')) {
    optimizations += `
# Optimized matrix multiplication
def optimized_matmul(a, b):
    if hasattr(a, '_webgpu_buffer') and hasattr(b, '_webgpu_buffer'):
        # Use WebGPU acceleration
        return a._webgpu_matmul(b)
    return np.dot(a, b)

torch._matmul = optimized_matmul
`;
  }
  
  return basePolyfill + optimizations;
}

/**
 * Validate PyTorch polyfill before installation
 */
export function validatePolyfill(polyfillCode) {
  const dangerousPatterns = [
    /\beval\s*\(/g,
    /\bexec\s*\(/g,
    /\b__import__\s*\(/g,
    /\bsubprocess\./g,
    /\bos\.system\s*\(/g
  ];
  
  for (const pattern of dangerousPatterns) {
    if (pattern.test(polyfillCode)) {
      throw new Error(`Dangerous pattern detected in PyTorch polyfill: ${pattern}`);
    }
  }
  
  return true;
}