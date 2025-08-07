/**
 * WebGPU PyTorch Runtime JavaScript Bridge
 * Provides the Python PyTorch runtime as a JavaScript string for Pyodide integration
 */

export function createWebGPUPyTorchRuntime() {
  return `# GreedJS WebGPU PyTorch Runtime
# Pure Python PyTorch implementation with direct WebGPU acceleration
# Users write: import torch; x = torch.tensor([1,2,3]) 
# Backend: All operations run on WebGPU compute shaders

import js
from typing import List, Optional, Union, Tuple
import array

class WebGPUDevice:
    """Device abstraction for WebGPU operations"""
    def __init__(self, device_type: str = 'webgpu'):
        self.type = device_type
        self.is_available = js.navigator.gpu is not js.undefined
    
    def __str__(self):
        return f"device(type='{self.type}')"
    
    def __repr__(self):
        return self.__str__()

# Global device instances
cuda = WebGPUDevice('webgpu')  # For PyTorch compatibility (cuda -> webgpu)
webgpu = WebGPUDevice('webgpu')

class WebGPUTensor:
    """Pure WebGPU PyTorch-compatible tensor implementation"""
    
    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        self.device = device or webgpu
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        
        # Convert input data to appropriate format
        if isinstance(data, WebGPUTensor):
            self._webgpu_id = data._webgpu_id
            self.shape = data.shape
            self.dtype = data.dtype
        elif isinstance(data, (list, tuple)):
            # Create WebGPU tensor from Python list/tuple
            self.shape = self._infer_shape(data)
            self.dtype = dtype or self._infer_dtype(data)
            self._webgpu_id = self._create_webgpu_tensor(data)
        else:
            # Handle scalar or other types
            if not isinstance(data, (list, tuple)):
                data = [data]
            self.shape = [len(data)] if isinstance(data, (list, tuple)) else [1]
            self.dtype = dtype or self._infer_dtype(data)
            self._webgpu_id = self._create_webgpu_tensor(data)
    
    def _create_webgpu_tensor(self, data):
        """Create WebGPU tensor through JavaScript bridge"""
        # Flatten the data for WebGPU
        flat_data = self._flatten_data(data)
        
        # Call JavaScript tensor bridge to create WebGPU tensor
        result = js.greedInstance.tensorBridge.createWebGPUTensor(
            flat_data,
            self.shape, 
            self.dtype,
            str(self.device)
        )
        
        return result.id
    
    def _flatten_data(self, data):
        """Recursively flatten nested lists/tuples to flat array"""
        def flatten(lst):
            result = []
            for item in lst:
                if isinstance(item, (list, tuple)):
                    result.extend(flatten(item))
                else:
                    result.append(float(item))
            return result
        
        return flatten(data)
    
    def _infer_shape(self, data):
        """Infer tensor shape from nested lists/tuples"""
        if not isinstance(data, (list, tuple)):
            return []
        
        shape = []
        current = data
        while isinstance(current, (list, tuple)) and len(current) > 0:
            shape.append(len(current))
            if isinstance(current[0], (list, tuple)):
                current = current[0]
            else:
                break
        
        return shape
    
    def _infer_dtype(self, data):
        """Infer PyTorch dtype from data"""
        flat = self._flatten_data(data) if isinstance(data, (list, tuple)) else [data]
        
        if not flat:
            return 'float32'
        
        sample = flat[0]
        if isinstance(sample, bool):
            return 'bool'
        elif isinstance(sample, int):
            return 'int64'
        elif isinstance(sample, float):
            return 'float32'
        else:
            return 'float32'
    
    # ===== TENSOR OPERATIONS =====
    
    def add(self, other):
        """Element-wise addition using WebGPU"""
        other_tensor = other if isinstance(other, WebGPUTensor) else WebGPUTensor(other)
        
        # Call WebGPU add shader through JavaScript
        result_id = js.greedInstance.tensorBridge.executeOperation(
            'add',
            self._webgpu_id,
            other_tensor._webgpu_id,
            {'shape': self.shape, 'dtype': self.dtype}
        )
        
        result = WebGPUTensor.__new__(WebGPUTensor)
        result._webgpu_id = result_id
        result.shape = self.shape
        result.dtype = self.dtype
        result.device = self.device
        result.requires_grad = self.requires_grad or other_tensor.requires_grad
        
        return result
    
    def __add__(self, other):
        return self.add(other)
    
    def sub(self, other):
        """Element-wise subtraction using WebGPU"""
        other_tensor = other if isinstance(other, WebGPUTensor) else WebGPUTensor(other)
        
        result_id = js.greedInstance.tensorBridge.executeOperation(
            'sub',
            self._webgpu_id,
            other_tensor._webgpu_id,
            {'shape': self.shape, 'dtype': self.dtype}
        )
        
        result = WebGPUTensor.__new__(WebGPUTensor)
        result._webgpu_id = result_id
        result.shape = self.shape
        result.dtype = self.dtype
        result.device = self.device
        result.requires_grad = self.requires_grad or other_tensor.requires_grad
        
        return result
    
    def __sub__(self, other):
        return self.sub(other)
    
    def mul(self, other):
        """Element-wise multiplication using WebGPU"""
        other_tensor = other if isinstance(other, WebGPUTensor) else WebGPUTensor(other)
        
        result_id = js.greedInstance.tensorBridge.executeOperation(
            'mul',
            self._webgpu_id,
            other_tensor._webgpu_id,
            {'shape': self.shape, 'dtype': self.dtype}
        )
        
        result = WebGPUTensor.__new__(WebGPUTensor)
        result._webgpu_id = result_id
        result.shape = self.shape
        result.dtype = self.dtype  
        result.device = self.device
        result.requires_grad = self.requires_grad or other_tensor.requires_grad
        
        return result
    
    def __mul__(self, other):
        return self.mul(other)
    
    def matmul(self, other):
        """Matrix multiplication using WebGPU"""
        other_tensor = other if isinstance(other, WebGPUTensor) else WebGPUTensor(other)
        
        # Calculate output shape for matrix multiplication
        if len(self.shape) == 2 and len(other_tensor.shape) == 2:
            output_shape = [self.shape[0], other_tensor.shape[1]]
        else:
            raise ValueError(f"matmul: incompatible shapes {self.shape} and {other_tensor.shape}")
        
        result_id = js.greedInstance.tensorBridge.executeOperation(
            'matmul',
            self._webgpu_id,
            other_tensor._webgpu_id,
            {'shape': output_shape, 'dtype': self.dtype}
        )
        
        result = WebGPUTensor.__new__(WebGPUTensor)
        result._webgpu_id = result_id
        result.shape = output_shape
        result.dtype = self.dtype
        result.device = self.device
        result.requires_grad = self.requires_grad or other_tensor.requires_grad
        
        return result
    
    def __matmul__(self, other):
        return self.matmul(other)
    
    def sum(self, dim=None, keepdim=False):
        """Sum reduction using WebGPU"""
        result_id = js.greedInstance.tensorBridge.executeOperation(
            'sum',
            self._webgpu_id,
            None,
            {
                'shape': self.shape,
                'dtype': self.dtype,
                'dim': dim,
                'keepdim': keepdim
            }
        )
        
        # Calculate output shape based on reduction
        if dim is None:
            output_shape = [1] if keepdim else []
        else:
            output_shape = self.shape.copy()
            if keepdim:
                output_shape[dim] = 1
            else:
                output_shape.pop(dim)
        
        result = WebGPUTensor.__new__(WebGPUTensor)
        result._webgpu_id = result_id
        result.shape = output_shape
        result.dtype = self.dtype
        result.device = self.device
        result.requires_grad = self.requires_grad
        
        return result
    
    def mean(self, dim=None, keepdim=False):
        """Mean reduction using WebGPU"""
        result_id = js.greedInstance.tensorBridge.executeOperation(
            'mean',
            self._webgpu_id,
            None,
            {
                'shape': self.shape,
                'dtype': self.dtype,
                'dim': dim,
                'keepdim': keepdim
            }
        )
        
        # Calculate output shape
        if dim is None:
            output_shape = [1] if keepdim else []
        else:
            output_shape = self.shape.copy()
            if keepdim:
                output_shape[dim] = 1
            else:
                output_shape.pop(dim)
        
        result = WebGPUTensor.__new__(WebGPUTensor)
        result._webgpu_id = result_id
        result.shape = output_shape
        result.dtype = self.dtype
        result.device = self.device
        result.requires_grad = self.requires_grad
        
        return result
    
    def relu(self):
        """ReLU activation using WebGPU"""
        result_id = js.greedInstance.tensorBridge.executeOperation(
            'relu',
            self._webgpu_id,
            None,
            {'shape': self.shape, 'dtype': self.dtype}
        )
        
        result = WebGPUTensor.__new__(WebGPUTensor)
        result._webgpu_id = result_id
        result.shape = self.shape
        result.dtype = self.dtype
        result.device = self.device
        result.requires_grad = self.requires_grad
        
        return result
    
    def backward(self, gradient=None):
        """Backpropagation using WebGPU autograd"""
        if not self.requires_grad:
            return
        
        if gradient is None:
            # Create gradient of ones with same shape
            gradient = ones_like(self)
        
        # Execute backward pass through JavaScript autograd
        js.greedInstance.tensorBridge.executeBackward(
            self._webgpu_id,
            gradient._webgpu_id if isinstance(gradient, WebGPUTensor) else gradient
        )
    
    # ===== TENSOR PROPERTIES =====
    
    def item(self):
        """Get scalar value from single-element tensor"""
        if len(self.shape) != 0 and self.shape != [1]:
            raise ValueError("item() only works on tensors with one element")
        
        # Get data from WebGPU tensor
        data = js.greedInstance.tensorBridge.getTensorData(self._webgpu_id)
        return data[0]
    
    def numpy(self):
        """Convert to numpy array (for debugging/compatibility)"""
        data = js.greedInstance.tensorBridge.getTensorData(self._webgpu_id)
        # Note: In real implementation, this would return actual numpy array
        return list(data)
    
    def size(self, dim=None):
        """Get tensor size"""
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]
    
    def numel(self):
        """Total number of elements"""
        result = 1
        for dim in self.shape:
            result *= dim
        return result
    
    def __repr__(self):
        """String representation"""
        data_str = str(self.numpy()[:10])  # Show first 10 elements
        if self.numel() > 10:
            data_str = data_str[:-1] + ", ...]\\"
        
        return f"tensor({data_str}, device='{self.device}', dtype={self.dtype})"
    
    def __str__(self):
        return self.__repr__()

# ===== TENSOR CREATION FUNCTIONS =====

def tensor(data, dtype=None, device=None, requires_grad=False):
    """Create tensor from data"""
    return WebGPUTensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def zeros(size, dtype='float32', device=None, requires_grad=False):
    """Create zero tensor using WebGPU zeros shader"""
    shape = size if isinstance(size, (list, tuple)) else [size]
    
    result_id = js.greedInstance.tensorBridge.executeCreationOperation(
        'zeros',
        {'shape': shape, 'dtype': dtype, 'device': str(device or webgpu)}
    )
    
    result = WebGPUTensor.__new__(WebGPUTensor)
    result._webgpu_id = result_id
    result.shape = shape
    result.dtype = dtype
    result.device = device or webgpu
    result.requires_grad = requires_grad
    
    return result

def ones(size, dtype='float32', device=None, requires_grad=False):
    """Create ones tensor using WebGPU"""
    shape = size if isinstance(size, (list, tuple)) else [size]
    
    result_id = js.greedInstance.tensorBridge.executeCreationOperation(
        'ones',
        {'shape': shape, 'dtype': dtype, 'device': str(device or webgpu)}
    )
    
    result = WebGPUTensor.__new__(WebGPUTensor)
    result._webgpu_id = result_id
    result.shape = shape
    result.dtype = dtype
    result.device = device or webgpu
    result.requires_grad = requires_grad
    
    return result

def randn(size, dtype='float32', device=None, requires_grad=False):
    """Create random normal tensor using WebGPU"""
    shape = size if isinstance(size, (list, tuple)) else [size]
    
    result_id = js.greedInstance.tensorBridge.executeCreationOperation(
        'randn',
        {'shape': shape, 'dtype': dtype, 'device': str(device or webgpu)}
    )
    
    result = WebGPUTensor.__new__(WebGPUTensor)
    result._webgpu_id = result_id
    result.shape = shape
    result.dtype = dtype
    result.device = device or webgpu
    result.requires_grad = requires_grad
    
    return result

def ones_like(tensor_like):
    """Create ones tensor with same shape as input"""
    return ones(tensor_like.shape, dtype=tensor_like.dtype, device=tensor_like.device)

def zeros_like(tensor_like):
    """Create zeros tensor with same shape as input"""
    return zeros(tensor_like.shape, dtype=tensor_like.dtype, device=tensor_like.device)

# ===== FUNCTIONAL OPERATIONS =====

def relu(input):
    """ReLU activation function"""
    return input.relu()

def matmul(input, other):
    """Matrix multiplication"""
    return input.matmul(other)

def add(input, other):
    """Element-wise addition"""
    return input.add(other)

# ===== NEURAL NETWORK MODULE =====

class Module:
    """Base class for neural network modules"""
    
    def __init__(self):
        self._parameters = {}
        self.training = True
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")
    
    def parameters(self):
        """Return all parameters"""
        params = []
        for param in self._parameters.values():
            params.append(param)
        return params
    
    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)

class Linear(Module):
    """Linear (fully connected) layer with WebGPU acceleration"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weights with Kaiming initialization
        self.weight = randn([out_features, in_features], requires_grad=True)
        if bias:
            self.bias = zeros([out_features], requires_grad=True)
        else:
            self.bias = None
        
        self._parameters['weight'] = self.weight
        if bias:
            self._parameters['bias'] = self.bias
    
    def forward(self, input):
        """Forward pass: input @ weight.T + bias"""
        # WebGPU linear layer operation
        result_id = js.greedInstance.tensorBridge.executeOperation(
            'linear',
            input._webgpu_id,
            self.weight._webgpu_id,
            {
                'bias_id': self.bias._webgpu_id if self.bias else None,
                'in_features': self.in_features,
                'out_features': self.out_features,
                'input_shape': input.shape,
                'dtype': input.dtype
            }
        )
        
        output_shape = input.shape[:-1] + [self.out_features]
        
        result = WebGPUTensor.__new__(WebGPUTensor)
        result._webgpu_id = result_id
        result.shape = output_shape
        result.dtype = input.dtype
        result.device = input.device
        result.requires_grad = True
        
        return result

class ReLU(Module):
    """ReLU activation function"""
    
    def forward(self, input):
        return relu(input)

class Sequential(Module):
    """Sequential container for modules"""
    
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = list(modules)
        
        # Collect parameters from all modules
        for i, module in enumerate(self.modules_list):
            if hasattr(module, '_parameters'):
                for name, param in module._parameters.items():
                    self._parameters[f'module_{i}_{name}'] = param
    
    def forward(self, input):
        for module in self.modules_list:
            input = module(input)
        return input

# Create nn submodule for PyTorch compatibility
class nn:
    Module = Module
    Linear = Linear
    ReLU = ReLU
    Sequential = Sequential

# ===== LOSS FUNCTIONS =====

class MSELoss(Module):
    """Mean Squared Error Loss"""
    
    def forward(self, input, target):
        # MSE = mean((input - target)^2)
        diff = input.sub(target)
        squared = diff.mul(diff)
        return squared.mean()

class CrossEntropyLoss(Module):
    """Cross Entropy Loss"""
    
    def forward(self, input, target):
        # Implement cross entropy using WebGPU
        result_id = js.greedInstance.tensorBridge.executeOperation(
            'cross_entropy',
            input._webgpu_id,
            target._webgpu_id,
            {'input_shape': input.shape, 'dtype': input.dtype}
        )
        
        result = WebGPUTensor.__new__(WebGPUTensor)
        result._webgpu_id = result_id
        result.shape = []  # Scalar loss
        result.dtype = input.dtype
        result.device = input.device
        result.requires_grad = True
        
        return result

# Add loss functions to nn
nn.MSELoss = MSELoss
nn.CrossEntropyLoss = CrossEntropyLoss

# ===== DEVICE UTILITIES =====

def is_available():
    """Check if WebGPU is available"""
    return js.navigator.gpu is not js.undefined

# Make cuda point to webgpu for PyTorch compatibility
class cuda_module:
    def is_available():
        return is_available()

cuda = cuda_module()

# Export torch module interface
__all__ = [
    'tensor', 'zeros', 'ones', 'randn', 'ones_like', 'zeros_like',
    'relu', 'matmul', 'add',
    'nn', 'cuda', 'is_available',
    'WebGPUTensor', 'WebGPUDevice'
]`;
}