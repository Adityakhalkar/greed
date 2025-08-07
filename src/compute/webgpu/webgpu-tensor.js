/**
 * WebGPUTensor - GPU-accelerated tensor implementation
 * Replaces the numpy-based WebGPUTensor with actual WebGPU operations
 */
import logger from '../../utils/logger.js';

export class WebGPUTensor {
  constructor(data, options = {}) {
    // Tensor properties
    this.device = options.device || 'webgpu';
    this.dtype = options.dtype || 'float32';
    this.requires_grad = options.requires_grad || false;
    this.grad = null;
    this.grad_fn = null;
    
    // Shape and data handling - enhanced for Python/Pyodide compatibility
    if (Array.isArray(data) || ArrayBuffer.isView(data)) {
      this.data = this._processInputData(data);
      this.shape = options.shape || this._inferShape(data);
    } else if (data instanceof ArrayBuffer) {
      this.data = new Float32Array(data);
      this.shape = options.shape || [this.data.length];
    } else if (data && typeof data === 'object' && typeof data.length !== 'undefined') {
      // Handle Pyodide/Python lists and array-like objects
      try {
        const arrayData = Array.from(data); // Convert to proper JavaScript array
        this.data = this._processInputData(arrayData);
        this.shape = options.shape || this._inferShape(arrayData);
      } catch (conversionError) {
        throw new Error(`Invalid tensor data: cannot convert to array. Got ${typeof data}, error: ${conversionError.message}`);
      }
    } else if (typeof data === 'number') {
      // Handle scalar values
      this.data = new Float32Array([data]);
      this.shape = options.shape || [1];
    } else {
      throw new Error(`Invalid tensor data type. Expected Array, TypedArray, ArrayBuffer, array-like object, or number. Got: ${typeof data} (${data})`);
    }
    
    // Derived properties
    this.ndim = this.shape.length;
    this.size = this.shape.reduce((a, b) => a * b, 1);
    
    // WebGPU compute engine reference
    this.computeEngine = options.computeEngine || null;
    this._buffer = null; // GPU buffer cache
    this._isOnGPU = false;
  }
  
  /**
   * Get reference to WebGPU compute engine
   */
  static setComputeEngine(engine) {
    WebGPUTensor.globalComputeEngine = engine;
  }
  
  get _engine() {
    return this.computeEngine || WebGPUTensor.globalComputeEngine;
  }
  
  /**
   * Execute WebGPU operation
   */
  async _executeGPUOperation(operation, other = null, options = {}) {
    if (!this._engine) {
      throw new Error('WebGPU compute engine not available');
    }
    
    const tensors = other ? [this.data, other.data || other] : [this.data];
    
    try {
      const result = await this._engine.execute(operation, tensors, {
        shape: this.shape,
        otherShape: other?.shape,
        dtype: this.dtype,
        ...options
      });
      
      const resultShape = this._calculateResultShape(operation, other, options);
      return new WebGPUTensor(result, {
        shape: resultShape,
        device: this.device,
        dtype: this.dtype,
        computeEngine: this._engine,
        requires_grad: this.requires_grad || (other?.requires_grad)
      });
    } catch (error) {
      logger.warn(`WebGPU operation ${operation} failed, falling back to CPU:`, {
        operation,
        error: error.message,
        tensorShape: this.shape,
        otherShape: other?.shape,
        fallbackReason: 'gpu-operation-failed'
      });
      return this._executeCPUFallback(operation, other, options);
    }
  }
  
  /**
   * CPU fallback for when WebGPU operations fail
   */
  _executeCPUFallback(operation, other, options) {
    // Implement basic CPU operations as fallback
    const result = this._cpuOperations[operation]?.(this.data, other?.data || other, options);
    if (!result) {
      throw new Error(`Operation ${operation} not supported`);
    }
    
    const resultShape = this._calculateResultShape(operation, other, options);
    return new WebGPUTensor(result, {
      shape: resultShape,
      device: 'cpu',
      dtype: this.dtype,
      requires_grad: this.requires_grad || (other?.requires_grad)
    });
  }
  
  // ===== ARITHMETIC OPERATIONS =====
  
  async add(other) {
    return this._executeGPUOperation('add', other);
  }
  
  async sub(other) {
    return this._executeGPUOperation('sub', other);
  }
  
  async mul(other) {
    return this._executeGPUOperation('mul', other);
  }
  
  async div(other) {
    return this._executeGPUOperation('div', other);
  }
  
  async pow(exponent) {
    return this._executeGPUOperation('pow', exponent);
  }
  
  // ===== MATRIX OPERATIONS =====
  
  async matmul(other) {
    if (this.ndim !== 2 || other.ndim !== 2) {
      throw new Error('matmul requires 2D tensors');
    }
    if (this.shape[1] !== other.shape[0]) {
      throw new Error(`Cannot multiply matrices of shapes ${this.shape} and ${other.shape}`);
    }
    return this._executeGPUOperation('matmul', other);
  }
  
  async bmm(other) {
    if (this.ndim !== 3 || other.ndim !== 3) {
      throw new Error('bmm requires 3D tensors');
    }
    return this._executeGPUOperation('bmm', other);
  }
  
  async transpose(dim0 = 0, dim1 = 1) {
    return this._executeGPUOperation('transpose', null, { dim0, dim1 });
  }
  
  // ===== ACTIVATION FUNCTIONS =====
  
  async relu() {
    return this._executeGPUOperation('relu');
  }
  
  async leaky_relu(negative_slope = 0.01) {
    return this._executeGPUOperation('leaky_relu', null, { negativeSlope: negative_slope });
  }
  
  async sigmoid() {
    return this._executeGPUOperation('sigmoid');
  }
  
  async tanh() {
    return this._executeGPUOperation('tanh');
  }
  
  async gelu() {
    return this._executeGPUOperation('gelu');
  }
  
  async softmax(dim = -1) {
    return this._executeGPUOperation('softmax', null, { dim });
  }
  
  // ===== REDUCTION OPERATIONS =====
  
  async sum(dim = null, keepdim = false) {
    return this._executeGPUOperation('sum', null, { dim, keepDim: keepdim });
  }
  
  async mean(dim = null, keepdim = false) {
    return this._executeGPUOperation('mean', null, { dim, keepDim: keepdim });
  }
  
  async max(dim = null, keepdim = false) {
    if (dim === null) {
      // Global max
      const reduced = await this._executeGPUOperation('max_reduce');
      return reduced;
    } else {
      // Max along dimension - returns values and indices
      return this._executeGPUOperation('max', null, { dim, keepDim: keepdim });
    }
  }
  
  async min(dim = null, keepdim = false) {
    if (dim === null) {
      const reduced = await this._executeGPUOperation('min_reduce');
      return reduced;
    } else {
      return this._executeGPUOperation('min', null, { dim, keepDim: keepdim });
    }
  }
  
  // ===== STATISTICAL OPERATIONS =====
  
  async std(dim = null, keepdim = false, unbiased = true) {
    return this._executeGPUOperation('std', null, { dim, keepDim: keepdim, unbiased });
  }
  
  async var(dim = null, keepdim = false, unbiased = true) {
    return this._executeGPUOperation('var', null, { dim, keepDim: keepdim, unbiased });
  }
  
  // ===== MATHEMATICAL FUNCTIONS =====
  
  async exp() {
    return this._executeGPUOperation('exp');
  }
  
  async log() {
    return this._executeGPUOperation('log');
  }
  
  async sqrt() {
    return this._executeGPUOperation('sqrt');
  }
  
  async abs() {
    return this._executeGPUOperation('abs');
  }
  
  // ===== TENSOR MANIPULATION =====
  
  view(...shape) {
    const newSize = shape.reduce((a, b) => a * b, 1);
    if (newSize !== this.size) {
      throw new Error(`Cannot reshape tensor of size ${this.size} to shape ${shape}`);
    }
    
    return new WebGPUTensor(this.data, {
      shape: shape,
      device: this.device,
      dtype: this.dtype,
      computeEngine: this._engine,
      requires_grad: this.requires_grad
    });
  }
  
  reshape(...shape) {
    return this.view(...shape);
  }
  
  unsqueeze(dim) {
    const newShape = [...this.shape];
    if (dim < 0) dim = newShape.length + dim + 1;
    newShape.splice(dim, 0, 1);
    return this.view(...newShape);
  }
  
  squeeze(dim = null) {
    let newShape;
    if (dim === null) {
      newShape = this.shape.filter(s => s !== 1);
    } else {
      if (this.shape[dim] !== 1) {
        throw new Error(`Cannot squeeze dimension ${dim} of size ${this.shape[dim]}`);
      }
      newShape = [...this.shape];
      newShape.splice(dim, 1);
    }
    return this.view(...newShape);
  }
  
  flatten(start_dim = 0, end_dim = -1) {
    if (end_dim === -1) end_dim = this.ndim - 1;
    
    const beforeDims = this.shape.slice(0, start_dim);
    const flattenDims = this.shape.slice(start_dim, end_dim + 1);
    const afterDims = this.shape.slice(end_dim + 1);
    
    const flattenedSize = flattenDims.reduce((a, b) => a * b, 1);
    const newShape = [...beforeDims, flattenedSize, ...afterDims];
    
    return this.view(...newShape);
  }
  
  // ===== DEVICE OPERATIONS =====
  
  to(device) {
    if (device === this.device) return this;
    
    return new WebGPUTensor(this.data.slice(), {
      shape: this.shape,
      device: device,
      dtype: this.dtype,
      computeEngine: this._engine,
      requires_grad: this.requires_grad
    });
  }
  
  cpu() {
    return this.to('cpu');
  }
  
  cuda() {
    return this.to('webgpu'); // Map CUDA to WebGPU
  }
  
  // ===== AUTOGRAD SUPPORT =====
  
  retain_grad() {
    if (!this.requires_grad) {
      throw new Error('can\'t retain_grad on Tensor that has requires_grad=False');
    }
    this._retain_grad = true;
    return this;
  }
  
  backward(gradient = null, retain_graph = false, create_graph = false) {
    if (!this.requires_grad) {
      return;
    }
    
    if (gradient === null) {
      if (this.size === 1) {
        gradient = new WebGPUTensor([1.0], { shape: this.shape });
      } else {
        throw new Error('grad can be implicitly created only for scalar outputs');
      }
    }
    
    // Initialize gradient if not present
    if (this.grad === null) {
      this.grad = new WebGPUTensor(new Float32Array(this.size).fill(0), { 
        shape: this.shape, 
        device: this.device, 
        dtype: this.dtype 
      });
    }
    
    // Accumulate gradient
    const gradData = gradient.data || gradient;
    for (let i = 0; i < this.grad.data.length; i++) {
      this.grad.data[i] += Array.isArray(gradData) ? gradData[i] : gradData;
    }
    
    if (this.grad_fn) {
      this.grad_fn(gradient);
    }
  }
  
  // ===== UTILITY METHODS =====
  
  numpy() {
    return this.data;
  }
  
  tolist() {
    if (this.ndim === 1) {
      return Array.from(this.data);
    }
    // For multi-dimensional arrays, recursively convert
    return this._arrayToNestedList(this.data, this.shape);
  }
  
  item() {
    if (this.size !== 1) {
      throw new Error('item() can only be called on tensors with one element');
    }
    return this.data[0];
  }
  
  clone() {
    return new WebGPUTensor(this.data.slice(), {
      shape: this.shape,
      device: this.device,
      dtype: this.dtype,
      computeEngine: this._engine,
      requires_grad: this.requires_grad
    });
  }
  
  detach() {
    const detached = this.clone();
    detached.requires_grad = false;
    detached.grad_fn = null;
    return detached;
  }
  
  // ===== PRIVATE METHODS =====
  
  _processInputData(data) {
    if (Array.isArray(data)) {
      return new Float32Array(this._flattenArray(data));
    } else if (ArrayBuffer.isView(data)) {
      return new Float32Array(data);
    } else {
      throw new Error('Unsupported data type');
    }
  }
  
  _flattenArray(arr) {
    const result = [];
    const flatten = (item) => {
      if (Array.isArray(item)) {
        item.forEach(flatten);
      } else {
        result.push(Number(item));
      }
    };
    flatten(arr);
    return result;
  }
  
  _inferShape(data) {
    if (!Array.isArray(data)) {
      return [data.length || 1];
    }
    
    const getShape = (arr) => {
      if (!Array.isArray(arr)) return [];
      const shape = [arr.length];
      if (arr.length > 0 && Array.isArray(arr[0])) {
        shape.push(...getShape(arr[0]));
      }
      return shape;
    };
    
    return getShape(data);
  }
  
  _calculateResultShape(operation, other, options) {
    switch (operation) {
      case 'matmul':
        return [this.shape[0], other.shape[1]];
      
      case 'bmm':
        return [this.shape[0], this.shape[1], other.shape[2]];
      
      case 'transpose':
        const newShape = [...this.shape];
        const { dim0 = 0, dim1 = 1 } = options;
        [newShape[dim0], newShape[dim1]] = [newShape[dim1], newShape[dim0]];
        return newShape;
      
      case 'sum':
      case 'mean':
        if (options.dim === null) {
          return options.keepDim ? this.shape : [1];
        } else {
          const newShape = [...this.shape];
          if (options.keepDim) {
            newShape[options.dim] = 1;
          } else {
            newShape.splice(options.dim, 1);
          }
          return newShape.length === 0 ? [1] : newShape;
        }
      
      case 'softmax':
        return this.shape;
      
      default:
        // Element-wise operations preserve shape
        return this.shape;
    }
  }
  
  _arrayToNestedList(data, shape) {
    if (shape.length === 1) {
      return Array.from(data);
    }
    
    const result = [];
    const stride = shape.slice(1).reduce((a, b) => a * b, 1);
    
    for (let i = 0; i < shape[0]; i++) {
      const start = i * stride;
      const end = start + stride;
      const subData = data.slice(start, end);
      result.push(this._arrayToNestedList(subData, shape.slice(1)));
    }
    
    return result;
  }
  
  // CPU fallback operations
  get _cpuOperations() {
    return {
      add: (a, b) => a.map((val, i) => val + (Array.isArray(b) ? b[i] : b)),
      sub: (a, b) => a.map((val, i) => val - (Array.isArray(b) ? b[i] : b)),
      mul: (a, b) => a.map((val, i) => val * (Array.isArray(b) ? b[i] : b)),
      div: (a, b) => a.map((val, i) => val / (Array.isArray(b) ? b[i] : b)),
      pow: (a, b) => a.map((val, i) => Math.pow(val, Array.isArray(b) ? b[i] : b)),
      exp: (a) => a.map(val => Math.exp(val)),
      log: (a) => a.map(val => Math.log(val)),
      sqrt: (a) => a.map(val => Math.sqrt(val)),
      abs: (a) => a.map(val => Math.abs(val)),
      relu: (a) => a.map(val => Math.max(0, val)),
      sigmoid: (a) => a.map(val => 1 / (1 + Math.exp(-val))),
      tanh: (a) => a.map(val => Math.tanh(val)),
      std: (a, options = {}) => {
        const mean = a.reduce((sum, val) => sum + val, 0) / a.length;
        const variance = a.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (options.unbiased ? a.length - 1 : a.length);
        return [Math.sqrt(variance)];
      },
      var: (a, options = {}) => {
        const mean = a.reduce((sum, val) => sum + val, 0) / a.length;
        const variance = a.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (options.unbiased ? a.length - 1 : a.length);
        return [variance];
      }
    };
  }
  
  toString() {
    return `WebGPUTensor(${this.shape.join('x')}, device=${this.device}, dtype=${this.dtype})`;
  }
  
  [Symbol.toPrimitive](hint) {
    if (hint === 'number' && this.size === 1) {
      return this.data[0];
    }
    return this.toString();
  }
  
  // ===== OPERATOR OVERLOADING =====
  
  // Matrix multiplication operator (@) - requires custom implementation
  ['@'](other) {
    return this.matmul(other);
  }
  
  // Add operator overloading for Python-style operations
  __add__(other) {
    return this.add(other);
  }
  
  __sub__(other) {
    return this.sub(other);
  }
  
  __mul__(other) {
    return this.mul(other);
  }
  
  __truediv__(other) {
    return this.div(other);
  }
  
  __matmul__(other) {
    return this.matmul(other);
  }
}

export default WebGPUTensor;