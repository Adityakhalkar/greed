/**
 * WebGPU Shaders - Complete collection of WGSL compute shaders for PyTorch operations
 * Replaces numpy operations with actual GPU-accelerated implementations
 */

export class WebGPUShaders {
  /**
   * Get comprehensive shader templates for all PyTorch operations
   */
  static getShaderTemplates() {
    return new Map([
      // ===== BASIC ARITHMETIC OPERATIONS =====
      ['add', (opts) => `
        @group(0) @binding(0) var<storage, read> input1: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> input2: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = input1[index] + input2[index];
        }
      `],

      ['sub', (opts) => `
        @group(0) @binding(0) var<storage, read> input1: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> input2: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = input1[index] - input2[index];
        }
      `],

      ['mul', (opts) => `
        @group(0) @binding(0) var<storage, read> input1: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> input2: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = input1[index] * input2[index];
        }
      `],

      ['div', (opts) => `
        @group(0) @binding(0) var<storage, read> input1: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> input2: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = input1[index] / input2[index];
        }
      `],

      ['pow', (opts) => `
        @group(0) @binding(0) var<storage, read> input1: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> input2: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = pow(input1[index], input2[index]);
        }
      `],

      // ===== MATRIX OPERATIONS =====
      ['matmul', (opts) => `
        @group(0) @binding(0) var<storage, read> input1: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> input2: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let row = global_id.x;
          let col = global_id.y;
          let M = params[0];  // rows of A
          let N = params[1];  // cols of B
          let K = params[2];  // cols of A, rows of B
          
          if (row >= M || col >= N) { return; }
          
          var sum = 0.0;
          for (var k = 0u; k < K; k = k + 1u) {
            sum = sum + input1[row * K + k] * input2[k * N + col];
          }
          output[row * N + col] = sum;
        }
      `],

      ['bmm', (opts) => `
        @group(0) @binding(0) var<storage, read> input1: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> input2: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let batch = global_id.z;
          let row = global_id.x;
          let col = global_id.y;
          
          let B = params[0];  // batch size
          let M = params[1];  // rows
          let N = params[2];  // cols of second matrix
          let K = params[3];  // cols of first matrix
          
          if (batch >= B || row >= M || col >= N) { return; }
          
          let batch_offset1 = batch * M * K;
          let batch_offset2 = batch * K * N;
          let batch_offset_out = batch * M * N;
          
          var sum = 0.0;
          for (var k = 0u; k < K; k = k + 1u) {
            sum = sum + input1[batch_offset1 + row * K + k] * input2[batch_offset2 + k * N + col];
          }
          output[batch_offset_out + row * N + col] = sum;
        }
      `],

      ['transpose', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let rows = params[0];
          let cols = params[1];
          let size = rows * cols;
          
          if (index >= size) { return; }
          
          let row = index / cols;
          let col = index % cols;
          let transposed_index = col * rows + row;
          
          output[transposed_index] = input[index];
        }
      `],

      // ===== ACTIVATION FUNCTIONS =====
      ['relu', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = max(input[index], 0.0);
        }
      `],

      ['leaky_relu', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          let negative_slope = bitcast<f32>(params[1]);
          if (index >= size) { return; }
          let val = input[index];
          output[index] = select(negative_slope * val, val, val > 0.0);
        }
      `],

      ['sigmoid', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = 1.0 / (1.0 + exp(-input[index]));
        }
      `],

      ['tanh', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = tanh(input[index]);
        }
      `],

      ['gelu', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          let x = input[index];
          // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
          let sqrt_2_over_pi = 0.7978845608;
          let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
          output[index] = 0.5 * x * (1.0 + tanh(inner));
        }
      `],

      ['softmax', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        var<workgroup> shared_max: f32;
        var<workgroup> shared_sum: f32;

        @compute @workgroup_size(${Math.min(opts.workgroupSize[0], 256)})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) workgroup_id: vec3<u32>) {
          let batch_size = params[0];
          let dim_size = params[1];
          let batch_idx = workgroup_id.x;
          let local_idx = local_id.x;
          
          if (batch_idx >= batch_size) { return; }
          
          let batch_offset = batch_idx * dim_size;
          
          // Find maximum for numerical stability
          var max_val = -3.4028235e+38; // -FLT_MAX
          for (var i = local_idx; i < dim_size; i = i + ${Math.min(opts.workgroupSize[0], 256)}u) {
            max_val = max(max_val, input[batch_offset + i]);
          }
          
          // Reduce maximum across workgroup
          workgroupBarrier();
          if (local_idx == 0u) {
            shared_max = max_val;
          }
          for (var stride = 1u; stride < ${Math.min(opts.workgroupSize[0], 256)}u; stride = stride * 2u) {
            workgroupBarrier();
            if (local_idx >= stride) {
              shared_max = max(shared_max, max_val);
            }
          }
          workgroupBarrier();
          
          // Compute exponentials and sum
          var sum = 0.0;
          for (var i = local_idx; i < dim_size; i = i + ${Math.min(opts.workgroupSize[0], 256)}u) {
            let exp_val = exp(input[batch_offset + i] - shared_max);
            sum = sum + exp_val;
            output[batch_offset + i] = exp_val;
          }
          
          // Reduce sum across workgroup
          workgroupBarrier();
          if (local_idx == 0u) {
            shared_sum = sum;
          }
          for (var stride = 1u; stride < ${Math.min(opts.workgroupSize[0], 256)}u; stride = stride * 2u) {
            workgroupBarrier();
            if (local_idx >= stride) {
              shared_sum = shared_sum + sum;
            }
          }
          workgroupBarrier();
          
          // Normalize
          for (var i = local_idx; i < dim_size; i = i + ${Math.min(opts.workgroupSize[0], 256)}u) {
            output[batch_offset + i] = output[batch_offset + i] / shared_sum;
          }
        }
      `],

      // ===== REDUCTION OPERATIONS =====
      ['sum', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        var<workgroup> shared_data: array<f32, ${opts.workgroupSize[0]}>;

        @compute @workgroup_size(${opts.workgroupSize[0]})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) workgroup_id: vec3<u32>) {
          let size = params[0];
          let local_idx = local_id.x;
          let global_idx = global_id.x;
          
          // Load data into shared memory
          var sum = 0.0;
          for (var i = global_idx; i < size; i = i + ${opts.workgroupSize[0]}u) {
            sum = sum + input[i];
          }
          shared_data[local_idx] = sum;
          
          workgroupBarrier();
          
          // Parallel reduction
          for (var stride = ${opts.workgroupSize[0] / 2}u; stride > 0u; stride = stride >> 1u) {
            if (local_idx < stride) {
              shared_data[local_idx] = shared_data[local_idx] + shared_data[local_idx + stride];
            }
            workgroupBarrier();
          }
          
          if (local_idx == 0u) {
            output[workgroup_id.x] = shared_data[0];
          }
        }
      `],

      ['mean', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        var<workgroup> shared_data: array<f32, ${opts.workgroupSize[0]}>;

        @compute @workgroup_size(${opts.workgroupSize[0]})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) workgroup_id: vec3<u32>) {
          let size = params[0];
          let local_idx = local_id.x;
          let global_idx = global_id.x;
          
          var sum = 0.0;
          for (var i = global_idx; i < size; i = i + ${opts.workgroupSize[0]}u) {
            sum = sum + input[i];
          }
          shared_data[local_idx] = sum;
          
          workgroupBarrier();
          
          for (var stride = ${opts.workgroupSize[0] / 2}u; stride > 0u; stride = stride >> 1u) {
            if (local_idx < stride) {
              shared_data[local_idx] = shared_data[local_idx] + shared_data[local_idx + stride];
            }
            workgroupBarrier();
          }
          
          if (local_idx == 0u) {
            output[workgroup_id.x] = shared_data[0] / f32(size);
          }
        }
      `],

      // ===== CONVOLUTION OPERATIONS =====
      ['conv2d', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> weight: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read> bias: array<${opts.dataType}>;
        @group(0) @binding(3) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(4) var<uniform> params: array<u32, 8>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let out_y = global_id.x;
          let out_x = global_id.y;
          let out_c = global_id.z;
          
          let batch_size = params[0];
          let in_channels = params[1];
          let in_height = params[2];
          let in_width = params[3];
          let out_channels = params[4];
          let out_height = params[5];
          let out_width = params[6];
          let kernel_size = params[7];
          
          if (out_y >= out_height || out_x >= out_width || out_c >= out_channels) { return; }
          
          var sum = 0.0;
          
          for (var in_c = 0u; in_c < in_channels; in_c = in_c + 1u) {
            for (var ky = 0u; ky < kernel_size; ky = ky + 1u) {
              for (var kx = 0u; kx < kernel_size; kx = kx + 1u) {
                let in_y = out_y + ky;
                let in_x = out_x + kx;
                
                if (in_y < in_height && in_x < in_width) {
                  let input_idx = in_c * in_height * in_width + in_y * in_width + in_x;
                  let weight_idx = out_c * in_channels * kernel_size * kernel_size + 
                                  in_c * kernel_size * kernel_size + ky * kernel_size + kx;
                  sum = sum + input[input_idx] * weight[weight_idx];
                }
              }
            }
          }
          
          sum = sum + bias[out_c];
          let output_idx = out_c * out_height * out_width + out_y * out_width + out_x;
          output[output_idx] = sum;
        }
      `],

      ['maxpool2d', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 8>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let out_y = global_id.x;
          let out_x = global_id.y;
          let c = global_id.z;
          
          let channels = params[0];
          let in_height = params[1];
          let in_width = params[2];
          let out_height = params[3];
          let out_width = params[4];
          let kernel_size = params[5];
          let stride = params[6];
          
          if (out_y >= out_height || out_x >= out_width || c >= channels) { return; }
          
          var max_val = -3.4028235e+38; // -FLT_MAX
          
          for (var ky = 0u; ky < kernel_size; ky = ky + 1u) {
            for (var kx = 0u; kx < kernel_size; kx = kx + 1u) {
              let in_y = out_y * stride + ky;
              let in_x = out_x * stride + kx;
              
              if (in_y < in_height && in_x < in_width) {
                let input_idx = c * in_height * in_width + in_y * in_width + in_x;
                max_val = max(max_val, input[input_idx]);
              }
            }
          }
          
          let output_idx = c * out_height * out_width + out_y * out_width + out_x;
          output[output_idx] = max_val;
        }
      `],

      // ===== MATHEMATICAL FUNCTIONS =====
      ['exp', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = exp(input[index]);
        }
      `],

      ['log', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = log(input[index]);
        }
      `],

      ['sqrt', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = sqrt(input[index]);
        }
      `],

      ['abs', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = abs(input[index]);
        }
      `],

      // ===== COMPARISON OPERATIONS =====
      ['max', (opts) => `
        @group(0) @binding(0) var<storage, read> input1: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> input2: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = max(input1[index], input2[index]);
        }
      `],

      ['min', (opts) => `
        @group(0) @binding(0) var<storage, read> input1: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> input2: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          output[index] = min(input1[index], input2[index]);
        }
      `],

      // ===== TENSOR MANIPULATION =====
      ['concat', (opts) => `
        @group(0) @binding(0) var<storage, read> input1: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> input2: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size1 = params[0];
          let size2 = params[1];
          let total_size = size1 + size2;
          
          if (index >= total_size) { return; }
          
          if (index < size1) {
            output[index] = input1[index];
          } else {
            output[index] = input2[index - size1];
          }
        }
      `],

      ['slice', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(2) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let start = params[0];
          let end = params[1];
          let step = params[2];
          let output_size = (end - start + step - 1u) / step;
          
          if (index >= output_size) { return; }
          
          let input_index = start + index * step;
          output[index] = input[input_index];
        }
      `],

      // ===== BATCH OPERATIONS =====
      ['batch_norm', (opts) => `
        @group(0) @binding(0) var<storage, read> input: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> running_mean: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read> running_var: array<${opts.dataType}>;
        @group(0) @binding(3) var<storage, read> weight: array<${opts.dataType}>;
        @group(0) @binding(4) var<storage, read> bias: array<${opts.dataType}>;
        @group(0) @binding(5) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(6) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let batch_size = params[0];
          let channels = params[1];
          let spatial_size = params[2];
          let eps = bitcast<f32>(params[3]);
          
          if (index >= batch_size * channels * spatial_size) { return; }
          
          let c = (index / spatial_size) % channels;
          let normalized = (input[index] - running_mean[c]) / sqrt(running_var[c] + eps);
          output[index] = normalized * weight[c] + bias[c];
        }
      `],

      // ===== LOSS FUNCTIONS =====
      ['cross_entropy', (opts) => `
        @group(0) @binding(0) var<storage, read> logits: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> targets: array<u32>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize[0]})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let batch_idx = global_id.x;
          let batch_size = params[0];
          let num_classes = params[1];
          
          if (batch_idx >= batch_size) { return; }
          
          let batch_offset = batch_idx * num_classes;
          let target_class = targets[batch_idx];
          
          // Find max for numerical stability
          var max_logit = -3.4028235e+38;
          for (var i = 0u; i < num_classes; i = i + 1u) {
            max_logit = max(max_logit, logits[batch_offset + i]);
          }
          
          // Compute log-sum-exp
          var sum_exp = 0.0;
          for (var i = 0u; i < num_classes; i = i + 1u) {
            sum_exp = sum_exp + exp(logits[batch_offset + i] - max_logit);
          }
          let log_sum_exp = log(sum_exp) + max_logit;
          
          // Cross entropy loss = -log(softmax[target])
          let target_logit = logits[batch_offset + target_class];
          output[batch_idx] = log_sum_exp - target_logit;
        }
      `],

      ['mse_loss', (opts) => `
        @group(0) @binding(0) var<storage, read> predictions: array<${opts.dataType}>;
        @group(0) @binding(1) var<storage, read> targets: array<${opts.dataType}>;
        @group(0) @binding(2) var<storage, read_write> output: array<${opts.dataType}>;
        @group(0) @binding(3) var<uniform> params: array<u32, 4>;

        @compute @workgroup_size(${opts.workgroupSize.join(', ')})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = params[0];
          if (index >= size) { return; }
          let diff = predictions[index] - targets[index];
          output[index] = diff * diff;
        }
      `]
    ]);
  }

  /**
   * Get optimal workgroup size for operation
   */
  static getOptimalWorkgroupSize(operation, tensorShape, deviceLimits) {
    const maxWorkgroupSize = deviceLimits?.maxComputeWorkgroupSizeX || 256;
    
    switch (operation) {
      case 'matmul':
      case 'bmm':
        return [Math.min(16, maxWorkgroupSize), Math.min(16, maxWorkgroupSize), 1];
      
      case 'conv2d':
        return [Math.min(8, maxWorkgroupSize), Math.min(8, maxWorkgroupSize), Math.min(8, maxWorkgroupSize)];
      
      case 'softmax':
        return [Math.min(256, maxWorkgroupSize), 1, 1];
      
      case 'sum':
      case 'mean':
        return [Math.min(256, maxWorkgroupSize), 1, 1];
      
      default:
        // For element-wise operations
        return [Math.min(64, maxWorkgroupSize), 1, 1];
    }
  }

  /**
   * Get buffer layout for operation
   */
  static getBufferLayout(operation, inputCount = 2, outputCount = 1) {
    const layouts = {
      // Standard binary operations
      add: { inputs: 2, outputs: 1, uniforms: 1 },
      sub: { inputs: 2, outputs: 1, uniforms: 1 },
      mul: { inputs: 2, outputs: 1, uniforms: 1 },
      div: { inputs: 2, outputs: 1, uniforms: 1 },
      pow: { inputs: 2, outputs: 1, uniforms: 1 },
      
      // Matrix operations
      matmul: { inputs: 2, outputs: 1, uniforms: 1 },
      bmm: { inputs: 2, outputs: 1, uniforms: 1 },
      
      // Unary operations
      relu: { inputs: 1, outputs: 1, uniforms: 1 },
      sigmoid: { inputs: 1, outputs: 1, uniforms: 1 },
      tanh: { inputs: 1, outputs: 1, uniforms: 1 },
      exp: { inputs: 1, outputs: 1, uniforms: 1 },
      log: { inputs: 1, outputs: 1, uniforms: 1 },
      sqrt: { inputs: 1, outputs: 1, uniforms: 1 },
      abs: { inputs: 1, outputs: 1, uniforms: 1 },
      
      // Complex operations
      conv2d: { inputs: 3, outputs: 1, uniforms: 1 }, // input, weight, bias
      batch_norm: { inputs: 5, outputs: 1, uniforms: 1 }, // input, mean, var, weight, bias
      cross_entropy: { inputs: 2, outputs: 1, uniforms: 1 }, // logits, targets
      
      // Reduction operations
      sum: { inputs: 1, outputs: 1, uniforms: 1 },
      mean: { inputs: 1, outputs: 1, uniforms: 1 }
    };
    
    return layouts[operation] || { inputs: inputCount, outputs: outputCount, uniforms: 1 };
  }

  /**
   * Generate parameter buffer for operation
   */
  static generateParams(operation, tensors, options = {}) {
    const params = new Uint32Array(4);
    
    switch (operation) {
      case 'matmul':
        // params: [M, N, K, reserved]
        params[0] = tensors[0].shape?.[0] || Math.sqrt(tensors[0].length);
        params[1] = tensors[1].shape?.[1] || Math.sqrt(tensors[1].length);
        params[2] = tensors[0].shape?.[1] || Math.sqrt(tensors[0].length);
        break;
      
      case 'bmm':
        // params: [B, M, N, K]
        params[0] = tensors[0].shape?.[0] || 1; // batch size
        params[1] = tensors[0].shape?.[1] || Math.sqrt(tensors[0].length);
        params[2] = tensors[1].shape?.[2] || Math.sqrt(tensors[1].length);
        params[3] = tensors[0].shape?.[2] || Math.sqrt(tensors[0].length);
        break;
      
      case 'conv2d':
        // params: [batch, in_ch, in_h, in_w, out_ch, out_h, out_w, kernel]
        const inShape = tensors[0].shape || [1, 1, 28, 28];
        const weightShape = tensors[1].shape || [32, 1, 3, 3];
        params[0] = inShape[0]; // batch
        params[1] = inShape[1]; // in_channels
        params[2] = inShape[2]; // in_height
        params[3] = inShape[3]; // in_width
        // Additional params would go in a second buffer for conv2d
        break;
      
      case 'softmax':
        // params: [batch_size, dim_size, reserved, reserved]
        const shape = tensors[0].shape || [1, tensors[0].length];
        params[0] = shape.length > 1 ? shape[0] : 1;
        params[1] = shape.length > 1 ? shape[1] : shape[0];
        break;
      
      case 'leaky_relu':
        // params: [size, negative_slope_bits, reserved, reserved]
        params[0] = tensors[0].length;
        params[1] = new Uint32Array(new Float32Array([options.negativeSlope || 0.01]).buffer)[0];
        break;
      
      default:
        // Standard element-wise operations: [size, reserved, reserved, reserved]
        params[0] = Array.isArray(tensors) ? tensors[0].length : tensors.length;
        break;
    }
    
    return params;
  }
}

export default WebGPUShaders;