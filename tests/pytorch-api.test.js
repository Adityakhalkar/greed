/**
 * Complete PyTorch API tests for v2.0
 */
import Greed from '../src/core/greed-v2.js';

describe('Complete PyTorch API', () => {
  let greed;
  
  beforeEach(async () => {
    greed = new Greed();
    await greed.initialize();
  });

  afterEach(async () => {
    if (greed && greed.isInitialized) {
      await greed.destroy();
    }
  });

  describe('Core Tensor Operations', () => {
    test('torch.tensor() should create tensors', async () => {
      const result = await greed.run(`
import torch
x = torch.tensor([1, 2, 3])
print(f"Tensor: {x}")
print(f"Type: {type(x)}")
print(f"Shape: {x.shape}")
x
      `, { useContext: true });
      expect(result.output).toContain('Tensor:');
      expect(result.output).toContain('Type:');
    });

    test('torch.zeros(), torch.ones(), torch.rand() should work', async () => {
      const result = await greed.run(`
import torch
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
rand_tensor = torch.rand(2, 3)
print(f"Zeros shape: {zeros.shape}")
print(f"Ones shape: {ones.shape}")
print(f"Random shape: {rand_tensor.shape}")
zeros
      `, { useContext: true });
      expect(result.output).toContain('Zeros shape: (2, 3)');
      expect(result.output).toContain('Ones shape: (2, 3)');
    });

    test('torch.eye() should create identity matrix', async () => {
      const result = await greed.run(`
import torch
eye = torch.eye(3)
print(f"Identity matrix shape: {eye.shape}")
print(f"Identity matrix:\\n{eye}")
eye
      `, { useContext: true });
      expect(result.output).toContain('Identity matrix shape: (3, 3)');
    });

    test('torch.cat() and torch.stack() should work', async () => {
      const result = await greed.run(`
import torch
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
cat_result = torch.cat([a, b])
stack_result = torch.stack([a, b])
print(f"Cat result: {cat_result}")
print(f"Stack result shape: {stack_result.shape}")
cat_result
      `, { useContext: true });
      expect(result.output).toContain('Cat result:');
      expect(result.output).toContain('Stack result shape:');
    });

    test('Reshape and view operations should work', async () => {
      const result = await greed.run(`
import torch
x = torch.tensor([[1, 2], [3, 4]])
reshaped = x.reshape(4)
viewed = x.view(-1)
print(f"Original shape: {x.shape}")
print(f"Reshaped: {reshaped.shape}")
print(f"Viewed: {viewed.shape}")
reshaped
      `, { useContext: true });
      expect(result.output).toContain('Original shape: (2, 2)');
      expect(result.output).toContain('Reshaped: (4,)');
    });

    test('Reduction operations should work', async () => {
      const result = await greed.run(`
import torch
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
mean_val = torch.mean(x)
sum_val = torch.sum(x)
max_val = torch.max(x)
print(f"Mean: {mean_val}")
print(f"Sum: {sum_val}")
print(f"Max: {max_val}")
mean_val
      `, { useContext: true });
      expect(result.output).toContain('Mean:');
      expect(result.output).toContain('Sum:');
    });
  });

  describe('Autograd Functionality', () => {
    test('requires_grad should enable gradient tracking', async () => {
      const result = await greed.run(`
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x.sum()
y.backward()
print(f"x.grad: {x.grad}")
print(f"x.requires_grad: {x.requires_grad}")
x.grad
      `, { useContext: true });
      expect(result.output).toContain('x.grad:');
      expect(result.output).toContain('x.requires_grad: True');
    });

    test('torch.no_grad() should disable gradients', async () => {
      const result = await greed.run(`
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
with torch.no_grad():
    y = x * 2
print(f"y.requires_grad: {y.requires_grad}")
y
      `, { useContext: true });
      expect(result.output).toContain('y.requires_grad: False');
    });
  });

  describe('Neural Network Layers', () => {
    test('nn.Linear should work', async () => {
      const result = await greed.run(`
import torch
import torch.nn as nn
linear = nn.Linear(3, 2)
x = torch.randn(1, 3)
output = linear(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Weight shape: {linear.weight.shape}")
output
      `, { useContext: true });
      expect(result.output).toContain('Input shape: (1, 3)');
      expect(result.output).toContain('Output shape: (1, 2)');
    });

    test('Activation functions should work', async () => {
      const result = await greed.run(`
import torch
import torch.nn as nn
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
x = torch.tensor([-1.0, 0.0, 1.0])
relu_out = relu(x)
sigmoid_out = sigmoid(x)
print(f"ReLU output: {relu_out}")
print(f"Sigmoid output: {sigmoid_out}")
relu_out
      `, { useContext: true });
      expect(result.output).toContain('ReLU output:');
      expect(result.output).toContain('Sigmoid output:');
    });

    test('nn.Sequential should work', async () => {
      const result = await greed.run(`
import torch
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(3, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
x = torch.randn(1, 3)
output = model(x)
print(f"Model: {model}")
print(f"Output shape: {output.shape}")
output
      `, { useContext: true });
      expect(result.output).toContain('Output shape: (1, 1)');
    });
  });

  describe('Loss Functions', () => {
    test('nn.MSELoss should work', async () => {
      const result = await greed.run(`
import torch
import torch.nn as nn
mse_loss = nn.MSELoss()
pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 2.5])
loss = mse_loss(pred, target)
print(f"MSE Loss: {loss}")
loss
      `, { useContext: true });
      expect(result.output).toContain('MSE Loss:');
    });

    test('nn.CrossEntropyLoss should work', async () => {
      const result = await greed.run(`
import torch
import torch.nn as nn
ce_loss = nn.CrossEntropyLoss()
logits = torch.randn(3, 5)  # 3 samples, 5 classes
targets = torch.tensor([1, 0, 4])
loss = ce_loss(logits, targets)
print(f"CrossEntropy Loss: {loss}")
loss
      `, { useContext: true });
      expect(result.output).toContain('CrossEntropy Loss:');
    });
  });

  describe('Optimizers', () => {
    test('optim.SGD should work', async () => {
      const result = await greed.run(`
import torch
import torch.nn as nn
import torch.optim as optim
model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(1, 2)
y = torch.randn(1, 1)
loss = nn.MSELoss()(model(x), y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"Optimizer: {optimizer}")
print(f"Loss: {loss}")
loss
      `, { useContext: true });
      expect(result.output).toContain('Optimizer:');
      expect(result.output).toContain('Loss:');
    });

    test('optim.Adam should work', async () => {
      const result = await greed.run(`
import torch
import torch.optim as optim
model = torch.nn.Linear(2, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"Adam optimizer: {optimizer}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
optimizer
      `, { useContext: true });
      expect(result.output).toContain('Adam optimizer:');
      expect(result.output).toContain('Learning rate:');
    });
  });

  describe('Device Management', () => {
    test('torch.device should work', async () => {
      const result = await greed.run(`
import torch
cpu_device = torch.device('cpu')
cuda_available = torch.cuda.is_available()
print(f"CPU device: {cpu_device}")
print(f"CUDA available: {cuda_available}")
cpu_device
      `, { useContext: true });
      expect(result.output).toContain('CPU device:');
      expect(result.output).toContain('CUDA available:');
    });

    test('tensor.to() should work for device transfer', async () => {
      const result = await greed.run(`
import torch
x = torch.tensor([1, 2, 3])
device = torch.device('cpu')
x_cpu = x.to(device)
print(f"Original device: {x.device}")
print(f"Moved to device: {x_cpu.device}")
x_cpu
      `, { useContext: true });
      expect(result.output).toContain('device:');
    });
  });

  describe('Data Loading', () => {
    test('TensorDataset should work', async () => {
      const result = await greed.run(`
import torch
from torch.utils.data import TensorDataset, DataLoader
x = torch.randn(100, 3)
y = torch.randn(100, 1)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
print(f"Dataset length: {len(dataset)}")
print(f"DataLoader batch size: {dataloader.batch_size}")
dataset
      `, { useContext: true });
      expect(result.output).toContain('Dataset length: 100');
      expect(result.output).toContain('DataLoader batch size: 10');
    });
  });

  describe('Save/Load Operations', () => {
    test('torch.save and torch.load should work', async () => {
      const result = await greed.run(`
import torch
import io
x = torch.tensor([1, 2, 3])
buffer = io.BytesIO()
torch.save(x, buffer)
buffer.seek(0)
loaded_x = torch.load(buffer)
print(f"Original: {x}")
print(f"Loaded: {loaded_x}")
print(f"Equal: {torch.equal(x, loaded_x)}")
loaded_x
      `, { useContext: true });
      expect(result.output).toContain('Original:');
      expect(result.output).toContain('Loaded:');
      expect(result.output).toContain('Equal: True');
    });
  });
});