/**
 * Context preservation and variable redefinition tests for v2.0
 */

import Greed from '../src/core/greed-v2.js';

describe('Context Preservation', () => {
  let greed;
  
  beforeEach(async () => {
    greed = new Greed({
      enableConsoleLogging: false,
      maxWorkers: 1,
      executionTimeout: 10000
    });
    await greed.initialize();
  });
  
  afterEach(async () => {
    if (greed) {
      await greed.cleanup();
    }
  });
  
  describe('Variable Persistence', () => {
    test('should preserve variables between executions', async () => {
      // Define a variable
      const result1 = await greed.run('x = 42', { useContext: true });
      expect(result1.success).toBe(true);
      
      // Use the variable in next execution
      const result2 = await greed.run('y = x * 2', { useContext: true });
      expect(result2.success).toBe(true);
      
      // Verify the result
      const result3 = await greed.run('result = y', { useContext: true });
      expect(result3.success).toBe(true);
      expect(result3.output).toBe(84);
    });
    
    test('should allow variable redefinition', async () => {
      // Define a variable
      const result1 = await greed.run('x = 10', { useContext: true });
      expect(result1.success).toBe(true);
      
      // Redefine the same variable
      const result2 = await greed.run('x = 20', { useContext: true });
      expect(result2.success).toBe(true);
      
      // Verify the new value
      const result3 = await greed.run('result = x', { useContext: true });
      expect(result3.success).toBe(true);
      expect(result3.output).toBe(20);
    });
    
    test('should preserve function definitions', async () => {
      // Define a function
      const result1 = await greed.run(`
def add_numbers(a, b):
    return a + b
      `, { useContext: true });
      expect(result1.success).toBe(true);
      
      // Use the function
      const result2 = await greed.run('result = add_numbers(5, 3)', { useContext: true });
      expect(result2.success).toBe(true);
      expect(result2.output).toBe(8);
    });
    
    test('should allow function redefinition', async () => {
      // Define a function
      const result1 = await greed.run(`
def multiply(a, b):
    return a * b
      `, { useContext: true });
      expect(result1.success).toBe(true);
      
      // Redefine the same function
      const result2 = await greed.run(`
def multiply(a, b):
    return a * b * 2
      `, { useContext: true });
      expect(result2.success).toBe(true);
      
      // Verify the new function behavior
      const result3 = await greed.run('result = multiply(3, 4)', { useContext: true });
      expect(result3.success).toBe(true);
      expect(result3.output).toBe(24); // 3 * 4 * 2
    });
    
    test('should preserve class definitions', async () => {
      // Define a class
      const result1 = await greed.run(`
class Calculator:
    def __init__(self, initial_value=0):
        self.value = initial_value
    
    def add(self, x):
        self.value += x
        return self.value
    
    def multiply(self, x):
        self.value *= x
        return self.value
      `, { useContext: true });
      expect(result1.success).toBe(true);
      
      // Create an instance
      const result2 = await greed.run('calc = Calculator(10)', { useContext: true });
      expect(result2.success).toBe(true);
      
      // Use the instance
      const result3 = await greed.run('result = calc.add(5)', { useContext: true });
      expect(result3.success).toBe(true);
      expect(result3.output).toBe(15);
      
      // Continue using the instance
      const result4 = await greed.run('result = calc.multiply(2)', { useContext: true });
      expect(result4.success).toBe(true);
      expect(result4.output).toBe(30);
    });
    
    test('should preserve imported modules', async () => {
      // Import a module
      const result1 = await greed.run('import math', { useContext: true });
      expect(result1.success).toBe(true);
      
      // Use the module
      const result2 = await greed.run('result = math.pi', { useContext: true });
      expect(result2.success).toBe(true);
      expect(result2.output).toBeCloseTo(3.14159, 4);
    });
    
    test('should preserve numpy arrays', async () => {
      // Create numpy array
      const result1 = await greed.run(`
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
      `, { useContext: true, packages: ['numpy'] });
      expect(result1.success).toBe(true);
      
      // Modify the array
      const result2 = await greed.run('arr = arr * 2', { useContext: true });
      expect(result2.success).toBe(true);
      
      // Verify the result
      const result3 = await greed.run('result = arr.tolist()', { useContext: true });
      expect(result3.success).toBe(true);
      expect(result3.output).toEqual([2, 4, 6, 8, 10]);
    });
  });
  
  describe('PyTorch Context Preservation', () => {
    test('should preserve torch tensors', async () => {
      // Create a tensor
      const result1 = await greed.run(`
import torch
x = torch.tensor([1.0, 2.0, 3.0])
      `, { useContext: true });
      expect(result1.success).toBe(true);
      
      // Modify the tensor
      const result2 = await greed.run('y = x * 2', { useContext: true });
      expect(result2.success).toBe(true);
      
      // Verify the result
      const result3 = await greed.run('result = y.tolist()', { useContext: true });
      expect(result3.success).toBe(true);
      expect(result3.output).toEqual([2.0, 4.0, 6.0]);
    });
    
    test('should preserve neural network models', async () => {
      // Define a model
      const result1 = await greed.run(`
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
      `, { useContext: true });
      expect(result1.success).toBe(true);
      
      // Use the model
      const result2 = await greed.run(`
x = torch.randn(1, 10)
output = model(x)
result = output.shape
      `, { useContext: true });
      expect(result2.success).toBe(true);
      expect(result2.output).toEqual([1, 1]);
    });
    
    test('should preserve optimizer state', async () => {
      // Create model and optimizer
      const result1 = await greed.run(`
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
      `, { useContext: true });
      expect(result1.success).toBe(true);
      
      // Perform optimization step
      const result2 = await greed.run(`
x = torch.randn(5, 10)
y = torch.randn(5, 1)
output = model(x)
loss = nn.MSELoss()(output, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
result = loss.item()
      `, { useContext: true });
      expect(result2.success).toBe(true);
      expect(typeof result2.output).toBe('number');
    });
    
    test('should preserve training state', async () => {
      // Initialize training
      const result1 = await greed.run(`
import torch
import torch.nn as nn

model = nn.Linear(2, 1)
model.train()
training_mode = model.training
      `, { useContext: true });
      expect(result1.success).toBe(true);
      
      // Switch to eval mode
      const result2 = await greed.run(`
model.eval()
eval_mode = model.training
result = {'training': training_mode, 'eval': eval_mode}
      `, { useContext: true });
      expect(result2.success).toBe(true);
      expect(result2.output.training).toBe(true);
      expect(result2.output.eval).toBe(false);
    });
  });
  
  describe('Global State Management', () => {
    test('should handle global variable updates', async () => {
      // Set global variable
      const result1 = await greed.run('global_var = "initial"', { useContext: true });
      expect(result1.success).toBe(true);
      
      // Update global variable
      const result2 = await greed.run('global_var = "updated"', { useContext: true });
      expect(result2.success).toBe(true);
      
      // Verify the update
      const result3 = await greed.run('result = global_var', { useContext: true });
      expect(result3.success).toBe(true);
      expect(result3.output).toBe("updated");
    });
    
    test('should handle complex data structures', async () => {
      // Create complex data structure
      const result1 = await greed.run(`
data = {
    'numbers': [1, 2, 3],
    'nested': {
        'value': 42,
        'text': 'hello'
    }
}
      `, { useContext: true });
      expect(result1.success).toBe(true);
      
      // Modify the data structure
      const result2 = await greed.run(`
data['numbers'].append(4)
data['nested']['value'] = 100
      `, { useContext: true });
      expect(result2.success).toBe(true);
      
      // Verify the modifications
      const result3 = await greed.run('result = data', { useContext: true });
      expect(result3.success).toBe(true);
      expect(result3.output.numbers).toEqual([1, 2, 3, 4]);
      expect(result3.output.nested.value).toBe(100);
    });
    
    test('should handle lambda functions', async () => {
      // Define lambda function
      const result1 = await greed.run('square = lambda x: x * x', { useContext: true });
      expect(result1.success).toBe(true);
      
      // Use lambda function
      const result2 = await greed.run('result = square(5)', { useContext: true });
      expect(result2.success).toBe(true);
      expect(result2.output).toBe(25);
      
      // Redefine lambda function
      const result3 = await greed.run('square = lambda x: x * x * x', { useContext: true });
      expect(result3.success).toBe(true);
      
      // Verify new behavior
      const result4 = await greed.run('result = square(3)', { useContext: true });
      expect(result4.success).toBe(true);
      expect(result4.output).toBe(27);
    });
  });
  
  describe('Context Isolation', () => {
    test('should not share context between different instances', async () => {
      const greed2 = new Greed({
        enableConsoleLogging: false,
        maxWorkers: 1
      });
      await greed2.initialize();
      
      try {
        // Set variable in first instance
        const result1 = await greed.run('x = 100', { useContext: true });
        expect(result1.success).toBe(true);
        
        // Try to access variable in second instance
        const result2 = await greed2.run('result = x', { useContext: true });
        expect(result2.success).toBe(false);
        expect(result2.error).toContain('NameError');
      } finally {
        await greed2.cleanup();
      }
    });
    
    test('should not share context between context and non-context executions', async () => {
      // Set variable with context
      const result1 = await greed.run('context_var = "from_context"', { useContext: true });
      expect(result1.success).toBe(true);
      
      // Try to access without context
      const result2 = await greed.run('result = context_var', { useContext: false });
      expect(result2.success).toBe(false);
      expect(result2.error).toContain('NameError');
    });
  });
});