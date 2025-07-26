/**
 * End-to-End Tests for Greed.js Manual Testing Environment
 * Tests various code execution scenarios using Playwright
 */

const { test, expect } = require('@playwright/test');
const path = require('path');

// Test configuration
const TEST_TIMEOUT = 120000; // 2 minutes for complex operations
const INIT_TIMEOUT = 60000;  // 1 minute for initialization

test.describe('Greed.js Manual Testing Environment', () => {
  let page;

  test.beforeEach(async ({ browser }) => {
    // Create a new page for each test
    page = await browser.newPage();
    
    // Enable console logging for debugging
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', error => console.error('PAGE ERROR:', error));
    
    // Navigate to the manual testing environment
    const testFilePath = path.resolve(__dirname, '../../manual-testing.html');
    await page.goto(`file://${testFilePath}`);
    
    // Wait for the page to load and libraries to initialize
    await page.waitForLoadState('networkidle');
    
    // Wait for Greed.js to initialize (check status indicator)
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: INIT_TIMEOUT });
    
    console.log('✅ Test environment initialized successfully');
  });

  test.afterEach(async () => {
    if (page) {
      await page.close();
    }
  });

  test('Environment loads and initializes properly', async () => {
    // Check that all UI elements are present
    await expect(page.locator('h1')).toContainText('Greed.js Manual Testing');
    await expect(page.locator('.version')).toContainText('v2.0.1');
    
    // Check editor is present and functional
    const editor = page.locator('#editor');
    await expect(editor).toBeVisible();
    
    // Check controls are present
    await expect(page.locator('#runBtn')).toBeVisible();
    await expect(page.locator('#runBtn')).toBeEnabled();
    await expect(page.locator('#clearEditor')).toBeVisible();
    await expect(page.locator('#examplesBtn')).toBeVisible();
    
    // Check output panel
    await expect(page.locator('#output')).toBeVisible();
    
    // Check status shows ready
    await expect(page.locator('#statusText')).toContainText('Ready');
    await expect(page.locator('.status-dot')).toHaveClass(/status-ready/);
    
    console.log('✅ Environment validation complete');
  });

  test('Basic Python execution works', async () => {
    const basicPython = `
# Basic Python test
print("Hello from Greed.js!")
x = 1 + 1
print(f"1 + 1 = {x}")

# Test basic Python features
numbers = [1, 2, 3, 4, 5]
print(f"Numbers: {numbers}")

total = sum(numbers)
print(f"Sum: {total}")
`;

    // Clear editor and input code
    await page.locator('#clearEditor').click();
    await page.locator('#editor').fill(basicPython);
    
    // Execute code
    await page.locator('#runBtn').click();
    
    // Wait for execution to complete
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: TEST_TIMEOUT });
    
    // Check for success output
    await expect(page.locator('#output')).toContainText('Code executed successfully');
    
    // Check execution time is displayed
    await expect(page.locator('#executionTime')).toContainText('ms');
    
    console.log('✅ Basic Python execution test passed');
  });

  test('PyTorch tensor operations work', async () => {
    const pytorchCode = `
# PyTorch tensor operations test
import torch
import numpy as np

print("Testing PyTorch tensor operations...")

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([2.0, 3.0, 4.0, 5.0])

print(f"x: {x}")
print(f"y: {y}")

# Basic operations
z = x + y  
print(f"x + y = {z}")

# Element-wise multiplication
w = x * y
print(f"x * y = {w}")

# Matrix operations
matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"Matrix: {matrix}")
print(f"Matrix shape: {matrix.shape}")

print("PyTorch test completed successfully!")
`;

    await page.locator('#clearEditor').click();
    await page.locator('#editor').fill(pytorchCode);
    
    await page.locator('#runBtn').click();
    
    // Wait for execution with longer timeout for PyTorch operations
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: TEST_TIMEOUT });
    
    // Check for successful PyTorch execution
    await expect(page.locator('#output')).toContainText('Code executed successfully');
    await expect(page.locator('#output')).toContainText('PyTorch test completed successfully');
    
    console.log('✅ PyTorch tensor operations test passed');
  });

  test('Neural network functionality works', async () => {
    const neuralNetworkCode = `
# Neural network test
import torch
import torch.nn as nn

print("Testing neural network functionality...")

# Test basic linear layer
linear = nn.Linear(3, 2)
input_data = torch.randn(1, 3)

print(f"Input: {input_data}")
print(f"Input shape: {input_data.shape}")

# Forward pass
output = linear(input_data)
print(f"Output: {output}")
print(f"Output shape: {output.shape}")

# Test activation function
activated = torch.nn.functional.relu(output)
print(f"ReLU activated: {activated}")

print("Neural network test completed!")
`;

    await page.locator('#clearEditor').click();
    await page.locator('#editor').fill(neuralNetworkCode);
    
    await page.locator('#runBtn').click();
    
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: TEST_TIMEOUT });
    
    await expect(page.locator('#output')).toContainText('Code executed successfully');
    await expect(page.locator('#output')).toContainText('Neural network test completed');
    
    console.log('✅ Neural network functionality test passed');
  });

  test('Matrix operations work correctly', async () => {
    const matrixCode = `
# Matrix operations test
import torch
import numpy as np

print("Testing matrix operations...")

# Create matrices
A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"Matrix A:\\n{A}")
print(f"Matrix B:\\n{B}")

# Matrix multiplication
C = torch.matmul(A, B)
print(f"A @ B =\\n{C}")

# Element-wise operations
element_wise = A * B
print(f"Element-wise A * B:\\n{element_wise}")

# Sum operations
sum_all = torch.sum(A)
print(f"Sum of all elements: {sum_all}")

print("Matrix operations test completed!")
`;

    await page.locator('#clearEditor').click();
    await page.locator('#editor').fill(matrixCode);
    
    await page.locator('#runBtn').click();
    
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: TEST_TIMEOUT });
    
    await expect(page.locator('#output')).toContainText('Code executed successfully');
    await expect(page.locator('#output')).toContainText('Matrix operations test completed');
    
    console.log('✅ Matrix operations test passed');
  });

  test('Examples dropdown works correctly', async () => {
    // Test examples dropdown functionality
    await page.locator('#examplesBtn').click();
    
    // Check dropdown is visible
    await expect(page.locator('#examplesContent')).toHaveClass(/show/);
    
    // Test loading a basic example
    await page.locator('[data-example="basic"]').click();
    
    // Check that code was loaded into editor
    await expect(page.locator('#editor')).toContainText('Basic PyTorch Operations');
    
    // Check that success message appears
    await expect(page.locator('#output')).toContainText('Loaded example: Basic PyTorch');
    
    // Test running the loaded example
    await page.locator('#runBtn').click();
    
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: TEST_TIMEOUT });
    await expect(page.locator('#output')).toContainText('Code executed successfully');
    
    console.log('✅ Examples dropdown test passed');
  });

  test('Error handling works correctly', async () => {
    const invalidCode = `
# This should cause an error
print("Testing error handling...")
undefined_variable = some_undefined_function()
print("This should not be reached")
`;

    await page.locator('#clearEditor').click();
    await page.locator('#editor').fill(invalidCode);
    
    await page.locator('#runBtn').click();
    
    // Wait for execution to complete (should fail quickly)
    await expect(page.locator('#statusText')).toContainText('Error occurred', { timeout: 30000 });
    
    // Check that error is displayed
    await expect(page.locator('#output')).toContainText('Execution error');
    
    // Check status dot shows error
    await expect(page.locator('.status-dot')).toHaveClass(/status-error/);
    
    console.log('✅ Error handling test passed');
  });

  test('Keyboard shortcuts work', async () => {
    const simpleCode = 'print("Testing keyboard shortcut")';
    
    await page.locator('#clearEditor').click();
    await page.locator('#editor').fill(simpleCode);
    
    // Test Ctrl+Enter (or Cmd+Enter on Mac)
    await page.locator('#editor').focus();
    await page.keyboard.press('Control+Enter');
    
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: TEST_TIMEOUT });
    await expect(page.locator('#output')).toContainText('Code executed successfully');
    
    console.log('✅ Keyboard shortcuts test passed');
  });

  test('Statistics functionality works', async () => {
    // Run some code first to generate stats
    await page.locator('#clearEditor').click();
    await page.locator('#editor').fill('print("Generating stats...")');
    await page.locator('#runBtn').click();
    
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: TEST_TIMEOUT });
    
    // Click stats button
    await page.locator('#statsBtn').click();
    
    // Check that stats are displayed
    await expect(page.locator('#output')).toContainText('Performance Statistics');
    await expect(page.locator('#output')).toContainText('Operations Count');
    
    console.log('✅ Statistics functionality test passed');
  });

  test('WebGPU capabilities test', async () => {
    const webgpuCode = `
# WebGPU capabilities test
import torch
import time

print("Testing WebGPU capabilities...")

# Create tensors
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.tensor([5, 4, 3, 2, 1], dtype=torch.float32)

print(f"x: {x}")
print(f"y: {y}")

# Time basic operations
start_time = time.time()
result = x + y
end_time = time.time()

print(f"Addition result: {result}")
print(f"Time: {(end_time - start_time) * 1000:.2f}ms")

# Test device info
print(f"Device: {x.device}")
print("WebGPU test completed!")
`;

    await page.locator('#clearEditor').click();
    await page.locator('#editor').fill(webgpuCode);
    
    await page.locator('#runBtn').click();
    
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: TEST_TIMEOUT });
    
    await expect(page.locator('#output')).toContainText('Code executed successfully');
    await expect(page.locator('#output')).toContainText('WebGPU test completed');
    
    console.log('✅ WebGPU capabilities test passed');
  });

  test('Clear functions work correctly', async () => {
    // Add some content first
    await page.locator('#editor').fill('print("Test content")');
    await page.locator('#runBtn').click();
    
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: TEST_TIMEOUT });
    
    // Test clear editor
    await page.locator('#clearEditor').click();
    await expect(page.locator('#editor')).toHaveValue('');
    await expect(page.locator('#output')).toContainText('Editor cleared');
    
    // Test clear output
    await page.locator('#clearOutput').click();
    
    // Check that output was cleared (should only contain the "Output cleared" message)
    const outputText = await page.locator('#output').textContent();
    expect(outputText).toContain('Output cleared');
    
    console.log('✅ Clear functions test passed');
  });
});

// Performance and reliability tests
test.describe('Performance and Reliability', () => {
  test('Multiple sequential operations work correctly', async ({ page }) => {
    const testFilePath = path.resolve(__dirname, '../../manual-testing.html');
    await page.goto(`file://${testFilePath}`);
    await page.waitForLoadState('networkidle');
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: INIT_TIMEOUT });

    // Run multiple operations in sequence
    const operations = [
      'print("Operation 1")',
      'x = torch.tensor([1, 2, 3]); print(f"Tensor: {x}")',
      'result = x + x; print(f"Addition: {result}")',
      'print("All operations completed")'
    ];

    for (let i = 0; i < operations.length; i++) {
      await page.locator('#clearEditor').click();
      await page.locator('#editor').fill(operations[i]);
      await page.locator('#runBtn').click();
      
      await expect(page.locator('#statusText')).toContainText('Ready', { timeout: TEST_TIMEOUT });
      await expect(page.locator('#output')).toContainText('Code executed successfully');
      
      console.log(`✅ Operation ${i + 1} completed successfully`);
    }
    
    console.log('✅ Multiple sequential operations test passed');
  });

  test('Large tensor operations work correctly', async ({ page }) => {
    const testFilePath = path.resolve(__dirname, '../../manual-testing.html');
    await page.goto(`file://${testFilePath}`);
    await page.waitForLoadState('networkidle');
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: INIT_TIMEOUT });

    const largeOperationCode = `
# Large tensor operations test
import torch
import numpy as np

print("Testing large tensor operations...")

# Create larger tensors
x = torch.randn(100, 100)
y = torch.randn(100, 100)

print(f"Created tensors of shape: {x.shape}")

# Matrix multiplication
result = torch.matmul(x, y)
print(f"Matrix multiplication result shape: {result.shape}")

# Sum operation
total = torch.sum(result)
print(f"Sum of all elements: {total}")

print("Large tensor operations completed!")
`;

    await page.locator('#clearEditor').click();
    await page.locator('#editor').fill(largeOperationCode);
    
    await page.locator('#runBtn').click();
    
    // Longer timeout for large operations
    await expect(page.locator('#statusText')).toContainText('Ready', { timeout: TEST_TIMEOUT * 2 });
    
    await expect(page.locator('#output')).toContainText('Code executed successfully');
    await expect(page.locator('#output')).toContainText('Large tensor operations completed');
    
    console.log('✅ Large tensor operations test passed');
  });
});