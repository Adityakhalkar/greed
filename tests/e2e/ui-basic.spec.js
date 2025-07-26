/**
 * Basic UI Tests for Greed.js Manual Testing Environment
 * Tests UI components without waiting for full initialization
 */

const { test, expect } = require('@playwright/test');
const path = require('path');

// Quick test configuration
const QUICK_TIMEOUT = 10000; // 10 seconds for UI tests
const LOAD_TIMEOUT = 30000;  // 30 seconds for page load

test.describe('Greed.js Manual Testing Environment - UI Components', () => {
  let page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();
    
    // Enable console logging
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', error => console.error('PAGE ERROR:', error));
    
    // Navigate to the manual testing environment
    const testFilePath = path.resolve(__dirname, '../../manual-testing.html');
    await page.goto(`file://${testFilePath}`);
    
    // Wait for the page to load
    await page.waitForLoadState('networkidle');
    
    console.log('✅ Page loaded successfully');
  });

  test.afterEach(async () => {
    if (page) {
      await page.close();
    }
  });

  test('Page loads and displays correctly', async () => {
    // Check page title
    await expect(page).toHaveTitle('Greed.js Manual Testing Environment');
    
    // Check main header
    await expect(page.locator('h1')).toContainText('Greed.js Manual Testing');
    await expect(page.locator('.version')).toContainText('v2.0.1');
    
    // Check description
    await expect(page.locator('.header p')).toContainText('WebGPU-accelerated Python runtime');
    
    console.log('✅ Page content validation complete');
  });

  test('UI layout and structure is correct', async () => {
    // Check main container
    await expect(page.locator('.container')).toBeVisible();
    
    // Check header section
    await expect(page.locator('.header')).toBeVisible();
    
    // Check main content grid
    await expect(page.locator('.main-content')).toBeVisible();
    
    // Check two panel layout
    await expect(page.locator('.editor-panel')).toBeVisible();
    await expect(page.locator('.output-panel')).toBeVisible();
    
    // Check status bar
    await expect(page.locator('.status-bar')).toBeVisible();
    
    console.log('✅ Layout structure validation complete');
  });

  test('Editor panel components are present', async () => {
    // Check panel header
    await expect(page.locator('.editor-panel .panel-header')).toContainText('Python Code Editor');
    
    // Check editor textarea
    const editor = page.locator('#editor');
    await expect(editor).toBeVisible();
    await expect(editor).toBeEnabled();
    
    // Check placeholder text
    await expect(editor).toHaveAttribute('placeholder');
    
    // Check controls
    await expect(page.locator('#runBtn')).toBeVisible();
    await expect(page.locator('#clearEditor')).toBeVisible();
    await expect(page.locator('#examplesBtn')).toBeVisible();
    await expect(page.locator('#clearOutput')).toBeVisible();
    await expect(page.locator('#statsBtn')).toBeVisible();
    
    console.log('✅ Editor panel validation complete');
  });

  test('Output panel components are present', async () => {
    // Check panel header
    await expect(page.locator('.output-panel .panel-header')).toContainText('Output & Results');
    
    // Check output container
    await expect(page.locator('#output')).toBeVisible();
    
    console.log('✅ Output panel validation complete');
  });

  test('Status bar components are present', async () => {
    // Check status indicator
    await expect(page.locator('.status-indicator')).toBeVisible();
    await expect(page.locator('#statusDot')).toBeVisible();
    await expect(page.locator('#statusText')).toBeVisible();
    
    // Check execution time display
    await expect(page.locator('#executionTime')).toBeVisible();
    await expect(page.locator('#executionTime')).toContainText('Execution time: --');
    
    console.log('✅ Status bar validation complete');
  });

  test('Examples dropdown functionality works', async () => {
    // Click examples button
    await page.locator('#examplesBtn').click();
    
    // Check dropdown appears
    await expect(page.locator('#examplesContent')).toHaveClass(/show/);
    
    // Check example items are present
    await expect(page.locator('[data-example="basic"]')).toBeVisible();
    await expect(page.locator('[data-example="neural-network"]')).toBeVisible();
    await expect(page.locator('[data-example="matrix-ops"]')).toBeVisible();
    await expect(page.locator('[data-example="webgpu-test"]')).toBeVisible();
    await expect(page.locator('[data-example="simple-test"]')).toBeVisible();
    
    // Check example titles and descriptions
    await expect(page.locator('[data-example="basic"] .example-title')).toContainText('Basic PyTorch');
    await expect(page.locator('[data-example="basic"] .example-desc')).toContainText('Tensor creation and operations');
    
    // Click outside to close dropdown
    await page.locator('body').click();
    await expect(page.locator('#examplesContent')).not.toHaveClass(/show/);
    
    console.log('✅ Examples dropdown validation complete');
  });

  test('Clear editor functionality works', async () => {
    // Add some text to editor
    await page.locator('#editor').fill('print("test content")');
    
    // Verify text is there
    await expect(page.locator('#editor')).toHaveValue('print("test content")');
    
    // Click clear button
    await page.locator('#clearEditor').click();
    
    // Verify editor is cleared
    await expect(page.locator('#editor')).toHaveValue('');
    
    // Check output message appears
    await expect(page.locator('#output')).toContainText('Editor cleared');
    
    console.log('✅ Clear editor functionality validation complete');
  });

  test('Clear output functionality works', async () => {
    // First add some content to output by clearing editor (which adds a message)
    await page.locator('#clearEditor').click();
    await expect(page.locator('#output')).toContainText('Editor cleared');
    
    // Then clear output
    await page.locator('#clearOutput').click();
    
    // Check that output now only contains the "Output cleared" message
    const outputText = await page.locator('#output').textContent();
    expect(outputText).toContain('Output cleared');
    
    console.log('✅ Clear output functionality validation complete');
  });

  test('Editor text input and editing works', async () => {
    const testCode = `# Test code
print("Hello World!")
x = 1 + 1
print(f"Result: {x}")`;

    // Clear editor first
    await page.locator('#clearEditor').click();
    
    // Type in editor
    await page.locator('#editor').fill(testCode);
    
    // Verify content
    await expect(page.locator('#editor')).toHaveValue(testCode);
    
    // Test appending text
    await page.locator('#editor').focus();
    await page.keyboard.press('End');
    await page.keyboard.type('\nprint("Additional line")');
    
    await expect(page.locator('#editor')).toContainText('Additional line');
    
    console.log('✅ Editor text input validation complete');
  });

  test('Tab support in editor works', async () => {
    // Focus editor
    await page.locator('#editor').focus();
    
    // Clear any existing content
    await page.keyboard.press('Control+a');
    await page.keyboard.press('Delete');
    
    // Type and press Tab
    await page.keyboard.type('def function():');
    await page.keyboard.press('Enter');
    await page.keyboard.press('Tab');
    await page.keyboard.type('return True');
    
    // Check that proper indentation was added
    const editorValue = await page.locator('#editor').inputValue();
    expect(editorValue).toContain('    return True'); // 4 spaces
    
    console.log('✅ Tab support validation complete');
  });

  test('Example loading functionality works', async () => {
    // Open examples dropdown
    await page.locator('#examplesBtn').click();
    
    // Click on basic example
    await page.locator('[data-example="basic"]').click();
    
    // Check that code was loaded into editor
    await expect(page.locator('#editor')).toContainText('Basic PyTorch Operations');
    await expect(page.locator('#editor')).toContainText('import torch');
    
    // Check success message
    await expect(page.locator('#output')).toContainText('Loaded example: Basic PyTorch');
    
    // Check dropdown closed
    await expect(page.locator('#examplesContent')).not.toHaveClass(/show/);
    
    console.log('✅ Example loading validation complete');
  });

  test('All example types can be loaded', async () => {
    const examples = [
      { id: 'simple-test', name: 'Simple Test' },
      { id: 'basic', name: 'Basic PyTorch' },
      { id: 'neural-network', name: 'Neural Network' },
      { id: 'matrix-ops', name: 'Matrix Operations' },
      { id: 'webgpu-test', name: 'WebGPU Test' }
    ];

    for (const example of examples) {
      // Open examples dropdown
      await page.locator('#examplesBtn').click();
      
      // Click on example
      await page.locator(`[data-example="${example.id}"]`).click();
      
      // Check that code was loaded
      const editorValue = await page.locator('#editor').inputValue();
      expect(editorValue.length).toBeGreaterThan(10); // Has some content
      
      // Check success message
      await expect(page.locator('#output')).toContainText(`Loaded example: ${example.name}`);
      
      console.log(`✅ ${example.name} example loaded successfully`);
    }
    
    console.log('✅ All examples validation complete');
  });

  test('Status indicator shows initialization state', async () => {
    // Initially should show "Initializing..."
    await expect(page.locator('#statusText')).toContainText('Initializing...');
    
    // Status dot should exist
    await expect(page.locator('#statusDot')).toBeVisible();
    
    console.log('✅ Status indicator validation complete');
  });

  test('Welcome messages appear in output', async () => {
    // Check for welcome messages
    await expect(page.locator('#output')).toContainText('Welcome to Greed.js Manual Testing Environment');
    await expect(page.locator('#output')).toContainText('Press Ctrl+Enter');
    await expect(page.locator('#output')).toContainText('Try the examples');
    
    console.log('✅ Welcome messages validation complete');
  });
});