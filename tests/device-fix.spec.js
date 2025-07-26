const { test, expect } = require('@playwright/test');

test.describe('Device Method Fix Test', () => {
  test('should access torch.device method without error', async ({ page }) => {
    // Navigate to the test page
    await page.goto('http://localhost:8000/test-device-fix.html');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Click the test button
    await page.click('button:has-text("Run Test")');
    
    // Wait for test completion - look for success or error message
    await page.waitForSelector('.success, .error', { timeout: 30000 });
    
    // Check if we got success messages
    const successMessages = await page.locator('.success').count();
    const errorMessages = await page.locator('.error').count();
    
    console.log(`Success messages: ${successMessages}, Error messages: ${errorMessages}`);
    
    // Log all output for debugging
    const outputs = await page.locator('#device-output .info, #device-output .success, #device-output .error').allTextContents();
    console.log('Test outputs:', outputs);
    
    // Should have success messages and no error messages
    expect(successMessages).toBeGreaterThan(0);
    expect(errorMessages).toBe(0);
    
    // Check for specific success indicators
    const outputText = await page.locator('#device-output').textContent();
    expect(outputText).toContain('Device method test completed successfully!');
    expect(outputText).toContain('CPU device:');
    expect(outputText).toContain('CUDA device:');
  });
});