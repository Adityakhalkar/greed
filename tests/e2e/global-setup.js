/**
 * Global setup for Playwright tests
 * Prepares the testing environment
 */

const { chromium } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

async function globalSetup() {
  console.log('🚀 Setting up Greed.js E2E testing environment...');
  
  // Ensure test results directory exists
  const testResultsDir = path.resolve(__dirname, '../../test-results');
  if (!fs.existsSync(testResultsDir)) {
    fs.mkdirSync(testResultsDir, { recursive: true });
  }
  
  // Check if manual-testing.html exists
  const testFile = path.resolve(__dirname, '../../manual-testing.html');
  if (!fs.existsSync(testFile)) {
    throw new Error(`Test file not found: ${testFile}`);
  }
  
  // Check if greed-standalone.js exists
  const greedStandalone = path.resolve(__dirname, '../../dist/greed-standalone.js');
  if (!fs.existsSync(greedStandalone)) {
    throw new Error(`Greed standalone build not found: ${greedStandalone}. Run 'npx webpack --config webpack.simple.config.js' first.`);
  }
  
  console.log('✅ Test file validation complete');
  
  // Test basic browser compatibility
  const browser = await chromium.launch({ 
    headless: true, // Use headless for setup check
    args: ['--enable-unsafe-webgpu', '--enable-features=VaapiVideoDecoder']
  });
  
  const context = await browser.newContext({
    // Remove webgpu permission as it's not recognized by Playwright
  });
  
  const page = await context.newPage();
  
  // Test if we can load the test page
  try {
    await page.goto(`file://${testFile}`, { waitUntil: 'domcontentloaded' });
    console.log('✅ Test page loads successfully');
  } catch (error) {
    console.error('❌ Failed to load test page:', error.message);
    throw error;
  } finally {
    await browser.close();
  }
  
  console.log('🎯 Global setup completed successfully');
  
  // Return setup data for tests
  return {
    testFile,
    greedStandalone,
    timestamp: new Date().toISOString()
  };
}

module.exports = globalSetup;