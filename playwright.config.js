/**
 * Playwright Configuration for Greed.js E2E Testing
 * Non-headless mode for visual testing as requested
 */

module.exports = {
  testDir: './tests/e2e',
  timeout: 120000, // 2 minutes per test
  expect: {
    timeout: 30000 // 30 seconds for assertions
  },
  
  // Run tests in parallel, but limit workers for stability
  fullyParallel: false,
  workers: 1, // Single worker for better debugging in non-headless mode
  
  // Reporter configuration
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['list']
  ],
  
  // Global test settings
  use: {
    // Browser settings
    headless: false, // Non-headless as requested
    viewport: { width: 1280, height: 720 },
    
    // Video and screenshot settings
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
    
    // Trace settings for debugging
    trace: 'retain-on-failure',
    
    // Slow down actions for better visibility
    actionTimeout: 10000,
    navigationTimeout: 30000,
    
    // Browser context settings
    ignoreHTTPSErrors: true,
    acceptDownloads: true,
    
    // Browser permissions (WebGPU enabled via launch args)
    // permissions: ['webgpu'], // Not supported by Playwright
    
    // Extra browser arguments for WebGPU support
    launchOptions: {
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=VaapiVideoDecoder',
        '--use-gl=egl',
        '--enable-gpu-sandbox'
      ]
    }
  },

  // Browser configurations
  projects: [
    {
      name: 'chromium',
      use: { 
        ...require('@playwright/test').devices['Desktop Chrome'],
        channel: 'chrome', // Use full Chrome for WebGPU support
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=VaapiVideoDecoder',
            '--use-gl=egl',
            '--enable-gpu-sandbox',
            '--disable-web-security', // For local file testing
            '--allow-running-insecure-content'
          ]
        }
      },
    },
    
    // Add Firefox for cross-browser testing (optional)
    {
      name: 'firefox',
      use: { 
        ...require('@playwright/test').devices['Desktop Firefox'],
        launchOptions: {
          firefoxUserPrefs: {
            'dom.webgpu.enabled': true,
            'gfx.webgpu.force-enabled': true
          }
        }
      },
    }
  ],

  // Global setup and teardown
  globalSetup: require.resolve('./tests/e2e/global-setup.js'),
  globalTeardown: require.resolve('./tests/e2e/global-teardown.js'),
  
  // Test output directory
  outputDir: 'test-results/',
  
  // Retry configuration
  retries: 2, // Retry failed tests up to 2 times
  
  // Test filtering
  testMatch: '**/*.spec.js',
  testIgnore: '**/node_modules/**',
};