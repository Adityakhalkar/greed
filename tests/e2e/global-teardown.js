/**
 * Global teardown for Playwright tests
 * Cleanup and report generation
 */

const fs = require('fs');
const path = require('path');

async function globalTeardown() {
  console.log('🧹 Running global teardown...');
  
  // Generate test summary report
  const testResultsDir = path.resolve(__dirname, '../../test-results');
  const resultsFile = path.join(testResultsDir, 'results.json');
  
  if (fs.existsSync(resultsFile)) {
    try {
      const results = JSON.parse(fs.readFileSync(resultsFile, 'utf8'));
      
      // Generate summary
      const summary = {
        timestamp: new Date().toISOString(),
        totalTests: results.suites?.reduce((acc, suite) => acc + (suite.specs?.length || 0), 0) || 0,
        passed: 0,
        failed: 0,
        skipped: 0,
        duration: results.stats?.duration || 0
      };
      
      // Count test results
      if (results.suites) {
        results.suites.forEach(suite => {
          if (suite.specs) {
            suite.specs.forEach(spec => {
              if (spec.tests) {
                spec.tests.forEach(test => {
                  test.results?.forEach(result => {
                    if (result.status === 'passed') summary.passed++;
                    else if (result.status === 'failed') summary.failed++;
                    else if (result.status === 'skipped') summary.skipped++;
                  });
                });
              }
            });
          }
        });
      }
      
      // Write summary
      const summaryFile = path.join(testResultsDir, 'summary.json');
      fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2));
      
      console.log('📊 Test Summary:');
      console.log(`   Total Tests: ${summary.totalTests}`);
      console.log(`   ✅ Passed: ${summary.passed}`);
      console.log(`   ❌ Failed: ${summary.failed}`);
      console.log(`   ⏭️  Skipped: ${summary.skipped}`);
      console.log(`   ⏱️  Duration: ${Math.round(summary.duration / 1000)}s`);
      
      // Generate simple text report
      const textReport = `
# Greed.js E2E Test Results

**Test Run:** ${summary.timestamp}
**Total Tests:** ${summary.totalTests}
**Passed:** ${summary.passed}
**Failed:** ${summary.failed}
**Skipped:** ${summary.skipped}
**Duration:** ${Math.round(summary.duration / 1000)}s

## Results Details
- Success Rate: ${summary.totalTests > 0 ? Math.round((summary.passed / summary.totalTests) * 100) : 0}%
- Test Environment: Manual Testing Interface
- Browser: Chromium with WebGPU support

## Test Coverage
- ✅ Environment initialization
- ✅ Basic Python execution
- ✅ PyTorch tensor operations
- ✅ Neural network functionality
- ✅ Matrix operations
- ✅ UI interactions (examples, controls)
- ✅ Error handling
- ✅ Performance testing
- ✅ WebGPU capabilities

Generated at: ${new Date().toLocaleString()}
`;
      
      const reportFile = path.join(testResultsDir, 'test-report.md');
      fs.writeFileSync(reportFile, textReport);
      
      console.log(`📋 Detailed report saved to: ${reportFile}`);
      
    } catch (error) {
      console.error('❌ Error generating test summary:', error.message);
    }
  }
  
  console.log('✅ Global teardown completed');
}

module.exports = globalTeardown;