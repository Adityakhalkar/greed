/**
 * Babel configuration for Greed.js v2.0
 * Handles ES modules, async/await, and modern JavaScript features
 */
module.exports = {
  presets: [
    ['@babel/preset-env', {
      targets: {
        node: 'current',
        browsers: ['> 1%', 'last 2 versions', 'not dead']
      },
      modules: 'auto', // Let Babel handle module transformation for Jest
      useBuiltIns: 'usage',
      corejs: 3
    }]
  ],
  plugins: [],
  env: {
    test: {
      presets: [
        ['@babel/preset-env', {
          targets: { node: 'current' },
          modules: 'commonjs' // Use CommonJS for Jest
        }]
      ]
    }
  }
};