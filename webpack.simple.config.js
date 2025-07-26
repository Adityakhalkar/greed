/**
 * Simple Webpack configuration for standalone Greed.js build
 * No chunk splitting - single file for easy testing
 */
const path = require('path');

module.exports = {
  entry: './src/core/greed-v2.js',
  
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'greed-standalone.js',
    library: {
      name: 'Greed',
      type: 'umd',
      export: 'default'
    },
    globalObject: 'this'
  },

  optimization: {
    splitChunks: {
      chunks: 'async' // Only split async chunks, keep main bundle together
    },
    usedExports: true,
    sideEffects: false,
    minimize: false // Keep unminified for easier debugging
  },

  resolve: {
    extensions: ['.js', '.mjs'],
    alias: {
      '@core': path.resolve(__dirname, 'src/core'),
      '@compute': path.resolve(__dirname, 'src/compute'),
      '@utils': path.resolve(__dirname, 'src/utils')
    }
  },

  module: {
    rules: [
      {
        test: /\.m?js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              ['@babel/preset-env', {
                targets: {
                  browsers: ['> 1%', 'last 2 versions', 'not dead']
                },
                modules: false,
                useBuiltIns: 'usage',
                corejs: 3
              }]
            ]
          }
        }
      }
    ]
  },

  externals: {
    'pyodide': {
      commonjs: 'pyodide',
      commonjs2: 'pyodide',
      amd: 'pyodide',
      root: 'loadPyodide'
    }
  },

  devtool: 'source-map',
  target: ['web', 'es2020']
};