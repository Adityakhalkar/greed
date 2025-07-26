/**
 * Webpack configuration for Greed.js v2.0 - Single unified build
 * Modular architecture with full backward compatibility
 */
const path = require('path');

module.exports = {
  entry: {
    'greed': './src/core/greed-v2.js',
    'greed.min': './src/core/greed-v2.js'
  },
  
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].js',
    library: {
      name: 'Greed',
      type: 'umd',
      export: 'default'
    },
    globalObject: 'this',
    clean: true // Clean dist folder on each build
  },

  optimization: {
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        // Core components
        core: {
          name: 'core',
          test: /[\\/]src[\\/]core[\\/]/,
          chunks: 'all',
          priority: 30
        },
        // Compute engines
        compute: {
          name: 'compute',
          test: /[\\/]src[\\/]compute[\\/]/,
          chunks: 'all',
          priority: 20
        },
        // Utilities
        utils: {
          name: 'utils',
          test: /[\\/]src[\\/]utils[\\/]/,
          chunks: 'all',
          priority: 10
        }
      }
    },
    usedExports: true,
    sideEffects: false,
    minimize: true
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
                modules: false, // Let webpack handle modules for tree shaking
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
    // Pyodide should be loaded separately
    'pyodide': {
      commonjs: 'pyodide',
      commonjs2: 'pyodide',
      amd: 'pyodide',
      root: 'loadPyodide'
    }
  },

  plugins: [],

  devtool: process.env.NODE_ENV === 'development' ? 'source-map' : false,

  devServer: {
    static: {
      directory: path.join(__dirname, 'dist'),
    },
    compress: true,
    port: 8080,
    hot: true,
    open: true,
    headers: {
      // Security headers for development
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin'
    }
  },

  performance: {
    hints: 'warning',
    maxAssetSize: 500000, // 500KB per asset
    maxEntrypointSize: 1000000 // 1MB total
  },

  target: ['web', 'es2020']
};