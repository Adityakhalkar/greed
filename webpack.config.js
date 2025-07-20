const path = require('path');

module.exports = {
  entry: './src/greed.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'greed.min.js',
    library: 'Greed',
    libraryTarget: 'umd',
    globalObject: 'this'
  },
  devtool: 'source-map', // Use source-map instead of eval to avoid size limitations
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      }
    ]
  },
  resolve: {
    extensions: ['.js']
  },
  devServer: {
    static: {
      directory: path.join(__dirname, './'),
    },
    compress: true,
    port: 3000,
    open: true,
    hot: true
  },
  optimization: {
    minimize: false // Disable minification for now to preserve PyTorch polyfill
  }
};