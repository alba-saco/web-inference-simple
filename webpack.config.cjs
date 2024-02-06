const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  entry: './main.js',
  mode: 'development',
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist'),
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: 'node_modules/onnxruntime-web/dist/**/*.wasm',
          to: 'onnxruntime-web/dist/[name][ext]',
        },
      ],
    }),
  ],
  resolve: {
    extensions: ['.js'],
    modules: [
      path.resolve(__dirname, 'node_modules'),
      path.resolve(__dirname, './node_modules/onnxruntime-web/dist'),
    ],
  },
};