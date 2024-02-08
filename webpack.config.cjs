const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  entry: './main.ts',
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
    extensions: ['.ts', '.js'],
    modules: [
      path.resolve(__dirname, 'node_modules'),
      path.resolve(__dirname, './node_modules/onnxruntime-web/dist'),
    ],
  },
};
