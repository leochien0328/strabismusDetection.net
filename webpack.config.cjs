const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');


module.exports = {
  mode:'production',
  entry:{ 
    main:'./src/index.js',
  },
  output: {
    filename: '[name].js',
    path: path.resolve(__dirname, 'dist'),
  },
  resolve: {
    fallback: {
      "assert": require.resolve("assert/"),
      "https": require.resolve("https-browserify"),
      "os": require.resolve("os-browserify/browser"),
      "buffer": require.resolve("buffer/"),
      "crypto": require.resolve("crypto-browserify"),
      "zlib": require.resolve("browserify-zlib"),
      "querystring": require.resolve("querystring-es3"),
      "path": require.resolve("path-browserify"),
      "crypto": require.resolve("crypto-browserify"),
      "http": require.resolve("stream-http"),
      "stream": require.resolve("stream-browserify"),
      "util": require.resolve("util/"),
      "url": require.resolve("url/"),
      "vm": require.resolve("vm-browserify"),
      "net": require.resolve("net"),
      "net": false,
      "fs": false,
      "async_hooks": false
    }
  },
  module: {
    rules: [
      {
        test: /\.css$/,
        use: [MiniCssExtractPlugin.loader, 'css-loader'],
      },
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env'],
          },
        },
      },
      {
        test: /express\/lib\/view\.js$/,
        use: 'null-loader'
      },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: './index.html',
    }),
    new MiniCssExtractPlugin({
      filename: '[name].css',
    }),
  ],
  devServer: {
    contentBase: path.join(__dirname, 'dist'),
    compress: true,
    port: 9000,
    proxy:{
      '/api':{
        target:'http://localhost:10000',
        changeOrigin:true,
      },
    },
  },
  optimization: {
    splitChunks: {
      chunks: 'all',
    },
  },
};