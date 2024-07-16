const express = require('express');
const cors = require('cors');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
app.use(cors());

app.use('/api', createProxyMiddleware({
    target: 'https://strabismusdetection-net.onrender.com/', // 替换为你的 API URL
    changeOrigin: true,
    pathRewrite: {
        '^/api': '', // 重写路径，使其与目标路径匹配
    },
}));
app.get('/', (req, res) => {
    res.send('Hello World!');
  });

const port = process.env.PORT || 10000;
app.listen(port, () => {
    console.log(`Proxy server running on port ${port}`);
});
