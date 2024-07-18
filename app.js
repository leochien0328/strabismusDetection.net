import express from 'express';
import cors from 'cors';
import { createProxyMiddleware } from 'http-proxy-middleware';



const app = express();
app.use(express.json()); // 解析 JSON 格式的请求体
app.use(express.urlencoded({ extended: true })); // 解析 URL 编码的请求体
app.use(cors({
    origin: 'https://strabismusdetection.com'
}));

app.use('/api', createProxyMiddleware({
    target: ' https://strabismusdetection-net.onrender.com', // 替换为你的 API URL
    changeOrigin: true,
    pathRewrite: {
        '^/api': '', // 重写路径，使其与目标路径匹配
    },
}));

app.get('/', (req, res) => {
    res.send('Hello World!');
});
app.post('/api/upload-photo', (req, res) => {
    const { image } = req.body;
    console.log('Received image data:', image);
    // 在這裡進行相應的處理，例如圖片處理、斜視檢測等等

    // 返回處理結果，這裡假設您返回一個 JSON 對象
    res.json({
        message: 'Image uploaded and processed successfully',
        result: 'your_result_here'
    });
});
const port = process.env.PORT || 10000;
app.listen(port, () => {
    console.log(`Proxy server running on port ${port}`);
});
