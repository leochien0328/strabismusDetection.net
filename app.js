import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import { createProxyMiddleware } from 'http-proxy-middleware';

const app = express();

// 啟用 CORS 和 body-parser，這裡可以合併兩個中介軟體來減少冗餘
app.use(cors({
    origin: 'https://strabismusdecation.com/', // 只允許特定來源，減少無用請求
    methods: ['POST', 'GET'],  // 只允許必要的請求方法
    optionsSuccessStatus: 204 // 減少 OPTIONS 請求的負擔
}));

app.use(bodyParser.json({ limit: '1mb' })); // 增加請求大小限制以防過大請求

// 建立 Proxy 中介軟體
app.use('/api', createProxyMiddleware({
    target: process.env.API_URL || 'https://strabismusdetection-net.onrender.com/api/upload-photo',
    changeOrigin: true,
    pathRewrite: { '^/api': '' }, // 簡化路徑重寫
    onProxyReq(proxyReq, req) {
        // 可以根據需要添加一些優化邏輯，例如檢查請求頭信息
        console.log(`Proxying request to: ${proxyReq.path}`);
    },
}));

// 接收上傳的圖片
app.post('/api/upload-photo', (req, res) => {
    const { image } = req.body;
    if (!image) {
        return res.status(400).json({ error: 'No image provided' });
    }
    console.log('Received image data');
    
    // 直接返回處理結果，這裡可以使用更高效的處理方式如異步處理圖片
    res.json({
        message: 'Image uploaded and processed successfully',
        result: 'your_result_here'
    });
});

// 避免不必要的路由
const port = process.env.PORT || 10000;
app.listen(port, () => {
    console.log(`Proxy server running on port ${port}`);
});

