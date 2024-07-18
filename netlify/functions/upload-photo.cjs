const fetch = require('node-fetch');
const FormData = require('form-data');
const { createProxyMiddleware } = require('http-proxy-middleware');
const express = require('express');
const app = express();

app.use(express.json()); // 解析 JSON 格式的请求体
app.use(express.urlencoded({ extended: true })); // 解析 URL 编码的请求体

export async function handler(event, context) {
    const raw_image_data = event.body;
    const apiUrl = process.env.API_URL || 'https://strabismusdetection-net.onrender.com/';

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/octet-stream' // 修改拼写错误
            },
            body: raw_image_data
        });

        const data = await response.json();
        return {
            statusCode: 200,
            body: JSON.stringify(data)
        };
    } catch (error) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: 'Failed to process the image' })
        };
    }
}

const port = process.env.PORT || 10000;
app.listen(port, () => {
    console.log(`Proxy server running on port ${port}`);
});
