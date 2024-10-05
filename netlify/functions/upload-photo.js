const fetch = require('node-fetch');

exports.handler = async (event) => {
    if (event.httpMethod !== 'POST') {
        return {
            statusCode: 405,
            body: JSON.stringify({ error: "Method not allowed" }),
        };
    }

    try {
        const body = JSON.parse(event.body);
        const image = body.image;

        if (!image) {
            return {
                statusCode: 400,
                body: JSON.stringify({ error: "No image data provided" }),
            };
        }

        const apiUrl = process.env.API_URL || 'https://strabismusdetection-net.onrender.com/api/upload-photo';

        const response = await Promise.race([
            fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': '70bdf7dde1abcefebd3f83b09656e340'  // 添加 API Key
                },
                body: JSON.stringify({ image })
            }),
            new Promise((_, reject) => setTimeout(() => reject(new Error('Request timed out')), 5000))
        ]);


        if (!response.ok) {
            throw new Error(`API responded with status: ${response.status}`);
        }

        const result = await response.json();

        return {
            statusCode: 200,
            body: JSON.stringify(result),
        };
    } catch (error) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: error.message }),
        };
    }
};

