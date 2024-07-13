const fetch = require('node-fetch');

exports.handler = async function(event, context) {
    const raw_image_data = event.body;
    const apiUrl = 'https://api.render.com/deploy/srv-cq5p1gmehbks73bsc580?key=Uax3XalWwX'; // 請替換為你的API URL

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/octect-stream'
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
};
