import fetch from 'node-fetch'; // 导入 fetch 模块

exports.handler = async function(event) {
    const raw_image_data = event.body;
    const apiUrl = process.env.API_URL || 'https://strabismusdetection-net.onrender.com/api/upload-photo';

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

