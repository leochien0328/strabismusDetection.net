import fetch from 'node-fetch';

exports.handler = async (event) => {
    try {
        console.log("Received event:", JSON.stringify(event));  // 打印整个事件对象

        if (!event.body) {
            throw new Error("Invalid request body");
        }

        const { image } = JSON.parse(event.body);

        if (!image) {
            throw new Error("No image data provided");
        }

        const apiUrl = process.env.API_URL || 'https://strabismusdetection-net.onrender.com/api/upload-photo';

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': '70bdf7dde1abcefebd3f83b09656e340'
            },
            body: JSON.stringify({ image })
        });

        const data = await response.json();
        console.log("Received response from API:", JSON.stringify(data));  // 添加日志
        return {
            statusCode: 200,
            body: JSON.stringify(data)
        };
    } catch (error) {
        console.error("Error:", error.message);  // 添加日志
        return {
            statusCode: 500,
            body: JSON.stringify({ error: error.message || 'Failed to upload photo' })
        };
    }
};
