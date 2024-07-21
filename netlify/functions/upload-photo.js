import fetch from 'node-fetch';

exports.handler = async (event) => {
    try {
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
        return {
            statusCode: 200,
            body: JSON.stringify(data)
        };
    } catch (error) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: error.message || 'Failed to upload photo' })
        };
    }
};
