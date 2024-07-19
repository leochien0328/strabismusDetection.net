import fetch from 'node-fetch';

exports.handler = async (event) => {
    const raw_image_data = JSON.parse(event.body).image;
    try {
        // 确保 event.body 存在并且是一个有效的 JSON 字符串
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
          body: JSON.stringify({ image: raw_image_data  })
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
