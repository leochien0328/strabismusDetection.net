import fetch from 'node-fetch'; // 导入 fetch 模块
exports.handler = async (event) => {
    const raw_image_data = JSON.parse(event.body).image;
    const apiUrl = process.env.API_URL || 'https://strabismusdetection-net.onrender.com/api/upload-photo';
  
    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': '70bdf7dde1abcefebd3f83b09656e340'
        },
        body: JSON.stringify({ image: raw_image_data })
      });
      const data = await response.json();
      return {
        statusCode: 200,
        body: JSON.stringify(data)
      };
    } catch (error) {
      return {
        statusCode: 500,
        body: JSON.stringify({ error: 'Failed to upload photo' })
      };
    }
  };

