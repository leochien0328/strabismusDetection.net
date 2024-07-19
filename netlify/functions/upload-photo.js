import fetch from 'node-fetch';

export const handler = async (event) => {
  try {
    console.log('Event Body:', event.body);
    const raw_image_data = JSON.parse(event.body).image;
    const apiUrl = process.env.API_URL || 'https://strabismusdetection-net.onrender.com/api/upload-photo';
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': '70bdf7dde1abcefebd3f83b09656e340'
      },
      body: JSON.stringify({ image: raw_image_data })
    });

    const data = await response.json();
    console.log('API Response:', data);

    return {
      statusCode: 200,
      body: JSON.stringify(data)
    };
  } catch (error) {
    console.error('Error:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Failed to upload photo' })
    };
  }
};
