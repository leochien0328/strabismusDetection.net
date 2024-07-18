// Importing necessary modules and styles
import '../styles.css'; // If you have any CSS files
import { upload_photo } from '../netlify/functions/upload-photo';

// You can initialize your main functionality here
document.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('uploadphoto');
    button.addEventListener('click', () => {
      // Assume you have raw_image_data available
      const raw_image_data = '';
      upload_photo(raw_image_data);
    });
});

