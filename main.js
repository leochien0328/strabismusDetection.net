document.addEventListener('DOMContentLoaded', function() {
    // 确保在操作元素前先检查它是否存在
    if (document.getElementById('my_camera')) {
        Webcam.set({
            width: 320,
            height: 240,
            image_format: 'jpeg',
            jpeg_quality: 90
        });

        document.getElementById('accesscamera').addEventListener('click', function() {
            Webcam.reset();
            Webcam.on('error', function() {
                swal({
                    title: 'Warning',
                    text: 'Please give permission to access your webcam',
                    icon: 'warning'
                });
            });
            Webcam.attach('#my_camera');
            document.getElementById('takephoto').classList.remove('d-none');
            document.getElementById('takephoto').classList.add('d-block');
        });

        document.getElementById('takephoto').addEventListener('click', take_snapshot);

        document.getElementById('retakephoto').addEventListener('click', function() {
            document.getElementById('my_camera').classList.add('d-block');
            document.getElementById('my_camera').classList.remove('d-none');
            document.getElementById('results').classList.add('d-none');
            document.getElementById('takephoto').classList.add('d-block');
            document.getElementById('takephoto').classList.remove('d-none');
            document.getElementById('retakephoto').classList.add('d-none');
            document.getElementById('retakephoto').classList.remove('d-block');
            document.getElementById('uploadphoto').classList.add('d-none');
            document.getElementById('uploadphoto').classList.remove('d-block');
        });

        document.getElementById('uploadphoto').addEventListener('click', function() {
            var raw_image_data = document.getElementById('photoStore').value;
            upload_photo(raw_image_data);
        });

        async function upload_photo(raw_image_data) {
            try {
                const response = await fetch('/api/upload-photo', {  // 使用代理服务器的路径
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': '70bdf7dde1abcefebd3f83b09656e340'  // 添加 API Key
                    },
                    body: JSON.stringify({ image: raw_image_data })
                });
                const data = await response.json();
                if (data.result > 3) {
                    $('#possibleStrabismusModal').modal('show');
                } else {
                    $('#noStrabismusModal').modal('show');
                }
                console.log('Solution:', data.solution);
            } catch (error) {
                console.error('Error:', error);
                swal({
                    title: 'Error',
                    text: 'Failed to upload photo. Please try again.',
                    icon: 'error'
                });
            }
        }

        function take_snapshot() {
            Webcam.snap(function(data_uri) {
                document.getElementById('results').innerHTML = '<img src="' + data_uri + '" class="d-block mx-auto rounded"/>';

                var raw_image_data = data_uri.replace(/^data:image\/\w+;base64,/, '');
                document.getElementById('photoStore').value = raw_image_data;
            });

            document.getElementById('my_camera').classList.remove('d-block');
            document.getElementById('my_camera').classList.add('d-none');
            document.getElementById('results').classList.remove('d-none');
            document.getElementById('takephoto').classList.remove('d-block');
            document.getElementById('takephoto').classList.add('d-none');
            document.getElementById('retakephoto').classList.remove('d-none');
            document.getElementById('retakephoto').classList.add('d-block');
            document.getElementById('uploadphoto').classList.remove('d-none');
            document.getElementById('uploadphoto').classList.add('d-block');
        }
    } else {
        console.error('Element #my_camera not found.');
    }
});

document.addEventListener('DOMContentLoaded', function() {
    // 取得 detailsButton 元素
    const detailsButton = document.getElementById('detailsButton');

    // 确认元素是否存在
    if (detailsButton) {
        // 如果存在，则添加 'singular' 类别
        detailsButton.classList.add('singular');
    } else {
        console.error('Element with id "detailsButton" not found.');
    }
});
