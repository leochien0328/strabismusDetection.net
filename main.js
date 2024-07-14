$(document).ready(function() {
    Webcam.set({
        width: 320,
        height: 240,
        image_format: 'jpeg',
        jpeg_quality: 90
    });

    $('#accesscamera').on('click', function() {
        Webcam.reset();
        Webcam.on('error', function() {
            swal({
                title: 'Warning',
                text: 'Please give permission to access your webcam',
                icon: 'warning'
            });
        });
        Webcam.attach('#my_camera');
        $('#takephoto').removeClass('d-none').addClass('d-block');
    });

    $('#takephoto').on('click', take_snapshot);

    $('#retakephoto').on('click', function() {
        $('#my_camera').addClass('d-block').removeClass('d-none');
        $('#results').addClass('d-none');
        $('#takephoto').addClass('d-block').removeClass('d-none');
        $('#retakephoto').addClass('d-none').removeClass('d-block');
        $('#uploadphoto').addClass('d-none').removeClass('d-block');
    });

    $('#uploadphoto').on('click', function() {
        var raw_image_data = $('#photoStore').val();
        upload_photo(raw_image_data);
    });

    async function upload_photo(raw_image_data) {
        console.log("Uploading photo...");  // 添加日志信息
        try {
            const response = await fetch('https://app-bq9j.onrender.com', {  // 确认 API URL 是否正确
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'rnd_O7xRsiVqeFJwSpSZh7Sbn1CajNWk'
                },
                body: JSON.stringify({ image: raw_image_data })
            });
            console.log("Response status:", response.status);  // 添加日志信息
            const data = await response.json();
            console.log("Response data:", data);  // 添加日志信息
            if (data.result > 3) {
                $('#noStrabismusModal').modal('show');
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
            $('#results').html('<img src="' + data_uri + '" class="d-block mx-auto rounded"/>');

            var raw_image_data = data_uri.replace(/^data:image\/\w+;base64,/, '');
            $('#photoStore').val(raw_image_data);
        });

        $('#my_camera').removeClass('d-block').addClass('d-none');
        $('#results').removeClass('d-none');
        $('#takephoto').removeClass('d-block').addClass('d-none');
        $('#retakephoto').removeClass('d-none').addClass('d-block');
        $('#uploadphoto').removeClass('d-none').addClass('d-block');
    }
});


