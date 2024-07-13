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
            //$('#photoModal').modal('hide');
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
        $('#my_camera').addClass('d-block');
        $('#my_camera').removeClass('d-none');

        $('#results').addClass('d-none');

        $('#takephoto').addClass('d-block');
        $('#takephoto').removeClass('d-none');

        $('#retakephoto').addClass('d-none');
        $('#retakephoto').removeClass('d-block');

        $('#uploadphoto').addClass('d-none');
        $('#uploadphoto').removeClass('d-block');
    });

    $('#uploadphoto').on('click', function() {
        var raw_image_data = $('#photoStore').val();
        upload_photo(raw_image_data);
    });

    function upload_photo(raw_image_data) {
        fetch('https://app-bq9j.onrender.com/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: raw_image_data })
        })
        .then(response => response.json())
        .then(data => {
            if (data.result >3) {
                $('#noStrabismusModal').modal('show');
            }  else {
                $('#noStrabismusModal').modal('show');
            }
            console.log('Solution:', data.solution);
        })
        .catch(error => {
            console.error('Error:', error);
            swal({
                title: 'Error',
                text: 'Failed to upload photo. Please try again.',
                icon: 'error'
            });
        });
    }
});

function take_snapshot() {
    Webcam.snap(function(data_uri) {
        $('#results').html('<img src="' + data_uri + '" class="d-block mx-auto rounded"/>');

        var raw_image_data = data_uri.replace(/^data\:image\/\w+\;base64\,/, '');
        $('#photoStore').val(raw_image_data);
    });

    $('#my_camera').removeClass('d-block');
    $('#my_camera').addClass('d-none');

    $('#results').removeClass('d-none');

    $('#takephoto').removeClass('d-block');
    $('#takephoto').addClass('d-none');

    $('#retakephoto').removeClass('d-none');
    $('#retakephoto').addClass('d-block');

    $('#uploadphoto').removeClass('d-none');
    $('#uploadphoto').addClass('d-block');
}
