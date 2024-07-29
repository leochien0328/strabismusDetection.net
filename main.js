document.addEventListener('DOMContentLoaded', function() {
    // 确保在操作元素前先检查它是否存在
    const myCameraElement = document.getElementById('my_camera');
    const accessCameraButton = document.getElementById('accesscamera');
    const takePhotoButton = document.getElementById('takephoto');
    const retakePhotoButton = document.getElementById('retakephoto');
    const uploadPhotoButton = document.getElementById('uploadphoto');
    const photoStoreInput = document.getElementById('photoStore');
    const resultsElement = document.getElementById('results');
    const detailsButton = document.getElementById('detailsButton');

    if (myCameraElement && accessCameraButton && takePhotoButton && retakePhotoButton && uploadPhotoButton && photoStoreInput && resultsElement) {
        Webcam.set({
            width: 320,
            height: 240,
            image_format: 'jpeg',
            jpeg_quality: 90
        });

        accessCameraButton.addEventListener('click', function() {
            Webcam.reset();
            Webcam.on('error', function() {
                swal({
                    title: 'Warning',
                    text: 'Please give permission to access your webcam',
                    icon: 'warning'
                });
            });
            Webcam.attach('#my_camera');
            takePhotoButton.classList.remove('d-none');
            takePhotoButton.classList.add('d-block');
        });

        takePhotoButton.addEventListener('click', take_snapshot);

        retakePhotoButton.addEventListener('click', function() {
            myCameraElement.classList.add('d-block');
            myCameraElement.classList.remove('d-none');
            resultsElement.classList.add('d-none');
            takePhotoButton.classList.add('d-block');
            takePhotoButton.classList.remove('d-none');
            retakePhotoButton.classList.add('d-none');
            retakePhotoButton.classList.remove('d-block');
            uploadPhotoButton.classList.add('d-none');
            uploadPhotoButton.classList.remove('d-block');
        });

        uploadPhotoButton.addEventListener('click', function() {
            var raw_image_data = document.getElementById('photoStore').value;
            upload_photo(raw_image_data);
        });

        async function upload_photo(raw_image_data) {
            try {
                const response = await fetch('https://strabismusdetection.com/.netlify/functions/upload-photo', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': '70bdf7dde1abcefebd3f83b09656e340'  // 添加 API Key
                    },
                    body: JSON.stringify({ image: raw_image_data })
                });
                const data = await response.json();
                if (data.result == 1) {
                    $('#exotropiaStrabismusModal').modal('show');
                }
                if(data.result==2){
                    $('#esotropiaStrabismusModal').modal('show');
                }
                if(data.result==3){
                    $('#hypotropiaStrabismusModal').modal('show');
                }
                if(data.result==4){
                    $('#hypotropiaStrabismusModal').modal('show');
                }
                if(data.result==0){
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
                while (raw_image_data.length % 4 !== 0) {
                    raw_image_data += '=';
                }
                document.getElementById('photoStore').value = raw_image_data;
            });

            myCameraElement.classList.remove('d-block');
            myCameraElement.classList.add('d-none');
            resultsElement.classList.remove('d-none');
            takePhotoButton.classList.remove('d-block');
            takePhotoButton.classList.add('d-none');
            retakePhotoButton.classList.remove('d-none');
            retakePhotoButton.classList.add('d-block');
            uploadPhotoButton.classList.remove('d-none');
            uploadPhotoButton.classList.add('d-block');
        }
        document.getElementById('photoForm').addEventListener('submit', async function(event) {
            event.preventDefault();
            const photoStore = document.getElementById('photoStore').value;

            const response = await fetch('/.netlify/functions/upload-photo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: photoStore })
            });

            const result = await response.json();
            console.log(result);
        });
    } else {
        console.error('One or more elements not found.');
    }

    if (detailsButton) {
        detailsButton.classList.add('singular');
    } else {
        console.error('Element with id "detailsButton" not found.');
    }
});
