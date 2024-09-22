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
        let videoStream = null;
        let model = null;

        // 設定視頻元素
        const video = document.getElementById('webcamVideo');

        // 獲取視頻流
        accessCameraButton.addEventListener('click', async function() {
            try {
                // 獲取用戶的攝像頭視頻流
                videoStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 }, 
                        height: { ideal: 480 }, 
                        aspectRatio: 4/3,
                        facingMode: 'user', 
                        frameRate: { ideal: 30 }, 
                    },
                    audio:false
                });
                video.srcObject = videoStream;
                video.play();

                // 顯示拍照按鈕
                takePhotoButton.classList.remove('d-none');
                takePhotoButton.classList.add('d-block');

                // 加載 facemesh 模型
                await loadModel();
            } catch (error) {
                console.error('Error accessing webcam:', error);
                swal({
                    title: 'Warning',
                    text: '請允許訪問您的攝像頭。',
                    icon: 'warning'
                });
            }
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

        async function loadModel() {
            try {
                model = await facemesh.load();
                detectFace();
            } catch (error) {
                console.error('Error loading facemesh model:', error);
                swal({
                    title: 'Error',
                    text: '無法加載面部檢測模型。',
                    icon: 'error'
                });
            }
        }
    
        async function detectFace() {
            if (!model) {
                console.error('Facemesh model not loaded.');
                return;
            }

            async function update() {
                if (video.readyState >= 2) { // HAVE_CURRENT_DATA
                    try {
                        const predictions = await model.estimateFaces(video);
                        
                        if (predictions.length > 0) {
                            const keypoints = predictions[0].scaledMesh;
                            const leftEye = keypoints[33];
                            const rightEye = keypoints[263];
                            const eyedistance = Math.sqrt(
                                Math.pow(leftEye[0] - rightEye[0], 2) +
                                Math.pow(leftEye[1] - rightEye[1], 2) +
                                Math.pow(leftEye[2] - rightEye[2], 2)
                            );
                            const actualEyeDistance = 6.3; // 厘米
                            const focalLength = 600;
                            const distanceToCamera = (actualEyeDistance * focalLength) / eyedistance;

                            document.getElementById('distanceInfo').innerText = '距離相機: ' + distanceToCamera.toFixed(2)+'cm';
                        } else {
                            document.getElementById('distanceInfo').innerText = '未檢測到面部';
                        }
                    } catch (error) {
                        console.error('Error during face detection:', error);
                        document.getElementById('distanceInfo').innerText = '檢測出錯';
                    }
                }
                requestAnimationFrame(update);
            }

            update();
        }
        
        async function upload_photo(raw_image_data) {
            try {
                const response = await fetch('https://strabismusdecation.com/.netlify/functions/upload-photo', {
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
                    $('#hypertropiaStrabismusModal').modal('show');
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
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const data_uri = canvas.toDataURL('image/jpeg');
            document.getElementById('results').innerHTML = '<img src="' + data_uri + '" class="d-block mx-auto rounded"/>';
            var raw_image_data = data_uri.replace(/^data:image\/\w+;base64,/, '');
            while (raw_image_data.length % 4 !== 0) {
                raw_image_data += '=';
            }
            document.getElementById('photoStore').value = raw_image_data;

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
    } else {
        console.error('One or more elements not found.');
    }

    if (detailsButton) {
        detailsButton.classList.add('singular');
        detailsButton.addEventListener('click', function() {
            swal({
                title: 'Details',
                text: '這裡可以顯示更多詳細信息。',
                icon: 'info'
            });
        });
    } else {
        console.error('Element with id "detailsButton" not found.');
    }
});
