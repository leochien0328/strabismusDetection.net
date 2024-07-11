// 初始化摄像头
const webcamElement = document.createElement('video');
const cameraBtn = document.getElementById('camera');
const webcam = new WebcamEasy({ videoElement: webcamElement, enablePause: false });

// 相机按钮点击事件
cameraBtn.addEventListener('click', async () => {
    try {
        // 打开摄像头
        await webcam.start();
        
        // 设置视频大小
        webcamElement.width = 300;
        webcamElement.height = 200;

        // 拍照
        const dataUrl = await webcam.snap();

        // 显示拍摄的照片
        const img = document.createElement('img');
        img.src = dataUrl;
        document.body.appendChild(img);

        // 停止摄像头
        webcam.stop();
    } catch (err) {
        console.error('Error accessing the camera:', err);
    }
});

