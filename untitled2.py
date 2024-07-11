import cv2

# 加载人脸和眼睛级联分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# 初始化摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头图像
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 遍历检测到的人脸
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # 检测眼睛
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # 遍历检测到的眼睛
        for (ex, ey, ew, eh) in eyes:
            eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]

            # 使用阈值化处理找到瞳孔
            _, eye_thresh = cv2.threshold(eye_roi_gray, 50, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(eye_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # 寻找最大的轮廓作为瞳孔
            if len(contours) > 0:
                pupil = max(contours, key=cv2.contourArea)
                x1, y1, w1, h1 = cv2.boundingRect(pupil)
                center = (int(x1 + w1/2), int(y1 + h1/2))

                # 在眼睛区域绘制瞳孔中心
                cv2.circle(eye_roi_color, center, 3, (255, 0, 0), -1)

        # 在眼睛周围绘制矩形
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Eye Tracking', frame)

    # 检测按键事件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
