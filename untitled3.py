import cv2
import mediapipe as mp
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
def detect_landmarks(image):
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(img2, 0)

    left_eye_roi = None
    right_eye_roi = None

    for face in faces:
        landmarks = predictor(img2, face)
        left_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        left_eye_rect = cv2.boundingRect(np.array(left_eye_landmarks))
        right_eye_rect = cv2.boundingRect(np.array(right_eye_landmarks))
        left_eye_roi = image[left_eye_rect[1]:left_eye_rect[1]+left_eye_rect[3], 
                            left_eye_rect[0]:left_eye_rect[0]+left_eye_rect[2]]
        right_eye_roi = image[right_eye_rect[1]:right_eye_rect[1]+right_eye_rect[3], 
                            right_eye_rect[0]:right_eye_rect[0]+right_eye_rect[2]]
        left_eye_roi = cv2.resize(left_eye_roi, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        right_eye_roi = cv2.resize(right_eye_roi, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
        for (x, y) in left_eye_landmarks:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        for (x, y) in right_eye_landmarks:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    return left_eye_roi, right_eye_roi

def resize_and_show(image, window_name='Resized Image'):
    # 定义新的图像尺寸
    new_width = 200  # 替换为你想要的新宽度
    new_height = 200  # 替换为你想要的新高度

    # 调整图像尺寸
    resized_image = cv2.resize(image, (new_width, new_height))

    # 显示调整后的图像
    cv2.imshow(window_name, resized_image)

def capture_photo():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        w = int(img.shape[1] * 0.5)
        h = int(img.shape[0] * 0.5)
        img = cv2.resize(img, (w, h))

        cv2.imshow('camera', img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 32:  # Space key to capture photo
            filename = 'photo.jpg'
            cv2.imwrite(filename, img)
            left_eye_roi, right_eye_roi = detect_landmarks(img)
            print("Photo captured!")
            img_processed=detection_eyes(img, w, h,left_eye_roi, right_eye_roi)  # 在拍照後立即執行眼睛偵測函式
            cv2.imwrite(filename, img_processed)
            return filename

    cap.release()
    cv2.destroyAllWindows()

def detection_eyes(image, w, h,left_eye_roi, right_eye_roi):
 with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:

        img=image.copy()
        #mask = np.zeros_like(image)
        size = image.shape   # 取得攝影機影像尺寸
        w = size[1]        # 取得畫面寬度
        h = size[0]        # 取得畫面高度
        img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img2)

        if results.detections:
            for detection in results.detections:
                #mp_drawing.draw_detection(image, detection)
                s = detection.location_data.relative_bounding_box     # 取得人臉尺寸
                eye = int(s.width*w*0.049)                            # 計算眼睛大小 ( 人臉尺寸*0.1 )
                a = detection.location_data.relative_keypoints[0]     # 取得左眼座標
                b = detection.location_data.relative_keypoints[1]     # 取得右眼座標
                ax, ay = int(a.x*w), int(a.y*h)                       # 計算左眼真正的座標 
                bx, by = int(b.x*w), int(b.y*h)                       # 計算右眼真正的座標
                left_eye=img[ay-eye:ay+eye,ax-eye:ax+eye]
                right_eye=img[by-eye:by+eye,bx-eye:bx+eye]
                #left_eye_mask = np.zeros_like(mask)
                #right_eye_mask = np.zeros_like(mask)
                #left_x, left_y, left_w, left_h = cv2.boundingRect(left_eye_roi)
                #right_x, right_y, right_w, right_h = cv2.boundingRect(right_eye_roi)
                #left_eye_mask[left_y:left_y + left_h, left_x:left_x + left_w] = 255
                #right_eye_mask[right_y:right_y + right_h, right_x:right_x + right_w] = 255
                # 将左眼区域添加到整体遮罩中
                #mask = cv2.bitwise_or(mask, left_eye_mask,right_eye_mask)
                # 将右眼区域添加到整体遮罩中
                #mask = cv2.bitwise_or(mask, right_eye_mask)
                #cv2.circle(mask, (ax,ay), (eye), (255, 255, 255), -1)       # 在遮罩上绘制左眼区域
                #cv2.circle(mask, (bx,by), (eye), (255, 255, 255), -1)       # 在遮罩上绘制右眼区域
                cv2.circle(img,(ax,ay),(eye),(255,255,255),1)       # 畫左眼白色大圓 ( 白眼球 )
                cv2.circle(img,(bx,by),(eye),(255,255,255),1)       # 畫右眼白色大圓 ( 白眼球 )
                enlarged_left_eye=cv2.resize(left_eye,None,fx=5.0,fy=5.0)
                enlarged_right_eye=cv2.resize(right_eye,None,fx=5.0,fy=5.0)

            img[10:10 + left_eye_roi.shape[0], 10:10 + left_eye_roi.shape[1]] = left_eye_roi
            img[10:10 + right_eye_roi.shape[0], w - 10 - right_eye_roi.shape[1]:w - 10] = right_eye_roi

        # 将左眼区域添加到遮罩中
            #mask[left_eye_roi[1]:left_eye_roi[1] + left_eye_roi[3], left_eye_roi[0]:left_eye_roi[0] + left_eye_roi[2]] = 255
        # 将右眼区域添加到遮罩中
            #mask[right_eye_roi[1]:right_eye_roi[1] + right_eye_roi[3], right_eye_roi[0]:right_eye_roi[0] + right_eye_roi[2]] = 255
            #img = cv2.bitwise_and(img, mask)

        print('l-x:',ax,'l-y:',ay,'r-x:',bx,'r-y:',by)
        cv2.imshow('oxxostudio', img)
        resize_and_show(left_eye,'Resized Left Eye')
        resize_and_show(right_eye,"Resized right eye")
        cv2.imshow("left eye",left_eye)
        cv2.imshow("right eye",right_eye)
        cv2.imshow("enlaged left eye",enlarged_left_eye)
        cv2.imshow("enlaged right eye",enlarged_right_eye)
        cv2.waitKey(0)       

capture_photo()
