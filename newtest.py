import os
import cv2
import numpy as np
import mediapipe as mp
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_landmarks(image):
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(img2)

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

def capture_photo():
    save_folder = 'photos'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    i = 1
    while os.path.exists(os.path.join(save_folder, f'photo{i}.jpg')):
        i += 1
    filename = os.path.join(save_folder, f'photo{i}.jpg')

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
            cv2.imwrite(filename, img)
            print(f"Photo captured and saved as {filename}!")
            left_eye_roi, right_eye_roi = detect_landmarks(img)
            img_processed = detection_eyes(img, w, h, left_eye_roi, right_eye_roi)
            cv2.imwrite(filename, img_processed)  # Save the processed image
            return filename

    cap.release()
    cv2.destroyAllWindows()

def detection_eyes(image, w, h, left_eye_roi, right_eye_roi):
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        
        results = face_detection.process(image)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                s = detection.location_data.relative_bounding_box     
                eye = int(s.width*w*0.049)                            
                a = detection.location_data.relative_keypoints[0]     
                b = detection.location_data.relative_keypoints[1]     
                ax, ay = int(a.x*w), int(a.y*h)                       
                bx, by = int(b.x*w), int(b.y*h)                       
                
                cv2.circle(image, (ax, ay), eye, (255, 255, 255), 1)  
                cv2.circle(image, (bx, by), eye, (255, 255, 255), 1)  
                
    print(a)
    print(b)
    print(eye)   
    cv2.imshow('photo',image)    
    return image

capture_photo()



