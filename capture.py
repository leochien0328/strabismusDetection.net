import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image
import os
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Initialize Mediapipe and OpenCV tools
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Eye and iris indices
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]
# capture.py
def capture_function(data):
    # 这里是capture.py中的一些功能实现
    return f"Capture function processed {data}"

# Capture photo route
@app.route('/capture', methods=['GET'])
def capture_photo():
    filename = capture_photos()
    if filename:
        return jsonify({'status': 'success', 'filename': filename})
    return jsonify({'status': 'failed'})

# Analyze image route
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'status': 'failed', 'message': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'failed', 'message': 'No selected file'})
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        result = detect_face_landmarks(filepath)
        return jsonify({'status': 'success', 'result': result})
    return jsonify({'status': 'failed'})

# Home route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Function to capture photos from the webcam
def capture_photos():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None

    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        w = int(img.shape[1] * 0.5)
        h = int(img.shape[0] * 0.5)
        img = cv2.resize(img, (w, h))

        cv2.imshow('camera', img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 32:
            filename = 'photo.jpg'
            cv2.imwrite(filename, img)
            print("Photo captured!")
            cap.release()
            return filename

    cap.release()
    cv2.destroyAllWindows()

# Function to calculate eye intersection coordinates
def eye_coordinates(mesh_points):
    left_eye_pts = np.array([mesh_points[i] for i in LEFT_EYE])
    right_eye_pts = np.array([mesh_points[i] for i in RIGHT_EYE])

    try:
        left_k1 = (left_eye_pts[1][1] - left_eye_pts[0][1]) / (left_eye_pts[1][0] - left_eye_pts[0][0])
        left_b1 = left_eye_pts[1][1] - left_k1 * left_eye_pts[1][0]
        left_k2 = (left_eye_pts[5][1] - left_eye_pts[4][1]) / (left_eye_pts[5][0] - left_eye_pts[4][0])
        left_b2 = left_eye_pts[5][1] - left_k2 * left_eye_pts[5][0]
        if left_k1!= left_k2:
            left_x_intersection = (left_b2 - left_b1) / (left_k1 - left_k2)
            left_y_intersection = left_k1 * left_x_intersection + left_b1
        else:
            left_x_intersection = None
            left_y_intersection = None
    except ZeroDivisionError:
        left_x_intersection = None
        left_y_intersection = None        

    try:
        right_k1 = (right_eye_pts[1][1] - right_eye_pts[0][1]) / (right_eye_pts[1][0] - right_eye_pts[0][0])
        right_b1 = right_eye_pts[1][1] - right_k1 * right_eye_pts[1][0]
        right_k2 = (right_eye_pts[5][1] - right_eye_pts[4][1]) / (right_eye_pts[5][0] - right_eye_pts[4][0])
        right_b2 = right_eye_pts[5][1] - right_k2 * right_eye_pts[5][0]
        if right_k1 != right_k2:
            right_x_intersection = (right_b2 - right_b1) / (right_k1 - right_k2)
            right_y_intersection = right_k1 * right_x_intersection + right_b1
        else:
            right_x_intersection = None
            right_y_intersection = None
    except ZeroDivisionError:
        right_x_intersection = None
        right_y_intersection = None

    return [left_x_intersection, left_y_intersection, right_x_intersection, right_y_intersection]

# Function to detect iris and draw on image
def detect_iris(image):
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(image, center_left, int(l_radius), (255, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(image, center_right, int(r_radius), (255, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(image, (int(l_cx), int(l_cy)), 2, (0, 0, 255), -1)
            cv2.circle(image, (int(r_cx), int(r_cy)), 2, (0, 0, 255), -1)
            return [l_cx, l_cy, r_cx, r_cy]

        return None

# Function to calculate distance vectors
def distance_vector(iris, coordinates, image):
    left_distance = math.sqrt((iris[0] - coordinates[0]) ** 2 + (iris[1] - coordinates[1]) ** 2)
    right_distance = math.sqrt((iris[2] - coordinates[2]) ** 2 + (iris[3] - coordinates[3]) ** 2)
    
    distances = [left_distance, right_distance]
    main(distances, image)
    return distances

# Function to detect face landmarks and analyze iris positions
def detect_face_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Cannot open the image file")
        return
    

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            for (x, y) in mesh_points:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            coordinates = eye_coordinates(mesh_points)

            iris_coords = detect_iris(image)
            if iris_coords:
                distance_vector(iris_coords, coordinates, image)

    cv2.imshow('Face Landmarks Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to analyze distances and display results
def main(distances, image):
    left_distance, right_distance = distances
    print("Left Distance:", left_distance)
    print("Right Distance:", right_distance)
    solution = abs(left_distance - right_distance)

    fontpath = 'C:\\Windows\\Fonts\\MSJH.TTC'
    font_size = 30  # Set font size here

    if not os.path.exists(fontpath):
        print("Font not found. Using default font.")
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(fontpath, font_size)  # Set font and size

    img_pil = Image.fromarray(image)  # Convert image to PIL format
    draw = ImageDraw.Draw(img_pil)  # Prepare to draw

    if solution > 3:
        print("疑似斜視")
        draw.text((0, 0), "疑似斜視", fill=(0, 0, 255), font=font)  # Draw text
    else:
        print("診斷正常")
        draw.text((0, 0), "診斷正常", fill=(0, 0, 255), font=font)  # Draw text

    img = np.array(img_pil)  # Convert back to numpy array

    size = 320
    height, width = img.shape[0], img.shape[1]
    scale = height/size
    width_size = int(width/scale)
    img=cv2.resize(img,(width_size, size))   

    cv2.imshow('Diagnosis', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
