from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image
import os

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
# readfile.py
def readfile_function(data):
    # 这里是readfile.py中的一些功能实现
    return f"Readfile function processed {data}"

# Calculate eye intersection coordinates
def eye_coordinates(mesh_points):
    left_eye_pts = np.array([mesh_points[i] for i in LEFT_EYE])
    right_eye_pts = np.array([mesh_points[i] for i in RIGHT_EYE])

    # Calculate slopes and intercepts for eye intersections
    left_k = (left_eye_pts[8][1] - left_eye_pts[0][1]) / (left_eye_pts[8][0] - left_eye_pts[0][0])
    left_b = left_eye_pts[8][1] - left_k * left_eye_pts[8][0]
    left_k1 = (left_eye_pts[12][1] - left_eye_pts[4][1]) / (left_eye_pts[12][0] - left_eye_pts[4][0])
    left_b1 = left_eye_pts[12][1] - left_k1 * left_eye_pts[12][0]

    right_k = (right_eye_pts[0][1] - right_eye_pts[8][1]) / (right_eye_pts[8][0] - right_eye_pts[0][0])
    right_b = right_eye_pts[8][1] - right_k * right_eye_pts[8][0]
    right_k1 = (right_eye_pts[12][1] - right_eye_pts[4][1]) / (right_eye_pts[12][0] - right_eye_pts[4][0])
    right_b1 = right_eye_pts[12][1] - right_k1 * right_eye_pts[12][0]

    left_x_intersection = (left_b1 - left_b) / (left_k - left_k1)
    left_y_intersection = left_k * left_x_intersection + left_b

    right_x_intersection = (right_b1 - right_b) / (right_k - right_k1)
    right_y_intersection = right_k * right_x_intersection + right_b

    return [left_x_intersection, left_y_intersection, right_x_intersection, right_y_intersection]

# Detect iris and draw on image
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

# Calculate distance vectors
def distance_vector(iris, coordinates, image):
    left_distance = math.sqrt((iris[0] - coordinates[0]) ** 2 + (iris[1] - coordinates[1]) ** 2)
    right_distance = math.sqrt((iris[2] - coordinates[2]) ** 2 + (iris[3] - coordinates[3]) ** 2)
    
    distances = [left_distance, right_distance]
    return distances

# Detect face landmarks and analyze iris positions
def detect_face_landmarks(image):
    img_h, img_w = image.shape[:2]

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            for (x, y) in mesh_points:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            coordinates = eye_coordinates(mesh_points)

            iris_coords = detect_iris(image)
            if iris_coords:
                distances = distance_vector(iris_coords, coordinates, image)
                return distances, image
    return None, image

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
        draw.text((0, 0), "診斷正常", fill=(0, 0, 0), font=font)  # Draw text

    img = np.array(img_pil)  # Convert back to numpy array

    size = 320
    height, width = img.shape[0], img.shape[1]
    scale = height/size
    width_size = int(width/scale)
    img = cv2.resize(img, (width_size, size))   

    return img

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        distances, annotated_image = detect_face_landmarks(image)
        if distances:
            result_image = main(distances, annotated_image)
            _, img_encoded = cv2.imencode('.png', result_image)
            response = img_encoded.tobytes()
            return response, 200, {'Content-Type': 'image/png'}

        return jsonify({'error': 'No face landmarks detected'}), 400

if __name__ == '__main__':
    app.run(debug=True)
