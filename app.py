from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import math

app = Flask(__name__)
CORS(app)

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]

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
            return [l_cx, l_cy, r_cx, r_cy], mesh_points

        return None, None

def eye_coordinates(mesh_points):
    left_eye_pts = np.array([mesh_points[i] for i in LEFT_EYE])
    right_eye_pts = np.array([mesh_points[i] for i in RIGHT_EYE])
    try:
        left_k1 = (left_eye_pts[1][1] - left_eye_pts[0][1]) / (left_eye_pts[1][0] - left_eye_pts[0][0])
        left_b1 = left_eye_pts[1][1] - left_k1 * left_eye_pts[1][0]
        left_k2 = (left_eye_pts[5][1] - left_eye_pts[4][1]) / (left_eye_pts[5][0] - left_eye_pts[4][0])
        left_b2 = left_eye_pts[5][1] - left_k2 * left_eye_pts[5][0]
        if left_k1 != left_k2:
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

@app.route('/')
def home():
    return "Strabismus Detection API"

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    iris_coords, mesh_points = detect_iris(image)
    if not iris_coords:
        return jsonify({"error": "No face detected"}), 400

    eye_coords = eye_coordinates(mesh_points)
    left_distance = math.sqrt((iris_coords[0] - eye_coords[0]) ** 2 + (iris_coords[1] - eye_coords[1]) ** 2)
    right_distance = math.sqrt((iris_coords[2] - eye_coords[2]) ** 2 + (iris_coords[3] - eye_coords[3]) ** 2)
    solution = abs(left_distance - right_distance)

    return jsonify({
        "solution": solution
    })

if __name__ == "__main__":
    app.run(debug=True)
