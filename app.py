from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import math
import base64
from io import BytesIO
from PIL import Image
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
@app.route('/api/upload-photo', methods=['POST'])
def upload_photo():
    if request.method == 'POST':
        try:
            if request.is_json:
                image_data = request.json.get('image')
                image = Image.open(BytesIO(base64.b64decode(image_data)))
                image = np.array(image)
                solution, dist_L, dist_R = detect_iris(image)
                result = 1 if solution > 3 else 0
                return jsonify({"result": result, "solution": solution, "dist_L": dist_L, "dist_R": dist_R}), 200
            else:
                return jsonify({"error": "Unsupported Media Type: Did not attempt to load JSON data because the request Content-Type was not 'application/json'."}), 415
        except Exception as e:
            return jsonify({"error": str(e)}), 500
@app.route('/', methods=['GET'])
def get_hello():
    return 'Hello World!'
if __name__ == "__main__":
    app.run(debug=True,port=10000)
