from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import math

app = Flask(__name__)
CORS(app)

# Initialize Mediapipe and OpenCV tools
mp_face_mesh = mp.solutions.face_mesh

# Eye and iris indices
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [469, 470, 471, 472, 468]
RIGHT_IRIS = [474, 475, 476, 477, 473]
NOSE_CENTER = [168, 6, 197]

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

def calculate_angle_and_direction(vector):
    """Calculate the angle of the vector and determine its direction."""
    angle = np.arctan2(vector[1], vector[0])
    degrees = np.degrees(angle)

    if vector[0] < 0:
        degrees += 180  # Adjust for second and third quadrants
    if vector[0] > 0 and vector[1] < 0:
        degrees = - degrees
    if vector[0] < 0 and vector[1] > 0:
        degrees = 360 - degrees          
    return degrees

def distance_vector(mesh_points):
    left_eye_pts = np.array([mesh_points[i] for i in LEFT_EYE])
    right_eye_pts = np.array([mesh_points[i] for i in RIGHT_EYE])
    nose_pns = np.array([mesh_points[i] for i in NOSE_CENTER])
    left_iris_pts = np.array([mesh_points[i] for i in LEFT_IRIS])
    right_iris_pts = np.array([mesh_points[i] for i in RIGHT_IRIS])

    origin = np.mean(nose_pns, axis=0)  # Calculate nose center
    lu, ld, lc = left_iris_pts[1], left_iris_pts[3], left_iris_pts[4]
    ru, rd, rc = right_iris_pts[1], right_iris_pts[3], right_iris_pts[4]

    left_iris_vector = lc - origin
    right_iris_vector = rc - origin
    left_iris_up_vector = lu - origin
    left_iris_down_vector = ld - origin
    right_iris_up_vector = ru - origin
    right_iris_down_vector = rd - origin
   
    

    # Calculate angles
    left_angle_degrees = calculate_angle_and_direction(left_iris_vector)
    left_angle_degrees_up = calculate_angle_and_direction(left_iris_up_vector)
    left_angle_degrees_down = calculate_angle_and_direction(left_iris_down_vector)
    right_angle_degrees = calculate_angle_and_direction(right_iris_vector)
    right_angle_degrees_up = calculate_angle_and_direction(right_iris_up_vector)
    right_angle_degrees_down = calculate_angle_and_direction(right_iris_down_vector) 

    left_vecs = calculate_vectors(lc[:2], left_eye_pts)
    right_vecs = calculate_vectors(rc[:2], right_eye_pts)

    left_distances = [np.linalg.norm(vec) for vec in left_vecs]
    right_distances = [np.linalg.norm(vec) for vec in right_vecs]

    coordinates = eye_coordinates(mesh_points)
    solution = distance_vector_from_coordinates(lc, rc, coordinates)

    return (left_distances, right_distances, left_angle_degrees, right_angle_degrees, left_angle_degrees_up, 
            left_angle_degrees_down, right_angle_degrees_up, right_angle_degrees_down, solution)

def calculate_vectors(iris_center, eye_pts):
    """Calculate vectors from iris to eye boundary points"""
    return [
        (iris_center[0] - eye_pts[8][0], iris_center[1] - eye_pts[8][1]),
        (iris_center[0] - eye_pts[5][0], iris_center[1] - eye_pts[5][1]),
        (iris_center[0] - eye_pts[0][0], iris_center[1] - eye_pts[0][1]),
        (iris_center[0] - eye_pts[12][0], iris_center[1] - eye_pts[12][1])
    ]

def distance_vector_from_coordinates(lc, rc, coordinates):
    left_distance = math.sqrt((lc[0] - coordinates[0]) ** 2 + (lc[1] - coordinates[1]) ** 2)
    right_distance = math.sqrt((rc[0] - coordinates[2]) ** 2 + (rc[1] - coordinates[3]) ** 2)

    solution = abs(left_distance - right_distance)
    return solution

def detect_face_landmarks(image):
    """Detect face landmarks and analyze iris positions."""
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1
    ) as face_mesh:
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            for (x, y) in mesh_points:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            
            return distance_vector(mesh_points)
         
        return None

def detect_strabismus(left_distances, right_distances, left_angle_degrees, right_angle_degrees, 
                      left_angle_degrees_up, left_angle_degrees_down, right_angle_degrees_up, right_angle_degrees_down, solution):
    """Determine the result of strabismus detection based on calculated distances and angles."""
    result = 0

    if(solution > 2.5):
        if ((left_distances[2] < left_distances[0] and left_distances[2] < right_distances[2]) or
            (right_distances[0] < right_distances[2] and right_distances[0] < left_distances[0])):
            if (left_angle_degrees >= 10 and right_angle_degrees >= 10):
                result = 1
        if ((left_distances[0] < left_distances[2] and left_distances[0] < right_distances[0]) or
            (right_distances[2] < right_distances[0] and right_distances[2] < left_distances[2])):
            if (left_angle_degrees >= 10 and right_angle_degrees >= 10):
                result = 2
        if ((left_angle_degrees_up > right_angle_degrees and left_angle_degrees > right_angle_degrees) or 
            (right_angle_degrees_up > left_angle_degrees and right_angle_degrees > left_angle_degrees)):
            if ((left_distances[1] < right_distances[1] and left_distances[3] > right_distances[3]) or
                (right_distances[1] < left_distances[1] and right_distances[3] > left_distances[3])):
                result = 3
        if ((left_angle_degrees_down < right_angle_degrees and left_angle_degrees < right_angle_degrees) or 
            (right_angle_degrees_down < left_angle_degrees and right_angle_degrees < left_angle_degrees)):
            if ((left_distances[3] < right_distances[3] and left_distances[1] > right_distances[1]) or
                (right_distances[3] < left_distances[3] and right_distances[1] > left_distances[1])):
                result = 4

    return result

@app.route('/api/upload-photo', methods=['POST'])
def upload_photo():
    if not request.is_json:
        return jsonify({"error": "Unsupported Media Type"}), 415

    try:
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image = np.array(image)
        h, w = image.shape[:2]
        new_w = 640
        new_h = int((new_w / w) * h)
        image = cv2.resize(image, (new_w, new_h))
        image = cv2.flip(image, 1)
        landmarks_result = detect_face_landmarks(image)

        if landmarks_result is None:
            return jsonify({"error": "No face landmarks detected"}), 400

        left_distances, right_distances, left_angle_degrees, right_angle_degrees, left_angle_degrees_up, left_angle_degrees_down, right_angle_degrees_up, right_angle_degrees_down, solution = landmarks_result

        result = detect_strabismus(left_distances, right_distances, 
                                   left_angle_degrees, right_angle_degrees, 
                                   left_angle_degrees_up, left_angle_degrees_down, 
                                   right_angle_degrees_up, right_angle_degrees_down, solution)

        return jsonify({"result": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def get_hello():
    return 'Hello World!'

if __name__ == "__main__":
    app.run(debug=True, port=10000)