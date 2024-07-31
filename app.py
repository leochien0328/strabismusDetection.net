from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image

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

def distance_vector(mesh_points):
    left_eye_pts = np.array([mesh_points[i] for i in LEFT_EYE])
    right_eye_pts = np.array([mesh_points[i] for i in RIGHT_EYE])
    nose_pns = np.array([mesh_points[i] for i in NOSE_CENTER])
    left_iris_pts = np.array([mesh_points[i] for i in LEFT_IRIS])
    right_iris_pts = np.array([mesh_points[i] for i in RIGHT_IRIS])
    origin = (nose_pns[1] + nose_pns[0] + nose_pns[2]) / 3  # Calculate nose center
    lu, ld, lc = left_iris_pts[3], left_iris_pts[1], left_iris_pts[4]
    ru, rd, rc = right_iris_pts[3], right_iris_pts[1], right_iris_pts[4]

    left_iris_up_vector = np.array(lu) - np.array(origin)
    left_iris_down_vector = np.array(ld) - np.array(origin)
    right_iris_up_vector = np.array(ru) - np.array(origin)
    right_iris_down_vector = np.array(rd) - np.array(origin)

    #left
    left_angle_vector = np.array(lc) - np.array(origin)
    left_angle = np.arctan2(left_angle_vector[1], left_angle_vector[0])
    if left_angle_vector[0] < 0 and left_angle_vector[1] >= 0:  # Second quadrant
        left_angle_degrees = 180 - np.degrees(left_angle)
    elif left_angle_vector[0] < 0 and left_angle_vector[1] < 0:  # Third quadrant
        left_angle_degrees = 180 +np.degrees(left_angle)
    # Handle other cases    
    else: 
        left_angle_degrees = 180 - np.degrees(left_angle)

    #left up
    left_angle_up = np.arctan2(left_iris_up_vector[1], left_iris_up_vector[0])
    if left_iris_up_vector[0] < 0 and left_iris_up_vector[1] >= 0:  # Second quadrant
        left_angle_degrees_up = 180 - np.degrees(left_angle_up)
    elif left_iris_up_vector[0] < 0 and left_iris_up_vector[1] < 0:  # Third quadrant
        left_angle_degrees_up = 180 + np.degrees(left_angle_up)
    # Handle other cases    
    else: 
        left_angle_degrees_up =180 - np.degrees(left_angle_up)

    #left down
    left_angle_down = np.arctan2(left_iris_down_vector[1], left_iris_down_vector[0])    
    if left_iris_down_vector[0] < 0 and left_iris_down_vector[1] >= 0:  # Second quadrant
        left_angle_degrees_down = 180 - np.degrees(left_angle_down)
    elif left_iris_down_vector[0] < 0 and left_iris_down_vector[1] < 0:  # Third quadrant
        left_angle_degrees_down = 180 + np.degrees(left_angle_down)
    # Handle other cases    
    else: 
        left_angle_degrees_down =180 - np.degrees(left_angle_down)    

    #right
    right_angle_vector = np.array(rc) - np.array(origin)
    right_angle = np.arctan2(right_angle_vector[1], right_angle_vector[0])
    if right_angle_vector[0] >= 0 and right_angle_vector[1] >= 0:  # First quadrant
        right_angle_degrees = np.degrees(right_angle)
    elif right_angle_vector[0] < 0 and right_angle_vector[1] >= 0:  # Fourth quadrant
        right_angle_degrees = 360 + np.degrees(right_angle)
    # Handle other cases    
    else:  
        right_angle_degrees = -np.degrees(right_angle)

    #right up    
    right_angle_up = np.arctan2(right_iris_up_vector[1], right_iris_up_vector[0])
    if right_iris_up_vector[0] >= 0 and right_iris_up_vector[1] >= 0:  # First quadrant
        right_angle_degrees_up = np.degrees(right_angle_up)
    elif right_iris_up_vector[0] < 0 and right_iris_up_vector[1] >= 0:  # Fourth quadrant
        right_angle_degrees_up = 360 + np.degrees(right_angle_up)
    # Handle other cases    
    else:  
        right_angle_degrees_up = -np.degrees(right_angle_up)

    #right down
    right_angle_down = np.arctan2(right_iris_down_vector[1], right_iris_down_vector[0])
    if right_iris_down_vector[0] >= 0 and right_iris_down_vector[1] >= 0:  # First quadrant
        right_angle_degrees_down = np.degrees(right_angle_down)
    elif right_iris_down_vector[0] < 0 and right_iris_down_vector[1] >= 0:  # Fourth quadrant
        right_angle_degrees_down = 360 + np.degrees(right_angle_down)
    # Handle other cases    
    else:  
        right_angle_degrees_down = -np.degrees(right_angle_down)    

    left_vecs = calculate_vectors(lc[:2], left_eye_pts)
    right_vecs = calculate_vectors(rc[:2], right_eye_pts)

    left_distances = [np.linalg.norm(vec) for vec in left_vecs]
    right_distances = [np.linalg.norm(vec) for vec in right_vecs]

    return left_distances, right_distances, left_angle_degrees, right_angle_degrees, left_angle_degrees_up, left_angle_degrees_down, right_angle_degrees_up, right_angle_degrees_down

def calculate_vectors(iris_center, eye_pts):
    """Calculate vectors from iris to eye boundary points"""
    return [
        (iris_center[0] - eye_pts[8][0], iris_center[1] - eye_pts[8][1]),
        (iris_center[0] - eye_pts[12][0], iris_center[1] - eye_pts[12][1]),
        (iris_center[0] - eye_pts[0][0], iris_center[1] - eye_pts[0][1]),
        (iris_center[0] - eye_pts[5][0], iris_center[1] - eye_pts[5][1])
    ]

def detect_face_landmarks(image):
    """Detect face landmarks and analyze iris positions."""
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as face_mesh:
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            for (x, y) in mesh_points:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            
            left_distances, right_distances, left_angle_degrees, right_angle_degrees, left_angle_degrees_up, left_angle_degrees_down, right_angle_degrees_up, right_angle_degrees_down = distance_vector(mesh_points)
            return diagnose(left_distances, right_distances, left_angle_degrees, right_angle_degrees, left_angle_degrees_up, left_angle_degrees_down, right_angle_degrees_up, right_angle_degrees_down)
         
        return None
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
        result = detect_face_landmarks(image)
        if result is None:
            return jsonify({"error": "No face landmarks detected"}), 400

        return jsonify({"result": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def diagnose(left_distances, right_distances, left_angle_degrees, right_angle_degrees, left_angle_degrees_up, left_angle_degrees_down, right_angle_degrees_up, right_angle_degrees_down):   
    
    if ((left_angle_degrees_down > right_angle_degrees and left_angle_degrees_up < right_angle_degrees) or 
        (right_angle_degrees_down > left_angle_degrees and right_angle_degrees_up < left_angle_degrees)):
        if ((left_distances[3] < right_distances[3] and left_distances[1] > right_distances[1]) or 
            (right_distances[3] < left_distances[3] and right_distances[1] > left_distances[1])):
            return 4
            
    elif ((left_angle_degrees_up > right_angle_degrees and left_angle_degrees_down < right_angle_degrees) or 
          (right_angle_degrees_up > left_angle_degrees and right_angle_degrees_down < left_angle_degrees)):
        if ((left_distances[1] < right_distances[1] and left_distances[3] > right_distances[3]) or 
            (right_distances[1] < left_distances[1] and right_distances[3] > left_distances[3])):
            return 3
        
    elif ((left_distances[0] < left_distances[2] and left_distances[0] < right_distances[0]) or 
        (right_distances[2] < right_distances[0] and right_distances[2] < left_distances[2])):
        if (left_angle_degrees >= 10 or right_angle_degrees >= 10):   
            return 2  
              
    elif ((left_distances[2] < left_distances[0] and left_distances[2] < right_distances[2]) or 
        (right_distances[0] < right_distances[2] and right_distances[0] < left_distances[0])):
        if (left_angle_degrees >= 10 or right_angle_degrees >= 10):       
            return 1
    else:
        return 0

@app.route('/', methods=['GET'])
def get_hello():
    return 'Hello World!'
if __name__ == "__main__":
    app.run(debug=True,port=10000)
