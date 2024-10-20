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

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157,
            158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
             388, 387, 386, 385, 384, 398]
LEFT_IRIS = [469, 470, 471, 472, 468]
RIGHT_IRIS = [474, 475, 476, 477, 473]
NOSE_CENTER = [168, 6, 197]

def extract_iris_data(iris_indices, binary_image, mesh_points, eye_x, eye_y):
    """Extract grayscale values and coordinates within the iris region."""
    iris_points = [(mesh_points[i][0] - eye_x, mesh_points[i][1] - eye_y) for i in iris_indices]

    x_min = min(iris_points, key=lambda p: p[0])[0]
    x_max = max(iris_points, key=lambda p: p[0])[0]
    y_min = min(iris_points, key=lambda p: p[1])[1]
    y_max = max(iris_points, key=lambda p: p[1])[1]

    iris_region = []

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if cv2.pointPolygonTest(np.array(iris_points), (x, y), False) >= 0:
                if 0 <= y < binary_image.shape[0] and 0 <= x < binary_image.shape[1]:
                    grayscale_value = binary_image[y, x]
                    iris_region.append((x + eye_x, y + eye_y, grayscale_value))

    return iris_region

def find_and_average_iris_white_pixels(iris_data, iris_indices, mesh_points, eye_x, eye_y):
    """Find coordinates with grayscale value 255, excluding iris landmark points, and calculate their average."""
    exclude_points = [(mesh_points[i][0], mesh_points[i][1]) for i in iris_indices]

    white_pixels = [
        (x, y) for x, y, grayscale in iris_data
        if grayscale == 255 and (x, y) not in exclude_points
    ]

    if not white_pixels:
        return "無反光點", None, None  # Return this message when no white pixels are found

    avg_x = sum(x for x, _ in white_pixels) / len(white_pixels)
    avg_y = sum(y for _, y in white_pixels) / len(white_pixels)

    return "反光點", avg_x, avg_y

def iris_catch(left_iris, right_iris):
    left_eye_gray = cv2.cvtColor(left_iris, cv2.COLOR_BGR2GRAY)
    right_eye_gray = cv2.cvtColor(right_iris, cv2.COLOR_BGR2GRAY)

    _, left_eye_binary = cv2.threshold(
        left_eye_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, right_eye_binary = cv2.threshold(
        right_eye_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return left_eye_binary, right_eye_binary

def left_horizontal_diagnose(La, Lb, Lc, Ld, Le, left_flash):
    left_dia_h = [0, 0]
    if left_flash is None:
        return left_dia_h
    if Lc[0] == left_flash[0]:
        left_dia_h = [0, 0]
    elif Lc[0] > left_flash[0]:  # eso
        if Le[0] < left_flash[0] and left_flash[0] < Ld[0]:
            left_dia_h = [2, abs(left_flash[0] - Lc[0]) * 15 /
                          (Ld[0] - Le[0]) + 15]
        elif Ld[0] <= left_flash[0] and left_flash[0] <= Lc[0]:
            left_dia_h = [2, abs(left_flash[0] - Lc[0]) * 15 /
                          (Lc[0] - Ld[0])]
    elif Lc[0] < left_flash[0]:  # exo
        if La[0] > left_flash[0] and left_flash[0] > Lb[0]:
            left_dia_h = [1, abs(left_flash[0] - Lc[0]) * 15 /
                          (La[0] - Lb[0]) + 15]
        elif La[0] >= left_flash[0] and left_flash[0] >= Lc[0]:
            left_dia_h = [1, abs(left_flash[0] - Lc[0]) * 15 /
                          (La[0] - Lb[0])]
    return left_dia_h

def left_vertical_diagnose(Lau, Lbu, Ldd, Led, Lc, left_flash):
    left_dia_v = [0, 0]
    if left_flash is None:
        return left_dia_v
    if Lc[1] == left_flash[1]:
        left_dia_v = [0, 0]
    elif Lc[1] > left_flash[1]:  # hypo
        if Lau[1] < left_flash[1] and left_flash[1] < Lbu[1]:
            left_dia_v = [3, abs(left_flash[1] - Lc[1]) * 15 /
                          (Lbu[1] - Lau[1]) + 15]
        elif Lbu[1] <= left_flash[1] and left_flash[1] <= Lc[1]:
            left_dia_v = [3, abs(left_flash[1] - Lc[1]) * 15 /
                          (Lc[1] - Lbu[1])]
    elif Lc[1] < left_flash[1]:  # hyper
        if Ldd[1] < left_flash[1] and left_flash[1] < Led[1]:
            left_dia_v = [4, abs(left_flash[1] - Lc[1]) * 15 /
                          (Led[1] - Ldd[1]) + 15]
        elif Lc[1] <= left_flash[1] and left_flash[1] <= Ldd[1]:
            left_dia_v = [4, abs(left_flash[1] - Lc[1]) * 15 /
                          (Ldd[1] - Lc[1])]
    return left_dia_v

def right_horizontal_diagnose(Ra, Rb, Rc, Rd, Re, right_flash):
    right_dia_h = [0, 0]
    if right_flash is None:
        return right_dia_h
    if Rc[0] == right_flash[0]:
        right_dia_h = [0, 0]
    elif Rc[0] < right_flash[0]:  # eso
        if Rd[0] < right_flash[0] and right_flash[0] < Re[0]:
            right_dia_h = [2, abs(right_flash[0] - Rc[0]) * 15 /
                           (Re[0] - Rd[0]) + 15]
        elif Rc[0] <= right_flash[0] and right_flash[0] <= Rd[0]:
            right_dia_h = [2, abs(right_flash[0] - Rc[0]) * 15 /
                           (Rd[0] - Rc[0])]
    elif Rc[0] > right_flash[0]:  # exo
        if Ra[0] < right_flash[0] and right_flash[0] < Rb[0]:
            right_dia_h = [1, abs(right_flash[0] - Rc[0]) * 15 /
                           (Rb[0] - Ra[0]) + 15]
        elif Rb[0] <= right_flash[0] and right_flash[0] <= Rc[0]:
            right_dia_h = [1, abs(right_flash[0] - Rc[0]) * 15 /
                           (Rc[0] - Rb[0])]
    return right_dia_h

def right_vertical_diagnose(Rau, Rbu, Rdd, Red, Rc, right_flash):
    right_dia_v = [0, 0]
    if right_flash is None:
        return right_dia_v
    if Rc[1] == right_flash[1]:
        right_dia_v = [0, 0]
    elif Rc[1] > right_flash[1]:  # hypo
        if Rau[1] < right_flash[1] and right_flash[1] < Rbu[1]:
            right_dia_v = [3, abs(right_flash[1] - Rc[1]) * 15 /
                           (Rbu[1] - Rau[1]) + 15]
        elif Rbu[1] <= right_flash[1] and right_flash[1] <= Rc[1]:
            right_dia_v = [3, abs(right_flash[1] - Rc[1]) * 15 /
                           (Rc[1] - Rbu[1])]
    elif Rc[1] < right_flash[1]:  # hyper
        if Rdd[1] < right_flash[1] and right_flash[1] < Red[1]:
            right_dia_v = [4, abs(right_flash[1] - Rc[1]) * 15 /
                           (Red[1] - Rdd[1]) + 15]
        elif Rc[1] <= right_flash[1] and right_flash[1] <= Rdd[1]:
            right_dia_v = [4, abs(right_flash[1] - Rc[1]) * 15 /
                           (Rdd[1] - Rc[1])]
    return right_dia_v

def iris_situation_define(mesh_points):
    left_iris_pts = np.array([mesh_points[i] for i in LEFT_IRIS])
    right_iris_pts = np.array([mesh_points[i] for i in RIGHT_IRIS])
    La = left_iris_pts[0]  # in
    Lc = left_iris_pts[4]  # center
    Le = left_iris_pts[2]  # out
    Lau = left_iris_pts[1]  # up
    Led = left_iris_pts[3]  # down
    Lb = (La - Lc) / 2 + Lc
    Ld = (Lc - Le) / 2 + Le
    Lbu = (Lc - Lau) / 2 + Lau
    Ldd = (Led - Lc) / 2 + Lc

    Ra = right_iris_pts[2]
    Rc = right_iris_pts[4]
    Re = right_iris_pts[0]
    Rau = right_iris_pts[1]
    Red = right_iris_pts[3]
    Rb = (Rc - Ra) / 2 + Ra
    Rd = (Re - Rc) / 2 + Rc
    Rbu = (Rc - Rau) / 2 + Rau
    Rdd = (Red - Rc) / 2 + Rc

    return La, Lb, Lc, Ld, Le, Lau, Lbu, Ldd, Led, \
           Ra, Rb, Rc, Rd, Re, Rau, Rbu, Rdd, Red

def reflection_dia(La, Lb, Lc, Ld, Le, Lau, Lbu, Ldd, Led,
                   Ra, Rb, Rc, Rd, Re, Rau, Rbu, Rdd, Red,
                   left_flash, right_flash):
    left_dia_h = left_horizontal_diagnose(La, Lb, Lc, Ld, Le, left_flash)
    left_dia_v = left_vertical_diagnose(Lau, Lbu, Ldd, Led, Lc, left_flash)
    right_dia_h = right_horizontal_diagnose(Ra, Rb, Rc, Rd, Re, right_flash)
    right_dia_v = right_vertical_diagnose(Rau, Rbu, Rdd, Red, Rc, right_flash)
    result = 0
    # Diagnosis logic based on the calculated values
    if ((left_dia_h[0] == right_dia_h[0] and left_dia_v[0] != right_dia_v[0]) or
        (left_dia_h[1] > 15 or right_dia_h[1] > 15 or left_dia_v[1] > 15 or right_dia_v[1] > 15)):
        # Exotropia
        if left_dia_h[0] == 1 and right_dia_h[0] == 1:
            if (left_dia_h[0] == 1 and left_dia_h[1] > 15) or (right_dia_h[0] == 1 and right_dia_h[1] > 15) or ((left_dia_h[0] == 1 and right_dia_h[0] == 1) and (left_dia_h[1] + right_dia_h[1]) >15):
                result = 1
        # Esotropia
        if left_dia_h[0] == 2 and right_dia_h[0] == 2:
            if (left_dia_h[0] == 2 and left_dia_h[1] > 15) or (right_dia_h[0] == 2 and right_dia_h[1] > 15) or ((left_dia_h[0] == 2 and right_dia_h[0] == 2) and (left_dia_h[1] + right_dia_h[1]) >15):
                result = 2
        # Hypotropia
        if left_dia_v[0] == 3 and right_dia_v[0] != 3:
            if (left_dia_v[0] == 3 and left_dia_v[1] > 15) or (right_dia_v[0] == 3 and right_dia_v[1] > 15) or ((left_dia_v[0] == 3 and right_dia_v[0] != 3) and (left_dia_v[1] + right_dia_v[1] > 15)):
                result = 3
        # Hypertropia
        if left_dia_v[0] == 4 and right_dia_v[0] != 4:
            if (left_dia_v[0] == 4 and left_dia_v[1] > 15) or (right_dia_v[0] == 4 and right_dia_v[1] > 15) or ((left_dia_v[0] == 4 and right_dia_v[0] != 4) and (left_dia_v[1] + right_dia_v[1] > 15)):
                result = 4
    return result
def calculate_angle_and_direction(vector):
    angle = np.arctan2(vector[1], vector[0])
    degrees = np.degrees(angle)
    degrees = (degrees + 360) % 360  # Normalize angle between 0 and 360
    return degrees

def calculate_vectors(iris_center, eye_pts):
    return [
        (iris_center[0] - eye_pts[8][0], iris_center[1] - eye_pts[8][1]),
        (iris_center[0] - eye_pts[4][0], iris_center[1] - eye_pts[4][1]),
        (iris_center[0] - eye_pts[0][0], iris_center[1] - eye_pts[0][1]),
        (iris_center[0] - eye_pts[12][0], iris_center[1] - eye_pts[12][1])
    ]

def distance_vector_from_coordinates(lc, rc, coordinates):
    if coordinates[0] is None or coordinates[1] is None or coordinates[2] is None or coordinates[3] is None:
        return 0
    left_distance = math.sqrt(
        (lc[0] - coordinates[0]) ** 2 + (lc[1] - coordinates[1]) ** 2)
    right_distance = math.sqrt(
        (rc[0] - coordinates[2]) ** 2 + (rc[1] - coordinates[3]) ** 2)
    solution = abs(left_distance - right_distance)
    return solution

def calculate_intersection(mesh_points):
    left_eye_pts = np.array([mesh_points[i] for i in LEFT_EYE])
    right_eye_pts = np.array([mesh_points[i] for i in RIGHT_EYE])

    # Left eye intersection calculation
    try:
        left_k1 = (left_eye_pts[8][1] - left_eye_pts[0][1]) / \
                  (left_eye_pts[8][0] - left_eye_pts[0][0])
        left_b1 = left_eye_pts[0][1] - left_k1 * left_eye_pts[0][0]
    except ZeroDivisionError:
        left_k1 = None
        left_b1 = None

    try:
        left_k2 = (left_eye_pts[12][1] - left_eye_pts[4][1]) / \
                  (left_eye_pts[12][0] - left_eye_pts[4][0])
        left_b2 = left_eye_pts[4][1] - left_k2 * left_eye_pts[4][0]
    except ZeroDivisionError:
        left_k2 = None
        left_b2 = None

    if left_k1 is not None and left_k2 is not None and left_k1 != left_k2:
        left_x_intersection = (left_b2 - left_b1) / (left_k1 - left_k2)
        left_y_intersection = left_k1 * left_x_intersection + left_b1
    else:
        left_x_intersection = None
        left_y_intersection = None

    # Right eye intersection calculation
    try:
        right_k1 = (right_eye_pts[8][1] - right_eye_pts[0][1]) / \
                   (right_eye_pts[8][0] - right_eye_pts[0][0])
        right_b1 = right_eye_pts[0][1] - right_k1 * right_eye_pts[0][0]
    except ZeroDivisionError:
        right_k1 = None
        right_b1 = None

    try:
        right_k2 = (right_eye_pts[12][1] - right_eye_pts[4][1]) / \
                   (right_eye_pts[12][0] - right_eye_pts[4][0])
        right_b2 = right_eye_pts[4][1] - right_k2 * right_eye_pts[4][0]
    except ZeroDivisionError:
        right_k2 = None
        right_b2 = None

    if right_k1 is not None and right_k2 is not None and right_k1 != right_k2:
        right_x_intersection = (right_b2 - right_b1) / (right_k1 - right_k2)
        right_y_intersection = right_k1 * right_x_intersection + right_b1
    else:
        right_x_intersection = None
        right_y_intersection = None

    return left_x_intersection, left_y_intersection, right_x_intersection, right_y_intersection

def distance_vector(mesh_points, image):
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

    coordinates = calculate_intersection(mesh_points)
    solution = distance_vector_from_coordinates(lc, rc, coordinates)

    # Calculate La, Lb, ..., Red
    La, Lb, Lc_pts, Ld, Le, Lau, Lbu, Ldd, Led, \
    Ra, Rb, Rc_pts, Rd, Re, Rau, Rbu, Rdd, Red = iris_situation_define(mesh_points)

    # Extract left and right iris regions
    left_eye_region = left_eye_pts[:, :2]
    right_eye_region = right_eye_pts[:, :2]

    left_eye_x, left_eye_y, left_eye_w, left_eye_h = cv2.boundingRect(left_eye_region)
    right_eye_x, right_eye_y, right_eye_w, right_eye_h = cv2.boundingRect(right_eye_region)

    left_eye_image = image[left_eye_y:left_eye_y + left_eye_h, left_eye_x:left_eye_x + left_eye_w]
    right_eye_image = image[right_eye_y:right_eye_y + right_eye_h, right_eye_x:right_eye_x + right_eye_w]

    left_eye_binary, right_eye_binary = iris_catch(left_eye_image, right_eye_image)

    left_iris_data = extract_iris_data(LEFT_IRIS, left_eye_binary, mesh_points, left_eye_x, left_eye_y)
    right_iris_data = extract_iris_data(RIGHT_IRIS, right_eye_binary, mesh_points, right_eye_x, right_eye_y)

    _, left_flash_x, left_flash_y = find_and_average_iris_white_pixels(left_iris_data, LEFT_IRIS, mesh_points, left_eye_x, left_eye_y)
    _, right_flash_x, right_flash_y = find_and_average_iris_white_pixels(right_iris_data, RIGHT_IRIS, mesh_points, right_eye_x, right_eye_y)

    left_flash = (left_flash_x, left_flash_y)
    right_flash = (right_flash_x, right_flash_y)

    return (left_distances, right_distances, left_angle_degrees, right_angle_degrees, left_angle_degrees_up,
            left_angle_degrees_down, right_angle_degrees_up, right_angle_degrees_down, solution,
            La, Lb, Lc_pts, Ld, Le, Lau, Lbu, Ldd, Led,
            Ra, Rb, Rc_pts, Rd, Re, Rau, Rbu, Rdd, Red,
            left_flash, right_flash)

def iris_dia(left_distances, right_distances, left_angle_degrees, right_angle_degrees,
             left_angle_degrees_up, left_angle_degrees_down, right_angle_degrees_up, right_angle_degrees_down, solution):
    result = 0

    if(solution > 2.5 or solution < 1.3):
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
            
            return distance_vector(mesh_points, image)
         
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
        h, w = image.shape[:2]
        new_w = 640
        new_h = int((new_w / w) * h)
        image = cv2.resize(image, (new_w, new_h))
        image = cv2.flip(image, 1)
        landmarks_result = detect_face_landmarks(image)

        if landmarks_result is None:
            return jsonify({"error": "No face landmarks detected"}), 400

        # Unpack all the variables
        left_distances, right_distances, left_angle_degrees, right_angle_degrees, \
        left_angle_degrees_up, left_angle_degrees_down, right_angle_degrees_up, right_angle_degrees_down, \
        solution, La, Lb, Lc, Ld, Le, Lau, Lbu, Ldd, Led, \
        Ra, Rb, Rc, Rd, Re, Rau, Rbu, Rdd, Red, left_flash, right_flash = landmarks_result

        # Call diagnosis functions
        iris_result = iris_dia(
            left_distances, right_distances, left_angle_degrees, right_angle_degrees,
            left_angle_degrees_up, left_angle_degrees_down, right_angle_degrees_up,
            right_angle_degrees_down, solution
        )

        reflection_result = reflection_dia(
            La, Lb, Lc, Ld, Le, Lau, Lbu, Ldd, Led,
            Ra, Rb, Rc, Rd, Re, Rau, Rbu, Rdd, Red,
            left_flash, right_flash
        )
        # Determine 'dia' based on 'iris_result'
        if (left_flash[0] is None) or (right_flash[0] is None):
            result = iris_result 
            dia = 0
        else:    
            result = reflection_result
            dia = 1

        
        return jsonify({"result": result, "dia":dia}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def get_hello():
    return 'Hello World!'

if __name__ == "__main__":
    app.run(debug=True, port=10000)