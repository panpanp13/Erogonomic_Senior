import math
import cv2
import mediapipe as mp
import numpy as np
import json
import pyk4a
from pyk4a import Config, PyK4A
from Rula import stepA1 , stepA2
# Function to calculate angles in 3D planes
def angle3d_planes(pointA, pointB, pointC):
    def calculate_2d_angle(v1, v2):
        """Calculate the angle between two 2D vectors."""
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return None  # Avoid division by zero
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
        cos_theta = max(-1, min(1, cos_theta))  # Clamp to [-1, 1]
        return math.degrees(math.acos(cos_theta))
    
    # Calculate vectors
    AB = (pointA[0] - pointB[0], pointA[1] - pointB[1], pointA[2] - pointB[2])
    BC = (pointC[0] - pointB[0], pointC[1] - pointB[1], pointC[2] - pointB[2])
    
    # Projections on each plane
    AB_xy = (AB[0], AB[1])
    BC_xy = (BC[0], BC[1])
    AB_yz = (AB[1], AB[2])
    BC_yz = (BC[1], BC[2])
    AB_xz = (AB[0], AB[2])
    BC_xz = (BC[0], BC[2])
    
    angle_xy = calculate_2d_angle(AB_xy, BC_xy)
    angle_yz = calculate_2d_angle(AB_yz, BC_yz)
    angle_xz = calculate_2d_angle(AB_xz, BC_xz)
    
    return {
        "XY Plane": angle_xy,
        "YZ Plane": angle_yz,
        "XZ Plane": angle_xz
    }

# Read calibration data
with open(r"KPOPENCVMEDIAPIPE/configset.json", "r") as f:
    calibration_data = json.load(f)
fx = calibration_data["intrinsic_matrix"][0] 
fy = calibration_data["intrinsic_matrix"][4] 
cx = calibration_data["intrinsic_matrix"][6]  
cy = calibration_data["intrinsic_matrix"][7] 
intrinsic_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])

# Kinect initialization
kinect = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_1080P,
                      depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                      synchronized_images_only=True))
kinect.start()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
coordinate_list = []
angles_list = []
freeze = False

# Angle configurations
upper_arm_angles = {
    "leftUpperArm": (11, 13, 23),
    "rightUpperArm": (12, 14, 24)  
}
lower_arm_angles = {
    "leftLowerArm": (13, 15, 11),
    "rightLowerArm": (14, 16, 12)
}
import pandas as pd

# Load Table A for RULA scoring
table_a = pd.read_csv(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\KPOPENCVMEDIAPIPE\Rula_score\TableA.csv")

# Tabela function definition
def tabela(A_1, A_2, A_3, A_4, table_a):
    """
    Find the score from Table A using A_1 (upper arm), A_2 (lower arm), A_3 (wrist twist),
    and A_4 (additional adjustment).

    Parameters:
        A_1 (int): Upper arm score.
        A_2 (int): Lower arm score.
        A_3 (int): Wrist twist score.
        A_4 (int): Additional adjustment score (+1 or 0).
        table_a (pd.DataFrame): Table A as a pandas DataFrame.

    Returns:
        int: Final score from Table A with adjustment.
    """
    try:
        # Adjust row and column indices (assuming the table is 1-indexed)
        row_index = A_1 - 1  # Upper arm score maps to rows
        col_index = (A_2 - 1) * 4 + (A_3 - 1)  # Lower arm and wrist twist map to columns
        
        # Retrieve the base score from Table A
        base_score = table_a.iloc[row_index, col_index]
        
        # Add the adjustment score
        final_score = base_score + A_4
        return final_score
    except IndexError:
        raise ValueError(f"Invalid indices for Table A: A_1={A_1}, A_2={A_2}, A_3={A_3}")
    except Exception as e:
        raise ValueError(f"Error in tabela function: {e}")

# Integration into main code
with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
    while True:
        if not freeze:
            capture = kinect.get_capture()
            if capture.color is None or capture.depth is None:
                print("Failed to capture frame")
                continue
            rgb_frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)
            depth_frame = capture.transformed_depth
            depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            results = pose.process(rgb_frame)
            coordinate_list.clear()
            angles_list.clear()

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                h, w, _ = rgb_frame.shape
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if landmark.visibility > 0.75 and 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                        cx_pixel, cy_pixel = int(landmark.x * w), int(landmark.y * h)
                        depth = depth_frame[cy_pixel, cx_pixel] / 1000.0 
                        real_x = (cx_pixel - cx) * depth / fx
                        real_y = (cy_pixel - cy) * depth / fy
                        real_z = depth
                        coordinate_list.append({
                            'landmark_id': idx,
                            'x_real': real_x,
                            'y_real': real_y,
                            'z_real': real_z
                        })
                left_upperangle = None
                right_upperangle = None
                left_lowerangle = None
                right_lowerangle = None
                y_offset = 50
                considered_side = None

                # Compute A_1 and A_2 as before
                # Example:
                A_1 = stepA1(left_upperangle if considered_side == "Left" else right_upperangle, considered_side.lower())
                A_2 = stepA2(left_lowerangle if considered_side == "Left" else right_lowerangle, considered_side.lower())
                A_3 = 1  # Wrist twist score
                A_4 = 1  # Adjustment score

                # Calculate final score using tabela
                try:
                    final_score = tabela(A_1, A_2, A_3, A_4, table_a)
                    print(f"Final RULA Score (Table A): {final_score}")
                    cv2.putText(rgb_frame, f"RULA Score: {final_score}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except ValueError as e:
                    print(f"Error calculating RULA score: {e}")
                
        cv2.imshow("Azure Kinect + MediaPipe Pose Tracking", rgb_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

kinect.stop()
cv2.destroyAllWindows()

