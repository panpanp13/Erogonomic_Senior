import cv2
import mediapipe as mp
import numpy as np
from Calvector3d import angle_upperarm_yz, angle_lowerarm_yz
from Rula import stepA1, stepA2, tabela, tabelb, tabelc, table_a, table_b1, table_b2, table_c
import json
import pyk4a
from pyk4a import Config, PyK4A
import time
from ultralytics import YOLO

# Load calibration data
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

# Initialize Azure Kinect
kinect = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_1080P,
                      depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                      synchronized_images_only=True))
kinect.start()

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")
freeze = False

# Define keypoint mappings for RULA
upper_arm_angles = {
    "leftUpperArm": (5, 7, 11),  # Left shoulder, left elbow, left hip
    "rightUpperArm": (6, 8, 12)  # Right shoulder, right elbow, right hip
}
lower_arm_angles = {
    "leftLowerArm": (7, 9, 5),  # Left elbow, left wrist, left shoulder
    "rightLowerArm": (8, 10, 6)  # Right elbow, right wrist, right shoulder
}

# Debug intrinsic parameters
print(f"Intrinsic Parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

while True:
    if not freeze:
        capture = kinect.get_capture()
        if capture.color is None or capture.depth is None:
            print("Failed to capture frame")
            continue

        # Get RGB and depth frames
        rgb_frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)
        depth_frame = capture.transformed_depth

        # Run YOLOv8 Pose
        try:
            results = model(rgb_frame)
        except Exception as e:
            print(f"YOLO processing error: {e}")
            continue

        # Debug YOLO results
        if results[0].keypoints:
            print(f"Detected {len(results[0].keypoints.data)} keypoints")
            keypoints_data = results[0].keypoints.data.cpu().numpy()
        else:
            print("No keypoints detected")
            keypoints_data = []

        frame = results[0].plot()

        # Initialize RULA variables
        coordinate_list = []
        angles_list = []
        left_upperangle = None
        right_upperangle = None
        left_lowerangle = None
        right_lowerangle = None
        considered_side = None

        A_1 = A_2 = A_3 = A_4 = 1
        B_9 = 1
        B_10 = 2
        B_11 = 1

        # Extract keypoints and process for RULA
        for idx, keypoint in enumerate(keypoints_data):
            if len(keypoint) >= 3:
                x, y, conf = keypoint[:3]
                conf = float(conf)  # Ensure conf is scalar
                if conf > 0.5:
                    # Map pixel coordinates to depth
                    cx_pixel, cy_pixel = int(x), int(y)
                    if 0 <= cx_pixel < depth_frame.shape[1] and 0 <= cy_pixel < depth_frame.shape[0]:
                        depth = depth_frame[cy_pixel, cx_pixel] / 1000.0  # Convert to meters
                        real_x = (cx_pixel - cx) * depth / fx
                        real_y = (cy_pixel - cy) * depth / fy
                        real_z = depth
                        coordinate_list.append({
                            'landmark_id': idx,
                            'x_real': real_x,
                            'y_real': real_y,
                            'z_real': real_z
                        })
                    else:
                        print(f"Keypoint {idx} is outside depth frame bounds")
                    # Draw keypoint on the frame
                    cv2.circle(frame, (cx_pixel, cy_pixel), 5, (0, 255, 0), -1)
            else:
                print(f"Keypoint {idx} has invalid format: {keypoint}")

        # Debug RULA calculations
        print(f"Coordinates collected: {coordinate_list}")

        # Calculate angles for upper and lower arms
        for angle_name, (landmark_id1, landmark_id2, landmark_id3) in upper_arm_angles.items():
            ids_in_list = {coord['landmark_id'] for coord in coordinate_list}
            if landmark_id1 in ids_in_list and landmark_id2 in ids_in_list and landmark_id3 in ids_in_list:
                upperangle = angle_upperarm_yz(landmark_id1, landmark_id2, landmark_id3, coordinate_list)
                if upperangle is not None:
                    angles_list.append({'name': angle_name, 'angle': upperangle})
                    if angle_name == "leftUpperArm":
                        left_upperangle = upperangle
                    elif angle_name == "rightUpperArm":
                        right_upperangle = upperangle

        if left_upperangle is not None and right_upperangle is not None:
            considered_side = "Right" if right_upperangle > left_upperangle else "Left"
            cv2.putText(frame, f"Consider: {considered_side}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if considered_side:
            if considered_side == "Left":
                A_1 = stepA1(left_upperangle, "left")
            else:
                A_1 = stepA1(right_upperangle, "right")

        for angle_name, (landmark_id1, landmark_id2, landmark_id3) in lower_arm_angles.items():
            ids_in_list = {coord['landmark_id'] for coord in coordinate_list}
            if landmark_id1 in ids_in_list and landmark_id2 in ids_in_list and landmark_id3 in ids_in_list:
                lowerangle = angle_lowerarm_yz(landmark_id1, landmark_id2, landmark_id3, coordinate_list)
                if lowerangle is not None:
                    angles_list.append({'name': angle_name, 'angle': lowerangle})
                    if angle_name == "leftLowerArm":
                        left_lowerangle = lowerangle
                    elif angle_name == "rightLowerArm":
                        right_lowerangle = lowerangle

        if considered_side:
            if considered_side == "Left":
                A_2 = stepA2(left_lowerangle, "left")
            else:
                A_2 = stepA2(right_lowerangle, "right")

        # Calculate RULA scores
        try:
            a_score = tabela(A_1, A_2, A_3, A_4, table_a)
            b_score = tabelb(B_9, B_10, B_11, table_b1, table_b2)
            c_score = tabelc(a_score + 0, b_score + 0, table_c)
            cv2.putText(frame, f"RULA Score: {c_score}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        except Exception as e:
            print(f"RULA calculation error: {e}")

        # Show the processed frame
        cv2.imshow("YTracking + RULA + Depth", frame)

    # Key handling for freeze and quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        freeze = True
    elif key == ord('c') and freeze:
        freeze = False
    elif key == ord('q'):
        break

kinect.stop()
cv2.destroyAllWindows()
