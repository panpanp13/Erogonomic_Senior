import cv2
import mediapipe as mp
import numpy as np
from Calvector3d import angle_upperarm_yz , angle_lowerarm_yz
from Rula import stepA1 , stepA2 , tabela, tabelb, tabelc , table_a ,table_b1, table_b2, table_c
import json
import pyk4a
from pyk4a import Config, PyK4A
import time
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
kinect = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_1080P,
                      depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                      synchronized_images_only=True))
kinect.start()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
coordinate_list = []
distances_list = []
angles_list = []
freeze = False
upper_arm_angles = {
    "leftUpperArm": (11, 13, 23),
    "rightUpperArm": (12, 14, 24)  
}
lower_arm_angles = {
    "leftLowerArm": (13, 15, 11),
    "rightLowerArm": (14, 16, 12)
}
previous_time=0
current_time=0
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
            current_time =time.time()
            fps = 1/(current_time-previous_time)
            previous_time= current_time
            cv2.putText(rgb_frame,str(round(fps,0)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),5)
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
                #######
                A_1 = 1
                A_2 = 1
                A_3 = 1
                A_4 = 1
                B_9 = 1
                B_10 = 1
                B_11 = 1
                #######
#----------------------------------START OF TABEL A--------------------------------------------#  
#############################A1#############################
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
                            # cv2.putText(rgb_frame, f"{angle_name} (Y-Z): {upperangle:.2f}", (10, y_offset),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                if left_upperangle is not None and right_upperangle is not None:
                    if right_upperangle > left_upperangle:
                        considered_side = "Right"
                    else:
                        considered_side = "Left"
                    cv2.putText(rgb_frame, f"Consider: {considered_side}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30

                if considered_side:
                    left_score = stepA1(left_upperangle, "left") if considered_side == "Left" else None
                    right_score = stepA1(right_upperangle, "right") if considered_side == "Right" else None

                    if considered_side == "Left" and left_score is not None:
                        A_1 = left_score
                        print(f"A1 :{A_1}")
                        # cv2.putText(rgb_frame, f"A1 Score: {A_1}", (10, y_offset),  # Display A1 score
                        #         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        # y_offset += 30
                    elif considered_side == "Right" and right_score is not None:
                        A_1 = right_score
                        print(f"A1 :{A_1}")
                        # cv2.putText(rgb_frame, f"A1 Score: {A_1}", (10, y_offset),  # Display A1 score
                        #         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        # y_offset += 30
##############################A2################################
                for angle_name, (landmark_id1, landmark_id2, landmark_id3) in lower_arm_angles.items():
                    ids_in_list = {coord['landmark_id'] for coord in coordinate_list}
                    if landmark_id1 in ids_in_list and landmark_id2 in ids_in_list and landmark_id3 in ids_in_list:
                        lowerangle = angle_lowerarm_yz(landmark_id1, landmark_id2, landmark_id3, coordinate_list)
                        print(f"Lower Arm Angle ({angle_name}): {lowerangle}")
                        if lowerangle is not None:
                            angles_list.append({'name': angle_name, 'angle': lowerangle})
                            if angle_name == "leftLowerArm":
                                left_lowerangle = lowerangle
                            elif angle_name == "rightLowerArm":
                                right_lowerangle = lowerangle
                if considered_side:
                    if considered_side == "Left":
                        print(f"Left Lower Angle Before StepA2: {left_lowerangle}")
                        left_score = stepA2(left_lowerangle, "left") if left_lowerangle is not None else None
                        if left_score is not None:
                            A_2 = left_score
                            print(f"A2 (Left): {A_2}")
                        else:
                            print(f"A2 (Left): None.")
                    elif considered_side == "Right":
                        print(f"Right Lower Angle Before StepA2: {right_lowerangle}")
                        right_score = stepA2(right_lowerangle, "right") if right_lowerangle is not None else None
                        if right_score is not None:
                            A_2 = right_score
                            print(f"A2 (Right): {A_2}")
                        else:
                            print(f"A2 (Right): None.")
                    if considered_side == "Left" and left_score is not None:
                        A_2 = left_score
                        print(f"A2 :{A_2}")
                    elif considered_side == "Right" and right_score is not None:
                        A_2 = right_score
                        print(f"A2 :{A_2}")
                A_3 = 1
                A_4 = 1
                y_offset = 50
                if A_1 is not None and A_2 is not None:
                    a_score = tabela(A_1, A_2, A_3, A_4, table_a)
                    print(f"A score: {a_score}")
                    # cv2.putText(rgb_frame, f"A score: {a_score}", (10, y_offset),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # y_offset += 30 
                asum = a_score + 0 + 0
#----------------------------------END OF TABEL A--------------------------------------------# 
#----------------------------------START OF TABEL B------------------------------------------#               
                B_9 = 1
                B_10 = 2
                B_11  = 1
                if B_9 is not None and B_10 is not None and B_11 is not None :
                    b_score = tabelb(B_9, B_10, B_11, table_b1, table_b2)
                    print(f"B score: {b_score}")
                    # cv2.putText(rgb_frame, f"B Score: {b_score}", (10, y_offset),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # y_offset += 30 
############################################
                bsum = b_score + 0 + 0
#----------------------------------END OF TABEL B------------------------------------------# 
                if asum is not None and bsum is not None :
                    c_score = tabelc(asum, bsum, table_c)
                    print(f"RULA score:{c_score}")
                    # cv2.putText(rgb_frame, f"C Score: {c_score}", (10, y_offset),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    # y_offset += 30 
                y_offset = 80
                cv2.putText(rgb_frame, f"A Score: {a_score}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 30
                cv2.putText(rgb_frame, f"B Score: {b_score}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 30
                cv2.putText(rgb_frame, f"RULA Score: {c_score}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 50 
                cv2.putText(rgb_frame, f"A_1: {A_1}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(rgb_frame, f"A_2: {A_2}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(rgb_frame, f"A_3: {A_3}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(rgb_frame, f"A_4: {A_4}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(rgb_frame, f"B_9: {B_9}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(rgb_frame, f"B_10: {B_10}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(rgb_frame, f"B_11: {B_11}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                if left_upperangle is not None:
                    cv2.putText(rgb_frame, f"Left Upper Arm Angle: {left_upperangle:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30
                if right_upperangle is not None:
                    cv2.putText(rgb_frame, f"Right Upper Arm Angle: {right_upperangle:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30
                if left_lowerangle is not None:
                    cv2.putText(rgb_frame, f"Left Lower Arm Angle: {left_lowerangle:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30
                if right_lowerangle is not None:
                    cv2.putText(rgb_frame, f"Right Lower Arm Angle: {right_lowerangle:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30
                
        cv2.imshow("Azure Kinect + MediaPipe Pose Tracking", rgb_frame)
        # cv2.imshow("Depth Frame", depth_colored)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            freeze = True
        elif key == ord('x') and freeze:
            freeze = False

kinect.stop()
cv2.destroyAllWindows()
