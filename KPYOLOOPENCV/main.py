import cv2
import json
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
from CAL3DYOLO import angle_upperarm_yz , angle_lowerarm_yz
from RULAYOLO import stepA1 , stepA2 , tabela, tabelb, tabelc , table_a1, table_a2 ,table_b1, table_b2, table_c
import math
import time
import matplotlib.pyplot as plt
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

YOLO_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]
upper_arm_angles = {
    "leftUpperArm": (5, 7, 11),
    "rightUpperArm": (6, 8, 12)
}
lower_arm_angles = {
    "leftLowerArm": (7, 9, 5), 
    "rightLowerArm": (8, 10, 6) 
}

model = YOLO('yolov8l-pose.pt')
kinect = PyK4A(Config(
    color_resolution=pyk4a.ColorResolution.RES_1080P,
    depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
    synchronized_images_only=True
))
coordinate_list = []
all_coordinate = []
angles_list = []
kinect.start()
freeze = False
c_scores = []
timestamps = []
# plt.ion()
# fig, ax = plt.subplots()
# line, = ax.plot([], [], label="C Score vs Time", color="blue")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("C Score")
# ax.set_title("RULA C Score vs Time")
# ax.legend()
# ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
start_time = time.time()
previous_time = 0
current_time = 0
while True:
    if not freeze:
        capture = kinect.get_capture()
        if capture.color is not None:
            frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)
            depth_image = capture.transformed_depth 
            results = model(frame)
            coordinate_list.clear()
            all_coordinate.clear()
            current_time =time.time()
            fps = 1/(current_time-previous_time)
            previous_time= current_time
            cv2.putText(frame,str(round(fps,0)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),5)
            for person_id, result in enumerate(results, start=1):
                keypoints = result.keypoints.data[0].cpu().numpy()
                for idx, (x, y, conf) in enumerate(keypoints):
                    x, y = int(x), int(y)
                    if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
                        depth = depth_image[y, x]
                        # print(depth)
                        if depth is not None and 0 < depth < 10000: 
                            z_real = depth / 1000.0
                            real_x = (x - cx) * z_real / fx
                            real_y = (y - cy) * z_real / fy
                            coordinate_list.append({
                                'landmark_id': idx,
                                'x_real': real_x,
                                'y_real': real_y,
                                'z_real': z_real
                            })
                            print(f"YOLOTRACK Person {person_id} - Keypoint {YOLO_KEYPOINTS[idx]} ({idx}): "
                                f"x={real_x:.3f}m, y={real_y:.3f}m, depth={z_real:.3f}m, confidence={conf}")
                        else:
                            continue 
                all_coordinate.append({
                    'person_id': person_id,
                    'keypoints': coordinate_list
                })
                left_upperangle = None
                right_upperangle = None
                left_lowerangle = None
                right_lowerangle = None
                y_offset = 50
                considered_side = "Right"
                A_1 = 1
                A_2 = 1
                A_3 = 1
                A_4 = 1
                B_9 = 1
                B_10 = 1
                B_11 = 1
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
                y_offset = 50
                if left_upperangle is not None and right_upperangle is not None:
                    if right_upperangle > left_upperangle:
                        considered_side = "Right"
                    else:
                        considered_side = "Left"
                    cv2.putText(frame, f"Consider: {considered_side}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30

                if considered_side:
                    left_score = stepA1(left_upperangle, "left") if considered_side == "Left" else None
                    right_score = stepA1(right_upperangle, "right") if considered_side == "Right" else None

                    if considered_side == "Left" and left_score is not None:
                        A_1 = left_score
                        print(f"A1 :{A_1}")
                        # cv2.putText(frame, f"A1 Score: {A_1}", (10, y_offset),  # Display A1 score
                        #         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        # y_offset += 30
                    elif considered_side == "Right" and right_score is not None:
                        A_1 = right_score
                        print(f"A1 :{A_1}")
                        # cv2.putText(frame, f"A1 Score: {A_1}", (10, y_offset),  # Display A1 score
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
                # if considered_side:
                #     left_score = stepA2(left_lowerangle, "left") if considered_side == "Left" else None
                #     right_score = stepA2(right_lowerangle, "right") if considered_side == "Right" else None
                #     if considered_side == "Left" and left_score is not None:
                #         A_2 = left_score
                #         print(f"A2 :{A_2}")
                #     elif considered_side == "Right" and right_score is not None:
                #         A_2 = right_score
                #         print(f"A2 :{A_2}")
                A_3 = 1
                A_4 = 1
                if A_1 is not None and A_2 is not None:
                    a_score = tabela(A_1, A_2, A_3, A_4, table_a1, table_a2)
                    print(f"A score: {a_score}")
                    # cv2.putText(frame, f"A score: {a_score}", (10, y_offset),
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
                    # cv2.putText(frame, f"B Score: {b_score}", (10, y_offset),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # y_offset += 30 
############################################
                bsum = b_score + 0 + 0
#----------------------------------END OF TABEL B------------------------------------------# 
                if asum is not None and bsum is not None :
                    c_score = tabelc(asum, bsum, table_c)
                    print(f"RULA score:{c_score}")
                    # cv2.putText(frame, f"RULA Score: {c_score}", (10, y_offset),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # y_offset += 30 
                    # c_scores.append(c_score)
                    # timestamps.append(current_time - start_time)
                    # line.set_xdata(timestamps)
                    # line.set_ydata(c_scores)
                    # ax.relim()
                    # ax.autoscale_view()
                    # plt.draw()
                    # plt.pause(0.01)
                y_offset = 80
                cv2.putText(frame, f"A Score: {a_score}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 30
                cv2.putText(frame, f"B Score: {b_score}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 30
                cv2.putText(frame, f"RULA Score: {c_score}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 50 
                cv2.putText(frame, f"A_1: {A_1}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(frame, f"A_2: {A_2}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(frame, f"A_3: {A_3}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(frame, f"A_4: {A_4}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(frame, f"B_9: {B_9}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(frame, f"B_10: {B_10}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(frame, f"B_11: {B_11}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
                if left_upperangle is not None:
                    cv2.putText(frame, f"Left Upper Arm Angle: {left_upperangle:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30
                if right_upperangle is not None:
                    cv2.putText(frame, f"Right Upper Arm Angle: {right_upperangle:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30
                if left_lowerangle is not None:
                    cv2.putText(frame, f"Left Lower Arm Angle: {left_lowerangle:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30
                if right_lowerangle is not None:
                    cv2.putText(frame, f"Right Lower Arm Angle: {right_lowerangle:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30
            frame = results[0].plot()
            cv2.imshow("Pose Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        freeze = True
        for i in all_coordinate:
            print(i)
    elif key == ord('c') and freeze:
        freeze = False
    elif key == ord('q'):
        break
kinect.stop()
cv2.destroyAllWindows()
