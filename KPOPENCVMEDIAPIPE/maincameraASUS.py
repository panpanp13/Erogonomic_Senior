import cv2
import mediapipe as mp
import numpy as np
from Calvector import calvector , calculate_angle
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
coordinate_list = []
distances_list = []
angles_list = []
freeze = False
custom_distances = {
    "upperRightArm": (11, 13),  
    "lowerRightArm": (13, 15),  
    "upperLeftArm": (12, 14),   
    "lowerLeftArm": (14, 16),    
    "torso": (11, 23),           
    "upperRightLeg": (23, 25), 
    "lowerRightLeg": (25, 27),  
    "upperLeftLeg": (24, 26),    
    "lowerLeftLeg": (26, 28)   
}
custom_angles = {
    "leftElbow": (12, 14, 16), 
    "rightElbow": (11, 13, 15),
    "leftKnee": (24, 26, 28), 
    "rightKnee": (23, 25, 27)  
}
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        if not freeze:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            coordinate_list.clear()
            distances_list.clear()
            angles_list.clear()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                h, w, _ = frame.shape
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if landmark.visibility > 0.5 and 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                        cv2.putText(frame, f"{idx}", (cx + 10, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        coordinate_list.append({
                            'landmark_id': idx,
                            'x_pixel': cx,
                            'y_pixel': cy,
                        })
                y_offset = 50 
                for name, (landmark_id1, landmark_id2) in custom_distances.items():
                    ids_in_list = {coord['landmark_id'] for coord in coordinate_list}
                    if landmark_id1 in ids_in_list and landmark_id2 in ids_in_list:
                        distance = calvector(landmark_id1, landmark_id2, coordinate_list)
                        distances_list.append({'name': name, 'distance': distance})
                        cv2.putText(frame, f"{name}: {distance:.2f}px", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_offset += 30
                        print(f"{name} (Landmarks {landmark_id1}-{landmark_id2}): {distance:.2f} pixels")
                    else :
                        print(f"Landmarks {landmark_id1} or {landmark_id2} not found for {name}")
                for angle_name, (landmark_id1, landmark_id2, landmark_id3) in custom_angles.items():
                    ids_in_list = {coord['landmark_id'] for coord in coordinate_list}
                    if landmark_id1 in ids_in_list and landmark_id2 in ids_in_list and landmark_id3 in ids_in_list:
                        angle = calculate_angle(landmark_id1, landmark_id2, landmark_id3, coordinate_list)
                        if angle is not None:
                            angles_list.append({'name': angle_name, 'angle': angle})
                            cv2.putText(frame, f"{angle_name}: {angle:.2f}", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            y_offset += 30
                            print(f"{angle_name} (Landmarks {landmark_id1}-{landmark_id2}-{landmark_id3}): {angle:.2f}")
                        else :
                            print(f"Landmarks {landmark_id1}, {landmark_id2}, or {landmark_id3} not found for {angle_name}")
        cv2.imshow("MediaPipe Pose Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            freeze = True
            print("Coordinates:")
            for coord in coordinate_list:
                print(coord)
            print("\nDistances:")
            for dist in distances_list:
                print(f"{dist['name']} Distance: {dist['distance']:.2f} pixels")
            print("\nAngles:")
            for ang in angles_list:
                print(f"{ang['name']} Angle: {ang['angle']:.2f} degrees")
        elif key == ord('x') and freeze:
            freeze = False

cap.release()
cv2.destroyAllWindows()
