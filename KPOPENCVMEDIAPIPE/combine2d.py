import cv2
import mediapipe as mp
import pandas as pd
import math

table_a = pd.read_csv(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\KPOPENCVMEDIAPIPE\Rula_score\TableA.csv")
table_b = pd.read_csv(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\KPOPENCVMEDIAPIPE\Rula_score\TableB.csv")
table_c = pd.read_csv(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\KPOPENCVMEDIAPIPE\Rula_score\TableC.csv")

def calculate_angle_3d(landmark1, landmark2, landmark3):
    vector_a = [
        landmark1.x - landmark2.x,
        landmark1.y - landmark2.y,
        landmark1.z - landmark2.z,
    ]
    vector_b = [
        landmark3.x - landmark2.x,
        landmark3.y - landmark2.y,
        landmark3.z - landmark2.z,
    ]
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a**2 for a in vector_a))
    magnitude_b = math.sqrt(sum(b**2 for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return None
    angle = math.acos(dot_product / (magnitude_a * magnitude_b))
    return math.degrees(angle)
def calculate_rula_score_from_tables(
    upper_arm, lower_arm, wrist_twist, neck, trunk, leg, muscle_use, load_force
):
    group_a_row = table_a[
        (table_a['UpperArm'] == upper_arm) & (table_a['LowerArm'] == lower_arm)
    ]
    if group_a_row.empty:
        raise ValueError("Invalid combination for Table A.")
    group_a_score = group_a_row[wrist_twist].values[0]
    group_a_score += muscle_use + load_force
    group_b_row = table_b[(table_b['Neck'] == neck)]
    if group_b_row.empty:
        raise ValueError("Invalid combination for Table B.")
    group_b_score = group_b_row[f"{trunk}{leg}"].values[0]
    final_row = table_c[table_c['Score'] == group_a_score]
    if final_row.empty:
        raise ValueError("Invalid Group A score in Table C.")
    rula_score = final_row[str(group_b_score)].values[0]
    return rula_score
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            upper_arm_angle = calculate_angle_3d(landmarks[12], landmarks[14], landmarks[16])  # Left Elbow
            lower_arm_angle = calculate_angle_3d(landmarks[14], landmarks[16], landmarks[15])  # Left Wrist
            upper_arm_score = 3 if upper_arm_angle > 90 else 2
            lower_arm_score = 2 if lower_arm_angle > 90 else 1
            wrist_twist = "3WT1" 
            neck_score = 3
            trunk_score = 2
            leg_score = 1
            muscle_use = 1
            load_force = 1
            try:
                rula_score = calculate_rula_score_from_tables(
                    upper_arm_score, lower_arm_score, wrist_twist, neck_score, trunk_score, leg_score, muscle_use, load_force
                )
                cv2.putText(frame, f"RULA Score: {rula_score}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"RULA Score: {rula_score}")
            except ValueError as e:
                print(f"Error calculating RULA score: {e}")

        cv2.imshow("Mediapipe Pose Tracking", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
