import math

def angle_upperarm_yz(landmark_id1, landmark_id2, landmark_id3, coordinate_list):
    shoulder = next((coord for coord in coordinate_list if coord['landmark_id'] == landmark_id1), None)
    elbow = next((coord for coord in coordinate_list if coord['landmark_id'] == landmark_id2), None)
    torso = next((coord for coord in coordinate_list if coord['landmark_id'] == landmark_id3), None)
    if shoulder is None or elbow is None or torso is None:
        print(f"Missing keypoints: shoulder={shoulder}, elbow={elbow}, torso={torso}")
        return None
    if None in (shoulder['z_real'], elbow['z_real'], torso['z_real']):
        print(f"Invalid z_real values: shoulder={shoulder['z_real']}, elbow={elbow['z_real']}, torso={torso['z_real']}")
        return None
    print(f"elbow: {elbow['y_real'], elbow['z_real']}")
    print(f"shoulder: {shoulder['y_real'], shoulder['z_real']}")
    print(f"torso: {torso['y_real'], torso['z_real']}")
    upper_arm_vector = [
        elbow['y_real'] - shoulder['y_real'],
        elbow['z_real'] - shoulder['z_real']
    ]
    torso_vector = [
        torso['y_real'] - shoulder['y_real'],
        torso['z_real'] - shoulder['z_real']
    ]
    print(f"Upper Arm Vector (YZ): {upper_arm_vector}")
    print(f"Torso Vector (YZ): {torso_vector}")

    if upper_arm_vector is not None and torso_vector is not None :
        magnitude_upper_arm = math.sqrt(sum(a**2 for a in upper_arm_vector))
        magnitude_torso = math.sqrt(sum(b**2 for b in torso_vector))
    if magnitude_upper_arm == 0 or magnitude_torso == 0:
        print("Zero-length vector detected. Skipping angle calculation.")
        return None
    dot_product = sum(a * b for a, b in zip(upper_arm_vector, torso_vector))
    cos_theta = dot_product / (magnitude_upper_arm * magnitude_torso)
    if not -1.0 <= cos_theta <= 1.0:
        print(f"Invalid cos_theta value: {cos_theta}")
        return None
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)
    print(angle_degrees)
    return angle_degrees


def angle_lowerarm_yz(landmark_id1, landmark_id2, landmark_id3, coordinate_list):
    wrist = next((coord for coord in coordinate_list if coord['landmark_id'] == landmark_id1), None)
    elbow = next((coord for coord in coordinate_list if coord['landmark_id'] == landmark_id2), None)
    shoulder = next((coord for coord in coordinate_list if coord['landmark_id'] == landmark_id3), None)
    if wrist is None or elbow is None or shoulder is None:
        print(f"Missing keypoints: wrist={wrist}, elbow={elbow}, shoulder={shoulder}")
        return None
    if None in (wrist['z_real'], elbow['z_real'], shoulder['z_real']):
        print(f"Invalid z_real values: wrist={wrist['z_real']}, elbow={elbow['z_real']}, shoulder={shoulder['z_real']}")
        return None
    print(f"wrist: {wrist['y_real'], wrist['z_real']}")
    print(f"elbow: {elbow['y_real'], elbow['z_real']}")
    print(f"shoulder: {shoulder['y_real'], shoulder['z_real']}")
    lower_arm_vector = [
        wrist['y_real'] - elbow['y_real'],
        wrist['z_real'] - elbow['z_real']
    ]
    upper_arm_vector = [
        shoulder['y_real'] - elbow['y_real'],
        shoulder['z_real'] - elbow['z_real']
    ]
    print(f"Lower Arm Vector (YZ): {lower_arm_vector}")
    print(f"Upper Arm Vector (YZ): {upper_arm_vector}")
    magnitude_lower_arm = math.sqrt(sum(a**2 for a in lower_arm_vector))
    magnitude_upper_arm = math.sqrt(sum(b**2 for b in upper_arm_vector))
    if magnitude_lower_arm == 0 or magnitude_upper_arm == 0:
        print("Zero-length vector detected. Skipping angle calculation.")
        return None
    dot_product = sum(a * b for a, b in zip(lower_arm_vector, upper_arm_vector))
    cos_theta = dot_product / (magnitude_lower_arm * magnitude_upper_arm)
    if not -1.0 <= cos_theta <= 1.0:
        print(f"Invalid cos_theta value: {cos_theta}")
        return None

    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)
    print(angle_degrees)
    return angle_degrees