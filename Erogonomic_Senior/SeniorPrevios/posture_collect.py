previous_angles = None
def posture_collect(current_angles):
    global previous_angles
    posture_changed = False
    if previous_angles is not None:
        for key, value in current_angles.items():
            if key in previous_angles and abs(previous_angles[key] - value) > 1:
                posture_changed = True
                print(f"Posture Changed Detected for {key}:")
                print(f"Old Angle: {previous_angles[key]}")
                print(f"New Angle: {value}")
                break
    previous_angles = current_angles.copy()
    if not posture_changed:
        print("No posture changes detected.")
    return posture_changed
