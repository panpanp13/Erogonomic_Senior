import json
import numpy as np
# Load intrinsic parameters from configset.json
with open(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\KPOPENCVMEDIAPIPE\configset.json", "r") as f:
    calibration_data = json.load(f)

# Extract intrinsic parameters
intrinsic_matrix_flat = calibration_data["intrinsic_matrix"]
intrinsic_matrix = np.array(intrinsic_matrix_flat).reshape(3, 3)
fx = intrinsic_matrix[0, 0]
fy = intrinsic_matrix[1, 1]
cx = intrinsic_matrix[0, 2]
cy = intrinsic_matrix[1, 2]
image_width = calibration_data["width"]
image_height = calibration_data["height"]

# Print loaded parameters for verification
print(f"Focal length (fx, fy): {fx}, {fy}")
print(f"Principal point (cx, cy): {cx}, {cy}")
print(f"Image dimensions (width, height): {image_width}, {image_height}")
