import numpy as np

def to_unitvec(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        print("Warning: Attempting to normalize a zero vector.")
        return np.zeros_like(vector)
    return vector / norm

def projection_on_plane(vector, normal):
    return vector - np.dot(vector, normal) * normal

# Core Body Vector (Unit Vector)
corebody = np.array([0, 20, 0])  # Core Body Vector
corebody_unit = to_unitvec(corebody)

# Camera and MidWaist
camera_point = np.array([0, 0, 0])  # Camera Position
midWaist = np.array([0, 0, 10])  # Midwaist

# 1. คำนวณ Normal Vector โดยใช้ Cross Product
camera_vector_waist = midWaist - camera_point
normalvector = np.cross(corebody, camera_vector_waist)  # Cross Product
normalvector_unit = to_unitvec(normalvector)  # ทำให้ Normal Vector เป็น Unit Vector

# Print Normal Vector and Plane Equation
print(f"Normal Vector: {normalvector_unit}")

# # 2. Define Plane Equation
# x0, y0, z0 = midWaist  # จุดที่ใช้ในสมการระนาบ
# a, b, c = normalvector_unit

# # สมการระนาบ: a(x - x0) + b(y - y0) + c(z - z0) = 0
# plane = f"{a}(x - {x0}) + {b}(y - {y0}) + {c}(z - {z0}) = 0"
# print(plane)

# 3. Shoulder to Elbow Vector
shoulder = np.array([5, 20, 10])
elbow = np.array([7, 12, 10])
shoulder_to_elbow = elbow - shoulder  # Vector from Shoulder to Elbow

# 4. ฉาย Shoulder → Elbow Vector ลงในระนาบ
projected_vector_shoulder = projection_on_plane(shoulder_to_elbow, normalvector_unit)
print(projected_vector_shoulder)
projected_vector_corebody = projection_on_plane(corebody, normalvector_unit)
print(projected_vector_corebody)
def angle(vector1, vector2):
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 'NULL'
    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_theta = np.clip(dot_product / magnitude_product, -1.0, 1.0)
    angle_radians = np.arccos(cos_theta)
    return np.rad2deg(angle_radians)
x = angle(projected_vector_corebody,projected_vector_shoulder)
print(x)

def normal_vector_from_plane(p1, p2, p3):
    """Compute the normal vector of a plane defined by three points."""
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        print("Warning: The three points are collinear, cannot define a plane.")
        return np.zeros(3)  # Return zero vector if points are collinear
    
    return normal / norm  # Normalize the normal vector

# Example usage:
R_Shoulder = np.array([1, 2, 0])
L_Shoulder = np.array([1, 0, 0])
R_Hip = np.array([0, 2, 0])

normal_vector = normal_vector_from_plane(R_Shoulder, L_Shoulder, R_Hip)
print("Normal Vector of the Plane:", normal_vector)