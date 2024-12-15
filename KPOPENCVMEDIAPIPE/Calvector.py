import math
def calvector(landmark_id1, landmark_id2, coordinate_list):
    point1 = next(coord for coord in coordinate_list if coord['landmark_id'] == landmark_id1)
    point2 = next(coord for coord in coordinate_list if coord['landmark_id'] == landmark_id2)
    x1, y1 = point1['x_pixel'], point1['y_pixel']
    x2, y2 = point2['x_pixel'], point2['y_pixel']
    D_pixel = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return D_pixel

def calculate_angle(landmark_id1, landmark_id2, landmark_id3, coordinate_list):
    point1 = next(coord for coord in coordinate_list if coord['landmark_id'] == landmark_id1)
    point2 = next(coord for coord in coordinate_list if coord['landmark_id'] == landmark_id2)
    point3 = next(coord for coord in coordinate_list if coord['landmark_id'] == landmark_id3)
    x1, y1 = point1['x_pixel'], point1['y_pixel']
    x2, y2 = point2['x_pixel'], point2['y_pixel']
    x3, y3 = point3['x_pixel'], point3['y_pixel']
    BA_x, BA_y = x1 - x2, y1 - y2
    BC_x, BC_y = x3 - x2, y3 - y2
    dot_product = (BA_x * BC_x + BA_y * BC_y)
    magnitude_BA = math.sqrt(BA_x**2 + BA_y**2)
    magnitude_BC = math.sqrt(BC_x**2 + BC_y**2)
    if magnitude_BA == 0 or magnitude_BC == 0:
        return None
    angle = math.acos(dot_product / (magnitude_BA * magnitude_BC))
    return math.degrees(angle)
