import numpy as np

def calculate_planepoint(pcd, picked, height):
    pts = np.asarray(pcd.points)
    point1 = np.asarray(pts[picked[0]]).copy()
    point2 = np.asarray(pts[picked[1]]).copy()
    point3 = np.asarray(pts[picked[2]]).copy()
    vector1 = point2 - point1
    vector2 = point3 - point2
    point4 = point1 + vector2
    nvector = np.cross(vector1, vector2)
    nvector_unit = nvector / np.linalg.norm(nvector)

    # Adjust points by the height of the plane
    point11 = point1 + nvector_unit * height
    point22 = point2 + nvector_unit * height
    point33 = point3 + nvector_unit * height
    point44 = point4 + nvector_unit * height

    return [point1, point2, point3, point4, point11, point22, point33, point44], nvector_unit

