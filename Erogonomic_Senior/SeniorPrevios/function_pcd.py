import open3d as o3d
import numpy as np
import cv2
def picks(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()

    line_set = o3d.geometry.LineSet()
    points = [[0, 0, 0], [0, 0, -3000], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
              [0, 1, 1], [1, 1, 1]]
    lines = [[0, 1]]
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors = [[255, 0, 0] for i in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    vis.add_geometry(pcd)
    vis.add_geometry(line_set)

    vis.run()
    vis.destroy_window()
    pick = vis.get_picked_points()
    return pick
def get_height():
    while True:
        height = input()
        try:
            height = int(height)
            break
        except:
            try:
                height = float(height)
                break
            except:
                print('not a number')
    print('height = '+str(height))
    return height
def projectpoint(point,nvector,point0):
    v = point-point0
    # dist = v[0]*nvector[0] + v[1]*nvector[1] + v[2]*nvector[2]
    dist = np.dot(v,nvector)
    proj = point - dist*nvector
    # print('distance = ',dist)
    return proj, dist

def check_inarea(point,point1,point2,point4):
    isinarea = False
    isinzone = False
    vector01 = point-point1
    vector12 = point2-point1
    vector14 = point4-point1
    if (0 < np.dot(vector01,vector12) < np.dot(vector12,vector12)) and (0 < np.dot(vector01,vector14) < np.dot(vector14,vector14)):
        # print('Point is inarea 2D area')
        isinarea = True
    elif (0 == np.dot(vector01,vector12)) or (0 == np.dot(vector01,vector14)):
        # print('Point is at point1')
        isinarea = True
    elif (np.dot(vector01,vector12)==np.dot(vector12,vector12)):
        # print('Point is at point2')
        isinarea = True
    elif (np.dot(vector01,vector14) == np.dot(vector14,vector14)):
        # print('Point is at point4')
        isinarea = True
    else:
        # print('Point is outside 2D area')
        # print('Safe')
        isinarea = False
    return isinarea, isinzone

def check_height(dist,height):
    isinzone = False
    if 0 <= dist <= height:
        # print('Danger')
        isinzone = True
    elif dist > height or dist < 0:
        # print('Safe')
        isinzone = False
    else:
        # print('Wrong: Dist =',dist,'Height = ',height)
        isinzone = False
    return isinzone