import open3d as o3d
import numpy as np
import copy
import cv2

# Load point cloud

# Check if normals exist
def Planar_patch_detect(pcd):
    if not pcd.has_normals():
        print("Computing normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # o3d.visualization.draw_geometries([pcd])

    # Verify again
    assert pcd.has_normals(), "Normals were not computed!"

    # using all defaults
    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0,
        min_num_points=0,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    print(oboxes)
    print("Detected {} patches".format(len(oboxes)))

    geometries = []
    for obox in oboxes:
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(obox.color)
        geometries.append(mesh)
        geometries.append(obox)
    geometries.append(pcd)
    print(geometries)
    o3d.visualization.draw_geometries(geometries,
                                    zoom=0.62,
                                    front=[0.4361, -0.2632, -0.8605],
                                    lookat=[2.4947, 1.7728, 1.5541],
                                    up=[-0.1726, -0.9630, 0.2071])


def align_floor_to_xz_plane(pcd_in, normal_vec,d_plane):
    """
    Segments the largest plane (assumed to be the floor) in a point cloud
    and rotates the cloud so that the plane aligns with the XZ-plane (i.e., 
    the floor's normal becomes the Y-axis).
    
    Parameters:
    -----------
    pcd_in : open3d.geometry.PointCloud
        The input point cloud.
    distance_threshold : float
        Maximum distance of a point to the plane model for it to be considered an inlier.
    ransac_n : int
        Number of initial points to consider for plane fitting.
    num_iterations : int
        Number of iterations for RANSAC.
    
    Returns:
    --------
    pcd_out : open3d.geometry.PointCloud
        A copy of the input point cloud, rotated so the largest plane is parallel to the XZ-plane.
    plane_model : list of float
        The plane equation coefficients [a, b, c, d].
    inliers : list of int
        Indices of inlier points that belong to the detected plane.
    rotation_matrix : numpy.ndarray of shape (3, 3)
        The 3x3 rotation matrix that was applied to align the plane.
    """

    # 1. Plane segmentation to find the largest plane (the floor).
    # pcd_in.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # plane_model, inliers = pcd_in.segment_plane(distance_threshold=distance_threshold,
    #                                            ransac_n=ransac_n,
    #                                            num_iterations=num_iterations)
    # Plane model is of the form ax + by + cz + d = 0
    # print('Plane_model: ',plane_model)
    pcd_in.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    a, b, c, d = normal_vec[0],normal_vec[1],normal_vec[2],d_plane
    print(f"Detected plane model: plane_eq (ax + by + cz + d = 0)")

    # 2. Compute the normal vector and normalize it.
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    print(f"Floor normal (unnormalized from plane): {[a, b, c]}")
    print(f"Normalized floor normal: {normal}")

    # 3. We want to align this normal with the Y-axis => target = [0, 1, 0].
    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Edge case: if the normal is already nearly parallel or anti-parallel to target
    dot_val = np.dot(normal, target)
    if np.isclose(dot_val, 1.0, atol=1e-6):
        # Already aligned with +Y
        rotation_matrix = np.eye(3)
    # elif np.isclose(dot_val, -1.0, atol=1e-6):
    #     # Aligned with -Y; rotate 180° around X or Z, for instance
    #     # 180° rotation around X => diag(1, -1, -1)
    #     rotation_matrix = np.array([[1, 0, 0],
    #                                 [0, -1, 0],
    #                                 [0, 0, -1]], dtype=np.float64)
    else:
        # General case: use axis-angle
        axis = np.cross(normal, target)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot_val)  # dot_val = normal . target
        print(angle)
        # Create the rotation matrix from axis-angle
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    # 4. Apply the rotation to the entire point cloud
    pcd_out = copy.deepcopy(pcd_in)
    pcd_out.rotate(rotation_matrix, center=(0, 0, 0))

    return pcd_out, rotation_matrix

# 2. Visualize and pick 3 points in the cloud.
#    The VisualizerWithEditing allows you to pick points and returns their indices.
def pick_point_from_pcd(pcd):
    # Create a sphere marker at the requested position
    # Set up the visualizer (for picking points from pcd)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Point Pick", width=1280, height=720)
    
    # Add both the point cloud and the sphere marker
    vis.add_geometry(pcd)
    # Block until the user closes the window or finishes picking
    vis.run()
    vis.destroy_window()
    
    # Retrieve the indices of the selected points
    picked_indices = vis.get_picked_points()
    # points = np.asarray(pcd.points)

    if picked_indices:
        picked_points = np.asarray(pcd.points)[picked_indices, :]
        print("Picked points (XYZ):\n", picked_points)
        return picked_points
    else:
        print("No points were picked.")
        return None
def create_plane_by_pick(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    print("Please pick exactly 3 points, then close the window.")
    vis.run()  # user picks points here
    vis.destroy_window()

    picked_ids = vis.get_picked_points()
    if len(picked_ids) < 3:
        raise ValueError("You must pick exactly 3 points for plane fitting.")

    # 3. Extract the chosen points.
    points = np.asarray(pcd.points)
    p1 = points[picked_ids[0]]
    p2 = points[picked_ids[1]]
    p3 = points[picked_ids[2]]

    # 4. Compute normal vector of the plane
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    # print(normal)
    # print(np.linalg.norm(normal))
    normal = normal / np.linalg.norm(normal)  # normal vector 1 หน่วยจาก plane
    # print(normal)

    # 5. Compute d in the plane equation n·x + d = 0
    d = -np.dot(normal, p2) #-np เพราะย้าย d มาลบ ax +by +cz =d
    print('d',d)

    print(f"Plane equation is {normal[0]:.3f}x + {normal[1]:.3f}y + {normal[2]:.3f}z + {d:.3f} = 0")

    # 6. Determine which points are close to that plane.
    threshold = 0.03 # จับระยะpoint ใกล้ๆ ถึง plane
    distances = np.abs(points @ normal + d)
    inlier_indices = np.where(distances < threshold)[0]
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # 7. Create inlier and outlier point clouds
    inlier_cloud = pcd.select_by_index(inlier_indices)
    outlier_cloud = pcd.select_by_index(inlier_indices, invert=True)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # color inliers red

    # 8. Visualize
    o3d.visualization.draw_geometries(
        [inlier_cloud, outlier_cloud],
        zoom=0.8,
        front=[-0.4999, -0.1659, -0.8499],
        lookat=[2.1813, 2.0619, 2.0999],
        up=[0.1204, -0.9852, 0.1215],
    )
    return normal,d
def visualization_draw_geometries(name,list_pcd):
    o3d.visualization.draw_geometries(
        list_pcd,f'{name}',1080,720)
    
def create_marker_at(point_3d, radius=0.01, color=(1, 1, 1)):
    """
    Create a small sphere at 'point_3d' with specified radius & color (RGB).
    # """
    # print(f'Type_point:{type(point_3d)} , point:{point_3d}')
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(point_3d)
    sphere_pcd = sphere.sample_points_uniformly(number_of_points=2000)
    sphere_pcd.paint_uniform_color(color)
    return sphere_pcd
def draw_box(point_list):
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(point_list),
    lines=o3d.utility.Vector2iVector(lines),
    )
#---------------------------------cam Yolo---------------------------------#
def detect_objects(frame, model):
    results = model(frame)
    return results
# def draw_detections(frame, depth_frame, results, pcd, intrin,show_pcd,rotation_matrix):
#     marker = None
#     pt_pcd = None
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = box.conf[0].item()
#             class_id = int(box.cls[0])

#             if confidence > 0.6 and class_id == 56: 
#                 # Draw the bounding box on the RGB frame
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                
#     return max(y1,y2)
def initialize_kinect(config_path):
    kinectsensor = o3d.io.AzureKinectSensor(o3d.io.read_azure_kinect_sensor_config(config_path))
    
    if not kinectsensor.connect(0):
        print("Failed to connect to Azure Kinect! Check if the device is plugged in.")
        exit()
    
    print("Azure Kinect connected successfully!")
    return kinectsensor


# Capture frame from Kinect
def capture_frame(kinectsensor):
    frame_data = kinectsensor.capture_frame(True)
    
    if frame_data is None or frame_data.color is None or frame_data.depth is None:
        # print("Error: Kinect did not return a valid frame. Retrying...")
        # time.sleep(1)
        return None, None, None, None

    raw_depth = np.asarray(frame_data.depth).astype(np.float32)  # Convert depth to float
    raw_color = np.asarray(frame_data.color)  # Get color data

    # Ensure valid depth values
    valid_depth = raw_depth[raw_depth > 0]  
    if valid_depth.size == 0:
        return None, None, None, None

    # Dynamically adjust min/max depth based on real-time data
    min_depth = np.percentile(valid_depth, 2)  # 2nd percentile to remove extreme noise
    max_depth = np.percentile(valid_depth, 98)  # 98th percentile to exclude outliers
    # print(f"Dynamic Min Depth: {min_depth:.1f} mm | Max Depth: {max_depth:.1f} mm")

    # Normalize depth for OpenCV visualization
    depth_visual = np.clip(raw_depth, min_depth, max_depth)
    depth_visual = ((depth_visual - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

    # Convert color to RGB format if necessary
    if raw_color.shape[-1] == 4:
        frame = cv2.cvtColor(raw_color, cv2.COLOR_BGRA2RGB)
    elif raw_color.shape[-1] == 3:
        frame = cv2.cvtColor(raw_color, cv2.COLOR_BGR2RGB)

    # Convert depth to meters for Open3D processing
    raw_depth_meters = raw_depth / 1000.0

    

    return frame, depth_colormap, raw_color, raw_depth_meters

#------------------------------------------------------------------------------------#
def depth2point(intrinsics,v,d):
    focalx, focaly = intrinsics.get_focal_length()         
    cx, cy = intrinsics.get_principal_point()
    z = d
    y = (v - cy) * z / focaly
    return y

       
if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud(r"C:\Users\pan\Desktop\My_o3d_code\Pcd_to_test.ply")  # or .pcd, .xyz, etc.
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().translate((0, 0, 0), relative=False)
    # o3d.visualization.draw_geometries([pcd,mesh])
    flip_transform_4x4 = np.array([
    [1,  0,  0, 0],
    [0, 1,  0, 0],
    [0,  0, 1, 0],
    [0,  0,  0, 1]
    ])
    normal_vec,d_plane = create_plane_by_pick(pcd)
    print(normal_vec)
    rotat_pcd,rotate_mat = align_floor_to_xz_plane(pcd,normal_vec,d_plane)
    print('rotate_mat',rotate_mat)
    
    
    #------------------------------ รุ่นพี่  ----------------------------------------
    # rotate_mat= np.array([[ 0.99935915 , 0.02097128  ,0.02900854 ],
    #             [ 0.02097128,  0.31373307 ,-0.9492796  ],
    #                 [-0.02900854 , 0.9492796  , 0.31309222]])
    
    # print('rotate_mat',rotate_mat,type(rotate_mat))
    # pcd_out = pcd.rotate(rotate_mat, center=(0, 0, 0))
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().translate((-0.17, -0.8, -2.1), relative=False)
    # o3d.visualization.draw_geometries([pcd_out,mesh])
    #-----------------------------------------------------------------------------
    # o3d.visualization.draw_geometries([pcd,mesh])
    # o3d.visualization.draw_geometries([rotat_pcd,mesh])
    # picl_org=pick_point_from_pcd(pcd)
    # o1=picl_org[0]
    # o2=picl_org[1]
    # o3=picl_org[2]
    # o4=picl_org[3]
    
    pickky=pick_point_from_pcd(rotat_pcd)
    o1=pickky[0]
    o2=pickky[1]
    o3=pickky[2]
    o4=pickky[3]
    
    # Extract (x, y, z)
    # print("o1 after flip:", o1_flipped)
    # Step 3: Apply the rotation matrix (3x3)
    # print(f'rotate_ma: {rotate_mat} \n o1={o1}')
    # o1_rotated = rotate_mat @ o1
    # o2_rotated = rotate_mat @ o2
    # o3_rotated = rotate_mat @ o3
    # o4_rotated = rotate_mat @ o4
    # print("o1 final rotated position:", o1_rotated)
    # marker1=create_marker_at(np.asarray(o1_rotated),radius=0.04, color=(1, 0,0 ))
    # marker2=create_marker_at(np.asarray(o2_rotated),radius=0.04, color=(1, 0,0 ))
    # marker3=create_marker_at(np.asarray(o3_rotated),radius=0.04, color=(1, 0,0 ))
    # marker4=create_marker_at(np.asarray(o4_rotated),radius=0.04, color=(1, 0,0 ))
    # o3d.visualization.draw_geometries([rotat_pcd,marker1,marker2,marker3])
    # o1_rotated_project = o1_rotated
    # list_newpoint_ori=[o1,o3,o4,o2]
    
    list_newpoint=[o1,o2,o3,o4]
    # c=0
    for new_point in range(4):
        # c=c+1
        print(new_point)
        new = list_newpoint[new_point].copy()  # Create a new copy of the list
        new[1] = -0.3
        list_newpoint.append(new)  # Append the modified copy
        # print(list_newpoint)
        # print(c)
#     points = [
#     [0, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [1, 1, 0],
#     [0, 0, 1],
#     [1, 0, 1],
#     [0, 1, 1],
#     [1, 1, 1],
# ]
    # pick_point_from_pcd(rotat_pcd)
    # print("rotate_matrix",rotate_mat)
    print("Let's draw a cubic using o3d.geometry.LineSet.")
    print('list_newpoint',list_newpoint)
    points = list_newpoint
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([rotat_pcd,line_set,mesh])
    