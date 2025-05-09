import cv2
import torch
import numpy as np
from ultralytics import YOLO
import open3d as o3d
import time
import logging
import mediapipe as mp
from Test_all import *
from vispy import app

# Capture frame from Kinect
def capture_frame(kinectsensor):
    frame_data = kinectsensor.capture_frame(True)
    
    if frame_data is None or frame_data.color is None or frame_data.depth is None:
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


# Run YOLO detection
def detect_objects(frame, model):
    results_yolo = model(frame)
    return results_yolo

def show_point_cloud(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Viewer", width=1280, height=720) # Create a window
    vis.add_geometry(pcd)

    # Get window size
    window_width = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic.width
    window_height = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic.height

    print(f"Point Cloud Window Size: {window_width} x {window_height}")

    vis.run()  # Run the visualization
    vis.destroy_window()
    
#----------------- Detect -----------------
def draw_detections(frame, depth_frame, results, pcd, intrin,show_pcd,rotation_matrix):
    marker = None
    pt_pcd = None
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])

            if confidence > 0.6 and class_id == 56:  # 56 chair 0 person
                # Draw the bounding box on the RGB frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                center_x, center_y = (x1 + x2)//2, (y1 + y2)//2
                # Depth in mm
                depth_mm = int(depth_frame[center_y, center_x]*1000)

                # Draw a red dot in the RGB image
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Show label in RGB with pixel coords + depth
                label = f"Robot: ({center_x}, {center_y}, {depth_mm} mm)"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # --- Convert (u, v, depth_mm) to 3D in PCD coords ---
                pt_cam = pixel_to_3d(center_x, center_y, depth_mm, intrin)
                if pt_cam is not None:
                    pt_pcd = apply_transform(pt_cam, FLIP_TRANSFORM)
                    pt_pcd =  rotation_matrix@pt_pcd
                    # print("3D coordinate in PCD space:", pt_pcd)

                    # Create a small marker (red sphere) at that 3D location
                    marker = create_marker_at(pt_pcd, radius=0.05, color=(1, 0, 0))
                    if show_pcd =="on":
                        pass
                    

    return frame,marker,pt_pcd

#-----------------Create Point Cloud from RGBD Image-----------------
def create_point_cloud(color_frame, depth_frame):
    intrin_path = r'C:/Users/pan/Desktop/pyk4a/Depth_map/intrin.json'
    intrin = o3d.io.read_pinhole_camera_intrinsic(intrin_path)

    # Convert depth to meters for Open3D
    depth_meters = o3d.geometry.Image((depth_frame * 1000.0).astype(np.uint16))  # Convert meters to mm
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_frame), depth_meters, depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrin)

    # Flip to Open3D coordinate system
    pcd.transform([[1, 0,  0, 0], 
                   [0, -1, 0, 0], 
                   [0, 0, -1, 0], 
                   [0, 0,  0, 1]])
    return pcd, rgbd_image
#-------------------convert 2D point to 3D-----------------
def pixel_to_3d(u, v, depth_mm, intrinsics):
    fx, fy = intrinsics.get_focal_length()         # e.g. (fx, fy)
    cx, cy = intrinsics.get_principal_point()      # e.g. (cx, cy)
    # print('u',u)
    # print('v',v)
    z_m = depth_mm / 1000.0  # Convert mm to meters
    if z_m <= 0:
        return None  # Invalid or zero depth

    X = (u - cx) * z_m / fx
    Y = (v - cy) * z_m / fy
    Z = z_m
    # print(np.array([X, Y, Z], dtype=float))
    return np.array([X, Y, Z], dtype=float)
def create_marker_at(point_3d, radius=0.05, color=(1, 1, 1)):
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

FLIP_TRANSFORM = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
], dtype=float)

def apply_transform(point_3d, transform_4x4):
    """
    Apply a 4x4 transform to a 3D point. Returns the transformed 3D coordinate.
    """
    p_hom = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    p_transformed = transform_4x4 @ p_hom
    return p_transformed[:3]


def combine_sphere__pcd(pcd,sphere_pcd):
    # o3d.visualization.draw_geometries([sphere_pcd])
    combined_pcd = o3d.geometry.PointCloud()

    # stack the points (and colors) from both
    combined_points = np.vstack((pcd.points, sphere_pcd.points))
    combined_colors = np.vstack((pcd.colors, sphere_pcd.colors))

    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    return combined_pcd
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
#   print(P1.get('P1'))
def check_in_zone(old_pick,coordinate_list,rotation_matrix,pcd):
    # print('old_pick',old_pick)
    if old_pick is not 0:
        a_np = old_pick
        # print(old_pick)
        max_col_0 = max(a_np[:, 0])  # Max of first column (a[i][0])
        min_col_0 = min(a_np[:, 0])
        max_col_2 = max(a_np[:, 2])  # Max of second column (a[i][1]
        min_col_2 = min(a_np[:, 2])
        # print(f'max_col_0 {max_col_0} \n',f'max_col_1 {max_col_1} \n',f'min_col_0 {min_col_0} \n',f'min_col_1 {min_col_1}\n')
        # print(coordinate_list)
        if  coordinate_list is not None:
            for member_landmark in coordinate_list:
                # print('member_landmark',member_landmark)
                member_landmark= [member_landmark['x'],member_landmark['y'],member_landmark['z']]
                member_landmark= rotation_matrix @member_landmark
                # print(member_landmark)
                # print(member_landmark)
                X_position = member_landmark[0]
                Z_position = member_landmark[2]
            # print(X_position,Y_position)
                # print(f'X coor : {min_col_0} <= {X_position} <= {max_col_0}')
                # print(f'Z coor: {min_col_2} <= {Z_position} <= {max_col_2}')
                if min_col_0<=X_position<=max_col_0 and min_col_2<= -Z_position <=max_col_2 :
                    print('Danger Now')
                    return True
                    # vis = o3d.visualization.Visualizer()
                    # vis.create_window(window_name="Danger")
                    # vis.add_geometry(pcd)
                    # vis.run()
    return None
# def depth2point(intrinsics,u,v,d):
#     focalx, focaly = intrinsics.get_focal_length()         
#     cx, cy = intrinsics.get_principal_point()
#     z = d
#     x = (u - cx) * z / focalx
#     y = (v - cy) * z / focaly
#     return [x,y,z]
def depth2point(u,v,d):
    intrin_path = r'C:/Users/pan/Desktop/pyk4a/Depth_map/intrin.json' 
    intrinsics = o3d.io.read_pinhole_camera_intrinsic(intrin_path)
    focalx, focaly = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()
    z = d
    x = (u - cx) * z / focalx
    y = (v - cy) * z / focaly
    return [x,y,z]

# def depth2point(intrinsics,u,v,d):
#     focalx, focaly = intrinsics.get_focal_length()         
#     cx, cy = intrinsics.get_principal_point()
#     z = d
#     x = (u - cx) * z / focalx
#     y = (v - cy) * z / focaly
#     return [x,y,z]

def media_pipe(intrinsics,frame,depth_frame):
    land_mark_avg = np.zeros((33,3,5)).tolist()
    land_mark = []
    kp_list=None
    land_mark_rula= None
    if frame is not None:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        coordinate_list = []
        land_mark_rula = []
        kp_list = []
        fx, fy = intrinsics.get_focal_length()         
        cx, cy = intrinsics.get_principal_point() 
        pose = mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.8,model_complexity=2)
        results = pose.process(frame)
        collect_landmark =[]
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            h, w, _ = frame.shape
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_cx = int(landmark.x * w)
                landmark_cy = int(landmark.y * h)
                if 0 <= landmark_cx < w and 0 <= landmark_cy < h:
                    depth = depth_frame[landmark_cy, landmark_cx]   # Valid depth
                else:
                    depth = 0
                cv2.circle(frame, (landmark_cx, landmark_cy), radius=5, color=(0, 255, 0), thickness=-1)  # Green dot
                # cv2.putText(frame, f"{idx}", (landmark_cx + 10, landmark_cy - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)    
                if len(land_mark_avg[idx][0]) >= 5:
                    if depth != 0:
                        land_mark_avg[idx][2].append(depth)
                    else:
                        depth = land_mark_avg[idx][2][-1]  # Fallback to last valid depth

                    land_mark_avg[idx][0].append(landmark_cx)
                    land_mark_avg[idx][1].append(landmark_cy)

                else:
                    land_mark_avg[idx][0].append(landmark_cx)
                    land_mark_avg[idx][1].append(landmark_cy)
                    land_mark_avg[idx][2].append(depth)
                land_mark.append({'x': landmark.x , 'y': landmark.y, 'visibility': landmark.visibility})
                pose_ = depth2point(landmark_cx, landmark_cy, depth)
                pose_.append(landmark.visibility)
                land_mark_rula.append(pose_)
                collect_landmark.append({
                    'x': float(f'{pose_[0]:.2f}'),
                    'y': float(f'{pose_[1]:.2f}'),
                    'z': float(f'{pose_[2]:.2f}')
                })
                coord_text = f"{idx}: ({pose_[0]:.2f}, {pose_[1]:.2f}, {pose_[2]:.2f}m)"
                # cv2.putText(frame, coord_text, (landmark_cx + 10, landmark_cy - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return frame,collect_landmark

def apply_red_overlay(frame, intensity=0.5):
    red_overlay = np.zeros_like(frame, dtype=np.uint8)
    red_overlay[:, :, 2] = 255  # Full red overlay
    return cv2.addWeighted(frame, 1 - intensity, red_overlay, intensity, 0)
def apply_orange_overlay(frame, intensity: float = 0.4):
    orange = (0, 140, 255)          # tweak green channel to taste
    overlay = np.full_like(frame, orange, dtype=np.uint8)

    return cv2.addWeighted(frame, 1 - intensity,
                           overlay, intensity,
                           0)
def clustering(pcd,center):

    # Use the first picked point.
    points = np.asarray(pcd.points)
    print("Picked point:", center)
    
    radius = 0.5  # Adjust as necessary.
    distances = np.linalg.norm(points - center, axis=1)
    local_indices = np.where(distances < radius)[0]
    print(f"Found {local_indices.shape[0]} points within a radius of {radius} from the picked point.")
    
    if local_indices.shape[0] == 0:
        print("No points found in the neighborhood of the picked point.")
        return center, None

    # Get current colors from the main point cloud.
    # If no colors are set, we initialize them to a default gray.
    if len(pcd.colors) != points.shape[0]:
        default_color = np.tile(np.array([[0.8, 0.8, 0.8]]), (points.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(default_color)
    else:
        default_color = np.asarray(pcd.colors)
    
    # Change the color of the local points to red [1.0, 0.0, 0.0].
    colors = np.asarray(pcd.colors)
    colors[local_indices] = [1.0, 0.0, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_cluster=pcd

    # Visualize the updated main point cloud with red neighborhood.
    print("Displaying main point cloud with the local neighborhood colored red...")
    
    return center, local_indices,pcd_cluster
#----------------------------------------gobal-----------------------------
# config_path = r'C:/Users/pan/Desktop/pyk4a/Depth_map/config.json'
# logging.getLogger("ultralytics").setLevel(logging.ERROR)
# # Initialize Kinect and YOLO
# kinectsensor = o3d.io.AzureKinectSensor(o3d.io.read_azure_kinect_sensor_config(config_path))
# kinectsensor.connect(0)
# model_path=r"yolov8n.pt"
# model=YOLO(model_path)
# show_pcd = 'off'
# list_ceterpcd=[]
# find_ceter=False
# old_pick = 0
# distance =0
# intrin_path = r'C:/Users/pan/Desktop/pyk4a/Depth_map/intrin.json' 
# intrin = o3d.io.read_pinhole_camera_intrinsic(intrin_path)
# rotate= False
# coundt_rt =1
# point_cloud = False
# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().translate((0,0,0), relative=False)
# prev_time = time.time()
# fps = 0
# frame_count = 0
# fps_update_time = time.time()
# count_danger=0
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name='point_cloud', width=640, height=480)
# # Initially add a dummy geometry so that the visualizer is ready
# dummy_pcd = o3d.geometry.PointCloud()
# vis.add_geometry(dummy_pcd)

# def update(event):
#     global rotate,rotation_matrix,list_ceterpcd,coundt_rt,find_ceter,old_pick,coordinate_list,mask
#     try:
        
#         frame, depth_frame, raw_color, raw_depth = capture_frame(kinectsensor)
#         if raw_color is not None and raw_depth is not None: 
#             # media_frame,coordinate_list = media_pipe(intrin,frame,raw_depth)
#             if point_cloud == False:
#                 pcd, rgbd_image = create_point_cloud(raw_color, raw_depth)
#                 # visualization_draw_geometries('rgbd_image',[rgbd_image])
                
#         #     # # visualization_draw_geometries([pcd,mesh])
#             if rotate is False:
#                 normal_vec,d_plane=create_plane_by_pick(pcd)
#                 pcd_out, rotation_matrix=align_floor_to_xz_plane(pcd,normal_vec,d_plane)
#         #         # check one time is that plane aline
#                 # visualization_draw_geometries('pcd_out,mesh',[pcd_out,mesh])
#                 rotate = True
#             else:
#                 rotation_matrix=rotation_matrix
#                 pcd_out=pcd.rotate(rotation_matrix, center=(0, 0, 0))  # Skip iteration if no valid frame
        
        
#         results_yolo = detect_objects(frame, model)
#         if raw_color is not None and raw_depth is not None:
#             frame,marker,pt_pcd =draw_detections(frame, raw_depth, results_yolo, pcd, intrin,show_pcd,rotation_matrix)
#             # if marker 
#             if pt_pcd is not None:
#                 if len(list_ceterpcd) == 0:
#                     list_ceterpcd = pt_pcd
#                 else :
#                     distance = list_ceterpcd - pt_pcd
#                     list_ceterpcd= pt_pcd
#                     # print('list_ceterpcd',list_ceterpcd)
                 

#             if marker is not None:
#                 pcd_out = combine_sphere__pcd(pcd_out, marker)
#                 # visualization_draw_geometries('clustering',[pcd_out])
#                 _,local_indices,pcd_out=clustering(pcd_out,pt_pcd)
#                 points = np.asarray(pcd_out.points)
#                 local_indices =points[local_indices]
#                 print('local_indices',local_indices)
#                 pcd_cluster = o3d.geometry.PointCloud()
#                 pcd_cluster.points = o3d.utility.Vector3dVector(local_indices)

#                 # Optional: assign a color to each point (e.g., white)
#                 pcd_cluster.paint_uniform_color([1, 0, 0])
                
#                 # Create a boolean mask where True indicates points to keep
#                 # visualization_draw_geometries('clustering',[pcd_cluster])
#                 # visualization_draw_geometries('clustering',[pcd_cluster,pcd_out])
#                 cluster_xmax,cluster_xmin = max(local_indices[:,0]),min(local_indices[:,0])
#                 cluster_zmax,cluster_zmin = max(local_indices[:,2]),min(local_indices[:,2])
#                 cluster_ymax,cluster_ymin = max(local_indices[:,1]),min(local_indices[:,1])
#                 x_max_index = np.argmax(local_indices[:, 0])
#                 x_min_index = np.argmin(local_indices[:, 0])

#                 z_max_index = np.argmax(local_indices[:, 2])
#                 z_min_index = np.argmin(local_indices[:, 2])

#                 y_max_index = np.argmax(local_indices[:, 1])
#                 y_min_index = np.argmin(local_indices[:, 1])
#                 # print(f"Cluster X max: {cluster_xmax} at index {x_max_index} ")
#                 # print(f"Cluster X min: {cluster_xmin} at index {x_min_index}")
#                 # print(f"Cluster Z max: {cluster_zmax} at index {z_max_index}")
#                 # print(f"Cluster Z min: {cluster_zmin} at index {z_min_index}")
#                 # print(f"Cluster Y max: {cluster_ymax} at index {y_max_index}")
#                 # print(f"Cluster Y max: {cluster_ymin} at index {y_min_index}")
#                 Traio_area=0.1
#                 cluster_xmax,cluster_zmax,cluster_ymax=cluster_xmax+0.1,cluster_zmax+0.1,cluster_ymax+0.1
#                 cluster_xmin,cluster_ymin,cluster_zmin=cluster_xmin,cluster_ymin-0.1,cluster_zmin-0.2
#                 point_1 = np.array([cluster_xmin,cluster_ymin,cluster_zmax])
#                 point_2 = np.array([cluster_xmin,cluster_ymin,cluster_zmin])
#                 point_3 = np.array([cluster_xmax,cluster_ymin,cluster_zmax])
#                 point_4 = np.array([cluster_xmax,cluster_ymin,cluster_zmin])
#                 point_5 = np.array([cluster_xmin,cluster_ymax,cluster_zmax])
#                 point_6 = np.array([cluster_xmin,cluster_ymax,cluster_zmin])
#                 point_7 = np.array([cluster_xmax,cluster_ymax,cluster_zmax])
#                 point_8 = np.array([cluster_xmax,cluster_ymax,cluster_zmin])
#                 # print('point_1',point_1)
#                 # print('point_2',point_2)
#                 # print('point_3',point_3)
#                 # print('point_4',point_4)
#                 # print('point_5',point_5)
#                 # print('point_6',point_6)
#                 # print('point_7',point_7)
#                 # print('point_8',point_8)
#                 points=[point_1,
#                         point_2,
#                         point_3,
#                         point_4,
#                         point_5,
#                         point_6,
#                         point_7,
#                         point_8]
#                 lines = [
#                     [0, 1],
#                     [0, 2],
#                     [1, 3],
#                     [2, 3],
#                     [4, 5],
#                     [4, 6],
#                     [5, 7],
#                     [6, 7],
#                     [0, 4],
#                     [1, 5],
#                     [2, 6],
#                     [3, 7],
#                 ]
#                 colors = [[1, 0, 0] for i in range(len(lines))]
#                 line_set = o3d.geometry.LineSet(
#                     points=o3d.utility.Vector3dVector(points),
#                     lines=o3d.utility.Vector2iVector(lines),
#                 )
#                 line_set.colors = o3d.utility.Vector3dVector(colors)
#                 ctr = vis.get_view_control()
#                 cam_params = ctr.convert_to_pinhole_camera_parameters()
#                 vis.clear_geometries()
#                 vis.add_geometry(pcd_out)
#                 vis.add_geometry(line_set)
                
#                 # Restore camera parameters
#                 ctr.convert_from_pinhole_camera_parameters(cam_params)
                
#                 vis.update_geometry(pcd_out)
#                 vis.poll_events()
#                 vis.update_renderer()   
#                 # if  coordinate_list is not None:
#                 #     print('coordinate_list',coordinate_list)
#                         # for member_landmark in coordinate_list:
#                         #     # print('member_landmark',member_landmark)
#                         #     member_landmark= [member_landmark['x'],member_landmark['y'],member_landmark['z']]
#                         #     member_landmark= rotation_matrix @member_landmark
#                         #     # print(member_landmark)
#                         #     # print(member_landmark)
#                         #     X_position = member_landmark[0]
#                         #     Z_position = member_landmark[2]

#                 #     if stage_danger is True or 0<count_danger<=6:
#                 #         media_frame = apply_red_overlay(media_frame, intensity=0.7)
#                 #         cv2.putText(media_frame, f"Danger", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(255,0, 0), thickness=3)
#                 #         count_danger= count_danger+1
#                 #     if count_danger>0:
#                 #         count_danger = count_danger+1
#                 #     if count_danger ==6:
#                 #         count_danger=0
                        
#         # coundt_rt= coundt_rt+1
#         # Show RGB and Depth frames
#         # cv2.imshow("Depth",depth_frame )
#         cv2.imshow("ํYolo_frame", frame)
#         # cv2.imshow("media_frame", media_frame)
#         cv2.imshow('Depth', depth_frame)
#         # Break on 'q' key
#     except Exception as e:
#         print(f"Error in update: {e}")
# timer = app.Timer(interval=1/30)
# timer.connect(update)
# timer.start()       

def main():
    config_path = r'C:/Users/pan/Desktop/pyk4a/Depth_map/config.json'
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    # Initialize Kinect and YOLO
    kinectsensor = o3d.io.AzureKinectSensor(o3d.io.read_azure_kinect_sensor_config(config_path))
    kinectsensor.connect(0)
    model_path=r"yolov8n.pt"
    model=YOLO(model_path)

    show_pcd = 'off'
    list_ceterpcd=[]
    find_ceter=False
    old_pick = 0
    distance =0
    intrin_path = r'C:/Users/pan/Desktop/pyk4a/Depth_map/intrin.json' 
    intrin = o3d.io.read_pinhole_camera_intrinsic(intrin_path)
    rotate= False
    coundt_rt =1
    point_cloud = False
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().translate((0,0,0), relative=False)
    prev_time = time.time()
    fps = 0
    frame_count = 0
    fps_update_time = time.time()
    count_danger=0
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='point_cloud', width=640, height=480)
    # Initially add a dummy geometry so that the visualizer is ready
    dummy_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(dummy_pcd)

    while True:

        frame, depth_frame, raw_color, raw_depth = capture_frame(kinectsensor)
        if raw_color is not None and raw_depth is not None: 
            # media_frame,coordinate_list = media_pipe(intrin,frame,raw_depth)
            if point_cloud == False:
                pcd, rgbd_image = create_point_cloud(raw_color, raw_depth)
                
        #     # # visualization_draw_geometries([pcd,mesh])
            if rotate is False:
                normal_vec,d_plane=create_plane_by_pick(pcd)
                pcd_out, rotation_matrix=align_floor_to_xz_plane(pcd,normal_vec,d_plane)
        #         # check one time is that plane aline
                visualization_draw_geometries('pcd_out,mesh',[pcd_out,mesh])
                rotate = True
            else:
                rotation_matrix=rotation_matrix
                pcd_out=pcd.rotate(rotation_matrix, center=(0, 0, 0))
        if frame is None or depth_frame is None:
            continue  # Skip iteration if no valid frame
        
        
        results_yolo = detect_objects(frame, model)
        if raw_color is not None and raw_depth is not None:
            frame,marker,pt_pcd =draw_detections(frame, raw_depth, results_yolo, pcd, intrin,show_pcd,rotation_matrix)
            # if marker 
            if pt_pcd is not None:
                if len(list_ceterpcd) == 0:
                    list_ceterpcd = pt_pcd
                else :
                    distance = list_ceterpcd - pt_pcd
                    list_ceterpcd= pt_pcd
                    print('list_ceterpcd',list_ceterpcd)
                 

            if marker is not None:
                pcd_out = combine_sphere__pcd(pcd_out, marker)
                _,local_indices,pcd_out=clustering(pcd_out,pt_pcd)
                points = np.asarray(pcd_out.points)
                # print('local_indices',points[local_indices])
                local_indices =points[local_indices]
                cluster_xmax,cluster_xmin = max(local_indices[:,0]),min(local_indices[:,0])
                cluster_zmax,cluster_zmin = max(local_indices[:,2]),min(local_indices[:,2])
                cluster_ymax = max(local_indices[:,1])
                print(f'cluster_xmax:{cluster_xmax}\ncluster_xmin{cluster_xmin}')
                print(f'cluster_zmax:{cluster_zmax}\ncluster_xmin{cluster_zmin}')
                print(f'cluster_ymax:{cluster_ymax}')
                if not find_ceter:  # Pick point only once
                    old_pick = pick_point_from_pcd(pcd_out)
                    useold=[old_pick[0],old_pick[1],old_pick[2],old_pick[3]]
                    for new_point in range(4):
                        new = useold[new_point].copy()  # Create a new copy of the list
                        new[1] = -0.3
                        useold.append(new)
                    points = useold
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
                    o3d.visualization.draw_geometries([pcd_out,line_set])
            
                    find_ceter = True
                else:
                    old_pick -= distance  # Use NumPy subtraction
                    useold=[old_pick[0],old_pick[1],old_pick[2],old_pick[3]]
                    for new_point in range(4):
                        new = useold[new_point].copy()  # Create a new copy of the list
                        new[1] = -0.3
                        useold.append(new)
                    points = useold
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
                # Create markers in a loop
                colors = [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 0, 0.7)]
                markers = [create_marker_at(np.asarray(pt), radius=0.02, color=c) for pt, c in zip(old_pick, colors)]
                
                # Combine all markers in one go
                marker_pcd = o3d.geometry.PointCloud()
                for m in markers:
                    marker_pcd = combine_sphere__pcd(marker_pcd, m)

                pcd_out = combine_sphere__pcd(pcd_out, marker_pcd)
        
                ctr = vis.get_view_control()
                cam_params = ctr.convert_to_pinhole_camera_parameters()
                vis.clear_geometries()
                vis.add_geometry(pcd_out)
                vis.add_geometry(line_set)
                
                # Restore camera parameters
                ctr.convert_from_pinhole_camera_parameters(cam_params)
                
                vis.update_geometry(pcd_out)
                vis.poll_events()
                vis.update_renderer()   
                # if coordinate_list != []:
                #     stage_danger=check_in_zone(old_pick,coordinate_list,rotation_matrix,pcd_out)

                #     if stage_danger is True or 0<count_danger<=6:
                #         media_frame = apply_red_overlay(media_frame, intensity=0.7)
                #         cv2.putText(media_frame, f"Danger", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(255,0, 0), thickness=3)
                #         count_danger= count_danger+1
                #     if count_danger>0:
                #         count_danger = count_danger+1
                #     if count_danger ==6:
                #         count_danger=0
                        
        coundt_rt= coundt_rt+1
        # Show RGB and Depth frames
        cv2.imshow("Depth",depth_frame )
 
        cv2.imshow("RGB", frame)
        # cv2.imshow('Depth', depth_frame)
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#     # Release Kinect
#     kinectsensor.disconnect()
#     cv2.destroyAllWindows()


# Run the script
if __name__ == "__main__":
    main()
    # app.run()

        
        
