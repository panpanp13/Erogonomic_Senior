import cv2
import torch
import numpy as np
from ultralytics import YOLO
import open3d as o3d
import time
import logging
import mediapipe as mp
from Test_fun.Test_all import *

def save_point_cloud(pcd, filename="output.ply"):
    """
    Saves the given point cloud to a PLY file.

    Parameters:
    -----------
    pcd : open3d.geometry.PointCloud
        The point cloud object to save.
    filename : str
        The output filename (should end in .ply).
    
    Returns:
    --------
    None
    """
    # Ensure the point cloud is valid
    if not isinstance(pcd, o3d.geometry.PointCloud):
        print("Error: Input is not a valid Open3D point cloud object.")
        return

    if len(pcd.points) == 0:
        print("Warning: Point cloud is empty. No file will be saved.")
        return

    # Save the point cloud to a PLY file
    try:
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Point cloud successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving point cloud: {e}")

def initialize_yolo(model_path="yolov8n.pt"):
    return YOLO(model_path)


# Initialize Kinect
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


# Run YOLO detection
def detect_objects(frame, model):
    results = model(frame)
    return results

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

            if confidence > 0.4 and class_id == 39:  # Suppose '39' = "cup"
                # Draw the bounding box on the RGB frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                center_x, center_y = (x1 + x2)//2, (y1 + y2)//2
                # Depth in mm
                depth_mm = int(depth_frame[center_y, center_x]*1000)

                # Draw a red dot in the RGB image
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Show label in RGB with pixel coords + depth
                label = f"Cup: ({center_x}, {center_y}, {depth_mm} mm)"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # --- Convert (u, v, depth_mm) to 3D in PCD coords ---
                pt_cam = pixel_to_3d(center_x, center_y, depth_mm, intrin)
                if pt_cam is not None:
                    pt_pcd = apply_transform(pt_cam, FLIP_TRANSFORM)
                    pt_pcd =  rotation_matrix@pt_pcd
                    print("3D coordinate in PCD space:", pt_pcd)

                    # Create a small marker (red sphere) at that 3D location
                    marker = create_marker_at(pt_pcd, radius=0.03, color=(0, 0, 0))
                    if show_pcd =="on":
                        pass
                        # print(f'object_ Po 3D : {pt_pcd}')
                        # o3d.visualization.draw_geometries([pcd, marker])

                    # Show the point cloud + marker
                    # WARNING: This will open a new O3D window each frame
                    # So typically you'd do this once or in a dedicated Visualizer
                    

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
    # flip Z to y 
    # pcd.transform([[1, 0,  0, 0], 
    #                [0, 0, 1, 0], 
    #                [0, 1, 0, 0], 
    #                [0, 0,  0, 1]])

    return pcd, rgbd_image
#-------------------convert 2D point to 3D-----------------
def pixel_to_3d(u, v, depth_mm, intrinsics):
    fx, fy = intrinsics.get_focal_length()         # e.g. (fx, fy)
    cx, cy = intrinsics.get_principal_point()      # e.g. (cx, cy)
    print('u',u)
    print('v',v)
    z_m = depth_mm / 1000.0  # Convert mm to meters
    if z_m <= 0:
        return None  # Invalid or zero depth

    X = (u - cx) * z_m / fx
    Y = (v - cy) * z_m / fy
    Z = z_m
    # print(np.array([X, Y, Z], dtype=float))
    return np.array([X, Y, Z], dtype=float)
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


    
# flip z to y 
# FLIP_TRANSFORM = np.array([
#     [1,  0,  0,  0],
#     [0, 0,  1,  0],
#     [0,  1, 0,  0],
#     [0,  0,  0,  1]
# ], dtype=float)
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
def cal_point(pt_pcd,pickpoint):
  #check X
  P1={'P1':[pt_pcd[0]-pickpoint[0][0],pt_pcd[1]-pickpoint[0][1]]}
  print(P1.get('P1'))
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
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name="Danger")
                    vis.add_geometry(pcd)
                    vis.run()
def depth2point(intrinsics,u,v,d):
    focalx, focaly = intrinsics.get_focal_length()         
    cx, cy = intrinsics.get_principal_point()
    z = d
    x = (u - cx) * z / focalx
    y = (v - cy) * z / focaly
    return [x,y,z]

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
        pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5,model_complexity=2)
        results = pose.process(frame)
        collect_landmark =[]
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            h, w, _ = frame.shape
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_cx = int(landmark.x * w)
                landmark_cy = int(landmark.y * h)
                print(landmark_cx)
                print(landmark_cy)
                if 0 <= landmark_cx < w and 0 <= landmark_cy < h:
                    depth = depth_frame[landmark_cy, landmark_cx]   # Valid depth
                else:
                    depth = 0
                    print(f"Warning: Landmark {idx} out of bounds: cx={landmark_cx}, cy={landmark_cy}")
                print(depth)
                cv2.circle(frame, (landmark_cx, landmark_cy), radius=5, color=(0, 255, 0), thickness=-1)  # Green dot
                cv2.putText(frame, f"{idx}", (landmark_cx + 10, landmark_cy - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)    
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
                pose_ = depth2point(intrinsics,landmark_cx, landmark_cy, depth)
                pose_.append(landmark.visibility)
                land_mark_rula.append(pose_)
                collect_landmark.append({
                    'x': float(f'{pose_[0]:.2f}'),
                    'y': float(f'{pose_[1]:.2f}'),
                    'z': float(f'{pose_[2]:.2f}')
                })
                coord_text = f"{idx}: ({pose_[0]:.2f}, {pose_[1]:.2f}, {pose_[2]:.2f}m)"
                cv2.putText(frame, coord_text, (landmark_cx + 10, landmark_cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    # print(f"land_mark_rula : {land_mark_rula}")
                    # coordinate_list.append({
                    #     # 'landmark_id': idx, #ไม่ใส่ index ง่ายกว่า
                    #     'x_real': real_x,
                    #     'y_real': real_y,
                    #     'z_real': real_z
                    # })
    return frame,collect_landmark



def main():
    config_path = r'C:/Users/pan/Desktop/pyk4a/Depth_map/config.json'
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    # Initialize Kinect and YOLO
    kinectsensor = initialize_kinect(config_path)
    model = initialize_yolo()
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
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="Point Cloud")
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().translate((0,0,0), relative=False)
    while True:
        # print('list_ceterpcd',list_ceterpcd)
        # Capture frame
        frame, depth_frame, raw_color, raw_depth = capture_frame(kinectsensor)
        if raw_color is not None and raw_depth is not None: 
            media_frame,coordinate_list = media_pipe(intrin,frame,raw_depth)
            if point_cloud == False:
                pcd, rgbd_image = create_point_cloud(raw_color, raw_depth)
            # visualization_draw_geometries([pcd,mesh])
            if rotate is False:
                normal_vec,d_plane=create_plane_by_pick(pcd)
                pcd_out, rotation_matrix=align_floor_to_xz_plane(pcd,normal_vec,d_plane)
                # check one time is that plane aline
                visualization_draw_geometries([pcd_out,mesh])
                rotate = True
            else:
                rotation_matrix=rotation_matrix
                pcd_out=pcd.rotate(rotation_matrix, center=(0, 0, 0))
                # pcd_out, rotation_matrix=align_floor_to_xz_plane(pcd,normal_vec,d_plane)
            # save_point_cloud(pcd,"Pcd_to_test.ply")
            # print(f'coundt_rt : {coundt_rt} , rotation_matrix : {rotation_matrix}')
            
            
            
        if frame is None or depth_frame is None:
            continue  # Skip iteration if no valid frame
        
        # Run YOLO Detection
        results = detect_objects(frame, model)
        if raw_color is not None and raw_depth is not None:
            frame,marker,pt_pcd =draw_detections(frame, raw_depth, results, pcd, intrin,show_pcd,rotation_matrix)
            if pt_pcd is not None:
                if len(list_ceterpcd) == 0:
                    list_ceterpcd = pt_pcd
                else :
                    distance = list_ceterpcd - pt_pcd
                    list_ceterpcd= pt_pcd
                    
            # print(type(pcd),f'pcd: {pcd}')
            # print(type(marker),f'Marker: {marker}')
            if marker is not None:
                # visualization_draw_geometries([pcd,marker,mesh])
                # visualization_draw_geometries([pcd_out,marker,mesh])
            #     print(type(pcd),f'pcd: {pcd}')
                # print(type(marker),f'Marker: {marker}')
                # picked_points = pick_point_from_pcd(pcd,marker)
                pcb=combine_sphere__pcd(pcd_out,marker)
                if find_ceter == False:
                    A=pick_point_from_pcd(pcb)
                    old_pick = A
                    find_ceter = True
                else:
                    # print('---------------------------------')
                    # print('ptc_point',pt_pcd)
                    # print('Distance',distance)
                    # print('old_pick[0]',old_pick[0])
                    # print('old_pick[1]',old_pick[1])
                    # print('old_pick[2]',old_pick[2])
                    # print('old_pick[3]',old_pick[3])
                    # print('---------------------------------')
                    old_pick[0]=old_pick[0] - distance
                    old_pick[1]=old_pick[1] - distance
                    old_pick[2]=old_pick[2] - distance
                    old_pick[3]=old_pick[3] - distance
                    # print('---------------------------------')
                    # print('ptc_point',pt_pcd)
                    # print('Distance',distance)
                    # print('old_pick[0]',old_pick[0])
                    # print('old_pick[1]',old_pick[1])
                    # print('old_pick[2]',old_pick[2])
                    # print('old_pick[3]',old_pick[3])
                    # print('---------------------------------')
                    o1 = old_pick[0]
                    o2 = old_pick[1]
                    o3=old_pick[2]
                    o4=old_pick[3]
                    #-----------------------------------Vis marker----------------------------------------------------------
                    marker1=create_marker_at(np.asarray(o1),radius=0.02, color=(1, 0, 0))
                    marker2 =create_marker_at(np.asarray(o2),radius=0.02, color=(1, 1, 0))
                    marker3=create_marker_at(np.asarray(o3),radius=0.02, color=(1, 0, 1))
                    marker4=create_marker_at(np.asarray(o4),radius=0.02, color=(1, 0, 0.7))
                    marker5=create_marker_at(np.asarray([0,0,0]),radius=0.02, color=(1, 1, 1))
                    # marker5=create_marker_at(np.asarray([0,0,0]),radius=0.03, color=(1, 0, 0.7))
                    pca=combine_sphere__pcd(pcb,marker1)
                    pcf=combine_sphere__pcd(pca,marker2)
                    pck=combine_sphere__pcd(pcf,marker3)
                    pcg=combine_sphere__pcd(pck,marker4)
                    pcd_out=pcg
                    if coordinate_list != []:
                         check_in_zone(old_pick,coordinate_list,rotation_matrix,pcg)
                    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().translate((-0.59184,-0.74272,-2.312), relative=False)
                    # pch=combine_sphere__pcd(pcg,marker5)
                    # o3d.visualization.draw_geometries([pcg,mesh])
                    # pcm=combine_sphere__pcd(pcg,marker5)
                    # o3d.visualization.draw_geometries([pcm,mesh])
                    # marker1=create_marker_at(apply_transform(old_pick[0], FLIP_TRANSFORM),radius=0.01, color=(1, 1, 1))
                    # print('marker1',marker1)
                    # pcd=combine_sphere__pcd(pcd,create_marker_at(old_pick[0],radius=0.01, color=(1, 1, 1)))
                    # # o3d.visualization.draw_geometries([create_marker_at(old_pick[0],radius=0.01, color=(1, 1, 1))])
                    # pcd=combine_sphere__pcd(pcd,create_marker_at(old_pick[1],radius=0.01, color=(1, 1, 1)))
                    # pcd=combine_sphere__pcd(pcd,create_marker_at(old_pick[2],radius=0.01, color=(1, 1, 1)))
                    # pcd=combine_sphere__pcd(pcd,create_marker_at(old_pick[3],radius=0.01, color=(1, 1, 1)))
                    
                # max_col_0,max_col_1,min_col_0,min_col_1=check_in_zone(old_pick)
                # print(f'max_col_0 {max_col_0}',f'max_col_1 {max_col_1}',f'min_col_0 {min_col_0}',f'min_col_1 {min_col_1}')
        coundt_rt= coundt_rt+1
        # if frame is not None and depth_frame is not None:
        #    media_frame,coordinate_list = media_pipe(intrin,frame,raw_depth)
        #    print('coordinate_list',coordinate_list)
        # if coordinate_list != []:
        #     check_in_zone(old_pick,coordinate_list,rotation_matrix,pcd_out)                   
                
            # **Pick a point from the point cloud**
            # print("Press any key in the Open3D window and select points using the mouse.")
            # if marker is not None:
            #     picked_points = pick_point_from_pcd(pcd,marker)
            #     if picked_points is not None:
            #         print(f"Selected 3D points: {picked_points}")
                # o3d.visualization.draw_geometries([marker])
        # print('center',pt_pcd)  
        # print('old_pick',old_pick)
        # print('type olde_pick',type(old_pick))
        # print('distance',distance)
        # Show RGB and Depth frames
        cv2.imshow("media_frame", media_frame)
        # print(coordinate_list)
        # cv2.imshow("Cup Detection", frame)
        # cv2.imshow('Depth', depth_frame)
        # print(f"Frame size: {raw_color.shape}")  # Shape of the RGB image
        # print(f"Depth frame size: {raw_depth.shape}")  # Shape of the depth map

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release Kinect
    kinectsensor.disconnect()
    cv2.destroyAllWindows()


# Run the script
if __name__ == "__main__":
    main()
    # config_path = r'C:/Users/pan/Desktop/pyk4a/Depth_map/config.json'
    # kinectsensor = initialize_kinect(config_path)
    # intrin_path = r'C:/Users/pan/Desktop/pyk4a/Depth_map/intrin.json'
    # intrin = o3d.io.read_pinhole_camera_intrinsic(intrin_path)
    # while True:
    #     frame, coordinate_list = media_pipe(intrin, kinectsensor)

    #     if frame is not None and coordinate_list is not None:
    #         for landmark  in coordinate_list:
    #             print(float(landmark['x_real'][0]))
    #         print('coordinate_list:', coordinate_list)
    #         cv2.imshow('MediaPipe Frame', frame)

    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break

    # cv2.destroyAllWindows()
         
        
        
