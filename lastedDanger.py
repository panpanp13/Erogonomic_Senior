import cv2
import torch
import numpy as np
from ultralytics import YOLO
import open3d as o3d
import time
import logging



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
        print("Error: Kinect did not return a valid frame. Retrying...")
        time.sleep(1)
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
def draw_detections(frame, depth_frame, results, pcd, intrin,show_pcd):
    marker = None
    pt_pcd = None
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])

            if confidence > 0.6 and class_id == 39:  # Suppose '39' = "cup"
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

    return pcd, rgbd_image
#-------------------convert 2D point to 3D-----------------
def pixel_to_3d(u, v, depth_mm, intrinsics):
    fx, fy = intrinsics.get_focal_length()         # e.g. (fx, fy)
    cx, cy = intrinsics.get_principal_point()      # e.g. (cx, cy)

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
    """
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
  
    while True:
        # print('list_ceterpcd',list_ceterpcd)
        # Capture frame
        frame, depth_frame, raw_color, raw_depth = capture_frame(kinectsensor)
        
        if raw_color is not None and raw_depth is not None: 
            pcd, rgbd_image = create_point_cloud(raw_color, raw_depth)
            intrin_path = r'C:/Users/pan/Desktop/pyk4a/Depth_map/intrin.json'
            intrin = o3d.io.read_pinhole_camera_intrinsic(intrin_path)
            
        if frame is None or depth_frame is None:
            continue  # Skip iteration if no valid frame
        
        # Run YOLO Detection
        results = detect_objects(frame, model)
        if raw_color is not None and raw_depth is not None:
            frame,marker,pt_pcd =draw_detections(frame, raw_depth, results, pcd, intrin,show_pcd)
            if pt_pcd is not None:
                if len(list_ceterpcd) == 0:
                    list_ceterpcd = pt_pcd
                else :
                    distance = list_ceterpcd - pt_pcd
                    list_ceterpcd= pt_pcd
                    
            # print(type(pcd),f'pcd: {pcd}')
            # print(type(marker),f'Marker: {marker}')
            if marker is not None:
            #     print(type(pcd),f'pcd: {pcd}')
                # print(type(marker),f'Marker: {marker}')
                # picked_points = pick_point_from_pcd(pcd,marker)
                pcb=combine_sphere__pcd(pcd,marker)
                if find_ceter == False:
                    A=pick_point_from_pcd(pcb)
                    old_pick = A
                    find_ceter = True
                else:
                    old_pick[0]=old_pick[0] + distance
                    old_pick[1]=old_pick[1] + distance
                    old_pick[2]=old_pick[2] + distance
                    old_pick[3]=old_pick[3] + distance
                    
            
                    
                # print(type(pcb),f'Pcb {len(pcb)}')
                
            # **Pick a point from the point cloud**
            # print("Press any key in the Open3D window and select points using the mouse.")
            # if marker is not None:
            #     picked_points = pick_point_from_pcd(pcd,marker)
            #     if picked_points is not None:
            #         print(f"Selected 3D points: {picked_points}")
                # o3d.visualization.draw_geometries([marker])
        print('center',pt_pcd)  
        print('old_pick',old_pick)
        print('distance',distance)
        # Show RGB and Depth frames
        cv2.imshow("Cup Detection", frame)
        cv2.imshow('Depth', depth_frame)
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
