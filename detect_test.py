import cv2
import torch
import numpy as np
from ultralytics import YOLO
import open3d as o3d
import time


    
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
    print(f"Dynamic Min Depth: {min_depth:.1f} mm | Max Depth: {max_depth:.1f} mm")

    # Normalize depth for OpenCV visualization
    depth_visual = np.clip(raw_depth, min_depth, max_depth)
    depth_visual = ((depth_visual - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

    # Convert color to RGB format if necessary
    if raw_color.shape[-1] == 4:
        raw_color = cv2.cvtColor(raw_color, cv2.COLOR_BGRA2RGB)
    elif raw_color.shape[-1] == 3:
        raw_color = cv2.cvtColor(raw_color, cv2.COLOR_BGR2RGB)

    # Convert depth to meters for Open3D processing
    raw_depth_meters = raw_depth / 1000.0  

    return raw_color, depth_colormap, raw_color, raw_depth_meters

# Run YOLO detection
def detect_objects(frame, model):
    results = model(frame)
    return results

# Process detections and visualize
def draw_detections(frame, depth_frame, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            # Assuming "cup" has class_id = 41 (COCO dataset) or your custom ID
            if confidence > 0.3 and class_id == 41:  
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Calculate center of box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Get depth at center of detected object
                object_depth = depth_frame[center_y, center_x]

                # Draw red point at center
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Add label with depth
                label = f"Cup: {center_x,object_depth} mm"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Create Point Cloud from RGBD Image
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

# Main function
def main():
    config_path = r'C:/Users/pan/Desktop/pyk4a/Depth_map/config.json'
    
    # Initialize Kinect and YOLO
    kinectsensor = initialize_kinect(config_path)
    model = initialize_yolo()
  
    

    while True:
        # Capture frame
        frame, depth_frame, raw_color, raw_depth = capture_frame(kinectsensor)
        
        if raw_color is not None and raw_depth is not None: 
            pcd, rgbd_image = create_point_cloud(raw_color, raw_depth)
            # Display RGBD and Point Cloud
            # o3d.visualization.draw_geometries([rgbd_image], window_name="RGBD Image")
            # o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

        if frame is None or depth_frame is None:
            continue  # Skip iteration if no valid frame

        # Run YOLO Detection
        results = detect_objects(frame, model)
        frame = draw_detections(frame, depth_frame, results)

        # Show RGB and Depth frames
        cv2.imshow("Cup Detection", frame)
        cv2.imshow('Depth', depth_frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release Kinect
    kinectsensor.disconnect()
    cv2.destroyAllWindows()

# Run the script
if __name__ == "__main__":
    main()
