import numpy as np
import open3d as o3d
import traceback
import time
from pyk4a import  PyK4A, Config, ColorResolution, DepthMode, K4AException,ImageFormat,CalibrationType
import cv2

# depth_cam = o3d.t.io.RealSenseSensor()
# depth_cam.start_capture()
depth_cam = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,  # 720p RGB resolution
        depth_mode=DepthMode.NFOV_2X2BINNED, 
        color_format=ImageFormat.COLOR_BGRA32,
        synchronized_images_only=True, 
    )
)
depth_cam.start()
device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
o3d_device = o3d.core.Device(device)
# intrinsic_matrix = o3d.core.Tensor(depth_cam.get_metadata().intrinsics.intrinsic_matrix, dtype=o3d.core.Dtype.Float32, device=o3d_device) 
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=1280, height=720, fx=607.4948, fy=607.2757, cx=639.5109, cy=369.9156
)
# Initialize the pointcloud viewer
pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud")
vis.add_geometry(pcd)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
vis.add_geometry(origin)
view_control = vis.get_view_control()
# view_control.set_zoom(0.7)
view_control.set_constant_z_far(100)
try:
    while True:
        capture = depth_cam.get_capture()
        depthshow=capture.depth
        depth_resized = cv2.resize(depthshow, (480, 640), interpolation=cv2.INTER_NEAREST)
        depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        color_image = o3d.geometry.Image(
        np.ascontiguousarray(capture.color[:, :, [2,1,0]])
        )
        depth_image = o3d.geometry.Image(
            np.ascontiguousarray(capture.transformed_depth)
        )
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
         color_image, depth_image, convert_rgb_to_intensity=False
        )
        new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,camera_intrinsic
        )
        new_pcd.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors
        pcd.normals = new_pcd.normals

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        cv2.imshow("Depth Stream", depth_colormap)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    vis.destroy_window()
    depth_cam.stop()