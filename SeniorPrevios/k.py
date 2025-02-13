from pyk4a import PyK4A, Config, ColorResolution
import open3d as o3d
import numpy as np
try:
    print("Initializing Azure Kinect...")
    k4a = PyK4A(Config(
        color_resolution=ColorResolution.RES_720P,  # Set to 720p
        depth_mode="NFOV_UNBINNED",                 # Narrow field of view for depth
        synchronized_images_only=True               # Ensure synchronized images
    ))
    k4a.start()
    print("Azure Kinect started successfully!")
except Exception as e:
    print(f"Device initialization failed: {e}")
    print("Check device connection, ensure no other application is using the Kinect.")
    raise

