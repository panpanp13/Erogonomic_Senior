import cv2
import torch
import mediapipe as mp
from math import floor
import numpy as np
import open3d as o3d
import pandas as pd
import threading
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mediapipe.framework.formats import landmark_pb2
from cal_angle2025 import angle_calc, rula_risk
from require_func import depth2point, check_length_hand 
from mp_plot import mp_plot
from multiprocessing import Process
from function_pcd import picks,get_height, projectpoint, check_inarea,check_height
import json
import pyk4a
from pyk4a import Config, PyK4A
from collections import Counter
from Stage import state_consider
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import threading

runcommand = None  # Global mode selection variable

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2,model_complexity=1)
config_file = r'Erogonomic_Senior\SeniorPrevios\config.json'
intrin_file = r'Erogonomic_Senior\SeniorPrevios\intrin.json'
np.set_printoptions(threshold=np.inf, suppress=True)
kinectsensor = o3d.io.AzureKinectSensor(o3d.io.read_azure_kinect_sensor_config(config_file))
kinectsensor.connect(0)
intrin = o3d.io.read_pinhole_camera_intrinsic(intrin_file)
focalx , focaly = intrin.get_focal_length()
cx , cy = intrin.get_principal_point()
print (focalx, focaly, cx, cy)
rula_array=[]
#-------------------------moving average---------------------------
window_size = 5
land_mark_avg = np.zeros((33,3,window_size)).tolist()
count = 0
padding = 0
#----------------------------------------------------time+plot-------------------------------
start0 = time.time() 
plt.ion()
defalut_angle=True
vector0=0
array_vector0=[]
head_theta0=0
array_head_theta0=[]
back_theta0=0
array_back_theta0=[]
#-------------------------------pointcloud----------------------------------------------
init_pcd = 1
draw_sphere=1
profile = 'no_profile'
#-------------------------------pointcloud------------------------
mediapipe_pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5,model_complexity=2)
import numpy as np
from vispy import scene, app, visuals
import time
from vispy.scene import visuals
from vispy.scene.visuals import Text
#-------------------------------------------------canvas-------------------------------
# canvas = scene.SceneCanvas(keys='interactive', show=True)
# grid = canvas.central_widget.add_grid()
# view1 = grid.add_view(row=0,col=0)
# view1.camera = 'turntable'  # or try 'arcball'
# view1.border_color = (0.5, 0.5, 0.5, 1)
# view1.camera.set_range(x=(-1000,1000), y=(-1000,1000), z = (-1000,1000))
# view1.add
# grid1 = scene.visuals.GridLines(parent=view1.scene)
# view2 = grid.add_view(row=0,col=1)
# view2.border_color = (0.5, 0.5, 0.5, 1)
# view2.camera = 'panzoom'
# grid2 = scene.visuals.GridLines(parent=view2.scene)
# scatter = scene.visuals.Markers()
# scatter2d = scene.visuals.Markers()
# scatter_rula = scene.visuals.Markers()
# Lineplot1 = scene.visuals.Line()
# Lineplot2 = scene.visuals.Line()
# vistext = scene.visuals.Text()
# vistext2 = scene.visuals.Text()
# view3 = grid.add_view(row=1,col=0,col_span=2)
# view3.border_color = (0.5, 0.5, 0.5, 1)
# view3.camera = 'panzoom'
# grid2 = scene.visuals.GridLines(parent=view3.scene)
# view3.camera.set_range(x=(-3,30), y=(0,8))
# axisx_view3 = scene.visuals.Axis()
# axisx_view3.pos = [[0, 0], [100, 0]]
# axisx_view3.domain = (0,100)
# view3.add(axisx_view3)
# axisy_view3 = scene.visuals.Axis()
# axisy_view3.pos = [[0, 0], [0, 7]]
# axisy_view3.domain = (0,7)
# view3.add(axisy_view3)
default_angle = True
# default_angle = False
vector0 = None
head_theta0 = None
back_theta0 = None
#-----------------------------------------------camera view------------------------------------
time_collect=[]
collect_start=time.time()
danger_array = []
start = time.time()
stable = False
guangle_odd = None
guangle_even = None
kpcount = 1
loop_time = None
last_change_time = None
kpn = False
pan = False
loopangle = []
stable_time = 0
loop_detected = False
state_history = []
max_count = 0
point_score = {}
rula_display = np.zeros((720, 240, 3), dtype=np.uint8)
rula_display_time = np.zeros((80, 240, 3), dtype=np.uint8)
muse = 0
#-----------------------------------------------DANGER ZONE------------------------------------
from Test_all import *
from pick_test import *
model_path=r"yolov8n.pt"
device= 0 if torch.cuda.is_available() else 'cpu'
model=YOLO(model_path).to(device)
show_pcd = 'off'
list_ceterpcd=[]
find_ceter=False
old_pick = 0
distance =0
intrin_path = r'Erogonomic_Senior\SeniorPrevios\intrin.json' 
intrin = o3d.io.read_pinhole_camera_intrinsic(intrin_path)
rotate = False
coundt_rt =1
point_cloud = False
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().translate((0,0,0), relative=False)
prev_time = 0
count_danger=0
logging.getLogger("ultralytics").setLevel(logging.ERROR)
pcd_check = True
#----------------------------------------------- Sound ------------------------------------
from Sound_Few import *
FORMAT = pyaudio.paInt16
CHANNELS = 7  # Azure Kinect has 7-mic array
RATE = 44100  # Sampling rate
CHUNK = 4096  # Buffer size
REFERENCE_PRESSURE = 2e-5  # 20 micropascals (standard reference pressure)
CALIBRATION_SENSITIVITY = 30  # Adjusted for real calibration
ALERT_THRESHOLD = 85  # dB SPL threshold for warnings
TARGET_CHANNEL = 0  # Select mic channel to measure
BACKGROUND_OFFSET = 0
NOISE_THRESHOLD = 0.003
audio = pyaudio.PyAudio()
device_index = None
audio = pyaudio.PyAudio()
device_index = None
print("Searching for Azure Kinect microphone...")
audio.get_device_count()
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if "Kinect" in info["name"]:
        device_index = i
        print(f"Azure Kinect mic found at index {i}: {info['name']}")
        break

if device_index is None:
    print("Error: Azure Kinect microphone not found.")
    audio.terminate()
    exit(1)

b_a, a_a = a_weighting(RATE)
#------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
rula = None
# Global variables to store RULA scores and timestamps
rula_times = []
rula_values = []
pie_chart =[]
pie_chart_dic = {}
# ----------------------------------------------plot --------------------------------------

fig, axs = plt.subplots(2, 1, figsize=(4, 9))
axs[0].set_title("RULA vs Time")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("RULA")
axs[0].set_ylim(0, 8)
axs[1].set_title("RULA Distribution")
pie_colors = {
    1: '#90EE90', 
    2: '#75F375',  
    3: '#F8F8B4',  
    4: '#FDFD38', 
    5: '#FDBF7F', 
    6: '#FF6347'   
}
# ------------------------------------------------------------------------------------------

def update(event):
    global count, window_size,padding,land_mark_avg, default_angle,vector0,array_vector0,head_theta0,array_head_theta0,back_theta0
    global array_back_theta0,init_pcd,draw_sphere,profile,land_mark_rula,point1,point2,point3,point4,point11,point22,point33,point44
    global nvector_unit,height,angle_dict, collect_start, Transformation, rula, t_rula, start, rula_dict, depth , stable, kpcount, guangle_even, guangle_odd, stable_time
    global last_change_time, kpn, elapsed_time ,loop_time, loopangle
    global state_check, state_history, state_change_count, state_count, previous_state, state, pan
    global rula_display, point_score
    global rotation_matrix,rotate,find_ceter,coundt_rt,list_ceterpcd,marker,pcd_out,frame,pt_pcd,old_pick
    global fps,prev_time,device
    global vis,dummy_pcd,pcd_check,pcd
    global cluster_xmax,cluster_xmin,cluster_ymax,cluster_ymin,cluster_zmax,cluster_zmin
    global rula_enabled, angle_enabled, danger_enabled, sound_enabled, muse , runcommand, pie_chart, pie_chart_dic, rula_display_time
    global FORMAT,CHANNELS,RATE,CHUNK,REFERENCE_PRESSURE,CALIBRATION_SENSITIVITY,ALERT_THRESHOLD,TARGET_CHANNEL,BACKGROUND_OFFSET,NOISE_THRESHOLD,device_index,audio 
    global data,audio_data,target_audio_data,weighted_audio_data,spl_value,useold,new,static_arr,static_mins,static_maxs
    if 'default_angle' not in globals() :
        print("Warning: 'default_angle' not initialized. Setting to True")
        default_angle = True

    #---------------------------------capture------------------------------------------------
    try :
        
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=CHUNK)
        print(f"Measuring noise levels at fs={RATE} Hz with A-weighting filter...")
        print("Press 'q' to stop.")
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
        target_audio_data = audio_data[:, TARGET_CHANNEL]
        weighted_audio_data = lfilter(b_a, a_a, target_audio_data)
        spl_value = calculate_spl(weighted_audio_data, CALIBRATION_SENSITIVITY, BACKGROUND_OFFSET)
        print(f"defa: {default_angle}")
        image = kinectsensor.capture_frame(True)
        if image is None :
            print("No frame capture")
        cimage_array = np.asarray(image.color)
        raw_depth= np.asarray(image.depth).astype(np.float32)
        depth_array = np.asarray(image.depth)
        depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        h,w,c = cimage_array.shape
        results = mediapipe_pose.process(cimage_array)
        results_hand = hands.process(cimage_array)
        cimage_array = cv2.cvtColor(cimage_array, cv2.COLOR_RGB2BGR)

        # ------------------------------------------- display sound-----------------------------------
        cimage_array=show_overlay_image(cimage_array, spl_value, ALERT_THRESHOLD)
        
        
        
 
        if init_pcd ==1 and runcommand.upper() == 'STATIC':
            if pcd_check is True:
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image.color, image.depth
                , depth_trunc=10000, convert_rgb_to_intensity=False)
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrin , project_valid_depth_only=True)
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            if rotate is False:
                normal_vec,d_plane=create_plane_by_pick(pcd)

                pcd_out, rotation_matrix=align_floor_to_xz_plane(pcd,normal_vec,d_plane)
                visualization_draw_geometries('pcd_out,mesh',[pcd_out,mesh])
                rotate = True
            else:
                rotation_matrix=rotation_matrix
                pcd_out=pcd.rotate(rotation_matrix, center=(0, 0, 0))
            Transformation = rotation_matrix
            if not find_ceter:  # Pick point only once
                    old_pick = pick_point_from_pcd(pcd_out)
                    useold=[old_pick[0],old_pick[1],old_pick[2],old_pick[3]]
                    print('useold',useold)
                    useold = sorted(useold, key=lambda a: a[2])
                    print('useold_sort',useold)
                    for new_point in range(4):
                        new = useold[new_point].copy()  # Create a new copy of the list
                        new[1] = 1
                        useold.append(new)
                    find_ceter=True
            if find_ceter:
                points = useold
                print('points',points)
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
                static_arr  = np.array(points)
                static_mins = static_arr.min(axis=0)
                static_maxs = static_arr.max(axis=0)  
                colors = [[1, 0, 0] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                line_set.colors = o3d.utility.Vector3dVector(colors)
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
        if  init_pcd ==1 and runcommand.upper() == 'DYNAMIC':
            if pcd_check is True:
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image.color, image.depth
                , depth_trunc=10000, convert_rgb_to_intensity=False)
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrin , project_valid_depth_only=True)
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            if rotate is False:
                normal_vec,d_plane=create_plane_by_pick(pcd)
                pcd_out, rotation_matrix=align_floor_to_xz_plane(pcd,normal_vec,d_plane)
                visualization_draw_geometries('pcd_out,mesh',[pcd_out,mesh])
                rotate = True
            else:
                rotation_matrix=rotation_matrix
                pcd_out=pcd.rotate(rotation_matrix, center=(0, 0, 0))
            Transformation = rotation_matrix
            results_yolo=model(cimage_array,device=device)
            cimage_array,marker,pt_pcd =draw_detections(cimage_array, raw_depth/1000, results_yolo, pcd, intrin,show_pcd,rotation_matrix)
            if pt_pcd is not None:
                if len(list_ceterpcd) == 0:
                    list_ceterpcd = pt_pcd
                else :
                    distance = list_ceterpcd - pt_pcd
                    list_ceterpcd= pt_pcd
    
            if marker is not None:
                pcd_out = combine_sphere__pcd(pcd_out, marker)
                _,local_indices,pcd_out=clustering(pcd_out,pt_pcd)
                points = np.asarray(pcd_out.points)
                local_indices =points[local_indices]
                cluster_xmax,cluster_xmin = max(local_indices[:,0]),min(local_indices[:,0])
                cluster_zmax,cluster_zmin = max(local_indices[:,2]),min(local_indices[:,2])
                cluster_ymax,cluster_ymin = max(local_indices[:,1]),min(local_indices[:,1])
                x_max_index = np.argmax(local_indices[:, 0])
                x_min_index = np.argmin(local_indices[:, 0])

                z_max_index = np.argmax(local_indices[:, 2])
                z_min_index = np.argmin(local_indices[:, 2])

                y_max_index = np.argmax(local_indices[:, 1])
                y_min_index = np.argmin(local_indices[:, 1])
                Traio_area=0.1
                cluster_xmax,cluster_zmax,cluster_ymax=cluster_xmax+0.1,cluster_zmax+0.1,cluster_ymax+0.1
                cluster_xmin,cluster_ymin,cluster_zmin=cluster_xmin,cluster_ymin,cluster_zmin-0.2
                point_1 = np.array([cluster_xmin,cluster_ymin,cluster_zmax])
                point_2 = np.array([cluster_xmin,cluster_ymin,cluster_zmin])
                point_3 = np.array([cluster_xmax,cluster_ymin,cluster_zmax])
                point_4 = np.array([cluster_xmax,cluster_ymin,cluster_zmin])
                point_5 = np.array([cluster_xmin,cluster_ymax,cluster_zmax])
                point_6 = np.array([cluster_xmin,cluster_ymax,cluster_zmin])
                point_7 = np.array([cluster_xmax,cluster_ymax,cluster_zmax])
                point_8 = np.array([cluster_xmax,cluster_ymax,cluster_zmin])

                points=[point_1,
                        point_2,
                        point_3,
                        point_4,
                        point_5,
                        point_6,
                        point_7,
                        point_8]
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
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time  
        if results.pose_landmarks is None :
            print("Pose landmarks not detected")
        else :
            hand_LR = {}
            land_mark = []
            land_mark_rula = []
            
            rh = np.array([results.pose_landmarks.landmark[20].x, results.pose_landmarks.landmark[20].y])
            lh = np.array([results.pose_landmarks.landmark[19].x, results.pose_landmarks.landmark[19].y])

            if results_hand.multi_hand_landmarks:
                hand_pose = []
                for handlm in results_hand.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(cimage_array, handlm, mpHands.HAND_CONNECTIONS)
                    hand_pose.append(np.array([handlm.landmark[9].x, handlm.landmark[9].y]))

                for hand in hand_pose:
                    if np.linalg.norm(rh - hand) > np.linalg.norm(lh - hand):
                        hand_LR[19] = hand
                        cv2.putText(cimage_array, 'L', (floor(hand[0] * w), floor(hand[1] * h)), 1, 2, (0, 0, 255), 2)
                    else:
                        hand_LR[20] = hand
                        cv2.putText(cimage_array, 'R', (floor(hand[0] * w), floor(hand[1] * h)), 1, 2, (0, 0, 255), 2)
            if results.pose_landmarks and results.pose_landmarks.landmark:
                for id, landmark in enumerate(results.pose_landmarks.landmark):
                    landmark_cx = int(landmark.x * w) 
                    landmark_cy = int(landmark.y * h)
                    if 0 <= landmark_cx < w and 0 <= landmark_cy < h:
                        depth = depth_array[landmark_cy, landmark_cx] / 1000.0  # Valid depth
                    else:
                        depth = 0
                        print(f"Warning: Landmark {id} out of bounds: cx={landmark_cx}, cy={landmark_cy}")
                    cv2.circle(cimage_array, (landmark_cx, landmark_cy), radius=5, color=(0, 255, 0), thickness=-1)  # Green dot
                    if id == 19 and 19 in hand_LR:
                        cv2.putText(cimage_array, 'L' , (floor(hand[0]*w),floor(hand[1]*h)) ,1,5, (0,0,255) ,5)
                        landmark_cx, landmark_cy = int(hand_LR[19][0] * w), int(hand_LR[19][1] * h)
                    if id == 20 and 20 in hand_LR:
                        cv2.putText(cimage_array, 'R' , (floor(hand[0]*w),floor(hand[1]*h)) ,1,5, (0,0,255) ,5)
                        landmark_cx, landmark_cy = int(hand_LR[20][0] * w), int(hand_LR[20][1] * h)

                    if len(land_mark_avg[id][0]) >= window_size:
                        if depth != 0:
                            land_mark_avg[id][2].append(depth)
                        else:
                            depth = land_mark_avg[id][2][-1]  # Fallback to last valid depth

                        land_mark_avg[id][0].append(landmark_cx)
                        land_mark_avg[id][1].append(landmark_cy)

                    else:
                        land_mark_avg[id][0].append(landmark_cx)
                        land_mark_avg[id][1].append(landmark_cy)
                        land_mark_avg[id][2].append(depth)

                    land_mark.append({'x': landmark.x , 'y': landmark.y, 'visibility': landmark.visibility})

                    pose_ = depth2point(landmark_cx, landmark_cy, depth)
                    pose_.append(landmark.visibility)
                    land_mark_rula.append(pose_)
                J = check_length_hand(np.array(land_mark_rula, dtype=np.float64))
                for j in J:
                    if len(land_mark_avg[j][0]) > 1:
                        land_mark_avg[j][0][-1] = land_mark_avg[j][0][-2]
                        land_mark_avg[j][1][-1] = land_mark_avg[j][1][-2]
                        land_mark_avg[j][2][-1] = land_mark_avg[j][2][-2]

                for id in range(33):
                    if len(land_mark_avg[id][0]) > 2: 
                        x_coord = np.mean(land_mark_avg[id][0][-3:]) 
                        y_coord = np.mean(land_mark_avg[id][1][-3:])
                        z_coord = np.mean(land_mark_avg[id][2][-3:])
                    else:
                        x_coord = land_mark_avg[id][0][-1]
                        y_coord = land_mark_avg[id][1][-1]
                        z_coord = land_mark_avg[id][2][-1]

                    if not np.isnan(x_coord) and not np.isnan(y_coord) and not np.isnan(z_coord):
                        land_mark_rula[id][:3] = depth2point(x_coord, y_coord, z_coord)
                    else:
                        print(f"Warning: Skipping ID {id} due to NaN values")

        #-------------------------------plot--------------------------------------------------------------------
                t = time.time()
        #-------------------------------plot-------------------------------------------------------------------- 
            cimage_array = cv2.copyMakeBorder(cimage_array,padding,padding,padding,padding, cv2.BORDER_CONSTANT, value=(0,0,0))
            landmark_subset = landmark_pb2.NormalizedLandmarkList(landmark = land_mark)
            mp_drawing.draw_landmarks(cimage_array, landmark_subset,mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #------------------------------------Default angle--------------------------------------------------------
            if land_mark_rula is None or not isinstance(land_mark_rula, list) or len(land_mark_rula) < 33:
                print("Error: Invalid or incomplete pose landmarks:", land_mark_rula)
                raise ValueError("Pose landmarks are missing or incomplete.")
            print(f"default_angle: {default_angle}")
            if default_angle : 
                try:
                    vector0, head_theta0, back_theta0 = angle_calc(kpn,land_mark_rula)
                    if vector0 is not None and head_theta0 is not None and back_theta0 is not None:
                        array_vector0.append(vector0)
                        array_head_theta0.append(head_theta0)
                        array_back_theta0.append(back_theta0)
                    else :
                        raise ValueError("Invalid output from angle_calc: One or more outputs are None.")
                    if len(array_vector0) > 50:
                        print("Trimming arrays...")
                        if len(array_vector0) >= 50:
                            array_vector0 = np.array(array_vector0[30:])
                            array_head_theta0 = np.array(array_head_theta0[20:])
                            array_back_theta0 = np.array(array_back_theta0[20:])
                        else:
                            raise ValueError("Array length is insufficient for trimming.")

                        head_theta0 = np.average(array_head_theta0[0])
                        back_theta0 = np.average(array_back_theta0[0])
                        for i in range(array_vector0.shape[1]):
                            vector0[i] = np.array([
                                np.average(array_vector0[:, i, 0]),
                                np.average(array_vector0[:, i, 1]),
                                np.average(array_vector0[:, i, 2]),
                            ])
                        default_angle = False
                        collect_start = time.time()
                        start = time.time()
                except Exception as e:
                    print("Error occurred in angle_calc with default_angle == True:")
                    print(e)
                    raise
            else:
                if vector0 is None or head_theta0 is None or back_theta0 is None:
                    rula, angle_dict, profile, necktwist = None, {}, None, None
                else:
                    try:
                        rula, angle_dict, profile, necktwist , point_score= angle_calc(kpn,
                            land_mark_rula, reference_vector=vector0, head_theta0=head_theta0, back_theta0=back_theta0
                        )
                        if loop_time == None and last_change_time == None :
                            loop_time = time.time()
                            last_change_time = time.time()
                        kpcount += 1
                        if kpcount% 2 == 0 :
                            guangle_even = angle_dict
                        else :
                            guangle_odd = angle_dict
                        if guangle_even and guangle_odd :
                            for key in guangle_even.keys() :
                                if key in guangle_odd and guangle_even[key] is not None and guangle_odd[key] is not None:
                                    diff = abs(guangle_even[key] - guangle_odd[key])
                                    if diff > 10 :  #############################
                                        print ("POSTURE CHANGE")
                                        kafew = True
                                        last_change_time = time.time()
                                        kpn = 'normal'
                                        stable = 'No muscle use condition'
                                        muse = 0
                                    else :
                                        print("POSTURE STABLE")
                                        kafew = False
                                else:
                                    print(f"Key '{key}' is missing or invalid in one of the dictionaries.")
                            stable_time = time.time() - last_change_time
                            if stable_time > 30 :
                                print("STABLE FOR 600 SECS")
                                kpn = 'extra'
                                stable = 'Muscle use condition I'
                                muse = 1
                        else:
                            print("One of guangle_even or guangle_odd is None. Skipping comparison.")
                        
                        elapsed_time = time.time() - loop_time
                        print(f"LOOP TIME : {loop_time}")
                        print(f"ELAPSED TIME : {elapsed_time}")
                        if elapsed_time < 60 :
                            state_check = state_consider(angle_dict)
                            state_history.append(state_check)
                            print(f"STATE HISTORY: {state_history}")
                            if pan == 'Muscle use' :
                                kpn = 'extra'
                                stable = 'Muscle use condition II'
                                print("EXTRA")
                                muse = 1
                        else :
                            print(len(state_history))
                            state_count = {}
                            previous_state = None
                            for state in state_history:
                                if state != previous_state: 
                                    if state not in state_count:
                                        state_count[state] = 1
                                    else:
                                        state_count[state] += 1  
                                previous_state = state  
                            for state, count in state_count.items():
                                print(f"{state}: {count} times")
                            if any(count > 4 for count in state_count.values()):
                                print(f"Warning: A state is more than 4 times")
                                kpn = 'extra'
                                stable = 'Muscle use condition II'
                                print("EXTRA") 
                                pan = 'Muscle use'
                                muse = 1
                            else:
                                kpn = 'normal'
                                stable = 'No muscle use'
                                print("NORMAL")
                                pan = 'No muscle use'
                                muse = 0
                            state_history.clear()
                            loop_time = time.time()
                    except Exception as e:
                        print("Error occurred in angle_calc with default_angle != True:")
                        if 'e' in locals():  # Check if 'e' exists before using it
                            print(f"Exception: {e}")
                        else:
                            print("Unknown error occurred.")
                        
                        rula, angle_dict, profile, necktwist = None, {}, None, None
                if rula is None:
                    print("RULA IS NONE.CHECK CALCULATION")

            try:
                pass
                Rupper_angle = angle_dict.get('Rshoulder')
                Rlower_angle = angle_dict.get('Relbow')
                Lupper_angle = angle_dict.get('Lshoulder')
                Llower_angle = angle_dict.get('Lelbow')
                Neck_angle = angle_dict.get('Neck')
                Trunk_angle = angle_dict.get('Trunk')
                pwp = state_consider(angle_dict)
                q = [angle_dict.get('Rshoulder'), angle_dict.get('Relbow'), angle_dict.get('Lshoulder'), angle_dict.get('Lelbow'), angle_dict.get('Neck')]
                A1 = point_score.get("upper_arm", "N/A")
                A2 = point_score.get("lower_arm", "N/A")
                A3 = point_score.get("wrist", "N/A")
                A4 = point_score.get("wrist_twist", "N/A")
                B9 = point_score.get("neck", "N/A")
                B10 = point_score.get("trunk", "N/A")
                B11 = point_score.get("legs", "N/A")
                base_y = 100
                text_position = (50, base_y)
                (text_width, text_height), baseline = cv2.getTextSize(f'RULA: {rula}', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                background_color = (0, 0, 0)
                cv2.rectangle(cimage_array, 
                            (text_position[0] - 5, text_position[1] + 5),  # Top-left corner
                            (text_position[0] + text_width + 5, text_position[1] - text_height - 5),  # Bottom-right corner
                            background_color, -1)
                cv2.putText(cimage_array, f'RULA: {rula}', text_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  
                rula_display[:] = (0, 0, 0)  # Clear to black

                # Prepare text info
                rula_info = {
                    "RULA Score": rula,
                    "A1 (Upper Arm)": point_score.get("upper_arm", "N/A"),
                    "A2 (Lower Arm)": point_score.get("lower_arm", "N/A"),
                    "A3 (Wrist)": point_score.get("wrist", "N/A"),
                    "A4 (Wrist Twist)": point_score.get("wrist_twist", "N/A"),
                    "B9 (Neck)": point_score.get("neck", "N/A"),
                    "B10 (Trunk)": point_score.get("trunk", "N/A"),
                    "B11 (Legs)": point_score.get("legs", "N/A"),
                    "Muscle Use condition": stable
                }

                # Drawing settings
                base_y = 35
                line_spacing = 26
                font_scale = 0.52
                font_thickness = 1
                font_color = (255, 255, 0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                max_line_width = 32  # Max chars before wrapping (you can tweak this)

                # Draw each line, support wrapping for long text
                for i, (key, value) in enumerate(rula_info.items()):
                    full_text = f"{key}: {value}"

                    # If text too long, split into two lines
                    if len(full_text) > max_line_width:
                        line1 = f"{key}:"
                        line2 = f"{value}"
                        pos1 = (10, base_y)
                        pos2 = (20, base_y + line_spacing)
                        if pos2[1] < rula_display.shape[0] - 5:
                            cv2.putText(rula_display, line1, pos1, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                            cv2.putText(rula_display, line2, pos2, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                            base_y += line_spacing * 2  # Move down for the next block
                    else:
                        pos = (10, base_y)
                        if pos[1] < rula_display.shape[0] - 5:
                            cv2.putText(rula_display, full_text, pos, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                            base_y += line_spacing
                rula_display_time[:] = (0, 0, 0)

                cv2.putText(rula_display_time, f'Stable Time: {stable_time:.2f} sec', 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(rula_display_time, f'Elapsed Time: {elapsed_time:.2f} sec', 
                            (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            except Exception as e: 
                rula=None
                # print(e)
            
        #--------------------------------------------danger-------------------------------
            if runcommand.upper() == 'STATIC':
                for pt in np.array(land_mark_rula):
                    pt= rotation_matrix@pt[:3]
                    print('pt',pt)
                    if static_mins[0]<=float(pt[0])<=static_maxs[0] and static_mins[2]<=-float(pt[2]) <=static_maxs[2]:
                        print('Danger Now')
                        cimage_array = apply_red_overlay(cimage_array, intensity=0.5)
                        break
            else:
                for pt in np.array(land_mark_rula):
                    print('pt',pt)
                    pt= rotation_matrix@pt[:3]
                    # print('pt',pt)
                    if cluster_xmin<=float(pt[0])<=cluster_xmax and cluster_zmin<=-float(pt[2]) <=cluster_zmax and -cluster_ymax<=float(pt[1])<=-cluster_ymin:
                        # print('Danger Now')
                        cimage_array = apply_red_overlay(cimage_array, intensity=0.5)
                        break
        #----------------------------vispy---------------------------------------------------------
            pos = np.array(land_mark_rula)[:,:3]
            color = np.ones((pos.shape[0],3))*[0.5,0.5,0.5]
            size = np.ones((pos.shape[0],))*10

            line_landmark1 = ((pos[20],pos[16]),(pos[16],pos[14]),(pos[14],pos[12]),(pos[12],pos[24]),(pos[24],pos[23]),
                                    (pos[23],pos[11]),(pos[11],pos[12]),(pos[12],pos[11]),(pos[11],pos[13]),(pos[13],pos[15]),
                                    (pos[15],pos[17]))
            line_landmark2 = ((pos[28],pos[26]),(pos[26],pos[24]),(pos[24],pos[23]),(pos[23],pos[25]),(pos[25],pos[27]))
            pose_=[]
            for i in np.array(land_mark_rula)[:,:3]:
                pose_.append(np.matmul(i,Transformation)[:2])
            pose_=np.array(pose_)

            color = np.ones((pose_.shape[0],3))*[0.5,0.5,0.5]
            size = np.ones((pose_.shape[0],))*10

            if rula:
                if rula!=None:
                    t_rula = time.time()-start
                    pos_rula = np.array([[t_rula,int(rula)]])
                    color = np.ones([pos_rula.shape[0],3])*[0.5,0.5,0.5]
                    rula_times.append(current_time)
                    rula_values.append(int(rula))
                    
                    pie_chart.append(int(rula))
                    pie_chart_dic = dict(Counter(pie_chart))
                    print(f"ASDICT {pie_chart_dic}")
                    axs[0].cla()  # Clear the RULA vs Time plot
                    axs[1].cla()  # Clear the pie chart plot
                    axs[0].axhspan(0, 2, color='green', alpha=0.3)  # Low risk
                    axs[0].axhspan(2, 4, color='yellow', alpha=0.3)  # Medium risk
                    axs[0].axhspan(4, 6, color='orange', alpha=0.3)  # High risk
                    axs[0].axhspan(6, 8, color='red', alpha=0.3) 
                    axs[0].plot(rula_times, rula_values, color='b', linestyle='-', linewidth=2)
                    axs[0].set_title("RULA vs Time")
                    axs[0].set_xlabel("Time (s)")
                    axs[0].set_ylabel("RULA")
                    axs[0].set_ylim(0, 8)  # Static Y-axis range from 0 to 8

                    # 2. Plot the pie chart on the second subplot
                    labels = [str(key) for key in pie_chart_dic.keys()]
                    sizes = list(pie_chart_dic.values())
                    total = sum(sizes)
                    pie_colors_list = [pie_colors.get(key, '#808080') for key in pie_chart_dic.keys()]  # Default to gray

                    # Custom autopct function to show both count and percentage
                    def make_autopct(values):
                        def my_autopct(pct):
                            count = int(round(pct * total / 100.0))
                            return f"{pct:.1f}%\n({count})"
                        return my_autopct

                    axs[1].cla()
                    axs[1].set_title("RULA Frequency")
                    axs[1].pie(
                        sizes,
                        labels=labels,
                        autopct=make_autopct(sizes),
                        startangle=90,
                        colors=pie_colors_list,
                        wedgeprops={'edgecolor': 'black', 'linewidth': 1}
                    )# Add black border to slices

                    # Adjust layout and update the figure
                    plt.tight_layout()
                    plt.draw()
                    plt.pause(0.1)
        cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)
        cv2.moveWindow('MediaPipe Pose', 640, 0)
        cv2.imshow('MediaPipe Pose',cimage_array)
        cv2.namedWindow('RULA Score & Stage', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RULA Score & Stage', 240, 720)
        cv2.moveWindow('RULA Score & Stage', 400, 0)
        cv2.imshow('RULA Score & Stage', rula_display)
        cv2.namedWindow('RULA Time Info', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RULA Time Info', 240, 80)
        cv2.moveWindow('RULA Time Info', 400, 740)  # Right below the main 720px-high RULA Score window
        cv2.imshow('RULA Time Info', rula_display_time)
        cv2.imshow('DEPTH IMAGE',depth_colored)
        ###########################################
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            while True:
                key2 = cv2.waitKey(1) & 0xFF
                if key2 == ord('c'):
                    break
    except Exception as e :
        print(f"Error in update: {e}")

def open_button_panel():
    global rula_enabled, angle_enabled, danger_enabled , sound_enabled

    panel = tk.Tk()
    panel.title("Quick Toggles")
    panel.geometry("200x220+100+100")

    rula_enabled = tk.BooleanVar(value=False)
    angle_enabled = tk.BooleanVar(value=False)
    danger_enabled = tk.BooleanVar(value=False)
    sound_enabled = tk.BooleanVar(value=False)
    
    def on_rula_toggle():
        print(f"[RULA] {'Enabled' if rula_enabled.get() else 'Disabled'}")

    def on_angle_toggle():
        print(f"[ANGLE] {'Enabled' if angle_enabled.get() else 'Disabled'}")

    def on_danger_toggle():
        print(f"[DANGER ZONE] {'Enabled' if danger_enabled.get() else 'Disabled'}")
    
    def on_sound_toggle():
        print(f"[SOUND] {'Enabled' if danger_enabled.get() else 'Disabled'}")

    tk.Checkbutton(panel, text="RULA", font=("Arial", 12),
                   variable=rula_enabled, command=on_rula_toggle).pack(pady=10)

    tk.Checkbutton(panel, text="Angle", font=("Arial", 12),
                   variable=angle_enabled, command=on_angle_toggle).pack(pady=10)

    tk.Checkbutton(panel, text="Danger Zone", font=("Arial", 12),
                   variable=danger_enabled, command=on_danger_toggle).pack(pady=10)
    
    tk.Checkbutton(panel, text="Sound", font=("Arial", 12),
                   variable=sound_enabled, command=on_sound_toggle).pack(pady=10)
    panel.attributes("-topmost", True)
    panel.mainloop()

def draw_overlay_RULA(cimage_array, rula, A1, A2, A3, A4, B9, B10, B11, stable):
    if rula_enabled and rula_enabled.get():
        base_y = 100
        line_height = 40
        base_y += line_height
        cv2.putText(cimage_array, f'A1: {A1}', (50, base_y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        base_y += line_height
        cv2.putText(cimage_array, f'A2: {A2}', (50, base_y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        base_y += line_height
        cv2.putText(cimage_array, f'A3: {A3}', (50, base_y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        base_y += line_height
        cv2.putText(cimage_array, f'A4: {A4}', (50, base_y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        base_y += line_height
        cv2.putText(cimage_array, f'B9: {B9}', (50, base_y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        base_y += line_height
        cv2.putText(cimage_array, f'B10: {B10}', (50, base_y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        base_y += line_height
        cv2.putText(cimage_array, f'B11: {B11}', (50, base_y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        base_y += line_height
        cv2.putText(cimage_array, f'POSTURECOLLECT: {stable}', (50, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)   

def draw_overlay_ANGLE(cimage_array, Rupper_angle, Rlower_angle, Lupper_angle, Llower_angle, Neck_angle, Trunk_angle,pwp):
    if angle_enabled and angle_enabled.get():
        base_y = 100
        line_height = 40
        base_y += line_height
        cv2.putText(cimage_array, f'Rupper_angle: {Rupper_angle}', (50, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        base_y += line_height
        cv2.putText(cimage_array, f'Rlower_angle: {Rlower_angle}', (50, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        base_y += line_height
        cv2.putText(cimage_array, f'LUpper_angle: {Lupper_angle}', (50, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        base_y += line_height
        cv2.putText(cimage_array, f'Llower_angle: {Llower_angle}', (50, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) 
        base_y += line_height
        cv2.putText(cimage_array, f'NECK: {Neck_angle}', (50, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)   
        base_y += line_height
        cv2.putText(cimage_array, f'Trunk: {Trunk_angle}', (50, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)   
        base_y += line_height
        cv2.putText(cimage_array, f'STATE: {pwp}', (50, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        base_y += line_height

def draw_overlay_SOUND(cimage_array, rula, A1, A2, A3, A4, B9, B10, B11, muse):
    if sound_enabled and sound_enabled.get():
        print("WAITING CODE SOUND")

def launch_static():
    global runcommand,vis
    runcommand = "STATIC"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='point_cloud', width=640, height=480)
    dummy_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(dummy_pcd)
    root.destroy()
    print("STATIC mode selected.")
    # threading.Thread(target=open_button_panel, daemon=True).start()
    timer = app.Timer(interval=1/50)
    timer.connect(update)
    timer.start()
    try:
        # canvas.show()
        app.run()
        kinectsensor.release()
    except:
        kinectsensor.release()
        df = pd.DataFrame({'rula':rula_array, 'time':time_collect})

def launch_dynamic():
    global runcommand,vis
    runcommand = "DYNAMIC"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='point_cloud', width=640, height=480)
    dummy_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(dummy_pcd)
    root.destroy()
    print("DYNAMIC mode selected.")
    # threading.Thread(target=open_button_panel, daemon=True).start()
    # You can still use timer + update, but skip checkboxes and overlays
    timer = app.Timer(interval=1/20)
    timer.connect(update)
    timer.start()
    try:
        # canvas.show()
        app.run()
        kinectsensor.release()
    except:
        kinectsensor.release()
        df = pd.DataFrame({'rula':rula_array, 'time':time_collect})

root = tk.Tk()
root.title("Select Mode")
root.geometry("720x480")
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

left_frame = tk.Frame(main_frame, width=360, height=480)
left_frame.pack(side="left", fill="both", expand=True)

image_path = os.path.abspath(r"C:\Users\pan\Downloads\istockphoto-1495829139-612x612.jpg")
try:
    img = Image.open(image_path)
    img = img.resize((360, 480), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    image_label = tk.Label(left_frame, image=photo)
    image_label.image = photo
    image_label.pack(fill="both", expand=True)
except Exception as e:
    print(f"Image load error: {e}")
    image_label = tk.Label(left_frame, text="Image not found", bg="gray", fg="white", font=("Arial", 16))
    image_label.pack(fill="both", expand=True)
    
right_frame = tk.Frame(main_frame, width=360, height=480, padx=40, pady=60)
right_frame.pack(side="right", fill="both", expand=True)
label = tk.Label(right_frame, text="Choose a mode to run:", font=("Arial", 16))
label.pack(pady=20)
btn_static = tk.Button(right_frame, text="STATIC", font=("Arial", 14), width=15, height=2, command=launch_static)
btn_static.pack(pady=10)
btn_dynamic = tk.Button(right_frame, text="DYNAMIC", font=("Arial", 14), width=15, height=2, command=launch_dynamic)

btn_dynamic.pack(pady=10)
root.mainloop()