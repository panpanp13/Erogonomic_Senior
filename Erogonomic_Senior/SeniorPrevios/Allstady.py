import cv2
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
from cal_angle2025 import angle_calc
from require_func import depth2point, check_length_hand 
from mp_plot import mp_plot
from multiprocessing import Process
from function_pcd import picks,get_height, projectpoint, check_inarea,check_height
import json
import pyk4a
from pyk4a import Config, PyK4A
from collections import Counter
from Stage import state
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2,model_complexity=1)
config_file = r'C:\Users\thinn\Desktop\CU\Senior project\Code\SeniorPrevios\config.json'
intrin_file = r'C:\Users\thinn\Desktop\CU\Senior project\Code\SeniorPrevios\intrin.json'
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
canvas = scene.SceneCanvas(keys='interactive', show=True)
grid = canvas.central_widget.add_grid()
view1 = grid.add_view(row=0,col=0)
view1.camera = 'turntable'  # or try 'arcball'
view1.border_color = (0.5, 0.5, 0.5, 1)
view1.camera.set_range(x=(-1000,1000), y=(-1000,1000), z = (-1000,1000))
view1.add
grid1 = scene.visuals.GridLines(parent=view1.scene)
view2 = grid.add_view(row=0,col=1)
view2.border_color = (0.5, 0.5, 0.5, 1)
view2.camera = 'panzoom'
grid2 = scene.visuals.GridLines(parent=view2.scene)
scatter = scene.visuals.Markers()
scatter2d = scene.visuals.Markers()
scatter_rula = scene.visuals.Markers()
Lineplot1 = scene.visuals.Line()
Lineplot2 = scene.visuals.Line()
vistext = scene.visuals.Text()
vistext2 = scene.visuals.Text()
view3 = grid.add_view(row=1,col=0,col_span=2)
view3.border_color = (0.5, 0.5, 0.5, 1)
view3.camera = 'panzoom'
grid2 = scene.visuals.GridLines(parent=view3.scene)
view3.camera.set_range(x=(-3,30), y=(0,8))
axisx_view3 = scene.visuals.Axis()
axisx_view3.pos = [[0, 0], [100, 0]]
axisx_view3.domain = (0,100)
view3.add(axisx_view3)
axisy_view3 = scene.visuals.Axis()
axisy_view3.pos = [[0, 0], [0, 7]]
axisy_view3.domain = (0,7)
view3.add(axisy_view3)
default_angle = True
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
loop_time = time.time()
last_change_time = time.time()
kpn = False
loopangle = []
stable_time = 0
loop_detected = False
state_history = []
def update(event):
    global count, window_size,padding,land_mark_avg, default_angle,vector0,array_vector0,head_theta0,array_head_theta0,back_theta0
    global array_back_theta0,init_pcd,draw_sphere,profile,land_mark_rula,point1,point2,point3,point4,point11,point22,point33,point44
    global nvector_unit,height,angle_dict, collect_start, Transformation, rula, t_rula, start, rula_dict, depth , stable, kpcount, guangle_even, guangle_odd, stable_time
    global last_change_time, kpn, elapsed_time ,loop_time, loopangle
    global state_check, state_history, state_change_count, set_state, state_count, sort_state, two_consider_state , state_change_count, previous_state
    if 'default_angle' not in globals() :
        print("Warning: 'default_angle' not initialized. Setting to True")
        default_angle = True
    #---------------------------------capture------------------------------------------------
    try :
        print(f"defa: {default_angle}")
        image = kinectsensor.capture_frame(True)
        if image is None :
            print("No frame capture")
        cimage_array = np.asarray(image.color)
        depth_array = np.asarray(image.depth)
        depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        h,w,c = cimage_array.shape
        results = mediapipe_pose.process(cimage_array)
        results_hand = hands.process(cimage_array)
        cimage_array = cv2.cvtColor(cimage_array, cv2.COLOR_RGB2BGR)
        # runcommand = input("Type 'STATIC' or 'DYNAMIC").strip()
        # if runcommand.upper() == 'STATIC' :
        if init_pcd:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image.color, image.depth\
            , depth_trunc=10000, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrin , project_valid_depth_only=True)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            picked = picks(pcd)
            height = get_height()
            pts = np.asarray(pcd.points)
            point1 = np.asarray(pts[picked[0]]).copy()
            point2 = np.asarray(pts[picked[1]]).copy()
            point3 = np.asarray(pts[picked[2]]).copy()
            vector1 = point2 - point1
            vector2 = point3 - point2
            point4 = point1 + vector2
            nvector = np.cross(vector1, vector2)
            if np.linalg.norm(nvector) == 0:
                print("Warning: nvector has zero magnitude. Using default unit vector.")
                nvector_unit = np.array([0, 0, 1])
            else:
                nvector_unit = nvector / np.linalg.norm(nvector)
            point11 = point1 + nvector_unit * height
            point22 = point2 + nvector_unit * height
            point33 = point3 + nvector_unit * height
            point44 = point4 + nvector_unit * height
            points = [point1,point2,point3,point4,point11,point22,point33,point44]
            lines = [[0,1],[0,3],[1,2],[2,3],[4,5],[4,7],[5,6],[6,7],[0,4],[1,5],[2,6],[3,7]]
            colors = [[0, 1, 0] for i in range(len(lines))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_set.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            init_pcd = 0
            p1 = point1*1000*[1,-1,-1]
            p2 = point2*1000*[1,-1,-1]
            p3 = point3*1000*[1,-1,-1]
            p4 = point4*1000*[1,-1,-1]
            p11 = point11*1000*[1,-1,-1]
            p22 = point22*1000*[1,-1,-1]
            p33 = point33*1000*[1,-1,-1]
            p44 = point44*1000*[1,-1,-1]
            v1 = p2-p1
            v2 = p3-p2
            #----------------------------------transformation matrix--------------------------------
            Transformation = np.stack((v1, v2, np.cross(v1,v2)), axis=1)
            line_pos = ((p11,p22),(p22,p33),(p33,p44),(p44,p11), (p11,p1),(p1,p2),(p2,p3),(p3,p4),(p4,p1), (p1,p2),(p2,p22),(p22,p33),(p33,p3),(p3,p4),(p4,p44))
            scene.visuals.Line(pos=line_pos, color=(1,0,0), parent=view1.scene, width=2)
            scene.visuals.Line(pos=((0,0,0),(7000,0,0)), color=(1,0,0), parent=view1.scene, width=2)
            scene.visuals.Line(pos=((0,0,0),(0,7000,0)), color=(0,1,0), parent=view1.scene, width=2)
            scene.visuals.Line(pos=((0,0,0),(0,0,7000)), color=(0,0,1), parent=view1.scene, width=2)
            view1.camera.elevation = 90 #for 3d (0,0,0)
            view1.camera.azimuth = 180
            view1.camera.roll = 180
            p1_proj = np.matmul(p1,Transformation)[:2]
            p2_proj = np.matmul(p2,Transformation)[:2]
            p3_proj = np.matmul(p3,Transformation)[:2]
            p4_proj = np.matmul(p4,Transformation)[:2]
            danger_xy = ((p1_proj[0],p1_proj[1]),(p2_proj[0],p2_proj[1]),(p3_proj[0],p3_proj[1]),(p4_proj[0],p4_proj[1]),(p1_proj[0],p1_proj[1]))
            xlim = (min(-int(p1_proj[0]*2),int(p1_proj[0]*2)), max(-int(p1_proj[0]*2),int(p1_proj[0]*2)))
            ylim = (min(-int(p1_proj[1]*2),int(p1_proj[1]*2)),max(-int(p1_proj[1]*2),int(p1_proj[1]*2)))
            print(xlim,ylim)
            view2.camera.set_range(x = xlim , y = ylim)
            scene.visuals.Line(pos=danger_xy, color=(1,0,0), parent=view2.scene, width=2)
            vistext2.text =['p1','p2','p3','p4']
            vistext2.color = (0,1,0)
            vistext2.font_size = 10
            vistext2.pos = (np.matmul(p11,Transformation)[:2],np.matmul(p22,Transformation)[:2],np.matmul(p33,Transformation)[:2],np.matmul(p44,Transformation)[:2])
            view2.add(vistext2) 
            danger_points = np.array([p1,p2,p3,p4,p11,p22,p33,p44])
            color_zone = np.ones((danger_points.shape[0],3))*[0.5,0.5,0.5]
            vistext.text = ['p11','p22','p33','p44','p1','p2','p3','p4']
            vistext.color = (1,0,0)
            vistext.font_size = 50000
            vistext.pos = (p11,p22,p33,p44, p1,p2,p3,p4)
            view1.add(vistext) 
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
                    cv2.putText(cimage_array, f"{id}", (landmark_cx + 10, landmark_cy - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)
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
                    
                    coord_text = f"{id}: ({pose_[0]:.2f}, {pose_[1]:.2f}, {pose_[2]:.2f}m)"
                    cv2.putText(cimage_array, coord_text, (landmark_cx + 10, landmark_cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    # print(land_mark_rula)
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
                    # print(f"Before angle_calc, R_Shoulder: {land_mark_rula[12]}")
                    vector0, head_theta0, back_theta0 = angle_calc(kpn,land_mark_rula)
                    # print(f"After angle_calc, R_Shoulder: {land_mark_rula[12]}")
                    # print("angle_calc output for default_angle == True:")
                    # print(f"vector0: {vector0}, shape: {vector0.shape if isinstance(vector0, np.ndarray) else 'Not an array'}")
                    # print(f"head_theta0: {head_theta0}")
                    # print(f"back_theta0: {back_theta0}")
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
                        # print("Arrays trimmed and parameters updated:")
                        # print(f"head_theta0: {head_theta0}")
                        # print(f"back_theta0: {back_theta0}")
                        # print(f"vector0: {vector0}")
                        # print("default_angle set to False, collection started.")
                except Exception as e:
                    print("Error occurred in angle_calc with default_angle == True:")
                    print(e)
                    raise
            else:
                if vector0 is None or head_theta0 is None or back_theta0 is None:
                    print("Invalid input parameters to angle_calc:")
                    print(f"vector0: {vector0}, head_theta0: {head_theta0}, back_theta0: {back_theta0}")
                    rula, angle_dict, profile, necktwist = None, {}, None, None
                else:
                    try:
                        # print(f"land_mark_rula before angle_calc: {land_mark_rula}")
                        # print(f"Before angle_calc, R_Shoulder: {land_mark_rula[12]}")
                        rula, angle_dict, profile, necktwist = angle_calc(kpn,
                            land_mark_rula, reference_vector=vector0, head_theta0=head_theta0, back_theta0=back_theta0
                        )
                        # print(f"After angle_calc, R_Shoulder: {land_mark_rula[12]}")
                        # print(last_change_time)
                        kpcount += 1
                        # print(kpcount)
                        if kpcount% 2 == 0 :
                            guangle_even = angle_dict
                            print(f"EVEN LOOP :{guangle_even}")
                        else :
                            guangle_odd = angle_dict
                            print(f"ODD LOOP :{guangle_odd}")
                        print("Success Achevive angle")
                        if guangle_even and guangle_odd :
                            print("Have odd and even")
                            print(f"NOW ANGLE COMPARE1 is : {guangle_even}")
                            print(f"NOW ANGLE COMPARE2 is : {guangle_odd}")
                            for key in guangle_even.keys() :
                                if key in guangle_odd and guangle_even[key] is not None and guangle_odd[key] is not None:
                                    diff = abs(guangle_even[key] - guangle_odd[key])
                                    if diff > 10 :  #############################
                                        print(f"Key '{key}' difference {diff:.2f} exceeds threshold 0.1.")
                                        print ("POSTURE CHANGE")
                                        kafew = True
                                        last_change_time = time.time()
                                        kpn = 'normal'
                                        stable = 'No muscle use'
                                    else :
                                        print("POSTURE STABLE")
                                        kafew = False
                                else:
                                    print(f"Key '{key}' is missing or invalid in one of the dictionaries.")
                            stable_time = time.time() - last_change_time
                            if stable_time > 5 :
                                print("STABLE FOR 5 SECS")
                                kpn = 'extra'
                                stable = 'Muscle use'
                        else:
                            print("One of guangle_even or guangle_odd is None. Skipping comparison.")
                        
                        # elapsed_time = time.time() - loop_time
                        # print(f"LOOP TIME : {loop_time}")
                        # print(f"ELAPSED TIME : {elapsed_time}")
                        # if elapsed_time <60 :
                        #     loopangle.append(list(angle_dict.values()))
                        #     loopangle = [[1,2,2,4,2,2],[1,2,2,4,2,2],[1,2,2,4,2,2]]
                        # else :
                        #     print("1 MINUTE is Pass")
                        elapsed_time = time.time() - loop_time
                        print(f"LOOP TIME : {loop_time}")
                        print(f"ELAPSED TIME : {elapsed_time}")
                        if elapsed_time < 60 :
                            state_check = state(angle_dict)
                            state_history.append(state_check)
                        else :
                            set_state = list(set(state_history))
                            state_count = {}
                            for i in set_state :
                                state_count[i] = state_history.count(i)
                            sort_state = sorted(state_count.items(), key=lambda x: x[1], reverse=True)
                            two_consider_state = [sort_state[i][0] for i in range(min(2, len(sort_state)))]
                            state_change_count = 0
                            previous_state = None
                            for state in state_history:
                                if previous_state is None:
                                    if state in two_consider_state:
                                        previous_state = state
                                    elif state != previous_state and state in two_consider_state:
                                        state_change_count += 1
                                        previous_state = state
                            if state_change_count > 8 :
                                kpn = 'extra'
                                stable = 'Muscle use'  
                            elif state_change_count < 8 :
                                kpn = 'normal'
                                stable = 'No muscle use'
                            state_history.clear()
                        # elapsed_time = time.time() - loop_time
                        # print(f"LOOP TIME : {loop_time}")
                        # print(f"ELAPSED TIME : {elapsed_time}")
                        # threshold = 10
                        # if elapsed_time < 60:
                        #     loopangle.append({key: value for key, value in angle_dict.items() if isinstance(value, (int, float))})
                        #     print("AAA_KAOPUN NOT 1 MINUTE")
                        # else:
                        #     print("1 minute Pass. CHECKING FOR LOOP ANGLE")
                        #     angle_count = {}
                        #     angles = [value for entry in loopangle for value in entry.values()]
                        #     for angle in angles:
                        #         found = False
                        #         for i in list(angle_count.keys()):
                        #             if abs(angle - i) <= threshold: 
                        #                 angle_count[i] += 1
                        #                 found = True
                        #                 break
                        #         if not found:
                        #             angle_count[angle] = 1
                        #     for angle, count in angle_count.items():
                        #         if count > 4:
                        #             kpn = 'extra'
                        #             print("LOOP FOUND MUSCLE USE CONDITION")
                        #     loop_time = time.time()
                        #     loopangle.clear()
                        
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
                rula_dict = {}
                print(f"KP_RULA :{rula_dict}")
                Rupper_angle = angle_dict.get('Rshoulder')
                Rlower_angle = angle_dict.get('Relbow')
                Lupper_angle = angle_dict.get('Lshoulder')
                Llower_angle = angle_dict.get('Lelbow')
                Neck_angle = angle_dict.get('Neck')
                Trunk_angle = angle_dict.get('Trunk')
                pwp = state(angle_dict)
                print(pwp)
                q = [angle_dict.get('Rshoulder'), angle_dict.get('Relbow'), angle_dict.get('Lshoulder'), angle_dict.get('Lelbow'), angle_dict.get('Neck')]
                print(angle_dict)
                base_y = 150
                line_height = 50
                cv2.putText(cimage_array, f'RULA: {rula}', (50, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(cimage_array, f'Rupper_angle: {Rupper_angle}', (50, base_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                line_height +=50
                cv2.putText(cimage_array, f'Rlower_angle: {Rlower_angle}', (50, base_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                line_height +=50
                cv2.putText(cimage_array, f'LUpper_angle: {Lupper_angle}', (50, base_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                line_height +=50
                cv2.putText(cimage_array, f'Llower_angle: {Llower_angle}', (50, base_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2) 
                line_height +=50
                cv2.putText(cimage_array, f'NECK: {Neck_angle}', (50, base_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)   
                line_height +=50
                cv2.putText(cimage_array, f'Trunk: {Trunk_angle}', (50, base_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)   
                line_height +=50
                cv2.putText(cimage_array, f'POSTURECOLLECT: {stable}', (50, base_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)   
                line_height +=50
                cv2.putText(cimage_array, f'STATE: {pwp}', (50, base_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                line_height +=50
                cv2.putText(cimage_array, f'Count Loop: {state_change_count}', (50, base_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)      
                text = f'Stable Time: {stable_time:.2f} sec'
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                position_x = cimage_array.shape[1] - text_width - 50 
                position_y = base_y 
                cv2.putText(cimage_array, text, (position_x, position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)     
                text = f'Elpased Time: {elapsed_time:.2f} sec'
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                position_x = cimage_array.shape[1] - text_width - 50 
                position_y = base_y + 50
                cv2.putText(cimage_array, text, (position_x, position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)   
            except Exception as e: 
                rula=None
                print(e)
        #--------------------------------------------danger-------------------------------
            for pt in np.array(land_mark_rula):
                proj,dist = projectpoint(pt[:3]*[1,-1,-1]/1000 ,nvector_unit,point1)
                inarea, isdanger = check_inarea(proj,point1,point2,point4)
                if inarea == True:
                    isdanger = check_height(dist,height)
                    if isdanger == True:
                        break
            if default_angle!=True: 
                rula_array.append(rula)
                time_collect.append(time.time()-collect_start)
                if isdanger==True: 
                    a='Danger'
                else: 
                    a='Safe'
                cv2.putText(cimage_array, f'{a}' , (100,50) ,1,5, (255,0,0) ,5)
        #----------------------------vispy---------------------------------------------------------
            pos = np.array(land_mark_rula)[:,:3]
            color = np.ones((pos.shape[0],3))*[0.5,0.5,0.5]
            size = np.ones((pos.shape[0],))*10
            scatter.set_data(pos=pos, edge_color=None, face_color=color, size=size) 
            view1.add(scatter)

            line_landmark1 = ((pos[20],pos[16]),(pos[16],pos[14]),(pos[14],pos[12]),(pos[12],pos[24]),(pos[24],pos[23]),
                                    (pos[23],pos[11]),(pos[11],pos[12]),(pos[12],pos[11]),(pos[11],pos[13]),(pos[13],pos[15]),
                                    (pos[15],pos[17]))
            Lineplot1.set_data(line_landmark1 , color=(0.5,0.5,0.5) , width=2)
            view1.add(Lineplot1)
            line_landmark2 = ((pos[28],pos[26]),(pos[26],pos[24]),(pos[24],pos[23]),(pos[23],pos[25]),(pos[25],pos[27]))
            Lineplot2.set_data(line_landmark2 , color=(0.5,0.5,0.5) , width=2)
            view1.add(Lineplot2)
            pose_=[]
            for i in np.array(land_mark_rula)[:,:3]:
                pose_.append(np.matmul(i,Transformation)[:2])
            pose_=np.array(pose_)

            color = np.ones((pose_.shape[0],3))*[0.5,0.5,0.5]
            size = np.ones((pose_.shape[0],))*10
            scatter2d.set_data(pos=pose_, edge_color=None, face_color=color, size=size) 
            view2.add(scatter2d)
            if rula:
                if rula!=None:
                    t_rula = time.time()-start
                    pos_rula = np.array([[t_rula,int(rula)]])
                    color = np.ones([pos_rula.shape[0],3])*[0.5,0.5,0.5]
                    if t_rula>30:
                        view3.camera.set_range(x=(-3,t_rula), y=(0,8))
                    visuals.LinePlot(data=pos_rula,color=color,line_kind='-' ,marker_size=10,width=10,parent=view3.scene)
        cv2.imshow('MediaPipe Pose',cimage_array)
        cv2.imshow("Depth Frame", depth_colored)
        ###########################################
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            while True:
                key2 = cv2.waitKey(1) & 0xFF
                if key2 == ord('c'):
                    break
        ###########################################
        # elif runcommand.upper() == 'DYNAMIC' :
        #     exit()
    except Exception as e :
        print(f"Error in update: {e}")
        

timer = app.Timer()
start = time.time()
timer.connect(update)
timer.start()
if __name__ =='__main__':
  try:
    canvas.show()
    app.run()
    kinectsensor.release()
  except:
    kinectsensor.release()
    df = pd.DataFrame({'rula':rula_array, 'time':time_collect})