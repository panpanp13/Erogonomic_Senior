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
from cal_angle2025 import rula_risk, rula_score, angle, angle_calc
from require_func import depth2point, check_length_hand , check_length_joint, point2depth
from mp_plot import mp_plot
from multiprocessing import Process
from function_pcd import picks,get_height, projectpoint, check_inarea,check_height
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
def update(event):
    global count, window_size,padding,land_mark_avg, default_angle,vector0,array_vector0,head_theta0,array_head_theta0,back_theta0
    global array_back_theta0,init_pcd,draw_sphere,profile,land_mark_rula,point1,point2,point3,point4,point11,point22,point33,point44
    global nvector_unit,height,angle_dict, collect_start, Transformation, rula, t_rula, start, rula_dict
    
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
            land_mark_rula=[]
            
            rh = np.array([results.pose_landmarks.landmark[20].x , results.pose_landmarks.landmark[20].y])
            lh = np.array([results.pose_landmarks.landmark[19].x , results.pose_landmarks.landmark[19].y])
            
            if results_hand.multi_hand_landmarks:
                hand_pose=[]
                for handlm in results_hand.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(cimage_array, handlm , mpHands.HAND_CONNECTIONS)
                    hand_pose.append(np.array([handlm.landmark[9].x , handlm.landmark[9].y]))
                    
                for hand in hand_pose:
                    if np.linalg.norm(rh - hand)>np.linalg.norm(lh - hand):
                        hand_LR[19] = hand
                        cv2.putText(cimage_array, 'L' , (floor(hand[0]*w),floor(hand[1]*h)) ,1,5, (0,0,255) ,5)
                    else: 
                        cv2.putText(cimage_array, 'R' , (floor(hand[0]*w),floor(hand[1]*h)) ,1,5, (0,0,255) ,5)
                        hand_LR[20] = hand
                        
            for id, lm in enumerate(results.pose_landmarks.landmark): # len=33 
                lmx = lm.x
                lmy = lm.y
                if id==19 and 19 in hand_LR:  
                    lmx,lmy = hand_LR[19]
                if id==20 and 20 in hand_LR: 
                    lmx,lmy = hand_LR[20]
                if not (count < window_size):
                    if 0<=floor(lmx*w) < w and 0<=floor(lmy*h) < h: 
                        z = depth_array[floor(lmy*h)][floor(lmx*w)]
                        if z!=0:
                            land_mark_avg[id][2].append(z)
                            land_mark_avg[id][2].pop(0)
                        else: 
                            z = land_mark_avg[id][2][-1]
                    else: 
                        z=None
                    land_mark_avg[id][0].append(lmx*w)
                    land_mark_avg[id][1].append(lmy*h)
                    land_mark_avg[id][0].pop(0)
                    land_mark_avg[id][1].pop(0)
                else:
                    land_mark_avg[id][0][count] = lmx*w
                    land_mark_avg[id][1][count] = lmy*h
                    if 0<=floor(lmx*w) < w and 0<=floor(lmy*h) < h:  
                        z = depth_array[floor(lmy*h)][floor(lmx*w)]
                    else: 
                        z=0
                    land_mark_avg[id][2][count] = z
                    if id==32: 
                        count +=1  
                land_mark.append({'x':lmx, 'y':lmy , 'visibility':lm.visibility})
                pose_ = depth2point(lmx*w, lmy*h , z)
                pose_.append(lm.visibility)
                land_mark_rula.append(pose_)
            J = check_length_hand(np.array(land_mark_rula, dtype=np.float64))
            for j in J:
                land_mark_avg[j][0][-1] = land_mark_avg[j][0][-2] 
                land_mark_avg[j][1][-1] = land_mark_avg[j][1][-2] 
                land_mark_avg[j][2][-1] = land_mark_avg[j][2][-2] 
    
            for id in range(33):
                land_mark_rula[id][:3] = depth2point(np.average(land_mark_avg[id][0]), np.average(land_mark_avg[id][1]), np.average(land_mark_avg[id][2]))
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
                    vector0, head_theta0, back_theta0 = angle_calc(land_mark_rula)
                    print("angle_calc output for default_angle == True:")
                    print(f"vector0: {vector0}, shape: {vector0.shape if isinstance(vector0, np.ndarray) else 'Not an array'}")
                    print(f"head_theta0: {head_theta0}")
                    print(f"back_theta0: {back_theta0}")
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
                        print("Arrays trimmed and parameters updated:")
                        print(f"head_theta0: {head_theta0}")
                        print(f"back_theta0: {back_theta0}")
                        print(f"vector0: {vector0}")
                        print("default_angle set to False, collection started.")
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
                        rula, angle_dict, profile, necktwist = angle_calc(
                            land_mark_rula, reference_vector=vector0, head_theta0=head_theta0, back_theta0=back_theta0
                        )
                        print("angle_calc output:")
                        print(f"rula: {rula}")
                        print(f"angle_dict: {angle_dict}")
                        print(f"profile: {profile}")
                        print(f"necktwist: {necktwist}")
                    except Exception as e:
                        print("Error occurred in angle_calc with default_angle != True:")
                        print(f"Exception: {e}")
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
        # danger_array.append(isdanger)
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