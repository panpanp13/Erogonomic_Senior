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
from cal_angle import rula_risk, rula_score, angle, angle_calc
from require_func import depth2point, check_length_hand , check_length_joint, point2depth
from mp_plot import mp_plot
from multiprocessing import Process
from function_pcd import picks,get_height, projectpoint, check_inarea,check_height

mp_drawing = mp.solutions.drawing_utils#modify_drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2,model_complexity=1)
# mpDraw = mp.solutions.drawing_utils11

np.set_printoptions(threshold=np.inf, suppress=True)
kinectsensor = o3d.io.AzureKinectSensor(o3d.io.read_azure_kinect_sensor_config(r'C:\Users\pan\Downloads\Senior_original\Senior_proj\config.json'))

kinectsensor.connect(0)
intrin = o3d.io.read_pinhole_camera_intrinsic(r'C:\Users\pan\Downloads\Senior_original\Senior_proj\intrin.json')
focalx , focaly = intrin.get_focal_length()
cx , cy = intrin.get_principal_point()
print(focalx,focaly,cx,cy)

rula_array=[]
#-------------------------moving average---------------------------
window_size = 5
land_mark_avg = np.zeros((33,3,window_size)).tolist()
count = 0
padding = 0

#----------------------------------------------------time+plot-------------------------------
start0 = time.time() 
plt.ion()
# mp_plot = mp_plot(do_plot3d=True,do_plot_signal=False, do_plot_depth=False, do_plot_topview=False)
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
#-----------------------------------------------camera view------------------------------------
time_collect=[]
collect_start=time.time()
danger_array = []
start = time.time()
def update(event):
    global count, window_size,padding,land_mark_avg,defalut_angle,vector0,array_vector0,head_theta0,array_head_theta0,back_theta0
    global array_back_theta0,init_pcd,draw_sphere,profile,land_mark_rula,point1,point2,point3,point4,point11,point22,point33,point44
    global nvector_unit,height,angle_dict, collect_start, Transformation, rula, t_rula, start
    #---------------------------------capture------------------------------------------------
    image = kinectsensor.capture_frame(True)
    if image != None:
      #---------------------------------Motion Tracking--------------------------------------------
      cimage_array = np.asarray(image.color)
      depth_array = np.asarray(image.depth)
      h,w,c = cimage_array.shape

      results = mediapipe_pose.process(cimage_array)
      results_hand = hands.process(cimage_array)
      cimage_array = cv2.cvtColor(cimage_array, cv2.COLOR_RGB2BGR)
      #--------------------------------------point cloud--------------------------------------------
      #----------------------------danger---------------------------------------------------------
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
        # point4 = np.asarray(pts[picked[3]])
        vector1 = point2-point1
        vector2 = point3-point2
        point4 = point1+vector2

        nvector = np.cross(vector1,vector2)
        #----------------------------------transformation matrix--------------------------------

        nvector_len = np.linalg.norm(nvector)
        nvector_unit = nvector/nvector_len

        point11 = point1 + nvector_unit*height
        point22 = point2 + nvector_unit*height
        point33 = point3 + nvector_unit*height
        point44 = point4 + nvector_unit*height

        points = [point1,point2,point3,point4,point11,point22,point33,point44]
        lines = [[0,1],[0,3],[1,2],[2,3],[4,5],[4,7],[5,6],[6,7],[0,4],[1,5],[2,6],[3,7]]
        colors = [[0, 1, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(line_set)

        # vis.add_geometry(pcd)
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

      #----------------------------danger---------------------------------------------------------
      # --------------------------------extract xyz--------------------------------------------------
      if results.pose_landmarks:
        hand_LR = {}
        land_mark = []
        land_mark_rula=[]
        rh = np.array([results.pose_landmarks.landmark[20].x , results.pose_landmarks.landmark[20].y])
        lh = np.array([results.pose_landmarks.landmark[19].x , results.pose_landmarks.landmark[19].y])
        #----------------------------------extract hand------------------------------------------------------------
        if results_hand.multi_hand_landmarks:
          hand_pose=[] #2 points
          for handlm in results_hand.multi_hand_landmarks:
              mp_drawing.draw_landmarks(cimage_array, handlm , mpHands.HAND_CONNECTIONS)
              hand_pose.append(np.array([handlm.landmark[9].x , handlm.landmark[9].y]))
          #--------------------------------------------detect left,right----------------------------------------
          for hand in hand_pose:
            if np.linalg.norm(rh - hand)>np.linalg.norm(lh - hand):
              hand_LR[19] = hand
              cv2.putText(cimage_array, 'L' , (floor(hand[0]*w),floor(hand[1]*h)) ,1,5, (0,0,255) ,5)
            else: 
              cv2.putText(cimage_array, 'R' , (floor(hand[0]*w),floor(hand[1]*h)) ,1,5, (0,0,255) ,5)
              hand_LR[20] = hand
        #--------------------------------------------extract pose-----------------------------------------------
        for id, lm in enumerate(results.pose_landmarks.landmark): # len=33 
          lmx = lm.x
          lmy = lm.y
          if id==19 and 19 in hand_LR:  lmx,lmy = hand_LR[19]
          if id==20 and 20 in hand_LR: lmx,lmy = hand_LR[20]

          if not (count < window_size):
            if 0<=floor(lmx*w) < w and 0<=floor(lmy*h) < h: 
              z = depth_array[floor(lmy*h)][floor(lmx*w)]
              if z!=0:
                land_mark_avg[id][2].append(z)
                land_mark_avg[id][2].pop(0)
              else: z = land_mark_avg[id][2][-1]
            else: z=None
            land_mark_avg[id][0].append(lmx*w)
            land_mark_avg[id][1].append(lmy*h)
            land_mark_avg[id][0].pop(0)
            land_mark_avg[id][1].pop(0)
          else:
            land_mark_avg[id][0][count] = lmx*w
            land_mark_avg[id][1][count] = lmy*h
            if 0<=floor(lmx*w) < w and 0<=floor(lmy*h) < h:  z = depth_array[floor(lmy*h)][floor(lmx*w)]
            else: z=0
            land_mark_avg[id][2][count] = z
            if id==32: count +=1  
          land_mark.append({'x':lmx, 'y':lmy , 'visibility':lm.visibility})
          pose_ = depth2point(lmx*w, lmy*h , z)
          pose_.append(lm.visibility)
          # mp_plot.xyz_real_update(pose_,id)
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
          # mp_plot.xyz_avg_update(land_mark_rula,id , t-start0)
        # mp_plot.plot_update(land_mark_rula , t_plot=mp_plot.t_plot, xyz_real=mp_plot.xyz_real, 
        #                     xyz_avg=mp_plot.xyz_avg , joint1=19 , depth_array=depth_array, Transformation=Transformation,
        #                     points_danger = [point1,point2,point3,point4])
        # mp_plot.plot_update(land_mark_rula , t_plot=mp_plot.t_plot, xyz_real=mp_plot.xyz_real, 
        #                     xyz_avg=mp_plot.xyz_avg , joint1=19)

        #-------------------------------plot-------------------------------------------------------------------- 
        cimage_array = cv2.copyMakeBorder(cimage_array,padding,padding,padding,padding, cv2.BORDER_CONSTANT, value=(0,0,0))
        landmark_subset = landmark_pb2.NormalizedLandmarkList(landmark = land_mark)
        # mp_drawing.draw_landmarks(cimage_array, landmark_subset,mp_pose.POSE_CONNECTIONS,
        #   landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(), padding_x =  padding, padding_y = padding)
        mp_drawing.draw_landmarks(cimage_array, landmark_subset,mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
          
        #------------------------------------Default angle--------------------------------------------------------
        if defalut_angle!=True: 
          rula, angle_dict, profile ,necktwist= angle_calc(land_mark_rula,vector0=vector0, head_theta0=head_theta0, back_theta0=back_theta0)
          
        else:
          vector0, head_theta0,back_theta0 = angle_calc(land_mark_rula) #return np.array([Back, Head, sagittal, transverse, coronal]) , head_theta0
          array_vector0.append(vector0)
          array_head_theta0.append(head_theta0)
          array_back_theta0.append(back_theta0)
          # cv2.putText(cimage_array, 'frame:{}'.format(len(array_vector0)) , (100,h+200) ,1,10, (0,0,255) ,10)
          if len(array_vector0)>50:
            array_vector0 = np.array(array_vector0[30:]) 
            array_head_theta0 = np.array(array_head_theta0[20:])
            array_back_theta0 = np.array(array_back_theta0[20:])
            head_theta0 = np.average(array_head_theta0[0])
            back_theta0 = np.average(array_back_theta0[0])
            for i in range(array_vector0.shape[1]):
              vector0[i] = np.array([np.average(array_vector0[:,i,0]) , np.average(array_vector0[:,i,1]) , 
                                    np.average(array_vector0[:,i,2])])
            defalut_angle=False
            collect_start=time.time()
            start = time.time()
        try:
          pass
          # angle = "Rs{},Re{},Rw{},Ls{},Le{},Lw{},N{},T{}".format(floor(angle_dict['Rshoulder']),floor(angle_dict['Relbow']),  \
          #                       floor(angle_dict['Rwrist']),floor(angle_dict['Lshoulder']),floor(angle_dict['Lelbow']), \
          #                       floor(angle_dict['Lwrist']),floor(angle_dict['Neck']),floor(angle_dict['Trunk']))

          # cv2.putText(cimage_array, f'{angle}' , (50,150) ,1,3, (0,0,255) ,3)
          # adjust = "Ntw{},Nsb{},Rsab{},Lsab{},Rlad{},LladP{},Ttw{},Tsb{}".format(angle_dict['Neck_twist'],angle_dict['Neck_sidebend'],angle_dict['Rshoulder_abduct'],  \
          #                                       angle_dict['Lshoulder_abduct'], angle_dict['Rlower_adjust'],angle_dict['Llower_adjust'], \
          #                                         angle_dict['trunktwist'], angle_dict['Trunk_sidebend']) 
          # cv2.putText(cimage_array, adjust , (50,250) ,1,3, (0,0,255) ,3)
          # adjust = "Neck : {}".format( floor(angle_dict['Neck'])) 
          # cv2.putText(cimage_array, adjust , (50,250) ,1,3, (0,0,255) ,3)


          cv2.putText(cimage_array, f'ru:{rula}' , (50,150) ,1,5, (255,0,0) ,5)
          # cv2.putText(cimage_array, f'{profile}' , (100,h-50) ,1,10, (0,0,255) ,10)


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
        if defalut_angle!=True: 
          # danger_array.append(isdanger)
          rula_array.append(rula)
          time_collect.append(time.time()-collect_start)
          if isdanger==True: a='Danger'
          else: a='Safe'
          cv2.putText(cimage_array, f'{a}' , (100,50) ,1,5, (255,0,0) ,5)

        #----------------------------vispy----------------------------------------------------------------------
        # # global start

        # # p1 = point1*1000*[1,-1,-1]
        # # p2 = point2*1000*[1,-1,-1]
        # # p3 = point3*1000*[1,-1,-1]
        # # p4 = point4*1000*[1,-1,-1]
        # # p11 = point11*1000*[1,-1,-1]
        # # p22 = point22*1000*[1,-1,-1]
        # # p33 = point33*1000*[1,-1,-1]
        # # p44 = point44*1000*[1,-1,-1]
        # # line_pos = ((p11,p22),(p22,p33),(p33,p44),(p44,p11),(p1,p2),(p2,p3),(p3,p4),(p4,p1),(p1,p11),(p2,p22),(p3,p33),(p4,p44))
        # # scene.visuals.Line(pos=line_pos, color=(1,0,0), parent=view.scene, width=2)

        # # danger_points = np.array([p1,p2,p3,p4,p11,p22,p33,p44])
        # pos = np.array(land_mark_rula)[:,:3]
        # # line_landmark = ((pos[18],pos[16]),(pos[18],pos[20]),(pos[20],pos[16]),(pos[14],pos[16]),
        # #                  (pos[2],pos[16]),(pos[14],pos[12]),(pos[24],pos[12]),(pos[12],pos[11]),
        # #                  (pos[24],pos[23]),(pos[11],pos[23]),(pos[11],pos[13]),(pos[13],pos[15]),
        # #                  (pos[15],pos[21]),(pos[15],pos[17]),(pos[15],pos[19]),(pos[19],pos[17]),
        # #                  (pos[24],pos[26]),(pos[26],pos[28]),(pos[28],pos[32]),(pos[28],pos[30]),
        # #                  (pos[32],pos[30]),(pos[23],pos[25]),(pos[25],pos[27]),(pos[27],pos[29]),
        # #                  (pos[27],pos[31]),(pos[29],pos[31]))
        # # scene.visuals.Line(pos=line_landmark, color=(1,0,0), parent=view.scene, width=5)

        # # color_body = np.ones((pos.shape[0],3))*[0.5,0.5,0.5]
        # # color_zone = np.ones((danger_points.shape[0],3))*[0,0,0]
        # # color = np.vstack((color_body,color_zone))
        # color = np.ones((pos.shape[0],3))*[0.5,0.5,0.5]

        
        # # size_body = np.ones((pos.shape[0],))*20
        # # size_zone = np.ones((danger_points.shape[0]))*10
        # # size = np.append(size_body,size_zone)
        # size = np.ones((pos.shape[0],))*20

        # # pos = np.vstack((pos,danger_points))

        # scatter.set_data(pos=pos, edge_color=None, face_color=color, size=size) 
        # # print('time taken per frame', time.time() - start,'fps:',1/(time.time() - start) , pos.shape,color.shape,size.shape)
        # # scatter.set_data(pos=pos, edge_color=None)
        
        # view.add(scatter)
        # # start = time.time()
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
        # try:
        if rula:
          if rula!=None:
            t_rula = time.time()-start
            pos_rula = np.array([[t_rula,int(rula)]])
            # print(pos_rula.shape,type(pos_rula))

            color = np.ones([pos_rula.shape[0],3])*[0.5,0.5,0.5]
            if t_rula>30:
              view3.camera.set_range(x=(-3,t_rula), y=(0,8))

            # scatter_rula.set_data(pos = pos_rula, size=20, face_color=np.array([[0.5,0.5,0.5]]))
            # view3.add(scatter_rula)
            visuals.LinePlot(data=pos_rula,color=color,line_kind='-' ,marker_size=10,width=10,parent=view3.scene)
        # except Exception as e:
        #   print(e)

      cv2.imshow('MediaPipe Pose',cimage_array)
  
    else:
      print('image is None')

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
    # print('asdasdsad-------------')
    # df = pd.DataFrame({'rula':rula_array, 'time':time_collect,'danger':danger_array})
    kinectsensor.release()
    df = pd.DataFrame({'rula':rula_array, 'time':time_collect})
    # df.to_csv(r'E:\Open3d_test\Biomechanics-Ai_ntl.ipynb\collectRula\test4.csv') #save validate rula score