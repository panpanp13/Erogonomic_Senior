import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
mp_pose = mp.solutions.pose

class mp_plot:
    def __init__(self, do_plot3d=False, do_plot_signal=False , do_plot_depth=False, do_plot_topview=False):
        self.do_plot3d = do_plot3d
        self.do_plot_signal = do_plot_signal
        self.do_plot_depth = do_plot_depth
        self.do_plot_topview = do_plot_topview
        self.sum_plot = sum([self.do_plot3d, self.do_plot_signal ,self.do_plot_depth,  self.do_plot_topview])
        
        self.fig = plt.figure(figsize=(10,10))#plt.figaspect(2.) =(30,15) figsize=plt.figaspect(0.2)
        # self.fig2 = plt.figure(figsize=(10,10))
        if do_plot3d:
            self.ax1 = self.fig.add_subplot(1,self.sum_plot,1,projection='3d')
            self.fig.subplots_adjust(wspace=0, hspace=0)
            self.ax1.view_init(elev=90, azim=0, roll=90)
            # self.ax1.view_init(elev=90, azim=180, roll=90)
            self.ax1.set_xlabel('x')
            self.ax1.set_ylabel('y')
            # self.ax1.set_aspect('equalxy')
        if do_plot_signal:
            self.ax2 = self.fig.add_subplot(1,self.sum_plot,2)
            self.y_plot1 = []
            self.y_plot2 = []
            self.x_plot = []
        if do_plot_depth:
            self.ax3 = self.fig.add_subplot(1,self.sum_plot,2)
        # if do_plot_topview:
        #     self.ax4 = self.fig2.add_subplot(1,1,1)
        #     self.ax4.set_xlim(0, 2000)
        #     self.ax4.set_ylim(0, 2000)
        #     self.ax4.set_title('top_view')
        
        self.xyz_real= np.zeros((33,3)).tolist()
        self.xyz_avg = np.zeros((33,3)).tolist()
        self.t_plot = []
        for i in range(33):
            for j in range(3):
                self.xyz_avg[i][j] = []
                self.xyz_real[i][j] = []

    def plot3d(self,pose1):
        x,y,z = [],[],[]
        for p in pose1:
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        Xn,Yn,Zn = [],[],[]
        Xn.append([x[28],x[32],x[30],x[28],x[26],x[24],x[12],x[14],x[16],x[18],x[20],x[16]])
        Yn.append([y[28],y[32],y[30],y[28],y[26],y[24],y[12],y[14],y[16],y[18],y[20],y[16]])
        Zn.append([z[28],z[32],z[30],z[28],z[26],z[24],z[12],z[14],z[16],z[18],z[20],z[16]])

        Xn.append([x[27],x[31],x[29],x[27],x[25],x[23],x[11],x[13],x[15],x[17],x[19],x[15]])
        Yn.append([y[27],y[31],y[29],y[27],y[25],y[23],y[11],y[13],y[15],y[17],y[19],y[15]])
        Zn.append([z[27],z[31],z[29],z[27],z[25],z[23],z[11],z[13],z[15],z[17],z[19],z[15]])

        Xn.append([x[8],x[6],x[5],x[4],x[0],x[1],x[2],x[3],x[7]])
        Yn.append([y[8],y[6],y[5],y[4],y[0],y[1],y[2],y[3],y[7]])
        Zn.append([z[8],z[6],z[5],z[4],z[0],z[1],z[2],z[3],z[7]])

        Xn.append([x[16],x[22]])
        Yn.append([y[16],y[22]])
        Zn.append([z[16],z[22]])

        Xn.append([x[11],x[12]])
        Yn.append([y[11],y[12]])
        Zn.append([z[11],z[12]])

        Xn.append([x[23],x[24]])
        Yn.append([y[23],y[24]])
        Zn.append([z[23],z[24]])

        Xn.append([x[15],x[21]])
        Yn.append([y[15],y[21]])
        Zn.append([z[15],z[21]])

        Xn.append([x[9],x[10]])
        Yn.append([y[9],y[10]])
        Zn.append([z[9],z[10]])

        n = 0
        while n < 8:
            self.ax1.plot3D(Xn[n],Yn[n],Zn[n],'green')
            n+=1
        self.ax1.scatter3D(x,y,z,'red')
        self.ax1.set_aspect('equalxy')
        for i in range(33):
            self.ax1.text(x[i] ,y[i], z[i] , str(i))
        self.ax1.set_xlim(-500,500)
        self.ax1.set_ylim(-1200,1200) 
        self.ax1.set_zlim(1300,2000) 

    def plot_signal(self, t_plot, xyz_real, xyz_avg , joint1):
        self.x_plot.append(t_plot[-1])
        self.y_plot1.append(xyz_real[joint1][2][-1])
        self.y_plot2.append(xyz_avg[joint1][2][-1])

        self.ax2.plot(self.x_plot, self.y_plot1 ,'b')
        self.ax2.plot(self.x_plot, self.y_plot2 ,'r--')
        self.ax2.legend(['Real joint {}'.format(joint1), 'filter joint {}'.format(joint1)])
        self.ax2.set_xlabel('time(s)')
        self.ax2.set_ylabel('z')
    def plot_deptharray(self,depth_array):
        self.ax3.imshow(depth_array)
    # def plot_update(self, pose1=None , t_plot=None, xyz_real=None, xyz_avg=None , joint1=None , depth_array=None, Transformation=None,points_danger=None):
    # def plot_update(self, pose1 , t_plot, xyz_real, xyz_avg , joint1 , depth_array, Transformation,points_danger):
    def plot_update(self, pose1 , t_plot, xyz_real, xyz_avg , joint1 ):
        if self.do_plot3d:
            self.plot3d(pose1)
            self.fig.canvas.flush_events()
            self.ax1.cla()
        if self.do_plot_signal:
            self.plot_signal(t_plot, xyz_real, xyz_avg , joint1)
        # if self.do_plot_depth:
        #     self.plot_deptharray(depth_array)
        # if self.do_plot_topview:
        #     self.show_topview(Transformation=Transformation , pose_=pose1, points_danger=points_danger)
        #     self.ax4.cla()
    def xyz_real_update(self,pose_,id):
        self.xyz_real[id][0].append(pose_[0])
        self.xyz_real[id][1].append(pose_[1])
        self.xyz_real[id][2].append(pose_[2])
    def xyz_avg_update(self, land_mark_rula,id ,time):
        self.xyz_avg[id][0].append(land_mark_rula[id][0])
        self.xyz_avg[id][1].append(land_mark_rula[id][1])
        self.xyz_avg[id][2].append(land_mark_rula[id][2])
        self.t_plot.append(time)
    # def show_topview(self,Transformation=None, pose_=None, points_danger=None):
    #     newpose = []
    #     p1 = np.matmul(points_danger[0],Transformation)[:2]*1000
    #     p2 = np.matmul(points_danger[1],Transformation)[:2]*1000
    #     p3 = np.matmul(points_danger[2],Transformation)[:2]*1000
    #     p4 = np.matmul(points_danger[3],Transformation)[:2]*1000

    #     danger_x = [p1[0], p2[0], p3[0], p4[0], p1[0]]
    #     danger_y = [p1[1], p2[1], p3[1], p4[1], p1[1]]
    #     self.ax4.plot(danger_x,danger_y)
    #     pose__ = np.array(pose_)[:,:3]
    #     for i in pose__[:1]:
    #         newpose.append(np.matmul(i*[1,-1,-1],Transformation)[:2])
    #     x = np.array(newpose)[:,0]
    #     y = np.array(newpose)[:,1]

    #     self.ax4.scatter(x,y)
    #     self.ax4.set_xlim(-4000, 4000)
    #     self.ax4.set_ylim(0, 4000)
    #     plt.draw()
    #     plt.pause(0.001)
    #     # self.fig.canvas.flush_events()
    #     self.ax4.cla()





