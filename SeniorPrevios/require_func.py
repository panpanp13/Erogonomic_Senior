import numpy as np
import open3d as o3d
intrin = o3d.io.read_pinhole_camera_intrinsic(r'intrin.json')
focalx , focaly = intrin.get_focal_length()
cx , cy = intrin.get_principal_point()
def check_length_hand(lm_):
    lm = np.array(lm_)
    joint = set()
    #------------------------------------------hand------------------------------------------
    thres_hand = 400
    if np.linalg.norm(lm[18][:3] - lm[16][:3])>thres_hand: 
      joint.add(16)
      joint.add(18)
    if np.linalg.norm(lm[20][:3] - lm[16][:3])>thres_hand:  joint.add(20)
    if np.linalg.norm(lm[22][:3] - lm[16][:3])>thres_hand:  joint.add(22)
    if np.linalg.norm(lm[17][:3] - lm[15][:3])>thres_hand: 
        joint.add(17)
        joint.add(15)
    if np.linalg.norm(lm[19][:3] - lm[15][:3])>thres_hand:  
      joint.add(19)
      # print(np.linalg.norm(lm[19][:3] - lm[15][:3]))
    if np.linalg.norm(lm[21][:3] - lm[15][:3])>thres_hand:  joint.add(21)
    #------------------------------------------hand------------------------------------------
    thres_lowerarm = 350
    # print(np.linalg.norm(lm[16][:3] - lm[14][:3]),np.linalg.norm(lm[14][:3] - lm[12][:3]))
    # print(np.linalg.norm(lm[16][:3] - lm[14][:3]))
    if np.linalg.norm(lm[16][:3] - lm[14][:3])>thres_lowerarm:
      joint.add(16)
      joint.add(14)
    if np.linalg.norm(lm[15][:3] - lm[13][:3])>thres_lowerarm:
      joint.add(15)
      joint.add(13)
    thres_upperarm = 350
    if np.linalg.norm(lm[14][:3] - lm[12][:3])>thres_upperarm:
      joint.add(14)
      joint.add(12)
    if np.linalg.norm(lm[13][:3] - lm[11][:3])>thres_upperarm:
      joint.add(13)
      joint.add(11)

    # thres_foot = 200
    # if np.linalg.norm(lm[32][:3] - lm[28][:3])>thres_foot:
    #   joint.add(32)
    #   joint.add(28)
    # if np.linalg.norm(lm[30][:3] - lm[28][:3])>thres_foot:
    #   joint.add(30)
    #   joint.add(28)
    # if np.linalg.norm(lm[32][:3] - lm[30][:3])>thres_foot:
    #   joint.add(32)
    #   joint.add(30)
    # if np.linalg.norm(lm[31][:3] - lm[27][:3])>thres_foot:
    #   joint.add(31)
    #   joint.add(27)
    # if np.linalg.norm(lm[31][:3] - lm[29][:3])>thres_foot:
    #   joint.add(31)
    #   joint.add(29)
    # if np.linalg.norm(lm[29][:3] - lm[27][:3])>thres_foot:
    #   joint.add(29)
    #   joint.add(27)
    return joint
def check_length_joint(lm_ , joint1,joint2):
  lm = np.array(lm_)
  return np.linalg.norm(lm[joint1][:3] - lm[joint2][:3])
def depth2point(u,v,d):
  try:
    z = d
    x = (u - cx) * z / focalx
    y = (v - cy) * z / focaly
    return [x,y,z]
  except: return [u,v,d] #return same value (d=None)
def point2depth(x,y,z):
  u2=0
  v2=0
  if z!=0:
    u2 = (x*focalx)/z+cx
    v2 = (y*focaly)/z+cy
  return [u2,v2]
