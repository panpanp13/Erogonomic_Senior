import math
import numpy as np
profile=""
import pandas as pd
tablea = pd.read_csv('C:\\Users\\kpnth\\OneDrive - Chulalongkorn University\\Desktop\\CU\\Senior project\\Code\\SeniorPrevios\\Rula_score\\TableA.csv')
tableb = pd.read_csv('C:\\Users\\kpnth\\OneDrive - Chulalongkorn University\\Desktop\\CU\\Senior project\\Code\\SeniorPrevios\\Rula_score\\TableB.csv')
tablec = pd.read_csv('C:\\Users\\kpnth\\OneDrive - Chulalongkorn University\\Desktop\\CU\\Senior project\\Code\\SeniorPrevios\\Rula_score\\TableC.csv')
#EDIT 11/12/2024#
def rula_risk(point_score):
    A_1 = point_score['upper_arm']
    A_2 = point_score['lower_arm']
    A_3 = point_score['wrist']
    A_4 = point_score['wrist_twist']
    B_9 = point_score['neck']
    B_10 = point_score['trunk']
    B_11 = point_score['legs']
    muscle_use_a = point_score['muscle_use_a']
    force_load_a = point_score['force_load_a']
    force_load_b = point_score['force_load_b']
    muscle_use_b = point_score['muscle_use_b']
    rula={}
    rula['score']='NULL'
    rula['risk']='NULL'
    if  A_1!=0 and A_2 !=0 and A_3!=0 and A_4!=0 and trunk!=0 and B_9!=0:
        #Table A:
        print(f"A1 :{A_1}")
        print(f"A2 :{A_2}")
        print(f"A3 :{A_3}")
        print(f"A4 :{A_4}")
        col_name=str(A_3)+'WT'+str(A_4)
        tablea_val=tablea[(tablea['UpperArm']== A_1) & (tablea['LowerArm']== A_2)]
        print(tablea_val[col_name],A_1, A_2)
        A_score=tablea_val[col_name].values[0]
        point_score['posture_score_a']=str(A_score)
        print(f"A Score = {A_score}")
        AA_score =A_score + muscle_use_a + force_load_a
        point_score['TABLE_ASCORE']=str(AA_score)
        print(f"RULA TABLE_A SUM:{AA_score}")
        #Table B:
        print(f"B9 :{B_9}")
        print(f"B10 :{B_10}")
        print(f"B11 :{B_11}")
        col_name=str(B_10)+str(B_11)
        tableb_val=tableb[(tableb['Neck']==B_9)]
        B_score=tableb_val[col_name].values[0]
        point_score['posture_score_b']=str(B_score)
        print(f"B Score:{B_score}")
        BB_score=B_score + muscle_use_b + force_load_b
        point_score['TABLE_BSCORE']=str(BB_score)
        print(f"RULA TABLE_B SUM:{BB_score}")
        #Table C:
        if AA_score>=8:
            AA_score=8
        if BB_score>=7:
            BB_score=7
        col_name=str(BB_score)
        tablec_val=tablec[(tablec['Score']==AA_score)]
        C_score=tablec_val[col_name].values[0]
        print(f"RULA TABLE_C SUM:{C_score}")
        if C_score==1 or C_score==2:
            rula['score']=str(C_score)
            rula['risk']='Negligible'
        elif C_score==3 or C_score==4:
            rula['score']=str(C_score)
            rula['risk']='Low risk'
        elif C_score==5 or C_score==6:
            rula['score']=str(C_score)
            rula['risk']='Medium risk'
        elif C_score>6:
            rula['score']=str(C_score)
            rula['risk']='Very high risk'
    return rula, point_score
#EDIT 11/12/2024#
#EDIT 11/12/2024#
def rula_score(angle_dict, pose,profile):
    global wrist, trunk, upper_Shoulder, lower_Limb, neck, wrist_twist, legs, muscle_use_a, force_load_a, force_load_b, muscle_use_b
    Nose = pose[0]
    R_Shoulder = pose[12]
    R_Elbow = pose[14]
    R_Wrist = pose[16]
    L_Shoulder = pose[11]
    L_Elbow = pose[13]
    L_Wrist = pose[15]
    R_Hip = pose[24]
    L_Hip = pose[23]
    R_Knee = pose[26]
    R_Ankle = pose[28]
    L_Knee = pose[25]
    L_Ankle = pose[27]
    R_Eye=pose[5]
    L_Eye=pose[2]
    R_Ear = pose[8]
    L_Ear = pose[7]
    L_Foot = pose[31]
    R_Foot = pose[32]
    R_Palm = pose[20]
    L_Palm = pose[19]
    point_score={}
    if profile:
        Rupper_Shoulder=0
        Lupper_Shoulder=0
        Rangle1 = angle_dict['Rshoulder']
        if str(Rangle1) !='NULL' or str(Rangle1)=='nan':
            if Rangle1 >= 0 and Rangle1 <=20:
                Rupper_Shoulder=1
            elif Rangle1 >20 and Rangle1 <=45:
                Rupper_Shoulder=2
            elif Rangle1 >45 and Rangle1 <=90:
                Rupper_Shoulder=3
            elif Rangle1 >90:
                Rupper_Shoulder=4
        else:
            Rupper_Shoulder=1
        Rupper_Shoulder += angle_dict['Rshoulder_abduct']
        Langle1 = angle_dict['Lshoulder']
        if str(Langle1) !='NULL' or str(Langle1)=='nan':
            if Langle1 >= 0 and Langle1 <=20:
                Lupper_Shoulder=1
            elif Langle1 >20 and Langle1 <=45:
                Lupper_Shoulder=2
            elif Langle1 >45 and Langle1 <=90:
                Lupper_Shoulder=3
            elif Langle1 >90:
                Lupper_Shoulder=4
            else:
                Lupper_Shoulder=1
        else:
            Lupper_Shoulder=1
        Lupper_Shoulder += angle_dict['Lshoulder_abduct']
        if Rupper_Shoulder>=Lupper_Shoulder:
            point_score['upper_arm'] = Rupper_Shoulder
            point_score['upper_arm_adjustment'] = angle_dict['Rshoulder_abduct']
            point_score['upperarm_side'] = 'Right'
        else:
            point_score['upper_arm'] = Lupper_Shoulder
            point_score['upper_arm_adjustment'] = angle_dict['Lshoulder_abduct']
            point_score['upperarm_side'] = 'Left'
        #-------------------------------------lower_arm-------------------------------------
        Rlower_Limb=0
        Llower_Limb=0
        Rangle2 = angle_dict['Relbow']
        if str(Rangle2) !='NULL' or str(Rangle2)=='nan':
            Rangle2=int(Rangle2)
            if Rangle2 >= 60 and Rangle2 <=100:
                Rlower_Limb=1

            elif Rangle2 >0 and Rangle2 <60: 
                Rlower_Limb=2
               
            elif Rangle2>100:
                Rlower_Limb=2
            else:
                Rlower_Limb=1
        else:
            Rlower_Limb=1
        Rlower_Limb_adjust = angle_dict['Rlower_adjust']
        Rlower_Limb += Rlower_Limb_adjust

        Langle2 = angle_dict['Lelbow']
        if str(Langle2) !='NULL' or str(Langle2)=='nan':
            Langle2=int(Langle2)
            if Langle2 >= 60 and Langle2 <=100:
                Llower_Limb=1

            elif Langle2 >0 and Langle2 <60: 
                Llower_Limb=2
               
            elif Langle2>100:
                Llower_Limb=2
            else:
                Llower_Limb=1
        else:
            Llower_Limb=1
        Llower_Limb_adjust = angle_dict['Llower_adjust']
        Llower_Limb += Llower_Limb_adjust
        if Rlower_Limb>=Llower_Limb:
            point_score['lower_arm'] = Rlower_Limb
            point_score['lower_arm_adjustment'] = Rlower_Limb_adjust
            point_score['lowerarm_side'] = 'Right'
        else:
            point_score['lower_arm'] = Llower_Limb
            point_score['lower_arm_adjustment'] = Llower_Limb_adjust
            point_score['lowerarm_side'] = 'Left'
        #-------------------------------------wrist-------------------------------------
        Rwrist=0
        Lwrist=0
        Rangle3 = angle_dict['Rwrist']
        if str(Rangle3) !='NULL' or str(Rangle3)=='nan':
            Rangle3=abs(int(Rangle3))
            Rwrist=1
            if Rangle3 >= 0 and Rangle3<=15:
                Rwrist=2
            elif Rangle3 >15:
                Rwrist=3
        else:
            Rwrist=1  
        Rwrist_adjust = 0
        Rwrist += Rwrist_adjust    
        Langle3 = angle_dict['Rwrist']
        if str(Langle3) !='NULL' or str(Langle3)=='nan':
            Langle3=abs(int(Langle3))
            Lwrist=1
            if Langle3 >= 0 and Langle3<=15:
                Lwrist=2
            elif Langle3 >15:
                Lwrist=3
        else:
            Lwrist=1  
        Lwrist_adjust = 0
        Lwrist += Lwrist_adjust    
        if Rwrist>=Lwrist:
            point_score['wrist'] = Rwrist
            point_score['wrist_adjust'] = Rwrist_adjust
            point_score['wrist_side'] = 'Right'
        else:
            point_score['wrist'] = Lwrist
            point_score['wrist_adjust'] = Lwrist_adjust
            point_score['wrist_side'] = 'Left'
        #EDIT 11/12/2024#
        neck_angle=angle_dict['Neck']
        if str(neck_angle)!='NULL' or str(neck_angle)=='nan':
            neck_angle=abs(int(neck_angle))
            if neck_angle >= 0 and neck_angle <=10:
                neck=1
            elif neck_angle >10 and neck_angle <=20:
                neck=2
            elif neck_angle >20:
                neck=3
            else:
                neck=1
        else:
            neck=1
        point_score['neck_adjust'] = angle_dict['Neck_twist'] + angle_dict['Neck_sidebend']
        point_score['neck'] = neck + point_score['neck_adjust']
        #EDIT 11/12/2024#

        trunk_angle=angle_dict['Trunk']
        if str(trunk_angle) !='NULL':
            trunk_angle=abs(int(trunk_angle))
            trunk=1
            if trunk_angle>0 and trunk_angle <= 20:
                trunk=2
            elif trunk_angle >20 and trunk_angle <=60:
                trunk=3
            elif trunk_angle >60:
                trunk=4
            else:
                trunk=1
        else:
            trunk=1
        point_score['trunk_adjust'] = angle_dict['Trunk_twist'] + angle_dict['Trunk_sidebend']
        point_score['trunk'] = trunk + point_score['trunk_adjust']
        #-------------------------------------wrist_twist-------------------------------------
        wrist_twist=1
        point_score['wrist_twist']=wrist_twist
        #-------------------------------------legs-------------------------------------
        legs=1
        point_score['legs']=legs
        muscle_use_a=0
        force_load_a=0
        muscle_use_b=0
        force_load_b=0
        point_score['muscle_use_a']=muscle_use_a
        point_score['force_load_a']=force_load_a
        point_score['muscle_use_b']=muscle_use_b
        point_score['force_load_b']=force_load_b
        rula, point_score = rula_risk(point_score)
        rula['point_score']=point_score
    else:
        rula={}
        rula['score']='NULL'
        rula['risk']='NULL'
        rula['point_score']={}
    return rula
#EDIT10/12/2024#
def validate_value(value) :
    return value is not None and str(value).lower() != 'null' and not pd.isna(value)
def angle(a,b):
    if not validate_value(a) or not validate_value(b) :
        return 'NULL'
    magnitudea = np.linalg.norm(a)
    magnitudeb = np.linalg.norm(b)
    if magnitudea == 0 or magnitudeb == 0:
        return 'NULL'
    dot_product = np.dot(a,b)
    magnitude_product = magnitudea * magnitudeb
    cos_theta = dot_product/magnitude_product
    clippedcos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_radians = np.arccos(clippedcos_theta)
    angle_degrees = np.rad2deg(angle_radians)
    return angle_degrees
#EDIT10/12/2024#

#EDIT10/12/2024# 
def project_onplane(normal, vector):
    normal_magnitude = np.linalg.norm(normal)
    if normal_magnitude == 0 :
        return 'NULL'
    unit_normal = normal/normal_magnitude
    vector_onnormal = np.dot(vector, unit_normal) * unit_normal
    projection = vector - vector_onnormal
    return projection
    # if np.linalg.norm(normal)!=0:
    #     unit_normal =  normal / np.linalg.norm(normal)
    #     v_onnormal = np.dot(v , unit_normal)*unit_normal
    #     projection = v - v_onnormal
    #     return projection
    # else: return 'NULL'
#EDIT10/12/2024#
# normal = np.array([0, 0, 1])
# vector = np.array([1, 1, 1])
# projection = project_onplane(normal, vector)
# print("Projected Vector:", projection)
#EDIT10/12/2024#
def rotate_vector(vector,vector_axis,theta):
    axis_magnitude = np.linalg.norm(vector_axis)
    if axis_magnitude == 0:
        return None
    axis_unit = vector_axis / axis_magnitude
    vector_parallel = np.dot(vector, axis_unit) * axis_unit
    vector_perpendicular = vector - vector_parallel
    if np.linalg.norm(vector_perpendicular) == 0:
        return None
    cross_product = np.cross(axis_unit, vector_perpendicular)
    rotated_perpendicular = (vector_perpendicular * np.cos(theta) + cross_product * np.sin(theta))
    rotated_vector = vector_parallel + rotated_perpendicular
    return rotated_vector
    # v_unit_axis = v_axis / np.linalg.norm(v_axis)
    # v_parallel = np.dot(v , v_unit_axis)*v_unit_axis
    # v_perpen = v-v_parallel
    # w = np.cross(v_axis,v_perpen)
    # x1 = np.cos(theta)/np.linalg.norm(v_perpen)
    # x2 = np.sin(theta)/np.linalg.norm(w)
    # v_perpen_theta = np.linalg.norm(v_axis)*(x1*v_perpen + x2*w)
    # v_rotate = v_perpen_theta + v_parallel
    # return v_rotate
#EDIT10/12/2024#
# vector = np.array([1, 0, 0])
# axis = np.array([0, 0, 1])
# angle_rad = np.pi / 2  
# rotated = rotate_vector(vector, axis, angle_rad)
# print("Rotated Vector:", rotated)
#EDIT10/12/ BY CREATED NEW DEF#
def to_unitvec(vector):
    return vector/np.linalg.norm(vector)
#EDIT10/12/ BY CREATED NEW DEF#

def projection_valid(original_vector, projected_vector, threshold=0.5):
    if np.linalg.norm(original_vector) == 0:
        return False
    return np.linalg.norm(projected_vector) / np.linalg.norm(original_vector) >= threshold

def twist_angle(reference_vector, vector1, vector2, threshold=30):
    proj1 = project_onplane(reference_vector, vector1)
    proj2 = project_onplane(reference_vector, vector2)
    if proj1 is None or proj2 is None or np.linalg.norm(proj1) == 0 or np.linalg.norm(proj2) == 0:
        return 0 
    return 1 if angle(proj1, proj2) > threshold else 0

def sidebend_angle(body_vector, projection_vector, reference_vector, threshold_ratio=0.4, angle_threshold=20):
    if body_vector is None or projection_vector is None or np.linalg.norm(body_vector) == 0:
        return 0 
    proj_ratio = np.linalg.norm(projection_vector) / np.linalg.norm(body_vector)
    if proj_ratio < threshold_ratio:
        return 0 
    return 1 if angle(projection_vector, reference_vector) > angle_threshold else 0
#EDIT11/12/2024#

def angle_calc(pose_list, reference_vector ='Nonee', head_theta0='None', back_theta0='None'):
    pose = np.array(pose_list)
    Nose = pose[0]
    R_Shoulder = pose[12]
    R_Elbow = pose[14]
    R_Wrist = pose[16]
    L_Shoulder = pose[11]
    L_Elbow = pose[13]
    L_Wrist = pose[15]
    R_Hip = pose[24]
    L_Hip = pose[23]
    R_Knee = pose[26]
    R_Ankle = pose[28]
    L_Knee = pose[25]
    L_Ankle = pose[27]
    R_Eye=pose[5]
    L_Eye=pose[2]
    R_Ear = pose[8]
    L_Ear = pose[7]
    L_Foot = pose[31]
    R_Foot = pose[32]
    R_Palm = pose[20]
    L_Palm = pose[19]
    left=0
    right=0
    front=0
    if abs(round(R_Elbow[3],2)-round(L_Elbow[3],2))<=0.2:
        front+=1
    elif round(R_Elbow[3],2)>round(L_Elbow[3],2)+0.2:
        right+=1
    else:
        left+=1
    if abs(round(R_Wrist[3],2)-round(L_Wrist[3],2))<=0.2:
        front+=1
    elif round(R_Wrist[3],2)>round(L_Wrist[3],2):
        right+=1
    else:
        left+=1
    if abs(round(R_Knee[3],2)-round(L_Knee[3],2))<=0.2:
        front+=1
    elif round(R_Knee[3],2)>round(L_Knee[3],2):
        right+=1
    else:
        left+=1
    if abs(round(R_Ankle[3],2)-round(L_Ankle[3],2))<=0.2:
        front+=1
    elif round(R_Ankle[3],2)>round(L_Ankle[3],2):
        right+=1
    else:
        left+=1
    if abs(round(R_Foot[3],2)-round(L_Foot[3],2))<=0.2:
        front+=1
    elif round(R_Foot[3],2)>round(L_Foot[3],2):
        right+=1
    else:
        left+=1
    if abs(round(R_Palm[3],2)-round(L_Palm[3],2))<0.2:
        front+=1
    elif round(R_Palm[3],2)>round(L_Palm[3],2):
        right+=1
    else:
        left+=1
    if left < right:
        if right > front:
            profile="Right"
        else:
            profile="Front"
    elif right < left:
        if left > front:
            profile="Left"
        else:
            profile="Front"
    else:
        profile="Front"
    angle_dict={}
    
    RupperArm = (R_Elbow - R_Shoulder)[:3]
    LupperArm = (L_Elbow - L_Shoulder)[:3]
    
    RlowerArm = (R_Wrist - R_Elbow)[:3]
    LlowerArm = (L_Wrist - L_Elbow)[:3]
    
    Rhand     = (R_Palm - R_Wrist)[:3]
    Lhand     = (L_Palm - L_Wrist)[:3]
    
    midNeck   = ((L_Shoulder + R_Shoulder)/2)[:3]
    midWaist  = ((L_Hip + R_Hip)/2)[:3]
    Head      = Nose[:3] - midNeck
    Back      = midNeck - midWaist
    
    Relbow    = angle(RlowerArm,RupperArm)
    Rwrist    = angle(Rhand,RlowerArm)
    
    Lelbow    = angle(LlowerArm,LupperArm)
    Lwrist    = angle(Lhand,LlowerArm)
    
    midAnkle  = ((L_Ankle + R_Ankle)/2)[:3]   ##############
    Leg       = midWaist - midAnkle #############  
    if str(reference_vector)!='Nonee':
        sagittal = project_onplane(reference_vector[3] ,L_Hip[:3] - R_Hip[:3])  
        coronal = to_unitvec(np.cross(sagittal, reference_vector[3]))
        body_proj_upperarm = project_onplane(sagittal,-Back)

        RupperArm_proj = project_onplane(sagittal,RupperArm)
        Rshoulder = angle(RupperArm_proj, body_proj_upperarm)
        print(f"Rshoulder angle: {Rshoulder}")
        
        LupperArm_proj = project_onplane(sagittal,LupperArm)
        Lshoulder = angle(LupperArm_proj, body_proj_upperarm)
        print(f"Lshoulder angle: {Lshoulder}")
        #EDIT10/12/ BY CREATED NEW DEF#
        if projection_valid(RupperArm, RupperArm_proj,): 
            Rshoulder = Rshoulder
        else :
            Rshoulder = 0
        if projection_valid(LupperArm, LupperArm_proj,): 
            Lshoulder = Lshoulder
        else :
            Lshoulder = 0
        #EDIT10/12/ BY USED NEW DEF#
        
        RupperArm_proj_abduct = project_onplane(reference_vector[3],RupperArm) 
        LupperArm_proj_abduct = project_onplane(reference_vector[3],LupperArm) 
        Hip_abduct = project_onplane(reference_vector[3],L_Hip[:3]-R_Hip[:3]) 
        
        if not projection_valid(RupperArm, RupperArm_proj_abduct):
            Rshoulder_abduct = 0
        elif angle(RupperArm_proj_abduct, -Hip_abduct) < 45:
            Rshoulder_abduct = 1
        else:
            Rshoulder_abduct = 0
        if not projection_valid(LupperArm, LupperArm_proj_abduct):
            Lshoulder_abduct = 0
        elif angle(LupperArm_proj_abduct, Hip_abduct) < 45:
            Lshoulder_abduct = 1
        else:
            Lshoulder_abduct = 0

        if np.linalg.norm(R_Palm - R_Hip) > np.linalg.norm(R_Palm - L_Hip): 
            Rlower_adjust = 1 
        else: 
            Rlower_adjust = 0
        if np.linalg.norm(L_Palm - L_Hip) > np.linalg.norm(L_Palm - R_Hip): 
            Llower_adjust = 1 
        else: 
            Llower_adjust = 0
            
        Back_default_angle = back_theta0  # Z vector reference
        Back_default = rotate_vector(reference_vector[3], -sagittal, np.deg2rad(Back_default_angle))
        Trunk = angle(Back, Back_default)
        print("Trunk Angle:", Trunk)
        
        Trunk_twist = twist_angle(reference_vector[3], L_Hip[:3] - R_Hip[:3], L_Shoulder[:3] - R_Shoulder[:3])
        print(f"Twist twist angle : {Trunk_twist}")

        Trunk_proj_coronal = project_onplane(coronal, Back)
        Trunk_sidebend = sidebend_angle(Back, Trunk_proj_coronal, reference_vector[3])
        print(f"Trunk bend angle:{Trunk_sidebend}")

        Head_default_angle = head_theta0
        Head_default = rotate_vector(coronal, -sagittal, np.deg2rad(90 - Head_default_angle))
        Head_proj_sag = project_onplane(sagittal, Head)
        Neck = angle(project_onplane(sagittal, Head_default), Head_proj_sag)
        Neck = abs(Neck - Trunk)

        Head_proj_transverse = project_onplane(reference_vector[3], Head)
        neck_Hip = project_onplane(reference_vector[3], L_Hip[:3] - R_Hip[:3])
        if angle(Head_proj_transverse, neck_Hip) < 45 or angle(Head_proj_transverse, -neck_Hip) < 45 :
            necktwist = 1
            print("Neck is Twist") 
        else: 
            necktwist = 0
            print("Neck is not Twist")
        Head_proj_coronal = project_onplane(coronal, Head)
        Neck_sidebend = sidebend_angle(Head, Head_proj_coronal, reference_vector[3])
        print(f"Neck sidebend:{Neck_sidebend}")
    else:
        neck_2d = midNeck.copy()
        neck_2d[2]=0
        waist_2d = midWaist.copy()
        waist_2d[2]=0
        transverse = to_unitvec(neck_2d - waist_2d)
        L_Hip_2d = L_Hip[:3].copy()
        L_Hip_2d[2]=0
        R_Hip_2d = R_Hip[:3].copy()
        R_Hip_2d[2]=0
        sagittal = project_onplane(transverse , L_Hip_2d - R_Hip_2d)
        coronal = np.cross(sagittal,transverse)
        head_theta0 = angle(Head,transverse)
        trunk_theta0 = angle(Back, transverse)
        return np.array([Back, Head, sagittal, transverse, coronal]) , head_theta0, trunk_theta0
    
    angle_dict = {'Rshoulder':Rshoulder, 
                  'Relbow': Relbow,
                  'Rwrist':Rwrist,
                  'Lshoulder':Lshoulder,
                  'Lelbow':Lelbow, 
                  'Lwrist':Lwrist,
                  'Neck':Neck,
                  'Trunk':Trunk,
                  'Neck_twist':necktwist,
                  'Rshoulder_abduct':Rshoulder_abduct,                  'Lshoulder_abduct':Lshoulder_abduct, 'Rlower_adjust':Rlower_adjust,
                  'Llower_adjust':Llower_adjust, 
                  'Neck_sidebend':Neck_sidebend,
                  'Trunk_twist':Trunk_twist,
                  'Trunk_sidebend':Trunk_sidebend}
    rula = rula_score(angle_dict,pose,profile)
    return rula['score'], angle_dict , profile, necktwist
