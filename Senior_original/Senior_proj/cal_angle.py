import math
import numpy as np
profile=""
import pandas as pd
tablea=pd.read_csv(r'C:\Users\pan\Downloads\Senior_original\Senior_proj\Rula_score\TableA.csv')
tableb=pd.read_csv(r'C:\Users\pan\Downloads\Senior_original\Senior_proj\Rula_score\TableB.csv')
tablec=pd.read_csv(r'C:\Users\pan\Downloads\Senior_original\Senior_proj\Rula_score\TableC.csv')

def rula_risk(point_score):
    upper_Shoulder = point_score['upper_arm']
    lower_Limb = point_score['lower_arm']
    wrist = point_score['wrist']
    wrist_twist = point_score['wrist_twist']
    neck = point_score['neck']
    trunk = point_score['trunk']
    legs = point_score['legs']

    muscle_use = point_score['muscle_use_a']
    force_load_a = point_score['force_load_a']
    force_load_b = point_score['force_load_b']
    upper_body_muscle = point_score['muscle_use_b']
    rula={}
    rula['score']='NULL'
    rula['risk']='NULL'
    if wrist!=0 and  trunk!=0 and upper_Shoulder!=0 and lower_Limb!=0 and neck!=0 and wrist_twist!=0:
        #Table A:
        col_name=str(wrist)+'WT'+str(wrist_twist)
        tablea_val=tablea[(tablea['UpperArm']==upper_Shoulder) & (tablea['LowerArm']==lower_Limb)]
        # print(tablea_val[col_name],upper_Shoulder, lower_Limb)
        tablea_val=tablea_val[col_name].values[0]
        point_score['posture_score_a']=str(tablea_val)
        tablea_val=tablea_val+muscle_use+force_load_a
        point_score['wrist_and_arm_score']=str(tablea_val)

        #Table B:
        col_name=str(trunk)+str(legs)
        tableb_val=tableb[(tableb['Neck']==neck)]
        tableb_val=tableb_val[col_name].values[0]
        point_score['posture_score_b']=str(tableb_val)
        tableb_val=tableb_val+force_load_b+upper_body_muscle
        point_score['neck_trunk_leg_score']=str(tableb_val)
        # print(tablea_val,tableb_val)
        #Table C
        if tablea_val>=8:
            tablea_val=8
        if tableb_val>=7:
            tableb_val=7

        col_name=str(tableb_val)
        tablec_val=tablec[(tablec['Score']==tablea_val)]
        tablec_val=tablec_val[col_name].values[0]

        if tablec_val==1 or tablec_val==2:
            rula['score']=str(tablec_val)
            rula['risk']='Negligible'
        elif tablec_val==3 or tablec_val==4:
            rula['score']=str(tablec_val)
            rula['risk']='Low risk'
        elif tablec_val==5 or tablec_val==6:
            rula['score']=str(tablec_val)
            rula['risk']='Medium risk'
        elif tablec_val>6:
            rula['score']=str(tablec_val)
            rula['risk']='Very high risk'

    return rula, point_score
def rula_score(angle_dict, pose,profile):
    global wrist, trunk, upper_Shoulder, lower_Limb, neck, wrist_twist, legs, muscle_use, force_load_a, force_load_b, upper_body_muscle
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
        #-------------------------------------upper_arm -------------------------------------
        Rupper_Shoulder=0
        Lupper_Shoulder=0
                #----------------------------RightUpperArm-----------------------------------
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
            # else:
            #     Rupper_Shoulder=1
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

        #-------------------------------------neck-------------------------------------
        angle4=angle_dict['Neck']
        if str(angle4)!='NULL' or str(angle4)=='nan':
            angle4=abs(int(angle4))
            if angle4 >= 0 and angle4 <=10:
                neck=1
            elif angle4 >10 and angle4 <=20:
                neck=2
            elif angle4 >20:
                neck=3
            else:
                neck=1
        else:
            neck=1
        neck += angle_dict['Neck_twist'] + angle_dict['Neck_sidebend']
        point_score['neck'] = neck
        point_score['neck_adjust'] = angle_dict['Neck_twist'] + angle_dict['Neck_sidebend']
        #-------------------------------------truck-------------------------------------
        angle5=angle_dict['Trunk']
        if str(angle5) !='NULL':
            angle5=abs(int(angle5))
            trunk=1
            if angle5>0 and angle5 <= 20:
                trunk=2
            elif angle5 >20 and angle5 <=60:
                trunk=3
            elif angle5 >60:
                trunk=4
            else:
                trunk=1
        else:
            trunk=1
        t_adjustment = angle_dict['trunktwist'] + angle_dict['Trunk_sidebend']
        point_score['trunk_adjustment']=t_adjustment
        trunk = trunk + t_adjustment
        point_score['trunk']=trunk
        #-------------------------------------wrist_twist-------------------------------------
        wrist_twist=1
        point_score['wrist_twist']=wrist_twist
        #-------------------------------------legs-------------------------------------
        legs=1
        point_score['legs']=legs
        muscle_use=0
        force_load_a=0
        force_load_b=0
        upper_body_muscle=0
        point_score['muscle_use_a']=muscle_use
        point_score['force_load_a']=force_load_a
        point_score['force_load_b']=force_load_b
        point_score['muscle_use_b']=upper_body_muscle

        rula, point_score = rula_risk(point_score)
        rula['point_score']=point_score
#     rula={}
#     rula['score']='NULL'
#     rula['risk']='NULL'
#     rula['point_score']={}
    else:
        rula={}
        rula['score']='NULL'
        rula['risk']='NULL'
        rula['point_score']={}

    return rula

def angle(a,b):
    suma = np.sqrt(a.dot(a))
    sumb = np.sqrt(b.dot(b))
    if np.array_equal(a, 'NULL') or np.array_equal(b, 'NULL'): 
        return 'NULL'
    if suma*sumb!=0:
        angle = np.arccos(np.dot(a,b)/(suma*sumb))
        angle = np.rad2deg(angle)
        return angle
    else:
        return 'NULL'
def project_onplane(normal, v):
    if np.linalg.norm(normal)!=0:
        unit_normal =  normal / np.linalg.norm(normal)
        v_onnormal = np.dot(v , unit_normal)*unit_normal
        projection = v - v_onnormal
        return projection
    else: return 'NULL'
    
def rotate_vector(v,v_axis,theta):
    v_unit_axis = v_axis / np.linalg.norm(v_axis)
    v_parallel = np.dot(v , v_unit_axis)*v_unit_axis
    v_perpen = v-v_parallel
    w = np.cross(v_axis,v_perpen)
    x1 = np.cos(theta)/np.linalg.norm(v_perpen)
    x2 = np.sin(theta)/np.linalg.norm(w)
    v_perpen_theta = np.linalg.norm(v_axis)*(x1*v_perpen + x2*w)
    v_rotate = v_perpen_theta + v_parallel
    return v_rotate

def to_unitvec(vector):
    return vector/np.linalg.norm(vector)
def angle_calc(pose_list, vector0 ='Nonee', head_theta0='None', back_theta0='None'):
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
    
    LupperArm = (L_Elbow - L_Shoulder)[:3]
    LlowerArm = (L_Wrist - L_Elbow)[:3]
    Lhand     = (L_Palm - L_Wrist)[:3]
    RupperArm = (R_Elbow - R_Shoulder)[:3]
    RlowerArm = (R_Wrist - R_Elbow)[:3]
    Rhand     = (R_Palm - R_Wrist)[:3]
    midNeck   = ((L_Shoulder + R_Shoulder)/2)[:3]
    Head      = Nose[:3] - midNeck


    midWaist  = ((L_Hip + R_Hip)/2)[:3]
    Back      = midNeck - midWaist

    Relbow    = angle(RlowerArm,RupperArm)
    Rwrist    = angle(Rhand,RlowerArm)
    Lelbow    = angle(LlowerArm,LupperArm)
    Lwrist    = angle(Lhand,LlowerArm)

    midAnkle  = ((L_Ankle + R_Ankle)/2)[:3]  
    Leg       = midWaist - midAnkle  

    #______________________________________________________________________________________________
    if str(vector0)!='Nonee': #vector0 = [Back, Head, sagittal, transverse, coronal], head_theta0, trunk_theta0
        sagittal = project_onplane(vector0[3] , L_Hip[:3] - R_Hip[:3])  #sagittal plane
        coronal = to_unitvec(np.cross(sagittal, vector0[3])   )             #coronal plane
        # if angle(L_Hip[:3] - R_Hip[:3],vector0[2])<45:
        #     profile_2='front'
        # else: profile_2 = 'side'
        # print(profile_2,angle(L_Hip[:3] - R_Hip[:3],sagittal))
        #------------------------------------upperarm--------------------------------------
        RupperArm_proj = project_onplane(sagittal,RupperArm)    #project upperarm onto sagittal plane
        body_proj_upperarm = project_onplane(sagittal,-Back)
        Rshoulder = angle(RupperArm_proj, body_proj_upperarm)    #angle between RupperArm_proj and vertical vector(transverse)
        LupperArm_proj = project_onplane(sagittal,LupperArm)
        Lshoulder = angle(LupperArm_proj, body_proj_upperarm)
        if np.linalg.norm(RupperArm_proj)/np.linalg.norm(RupperArm)<0.5: Rshoulder=0
        if np.linalg.norm(LupperArm_proj)/np.linalg.norm(LupperArm)<0.5: Lshoulder=0

        # RupperArm_proj_abduct = project_onplane(coronal,RupperArm) 
        # LupperArm_proj_abduct = project_onplane(coronal,LupperArm) 
        # Back_abduct = project_onplane(coronal,-Back) 
        # print(np.linalg.norm(RupperArm_proj_abduct)/np.linalg.norm(RupperArm),'--------')
        # if np.linalg.norm(RupperArm_proj_abduct)/np.linalg.norm(RupperArm)<0.4: Rshoulder_abduct = 0
        # elif angle(RupperArm_proj_abduct, Back_abduct)>45 and angle(RupperArm_proj_abduct, Back_abduct)!='NULL': 
        #     Rshoulder_abduct = 1 
        # else: Rshoulder_abduct=0
        # if np.linalg.norm(LupperArm_proj_abduct)/np.linalg.norm(LupperArm)<0.4: Lshoulder_abduct = 0
        # elif angle(LupperArm_proj_abduct, Back_abduct)>45 and angle(LupperArm_proj_abduct, Back_abduct)!='NULL': 
        #     Lshoulder_abduct = 1 
        # else: Lshoulder_abduct=0

        RupperArm_proj_abduct = project_onplane(vector0[3],RupperArm) 
        LupperArm_proj_abduct = project_onplane(vector0[3],LupperArm) 
        Hip_abduct = project_onplane(vector0[3],L_Hip[:3]-R_Hip[:3]) 
        if np.linalg.norm(RupperArm_proj_abduct)/np.linalg.norm(RupperArm)<0.5:Rshoulder_abduct=0
        elif angle(RupperArm_proj_abduct, -Hip_abduct)<45: Rshoulder_abduct = 1 
        else: Rshoulder_abduct = 0

        if np.linalg.norm(LupperArm_proj_abduct)/np.linalg.norm(LupperArm)<0.5:Lshoulder_abduct=0
        elif angle(LupperArm_proj_abduct, Hip_abduct)<45: Lshoulder_abduct = 1 
        else: Lshoulder_abduct = 0

        #-----------------------------------lower_arm----------------------------------------
        if np.linalg.norm(R_Palm - R_Hip)>np.linalg.norm(R_Palm - L_Hip): Rlower_adjust = 1 
        else: Rlower_adjust = 0
        if np.linalg.norm(L_Palm - L_Hip)>np.linalg.norm(L_Palm - R_Hip): Llower_adjust = 1 
        else: Llower_adjust = 0
        # RlowerArm_proj_abduct = project_onplane(coronal,RlowerArm) 
        # LlowerArm_proj_abduct = project_onplane(coronal,LlowerArm) 
        # if np.linalg.norm(RlowerArm_proj_abduct)/np.linalg.norm(RlowerArm)<0.4: Rlower_adjust = 0
        # elif angle(RlowerArm_proj_abduct, -vector0[3])>10 and angle(RlowerArm_proj_abduct, -vector0[3])!='NULL': 
        #     Rlower_adjust = 1 
        # else: Rlower_adjust=0
        # if np.linalg.norm(LlowerArm_proj_abduct)/np.linalg.norm(LlowerArm)<0.4: Llower_adjust = 0
        # elif angle(LlowerArm_proj_abduct, -vector0[3])>10 and angle(LlowerArm_proj_abduct, -vector0[3])!='NULL': 
        #     Llower_adjust = 1 
        # else: Llower_adjust=0
        #-----------------------------------Trunk----------------------------------------
        Back_default_angle = back_theta0
        Back_default = rotate_vector(vector0[3] , -sagittal, np.deg2rad(Back_default_angle))
        Trunk = angle(Back, Back_default) #angle between back and vertical plane

        #detect trunk twist
        LRHip_proj_transverse =  project_onplane(vector0[3] , L_Hip[:3] - R_Hip[:3])
        LRBack_proj_transverse = project_onplane(vector0[3],L_Shoulder[:3]-R_Shoulder[:3])
        if angle(LRHip_proj_transverse, LRBack_proj_transverse)>30:  
            trunktwist = 1
        else: trunktwist=0

        #detect trunk side bending 
        Trunk_proj_coronal = project_onplane(coronal,Back)
        if np.linalg.norm(Trunk_proj_coronal)/np.linalg.norm(Back)<0.4: Trunk_sidebend = 0
        elif angle(Trunk_proj_coronal, vector0[3])>20 and angle(Trunk_proj_coronal, vector0[3])!='NULL': 
            Trunk_sidebend = 1 
        else: Trunk_sidebend=0
        #------------------------------------Neck---------------------------------------
        #define head default vector
        Head_default_angle = head_theta0
        Head_default = rotate_vector(coronal , -sagittal, np.deg2rad(90-Head_default_angle))

        Head_proj_sag = project_onplane(sagittal,Head)    #project neck to sagittal plane
        Neck = angle(project_onplane(sagittal,Head_default) , Head_proj_sag) #angle between default head proj and neck proj
        # delta_trunk = angle(Back, vector0[0])
        Neck = abs(Neck-Trunk)

        #detect neck twist
        Head_proj_transverse = project_onplane(vector0[3],Head)    #project neck onto transverse plane
        neck_Hip = project_onplane(vector0[3],L_Hip[:3] - R_Hip[:3]) 
        if angle(Head_proj_transverse, neck_Hip)<45 or angle(Head_proj_transverse, -neck_Hip)<45:  #check neck twist
            necktwist = 1
        else: necktwist=0

        #detect neck side bending 
        Head_proj_coronal = project_onplane(coronal,Head)
        if np.linalg.norm(Head_proj_coronal)/np.linalg.norm(Head)<0.4: Neck_sidebend = 0
        elif angle(Head_proj_coronal, vector0[3])>20 and angle(Head_proj_coronal, vector0[3])!='NULL': 
            Neck_sidebend = 1 
        else: Neck_sidebend=0


    else: #-----------------------initial collection----------------------
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
        # normal_body = np.cross(R_Hip[:3]-R_Shoulder[:3], L_Shoulder[:3]-R_Shoulder[:3])
        # if angle(normal_body, )
        trunk_theta0 = angle(Back, transverse)
        return np.array([Back, Head, sagittal, transverse, coronal]) , head_theta0, trunk_theta0
    # #______________________________________________________________________________________________

    angle_dict = {'Rshoulder':Rshoulder , 'Relbow': Relbow , 'Rwrist':Rwrist , 'Lshoulder':Lshoulder , 'Lelbow':Lelbow, 
    'Lwrist':Lwrist , 'Neck':Neck , 'Trunk':Trunk  , 'Neck_twist':necktwist, 'Rshoulder_abduct':Rshoulder_abduct,
    'Lshoulder_abduct':Lshoulder_abduct
    , 'Rlower_adjust':Rlower_adjust,'Llower_adjust':Llower_adjust, 'Neck_sidebend':Neck_sidebend,'trunktwist':trunktwist,'Trunk_sidebend':Trunk_sidebend}

    rula = rula_score(angle_dict,pose,profile)

    return rula['score'], angle_dict , profile, necktwist
