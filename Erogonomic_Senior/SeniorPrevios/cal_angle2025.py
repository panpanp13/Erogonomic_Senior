import math
import numpy as np
profile=""
import pandas as pd
tablea = pd.read_csv(r'C:\Users\pan\Desktop\My_o3d_code\Erogonomic_Senior\SeniorPrevios\Rula_score\TableA.csv')
tableb = pd.read_csv(r'C:\Users\pan\Desktop\My_o3d_code\Erogonomic_Senior\SeniorPrevios\Rula_score\TableB.csv')
tablec = pd.read_csv(r'C:\Users\pan\Desktop\My_o3d_code\Erogonomic_Senior\SeniorPrevios\Rula_score\TableC.csv')
#EDIT 28/12/2024#
previous_angles = None
muscle_use_a = 0
muscle_use_b = 0
#EDIT 11/12/2024#
def rula_risk(point_score):
    print("start calculate")
    A_1 = point_score['upper_arm']
    A_2 = point_score['lower_arm']
    A_3 = point_score['wrist']
    A_4 = point_score['wrist_twist']
    B_9 = point_score['neck']
    B_10 = point_score['trunk']
    B_11 = point_score['legs']     
    force_load_a = point_score['force_load_a']
    force_load_b = point_score['force_load_b']
    point_score['muscle_use_a']=muscle_use_a
    point_score['muscle_use_b']=muscle_use_b
    # print(muscle_use_a)
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
        # print(tablea_val[col_name],A_1, A_2)
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
def rula_score(angle_dict, pose, profile, condition):
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
        # print("Found profile is True")
        Rupper_Shoulder=0
        Lupper_Shoulder=0
        Rangle1 = angle_dict['Rshoulder']
        # print(f"Rangle1 :{Rangle1}")
        if str(Rangle1) !='NULL' or str(Rangle1)=='nan':
            if Rangle1 >= 0 and Rangle1 <=20:
                Rupper_Shoulder=1
                # print(f"RUPPER: {Rupper_Shoulder}")
            elif Rangle1 >20 and Rangle1 <=45:
                Rupper_Shoulder=2
                # print(f"RUPPER: {Rupper_Shoulder}")
            elif Rangle1 >45 and Rangle1 <=90:
                Rupper_Shoulder=3
                # print(f"RUPPER: {Rupper_Shoulder}")
            elif Rangle1 >90:
                Rupper_Shoulder=4
                # print(f"RUPPER: {Rupper_Shoulder}")
        else:
            Rupper_Shoulder=1
            # print(f"RUPPER: {Rupper_Shoulder}")
        # Rupper_Shoulder += angle_dict['Rshoulder_abduct']
        # print(f"RUPPER2: {Rupper_Shoulder}")
        # print("PASS RIGHT UPPER")
        
        Langle1 = angle_dict['Lshoulder']
        # print(f"Langle1 : {Langle1}")
        if str(Langle1) !='NULL' or str(Langle1)=='nan':
            if Langle1 >= 0 and Langle1 <=20:
                Lupper_Shoulder=1
                # print(f"LUPPER: {Lupper_Shoulder}")
            elif Langle1 >20 and Langle1 <=45:
                Lupper_Shoulder=2
                # print(f"LUPPER: {Lupper_Shoulder}")
            elif Langle1 >45 and Langle1 <=90:
                Lupper_Shoulder=3
                # print(f"LUPPER: {Lupper_Shoulder}")
            elif Langle1 >90:
                Lupper_Shoulder=4
                # print(f"LUPPER: {Lupper_Shoulder}")
            else:
                Lupper_Shoulder=1
                # print(f"LUPPER: {Lupper_Shoulder}")
        else:
            Lupper_Shoulder=1
            # print(f"LUPPER: {Lupper_Shoulder}")
        
        # Lupper_Shoulder += angle_dict['Lshoulder_abduct']
        # print(f"LUPPER2: {Lupper_Shoulder}")
        # print("PASS LEFT UPPER")
        if Rupper_Shoulder>=Lupper_Shoulder:
            # print("R NA")
            point_score['upper_arm'] = Rupper_Shoulder
            point_score['upper_arm_adjustment'] = angle_dict['Rshoulder_abduct']
            point_score['upperarm_side'] = 'Right'
        else:
            # print("L NA")
            point_score['upper_arm'] = Lupper_Shoulder
            point_score['upper_arm_adjustment'] = angle_dict['Lshoulder_abduct']
            point_score['upperarm_side'] = 'Left'
        # CLEAR------9/1/2025
        #-------------------------------------lower_arm-------------------------------------
        Rlower_Limb=0
        Llower_Limb=0
        Rangle2 = angle_dict['Relbow']
        # print(f"Rangle2 :{Rangle2}")
        if str(Rangle2) !='NULL' or str(Rangle2)=='nan':
            Rangle2=int(Rangle2)
            if Rangle2 >= 60 and Rangle2 <=100:
                Rlower_Limb=1
                # print(f"RLOWER: {Rlower_Limb}")
            elif Rangle2 >0 and Rangle2 <60: 
                Rlower_Limb=2
                # print(f"RLOWER: {Rlower_Limb}")
            elif Rangle2>100:
                Rlower_Limb=2
                # print(f"RLOWER: {Rlower_Limb}")
            else:
                Rlower_Limb=1
                # print(f"RLOWER: {Rlower_Limb}")
        else:
            Rlower_Limb=1
            # print(f"RLOWER: {Rlower_Limb}")
        # Rlower_Limb_adjust = angle_dict['Rlower_adjust']
        # Rlower_Limb += Rlower_Limb_adjust
        # print(f"RLOWER2: {Rlower_Limb}")
        # print("PASS RIGHT LOWER")
        Langle2 = angle_dict['Lelbow']
        # print(f"Langle2 : {Langle2}")
        if str(Langle2) !='NULL' or str(Langle2)=='nan':
            Langle2=int(Langle2)
            if Langle2 >= 60 and Langle2 <=100:
                Llower_Limb=1
                # print(f"LLOWER: {Llower_Limb}")
            elif Langle2 >0 and Langle2 <60: 
                Llower_Limb=2
                # print(f"LLOWER: {Llower_Limb}")
            elif Langle2>100:
                Llower_Limb=2
                # print(f"LLOWER: {Llower_Limb}")
            else:
                Llower_Limb=1
                # print(f"LLOWER: {Llower_Limb}")
        else:
            Llower_Limb=1
            # print(f"LLOWER: {Llower_Limb}")
        # Llower_Limb_adjust = angle_dict['Llower_adjust']
        # Llower_Limb += Llower_Limb_adjust
        # print(f"LLOWER: {Llower_Limb}")
        # print("PASS LEFT LOWER")
        if Rlower_Limb>=Llower_Limb:
            point_score['lower_arm'] = Rlower_Limb
            # point_score['lower_arm_adjustment'] = Rlower_Limb_adjust
            point_score['lowerarm_side'] = 'Right'
        else:
            point_score['lower_arm'] = Llower_Limb
            # point_score['lower_arm_adjustment'] = Llower_Limb_adjust
            point_score['lowerarm_side'] = 'Left'
        #Clear------12/1/2025
        #-------------------------------------wrist-------------------------------------
        Rwrist=0
        Lwrist=0
        Rangle3 = angle_dict['Rwrist']
        # print(f"Rangle3 :{Rangle3}")
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
        # print(f"Langle3 : {Langle3}")
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
        Rwrist = 1
        Lwrist = 1
        if Rwrist>=Lwrist:
            point_score['wrist'] = Rwrist
            point_score['wrist_adjust'] = Rwrist_adjust
            point_score['wrist_side'] = 'Right'
            # print("PASS RIGHT WRIST")
            # print(f"Rwrist : {Rwrist}")
        else:
            point_score['wrist'] = Lwrist
            point_score['wrist_adjust'] = Lwrist_adjust
            point_score['wrist_side'] = 'Left'
            # print("PASS LEFT WRIST")
            # print(f"Lwrist : {Lwrist}")
        #CLEAR 12/1/2025------------------   
        #EDIT 11/12/2024#
        neck_angle = angle_dict['Neck']
        neck = 1
        # print(f"Neck_angle :{neck_angle}")
        if str(neck_angle)!='NULL' or str(neck_angle)=='nan':
            neck_angle=abs(int(neck_angle))
            if neck_angle >= 0 and neck_angle <=10:
                neck=1
            elif neck_angle >10 and neck_angle <=20:
                neck=2
            elif neck_angle >20:
                neck=3
            elif neck_angle < 0:
                neck=4
        else:
            neck=1
        #NOT CLEAR THIS ONE
        point_score['neck_adjust'] = int(angle_dict['Neck_twist']) + int(angle_dict['Neck_sidebend'])
        point_score['neck'] = neck #+ point_score['neck_adjust']
        # print(f"PS Nexk_twist: {angle_dict['Neck_twist']}")
        # print(f"PS Nexk_sidebend : {angle_dict['Neck_sidebend']}")
        # print(f"PS Nexk_adjust : {point_score['neck_adjust']}")
        # print(f"PS Nexk : {point_score['neck']}")
        #NOT CLEAR THIS ONE
        #EDIT 11/12/2024#

        trunk_angle=angle_dict['Trunk']
        print(f"Trunk_angle : {trunk_angle}")
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
        # point_score['trunk_adjust'] = int(angle_dict['Trunk_twist']) + int(angle_dict['Trunk_sidebend'])
        point_score['trunk'] = int(trunk) #+ int(point_score['trunk_adjust'])
        # print(f"PS Trunk_adjust: {point_score['trunk_adjust']}")
        # print(f"PS Trunk: {point_score['trunk']}")
        #-------------------------------------wrist_twist-------------------------------------
        wrist_twist=1
        point_score['wrist_twist']=wrist_twist
        #-------------------------------------legs-------------------------------------
        legs=1
        R_knee_rula = angle_dict['Knee_R']
        L_knee_rula = angle_dict['Knee_L']
        if R_knee_rula - L_knee_rula > 20 :
            legs = 2
            point_score['legs']=legs
        point_score['legs']=legs
        if condition == 'extra' :
            muscle_use_a = 1
            muscle_use_b = 1
            print("EXTRA CONDITION")
        if condition == 'normal' :
            muscle_use_a = 0
            muscle_use_b = 0
            print("NORMAL CONDITION")  
        force_load_a=0
        force_load_b=0
        point_score['muscle_use_a']=muscle_use_a
        point_score['force_load_a']=force_load_a
        point_score['muscle_use_b']=muscle_use_b
        point_score['force_load_b']=force_load_b
        rula, point_score = rula_risk(point_score)
        rula['point_score']=point_score
        print("COMPLETE PROFILE")
    else:
        print("No Profile False")
        rula={}
        rula['score']='NULL'
        rula['risk']='NULL'
        rula['point_score']={}
    return rula, point_score
#EDIT10/12/2024#
def validate_value(value):
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return np.array([validate_value(v) for v in value])
    if value is None:
        return False
    try:
        if str(value).lower() == 'null':
            return False
    except Exception:
        return False
    return not pd.isna(value)
###########################################################
# test_cases = [
#     (None, False),               # Test with None
#     ('null', False),             # Test with 'null' string
#     ('NULL', False),             # Test with 'NULL' string (case insensitive)
#     (float('nan'), False),       # Test with NaN
#     ('NaN', False),              # Test with 'NaN' string
#     (42, True),                  # Test with an integer
#     (0, True),                   # Test with zero (valid numeric)
#     (3.14, True),                # Test with a float
#     ('hello', True),             # Test with a normal string
#     ('', True),                  # Test with an empty string
#     (' ', True),                 # Test with a whitespace string
#     ([1, 2, 3], True),           # Test with a list
#     ({"key": "value"}, True),    # Test with a dictionary
#     (True, True),                # Test with a boolean True
#     (False, True),               # Test with a boolean False
# ]
# # Run test cases
# for idx, (input_value, expected_output) in enumerate(test_cases):
#     result = validate_value(input_value)
#     print(f"Test {idx + 1}: Input={input_value} | Expected={expected_output} | Result={result}")
#############################################################
def angle(vector1, vector2):
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 'NULL'
    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_theta = np.clip(dot_product / magnitude_product, -1.0, 1.0)
    angle_radians = np.arccos(cos_theta)
    return np.rad2deg(angle_radians)
#############################################################
# print(angle([1, 0, 0], [0, 1, 0]))
# print(angle([1, 0, 0], [1, 0, 0]))
# print(angle([1, 0, 0], [-1, 0, 0]))
# print(angle([0, 0, 0], [1, 0, 0]))
#EDIT10/12/2024#
#############################################################
#EDIT10/12/2024# 
def project_onplane(normal, vector):
    if len(normal) != len(vector):
        raise ValueError("The normal and vector must have the same dimension.")
    normal_magnitude = np.linalg.norm(normal)
    if normal_magnitude == 0:
        print("Returning zero projection.")
        return np.zeros_like(vector)
    unit_normal = normal / normal_magnitude
    vector_onnormal = np.dot(vector, unit_normal) * unit_normal
    projection = vector - vector_onnormal
    # print("Normal vector :", unit_normal)
    # print("Projection of vector onto plane:", projection)
    return projection
######################################################
# normal = np.array([0, 0, 1])
# vector = np.array([1, 1, 1]) 
# projected_vector = project_onplane(normal, vector)
# print("Projected vector:", projected_vector)
# CLEAR NOTTHING WRONG WITH THIS ONE
######################################################
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
def rotate_vector(vector, axis, theta):
    axis_magnitude = np.linalg.norm(axis)
    if axis_magnitude == 0:
        return None
    axis_unit = axis / axis_magnitude
    vector_parallel = np.dot(vector, axis_unit) * axis_unit
    vector_perpendicular = vector - vector_parallel
    if np.linalg.norm(vector_perpendicular) == 0:
        return None
    cross_product = np.cross(axis_unit, vector_perpendicular)
    rotated_perpendicular = (vector_perpendicular * np.cos(theta) + cross_product * np.sin(theta))
    return vector_parallel + rotated_perpendicular
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
    norm = np.linalg.norm(vector)
    if norm == 0:
        print("Warning: Attempting to normalize a zero vector.")
        return np.zeros_like(vector)
    return vector / norm
#EDIT10/12/ BY CREATED NEW DEF#

def projection_valid(original_vector, projected_vector, threshold=0.5):
    if np.linalg.norm(original_vector) == 0:
        return False
    return np.linalg.norm(projected_vector) / np.linalg.norm(original_vector) >= threshold

# def twist_angle(reference_vector, vector1, vector2, threshold=30):
#     proj1 = project_onplane(reference_vector, vector1)
#     proj2 = project_onplane(reference_vector, vector2)
#     if proj1 is None or proj2 is None or np.linalg.norm(proj1) == 0 or np.linalg.norm(proj2) == 0:
#         return 0 
#     return 1 if angle(proj1, proj2) > threshold else 0

# def sidebend_angle(body_vector, projection_vector, reference_vector, threshold_ratio=0.4, angle_threshold=20):
#     if body_vector is None or projection_vector is None or np.linalg.norm(body_vector) == 0:
#         return 0 
#     proj_ratio = np.linalg.norm(projection_vector) / np.linalg.norm(body_vector)
#     if proj_ratio < threshold_ratio:
#         return 0 
#     if angle(projection_vector, reference_vector) > angle_threshold :
#         return 1
#     else:
#         return 0
#EDIT11/12/2024#
def normal_vector_from_plane(p1, p2, p3):
    """Compute the normal vector of a plane defined by three points."""
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    
    norm = np.linalg.norm(normal)
    if norm == 0:
        print("Warning: The three points are collinear, cannot define a plane.")
        return np.zeros(3)  # Return zero vector if points are collinear
    
    return normal / norm 
def angle_calc(kpn, pose_list, reference_vector = None , head_theta0= None, back_theta0= None, vector0 = None):
    condition = kpn
    # print(f"Inside angle_calc, full pose_list: {pose_list}")
    # print(f"Inside angle_calc, R_Shoulder: {pose_list[12]}")
    try :
        pose = np.array(pose_list)
        # print(f"Original pose_list[12]: {pose_list[12]}")
        # print(f"Converted pose[12]: {pose[12]}")

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
        L_Foot = pose[31]
        R_Foot = pose[32]
        R_Palm = pose[20]
        L_Palm = pose[19]
        R_knee = pose[26]
        L_knee = pose[25]
        R_ankle = pose[28]
        L_ankle = pose[27]
        left=0
        right=0
        front=0
        # print("Debugging angle_calc:")
        # print(f"pose_list (type: {type(pose_list)}): {pose_list[:5] if isinstance(pose_list, list) else pose_list}")
        # print(f"reference_vector (type: {type(reference_vector)}): {reference_vector}")
        # print(f"head_theta0 (type: {type(head_theta0)}): {head_theta0}")
        # print(f"back_theta0 (type: {type(back_theta0)}): {back_theta0}")
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
        print("KP RN CHECKING")
        # print(f"Rshoulder : {R_Shoulder}")
        # print(f"RElbow : {R_Elbow}")
        RlowerArm = (R_Wrist - R_Elbow)[:3]
        LlowerArm = (L_Wrist - L_Elbow)[:3]
        Rhand     = (R_Palm - R_Wrist)[:3]
        Lhand     = (L_Palm - L_Wrist)[:3]
        midNeck   = ((L_Shoulder + R_Shoulder)/2)[:3]
        midWaist  = ((L_Hip + R_Hip)/2)[:3]
        Head      = Nose[:3] - midNeck
        Back      = midNeck - midWaist
        # print(f"midNeck : {midNeck}")
        # print(f"midWaist : {midWaist}")
        # print(f"Head : {Head}")
        # print(f"Back : {Back}")
        # Relbow    = angle(RlowerArm,RupperArm)
        # Rwrist    = angle(Rhand,RlowerArm)
        corebody = midNeck - midWaist
        # Lelbow    = angle(LlowerArm,LupperArm)
        # Lwrist    = angle(Lhand,LlowerArm)
        if reference_vector is not None:
            print(reference_vector)
            # 1 FIND COREBODY VECTOR
            corebody = corebody
            corebody_unit = to_unitvec(corebody)
            # 2 FIND REFERENCE VECTOR AROOWFACE
            if R_Shoulder is None or L_Shoulder is None or R_Hip is None :
                print("CANNOT FIND ARROWBODY")
                pass
            else :
                print(f"R_Shoulder : {R_Shoulder}")
                print(f"L_Shoulder : {L_Shoulder}")
                print(f"R_Hip : {R_Hip}")
                arrowbody = normal_vector_from_plane(R_Shoulder[:3],L_Shoulder[:3],R_Hip[:3])
                print(f"Arrowbody : {arrowbody}")
                arrowbody = to_unitvec(arrowbody)
                # arrowface = Nose[:3]-midNeck
                # arrowface_noy = [arrowface[0], 0, arrowface[2]]
                # print(f"Arrow face : {arrowface}")
                # print(f"Arrow face Plane XZ : {arrowface_noy}")
                # 3 Normal vector
                print(RupperArm)
                RupperArm = to_unitvec(RupperArm)
                print(RupperArm)
                normalvector = np.cross(corebody_unit, arrowbody)
                normalvector_unit = to_unitvec(normalvector)
                # print(f"Normal vector : {normalvector}")
                # print(f"Normal vector_unit : {normalvector_unit}")
                # dot_product = np.dot(corebody_unit, arrowbody)
                # print(f"Dot product (should be ~0 if perpendicular): {dot_product}")
                projected_vector_corebody = project_onplane(normalvector_unit, corebody_unit)
                RupperArm_proj = project_onplane(normalvector_unit, RupperArm)
                print(RupperArm_proj)
                print(projected_vector_corebody)
                Rshoulder = angle(-projected_vector_corebody,RupperArm_proj)
                # dot_check = np.dot(projected_vector_corebody, normalvector_unit)
                # print(f"Dot product after projection (should be ~0): {dot_check}")
                print(f"Rshoulder : {Rshoulder}")
                # print("---------------")
                # print("Rupperarm", RupperArm)
                # print("Corebody", corebody)
                # print("Arrowbody", arrowbody)
                # print("Normal coss", normalvector)
                # print("Rupperarmproj", RupperArm_proj)
                # print("Corebodyproj", projected_vector_corebody)
                # print(f"Arrowbody before normalization: {arrowbody}")
                # print(f"Dot product (corebody vs. arrowbody): {np.dot(corebody_unit, arrowbody)}")  # Should be ~0
                # print(f"Normal vector before normalization: {normalvector}")
                # print(f"Projected Corebody: {projected_vector_corebody}")
                # print(f"Projected RupperArm: {RupperArm_proj}")
                # print(f"Dot product after projection (should be ~0): {np.dot(projected_vector_corebody, normalvector_unit)}")
                # print(f"Computed Shoulder Angle: {Rshoulder}")
                # print("---------------")
                LupperArm_proj = project_onplane(normalvector_unit, LupperArm)
                print(LupperArm_proj)
                Lshoulder = angle(-projected_vector_corebody,LupperArm_proj)
                print(f"Lshoulder : {Lshoulder}")
                # 
                RupperArm = to_unitvec(RupperArm)
                RlowerArm = to_unitvec(RlowerArm)
                Relbow = angle(RlowerArm,RupperArm)
                Rwrist    = angle(Rhand,RlowerArm)
                #
                LupperArm = to_unitvec(LupperArm)
                LlowerArm = to_unitvec(LlowerArm)
                Lelbow = angle(LlowerArm,LupperArm)
                Lwrist = angle(Lhand,LlowerArm)
                #
                
                #EDIT10/12/ BY CREATED NEW DEF#
                # if projection_valid(RupperArm, RupperArm_proj): 
                #     Rshoulder = Rshoulder
                #     print("R shoulder is valid")
                # else :
                #     Rshoulder = 0
                #     print("R shoudler is not valid")
                # if projection_valid(LupperArm, LupperArm_proj): 
                #     Lshoulder = Lshoulder
                #     print("L shoulder is valid")
                # else :
                #     Lshoulder = 0
                #     print("L shoulder is not valid")
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
                #############################################
                #Trunk angle
                corebody = corebody  
                # Back_default = rotate_vector(reference_vector[3], -sagittal, np.deg2rad(Back_default_angle))
                # Trunk = angle(Back, Back_default)
                # Trunk = angle(corebody_unit, Back_default_angle)
                # print("Trunk Angle:", Trunk)
                Back_default_angle = [0,1,0]
                Trunk_projside = project_onplane(normalvector_unit,corebody_unit)
                Back_default_angle_projside = project_onplane(normalvector_unit,Back_default_angle)
                Trunk = angle(Trunk_projside,Back_default_angle_projside)
                Trunk = 180 - Trunk
                print("THIS IS NOW TRUNK ANGLE",Trunk)
                # Trunk_twist = 0
                # # Trunk_twist = twist_angle(reference_vector[3], L_Hip[:3] - R_Hip[:3], L_Shoulder[:3] - R_Shoulder[:3])
                # print(f"Twist twist angle : {Trunk_twist}")
                # Trunk_sidebend = 0
                ###########################################
                #Trunk twist
                vector_rightelbow_2_leftelbow = R_Shoulder[:3]-L_Shoulder[:3]
                vector_rightelbow_2_leftelbow = to_unitvec(vector_rightelbow_2_leftelbow)
                # print("R2Lelbow", vector_rightelbow_2_leftelbow)
                vector_righthip_2_lefthip = R_Hip[:3]-L_Hip[:3]
                vector_righthip_2_lefthip = to_unitvec(vector_righthip_2_lefthip)
                # print("R2Lhip", vector_righthip_2_lefthip)
                R2Lelbowy_plane = project_onplane(corebody_unit,vector_rightelbow_2_leftelbow)
                R2Lhipy_plane = project_onplane(corebody_unit,vector_righthip_2_lefthip)
                Trunk_twist = angle(R2Lelbowy_plane,R2Lhipy_plane)
                # print(f"Twist twist angle : {Trunk_twist}")
                ##########################################
                #Trunk sidebend
                Trunk_projfront = project_onplane(arrowbody,corebody_unit)
                Back_default_angle_projfront = project_onplane(arrowbody,Back_default_angle)
                Trunk_sidebend = angle(Trunk_projfront,Back_default_angle_projfront)
                # Trunk_proj_coronal = project_onplane(coronal, Back)
                # Trunk_sidebend = sidebend_angle(Back, Trunk_proj_coronal, reference_vector[3])
                Trunk_sidebend = 180- Trunk_sidebend
                # print(f"Trunk bend angle:{Trunk_sidebend}")
                ########################################3
                # Trunk = 0
                Trunk_sidebend = 0
                Trunk_twist = 0
                #############################################
                # Neck angle
                arrowface = Nose[:3]- midNeck
                # print(f"Arrowface Original: {arrowface}")
                Neckvector =  project_onplane(arrowbody,arrowface)
                Neckvector = to_unitvec(Neckvector)
                # print("NECK UPUP",Neckvector)
                neckvectorside = project_onplane(normalvector_unit,Neckvector)
                angleneckandcorebody = angle(neckvectorside,projected_vector_corebody)
                # print("ANGLE NECK AND COREBODY SIDE",angleneckandcorebody)
                Neck = angleneckandcorebody
                # print("NECK ANGLE",Neck)
                #############################################
                #neck twisted
                arrowbody_xz = project_onplane(corebody_unit, arrowbody)
                print(f"Arrowbody_XZ: {arrowbody_xz}")
                arrowface_xz = project_onplane(corebody_unit, arrowface)
                print(f"Arrowface_XZ : {arrowface_xz}")
                Neck_twist = angle(arrowbody_xz,arrowface_xz)
                if Neck_twist > 60 :
                    print("NEck twist",Neck_twist)
                    necktwist = 1
                    print("Neck is Twist") 
                else:
                    print("NEck twist",Neck_twist) 
                    necktwist = 0
                    print("Neck is not Twist")
                ##########################################
                #neck sidebend
                arrowface = Nose[:3]- midNeck
                Neck_projfront = project_onplane(arrowbody,arrowface)
                Neck_projfront = to_unitvec(Neck_projfront)
                Corebody_projfront = project_onplane(arrowbody,corebody_unit)
                Corebody_projfront = to_unitvec(Corebody_projfront)
                Neck_sidebend = angle(Neck_projfront,Corebody_projfront)
                print("Neck SIDEBEND",Neck_sidebend)
                ########################################
                # Neck = 0
                necktwist = 0
                Neck_sidebend = 0
                #########################################
                #leg
                Rlegupper = R_Hip[:3] - R_knee[:3]
                Llegupper = L_Hip[:3] - L_Knee[:3]
                Rleglower = R_knee[:3] - R_ankle[:3]
                Lleglower = L_knee[:3] - L_ankle[:3]
                angle_knee_R = angle(Rlegupper,Rleglower)
                angle_knee_L = angle(Llegupper,Lleglower) 
                ########################################
                # Head_default_angle = head_theta0
                # Head_default = rotate_vector(coronal, -sagittal, np.deg2rad(90 - Head_default_angle))
                # Head_proj_sag = project_onplane(sagittal, Head)
                # Neck = angle(project_onplane(sagittal, Head_default), Head_proj_sag)
                # Neck = abs(Neck - Trunk)

                # Head_proj_transverse = project_onplane(reference_vector[3], Head)
                # neck_Hip = project_onplane(reference_vector[3], L_Hip[:3] - R_Hip[:3])
                # if angle(Head_proj_transverse, neck_Hip) > 60 or angle(Head_proj_transverse, -neck_Hip) >60 :
                #     necktwist = 1
                #     print("Neck is Twist") 
                # else: 
                #     necktwist = 0
                #     print("Neck is not Twist")
                # Head_proj_coronal = project_onplane(coronal, Head)
                # Neck_sidebend = sidebend_angle(Head, Head_proj_coronal, reference_vector[3])
                # print(f"Neck sidebend:{Neck_sidebend}")
        else:
            try:
                print("Problem is in else")
                neck_2d = midNeck.copy()
                neck_2d[2] = 0
                waist_2d = midWaist.copy()
                waist_2d[2] = 0
                transverse = to_unitvec(neck_2d - waist_2d)
                if transverse is None or np.linalg.norm(transverse) == 0:
                    print("Warning: Transverse vector is zero. Using default vector.")
                    transverse = np.array([0, 1, 0]) 

                sagittal = project_onplane(transverse, L_Hip[:3] - R_Hip[:3]) #ลูกศร
                if sagittal is None or np.linalg.norm(sagittal) == 0:
                    print("Warning: Sagittal vector is zero. Using default vector.")
                    sagittal = np.array([1, 0, 0]) 

                coronal = np.cross(sagittal, transverse)
                if coronal is None or np.linalg.norm(coronal) == 0:
                    print("Warning: Coronal vector is zero. Using default vector.")
                    coronal = np.array([0, 0, 1]) 

                print(f"neck_2d: {neck_2d}, waist_2d: {waist_2d}")
                print(f"transverse: {transverse}, sagittal: {sagittal}, coronal: {coronal}")
                if np.linalg.norm(transverse) > 0:
                    head_theta0 = angle(Head, transverse)
                else:
                    head_theta0 = 0

                if np.linalg.norm(transverse) > 0:
                    trunk_theta0 = angle(Back, transverse)
                else:
                    trunk_theta0 = 0
                return np.array([Back, Head, sagittal, transverse, coronal]), head_theta0, trunk_theta0

            except Exception as e:
                print("Error in angle_calc 'else' block:", e)
                raise
            
        angle_dict = {'Rshoulder':Rshoulder, 
                    'Relbow': Relbow,
                    'Lshoulder':Lshoulder,
                    'Lelbow':Lelbow, 
                    'Neck':Neck,
                    'Trunk':Trunk,
                    'Rwrist':Rwrist,
                    'Lwrist':Lwrist,
                    'Neck_twist':necktwist,
                    'Rshoulder_abduct':Rshoulder_abduct,
                    'Lshoulder_abduct':Lshoulder_abduct, 
                    'Rlower_adjust':Rlower_adjust,
                    'Llower_adjust':Llower_adjust, 
                    'Neck_sidebend':Neck_sidebend,
                    'Trunk_twist':Trunk_twist,
                    'Trunk_sidebend':Trunk_sidebend,
                    'Knee_R': angle_knee_R,
                    'Knee_L' : angle_knee_L}
        rula, pointscore = rula_score(angle_dict,pose,profile,condition)
        return rula['score'], angle_dict , profile, necktwist, pointscore
    except :
        print(f"Error in angle_calc: {e}")
        return None,None,None

