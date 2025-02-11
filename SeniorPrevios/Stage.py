def state_consider(angle_dict):
    threshold = 40
    states = [
        ['Stand up', 0, 0, 0, 0, 0, 0],
        ['Raise Hand', 180, 0,180, 0, 0, 0],
        ['Raise Hand R 90',90,0,0,0,0,0],
        ['Raise Hand L 90',0,0,90,0,0,0],
        ["Hand on Right", 0, 90, 0, 0, 0, 0], 
        ["Hand on Left", 0, 0, 0, 90, 0, 0], 
        ["One Arm Raised Right", 180, 0, 0, 0, 30, 0], 
        ["One Arm Raised Left", 0, 0, 180, 0, 30, 0],  
        ["Elbows Bent 90 Both", 0, 90, 0, 90, 0, 0],  
        ["Stretching Arms Forward", 90, 0, 90, 0, 0, 0], 
    ]
    for state in states:
        state_name = state[0]
        state_values = state[1:]
        if all(any(abs(angle - value) <= threshold for value in state_values) for angle in angle_dict.values()):
            return state_name

    return None


# angle_dict1 = {'Rshoulder': 25, 'Relbow': 20, 'Rwrist': 15}  
# angle_dict2 = {'Rshoulder': 185, 'Relbow': 5, 'Rwrist': 10}  
# print(state(angle_dict1)) 
# print(state(angle_dict2))