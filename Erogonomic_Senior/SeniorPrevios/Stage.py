def state_consider(angle_dict):
    # Define individual thresholds for each joint angle
    thresholds = [40, 30, 40, 30, 20, 20]  # Adjust these values as needed
    states = [
        ['Stand up', 0, 0, 0, 0, 0, 0],
        ['Raise Hand', 180, 0, 180, 0, 0, 0],
        ['Raise Hand R 90', 90, 0, 0, 0, 0, 0],
        ['Raise Hand L 90', 0, 0, 90, 0, 0, 0],
        ["Hand on Right", 0, 90, 0, 0, 0, 0], 
        ["Hand on Left", 0, 0, 0, 90, 0, 0], 
        ["One Arm Raised Right", 180, 0, 0, 0, 30, 0], 
        ["One Arm Raised Left", 0, 0, 180, 0, 30, 0],  
        ["Elbows Bent 90 Both", 0, 90, 0, 90, 0, 0],  
        ["Stretching Arms Forward", 90, 0, 90, 0, 0, 0], 
    ]

    for state in states:
        state_name = state[0]
        state_values = state[1:]  # Exclude state name
        match = all(
            abs(angle - value) <= threshold
            for angle, value, threshold in zip(angle_dict.values(), state_values, thresholds)
        )
        if match:
            return state_name 
    return None 
