import numpy as np

pose_list_valid = [
    [0, 0, 0, 0.98],  
    [1, 1, 1, 0.95], 
]
pose_list_valid.extend([[0, 0, 0, 0.95]] * (33 - len(pose_list_valid)))  # Fill to 33

reference_vector_valid = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0]])
head_theta0_valid = 45
back_theta0_valid = 30

pose_list_short = [[0, 0, 0, 0.98]] * 10  # Fewer than 33 keypoints
pose_list_empty = []
reference_vector_invalid = None
pose_list_none = None
reference_vector_none = None
head_theta0_none = None
back_theta0_none = None

# Test cases
def test_angle_calc():
    test_cases = [
        {
            "name": "Valid Inputs",
            "pose_list": pose_list_valid,
            "reference_vector": reference_vector_valid,
            "head_theta0": head_theta0_valid,
            "back_theta0": back_theta0_valid,
        },
        {
            "name": "Pose List Too Short",
            "pose_list": pose_list_short,
            "reference_vector": reference_vector_valid,
            "head_theta0": head_theta0_valid,
            "back_theta0": back_theta0_valid,
        },
        {
            "name": "Empty Pose List",
            "pose_list": pose_list_empty,
            "reference_vector": reference_vector_valid,
            "head_theta0": head_theta0_valid,
            "back_theta0": back_theta0_valid,
        },
        {
            "name": "Invalid Reference Vector",
            "pose_list": pose_list_valid,
            "reference_vector": reference_vector_invalid,
            "head_theta0": head_theta0_valid,
            "back_theta0": back_theta0_valid,
        },
        {
            "name": "All Inputs None",
            "pose_list": pose_list_none,
            "reference_vector": reference_vector_none,
            "head_theta0": head_theta0_none,
            "back_theta0": back_theta0_none,
        },
    ]

    for test in test_cases:
        print(f"Running Test: {test['name']}")
        try:
            result = angle_calc(
                pose_list=test["pose_list"],
                reference_vector=test["reference_vector"],
                head_theta0=test["head_theta0"],
                back_theta0=test["back_theta0"],
            )
            print(f"Result for {test['name']}: {result}\n")
        except Exception as e:
            print(f"Exception in {test['name']}: {e}\n")

# Run tests
if __name__ == "__main__":
    test_angle_calc()
