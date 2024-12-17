import pandas as pd
table_a = pd.read_csv(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\KPOPENCVMEDIAPIPE\Rula_score\TableA.csv")
table_a1 = pd.read_csv(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\KPYOLOOPENCV\Rula_score\TableA1.csv", header=0)
table_a2 = pd.read_csv(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\KPYOLOOPENCV\Rula_score\TableA2.csv", header=0)
table_b1 = pd.read_csv(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\KPOPENCVMEDIAPIPE\Rula_score\TableB1.csv", index_col=0)
table_b2 = pd.read_csv(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\KPOPENCVMEDIAPIPE\Rula_score\TableB2.csv", index_col=0)
table_c = pd.read_csv(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\KPOPENCVMEDIAPIPE\Rula_score\TableC.csv")
def stepA1(angle, side):
    if side.lower() not in ["left", "right"]:
        raise ValueError("Invalid side. Must be 'left' or 'right'.")
    if angle is None:
        print(f"Step A1 ({side.capitalize()} Upper Arm): Angle is None. Cannot calculate score.")
        return None
    if angle < 20:
        A_1 = 1
    elif 20 <= angle < 45:
        A_1 = 2
    elif 45 <= angle < 90:
        A_1 = 3
    else: 
        A_1 = 4
    print(f"Step A1 ({side.capitalize()} Upper Arm): Angle = {angle}° -> Score = {A_1}")
    return A_1
def stepA2(angle, side):
    if side.lower() not in ["left", "right"]:
        raise ValueError("Invalid side. Must be 'left' or 'right'.")
    if angle is None:
        print(f"Step A2 ({side.capitalize()} Lower Arm): Angle is None. Cannot calculate score.")
        return None 
    if 60 < angle < 100:
        A_2 = 1
    elif angle < 50 or angle >= 100:
        A_2 = 2
    else:
        A_2 = 2
    print(f"Step A2 ({side.capitalize()} Lower Arm): Angle = {angle}° -> Score = {A_2}")
    return A_2

table_a1.columns = table_a1.columns.str.strip()
table_a1['Upperarm'] = table_a1['Upperarm'].astype(int)
table_a1['Lowerarm'] = table_a1['Lowerarm'].astype(int)

table_a2.columns = table_a2.columns.str.strip()
table_a2['Upperarm'] = table_a2['Upperarm'].astype(int)
table_a2['Lowerarm'] = table_a2['Lowerarm'].astype(int)

def tabela(A1, A2, A3, A4, table_a1, table_a2):
    if not (1 <= A1 <= 6) or not (1 <= A2 <= 3):
        raise ValueError(f"Invalid values: A1 = {A1} or A2 = {A2}. Must be between 1 and 6.")
    if A4 == 1:
        table_to_use = table_a1
    elif A4 == 2:
        table_to_use = table_a2
    else:
        raise ValueError(f"Invalid value for A4: {A4}. Must be 1 or 2.")
    valid_rows = table_to_use[table_to_use['Upperarm'] == A1]
    if valid_rows.empty:
        raise ValueError(f"No rows found with Upperarm = {A1}.")
    valid_rows = valid_rows[valid_rows['Lowerarm'] == A2]
    if valid_rows.empty:
        raise ValueError(f"No rows found with Lowerarm = {A2}.")
    wrist_column = f"Wrist_{A3}"
    if wrist_column not in table_to_use.columns:
        raise ValueError(f"Invalid wrist column: {wrist_column}. Ensure that the wrist columns exist.")
    score = valid_rows.iloc[0][wrist_column]
    print(f"Retrieved score: {score}")
    return score

# def tabela(A_1, A_2, A_3, A_4, table_a):
#     row_index = A_1 - 1
#     col_index = (A_2 - 1) * 4 + (A_3 - 1)
#     # print(f"Row index: {row_index}, Column index: {col_index}")
#     # print(f"Table A Shape: {table_a.shape}")
#     base_score = table_a.iloc[row_index, col_index]
#     # print(f"Base Score: {base_score}, A_4: {A_4}")
#     final_score = base_score + A_4
#     print(f"Final Score: {final_score}")
#     return final_score

def tabelb(B_9, B_10, B_11, table_b1, table_b2):
    if B_11 == 1:
        selected_table = table_b1
        table_name = "Table B1"
        print("USE TABLE B1")
    elif B_11 == 2:
        selected_table = table_b2
        table_name = "Table B2"
        print("USE TABLE B2")
    else:
        raise ValueError(f"Invalid leg score: {B_11}. Must be 1 or 2.")
    column_name = f"Trunk_{B_10}"
    # print(f"Using {table_name} for calculation.")
    # print(f"Row index (Neck): Neck_{B_9}, Column: {column_name}")
    b_score = selected_table.loc[f"Neck_{B_9}", column_name]
    print(f"Score retrieved from {table_name}: {b_score}")
    
    return int(b_score)

def tabelc(asum, bsum, table_c):
    row_index = asum - 1
    col_label = f"NTL_{bsum}" 
    # print(f"Row index (asum): {row_index}, Column label (bsum): {col_label}")
    # print(f"Table C Shape: {table_c.shape}")
    # print(f"Table C Columns: {list(table_c.columns)}")
    final_score = table_c.iloc[row_index][col_label]
    print(f"Final RULA Score from Table C: {final_score}")
    return final_score
