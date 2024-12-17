import pandas as pd

# Load your table (ensure the correct path)
table_a1 = pd.read_csv(r"C:\Users\kpnth\OneDrive - Chulalongkorn University\Desktop\CU\Senior project\Code\SeniorPrevios\Rula_score\TableA.csv", header=0)

def tabela(A1, A2, A3, A4, table_a1):
    """
    This function retrieves the value from Table A1 based on the Upper Arm (A1),
    Lower Arm (A2), Wrist (A3), and Wrist Column Set (A4).
    
    A1: Upper Arm score (1-6)
    A2: Lower Arm score (1-6)
    A3: Wrist group (1, 2, 3, or 4) - corresponds to columns like 1WT1, 1WT2, etc.
    A4: Column Set (1 or 2) - used to select between WT1 and WT2 for the same wrist number
    
    Returns:
        The value from Table A1 corresponding to the given inputs.
    """
    
    # Ensure A1 and A2 are valid indices for rows
    if not (1 <= A1 <= 6) or not (1 <= A2 <= 6):
        raise ValueError(f"Invalid values: A1 = {A1} or A2 = {A2}. Must be between 1 and 6.")
    
    # Define the wrist column based on A3 and A4 (e.g., 2WT1, 2WT2)
    wrist_column = f"{A3}WT{A4}"
    
    # Check if the wrist column exists in the table
    if wrist_column not in table_a1.columns:
        raise ValueError(f"Invalid wrist column: {wrist_column}. Ensure that the wrist columns exist.")
    
    try:
        # Retrieve the value from the table based on the row and column
        row_index = A1 - 1  # row index (based on A1)
        
        # The correct column index: (A3 - 1) gives the group (1, 2, 3, or 4)
        # For each wrist group, there are two columns (WT1 and WT2). A4 selects the column set.
        column_index = table_a1.columns.get_loc(wrist_column)  # find the correct column for wrist group
        
        value = table_a1.iloc[row_index, column_index]  # Get the value from the table
        print(f"Value from Table A1 for A1 = {A1}, A2 = {A2}, {wrist_column}: {value}")
        return value
    except KeyError as e:
        raise ValueError(f"Error retrieving value: {e}. Make sure the table and inputs are correct.")
    except IndexError as e:
        raise ValueError(f"Index error: {e}. Make sure A1 and A2 are within the valid range.")

# Example usage
A1 = 5  # Upper arm score
A2 = 2  # Lower arm score
A3 = 2  # Wrist group number (e.g., 2)
A4 = 1  # Column set (1 or 2) for Wrist_2

result = tabela(A1, A2, A3, A4, table_a1)  # Example call
print(f"Result: {result}")
