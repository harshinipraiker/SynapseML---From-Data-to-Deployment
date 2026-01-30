import pandas as pd

def identify_task(df: pd.DataFrame, target_column: str):
    """
    Identifies the machine learning task based on the target column's properties.

    Heuristics:
    - If the target column is of type 'object' or 'category', it's classification.
    - If the number of unique values is very small (e.g., <= 20 and is integer), it's likely classification.
    - Otherwise, it's regression.

    Returns:
        str: 'classification' or 'regression'
    """
    target_dtype = df[target_column].dtype
    unique_values = df[target_column].nunique()

    if target_dtype in ['object', 'category', 'bool']:
        print("Task identified as Classification (target is object/category).")
        return 'classification'

    # If the column is numeric
    if pd.api.types.is_numeric_dtype(target_dtype):
        # If it has very few unique values, treat it as classification (e.g., predicting a class label like 0, 1, 2)
        if unique_values <= 20: 
            print(f"Task identified as Classification (target is numeric with {unique_values} unique values).")
            return 'classification'
        else:
            print(f"Task identified as Regression (target is numeric with {unique_values} unique values).")
            return 'regression'
    
    # Fallback for other types
    print("Could not definitively identify task. Defaulting to Regression.")
    return 'regression'