import os
import pandas as pd

def validate_dataset():
    data_path = "tourism_project/data/tourism.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print("================ DATA REGISTRATION & AUDIT ================")
    
    target_col = 'ProdTaken'
    if target_col not in df.columns:
        raise ValueError(f"Critical Error: Target '{target_col}' missing!")
    
    null_counts = df.isnull().sum()
    missing_data = null_counts[null_counts > 0]
    if not missing_data.empty:
        print("Missing values detected:\n", missing_data.to_string())

    for col in ['Gender', 'MaritalStatus']:
        if col in df.columns:
            print(f"{col} unique values: {df[col].unique()}")

if __name__ == "__main__":
    validate_dataset()
