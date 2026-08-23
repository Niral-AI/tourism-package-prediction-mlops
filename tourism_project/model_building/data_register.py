import os
import pandas as pd

def validate_dataset():
    """
    Acts as an automated data validation gate. Reads the raw dataset from the
    repository, verifies the presence of mandatory target columns, audits for
    missing values, and logs categorical distributions to ensure data integrity.
    """
    data_path = "tourism_project/data/tourism.csv"
    
    # 1. Verify existence of the data source
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Critical Error: Dataset not found at {data_path}")

    # 2. Load dataset
    df = pd.read_csv(data_path)
    print("================ DATA REGISTRATION & AUDIT ================")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

    # 3. Schema & Target Column Validation
    target_col = 'ProdTaken'
    if target_col not in df.columns:
        raise ValueError(f"Critical Pipeline Failure: Target column '{target_col}' missing!")
    print(f"✅ Target column '{target_col}' verified.")

    # 4. Audit Missing / Null Values across all features
    null_counts = df.isnull().sum()
    missing_data = null_counts[null_counts > 0]
    if not missing_data.empty:
        print("\nMissing values detected (will be handled by preprocessing pipeline):")
        print(missing_data.to_string())
    else:
        print("✅ No missing values detected in the raw dataset.")

    # 5. Log categorical distributions for survey consistency checks
    for col in ['Gender', 'MaritalStatus']:
        if col in df.columns:
            print(f"Unique values in '{col}': {df[col].unique()}")

if __name__ == "__main__":
    validate_dataset()
