import os
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

def register_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("=================== DATASET REGISTRATION SUMMARY ===================")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    
    expected_cols = [
        'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
        'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
        'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
        'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
    ]
    
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed! Missing columns: {missing_cols}")
        
    print("Schema Check: PASSED. All expected columns exist.")
    print("\nClass Distribution for 'ProdTaken':")
    print(df['ProdTaken'].value_counts(normalize=True))
    print("\nMissing Values Count:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("====================================================================")

if __name__ == "__main__":
    register_dataset()
