import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    """
    Loads raw tourism dataset, performs data cleaning and standardization,
    removes non-predictive identifiers, and splits data into train/test sets
    using stratified sampling to maintain target class distribution.
    """
    # 1. Load the registered dataset
    df = pd.read_csv("tourism_project/data/tourism.csv")

    # 2. Remove non-predictive identifier columns to prevent noise
    drop_cols = [c for c in ['CustomerID', 'Unnamed: 0'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 3. Standardize categorical anomalies from survey entry
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'}).str.strip()
    if 'MaritalStatus' in df.columns:
        df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'}).str.strip()

    # 4. Separate features (X) and target variable (y)
    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']

    # 5. Perform an 80/20 Stratified Train-Test Split
    # Stratification ensures both train and test sets preserve the exact target class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    # 6. Save split artifacts locally for the model training stage
    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)
    print("✅ Data preparation complete. Split files saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

if __name__ == "__main__":
    prepare_data()
