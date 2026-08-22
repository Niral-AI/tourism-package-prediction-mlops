import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    df = pd.read_csv("tourism_project/data/tourism.csv")
    
    # Drop identifier and indexing columns
    drop_cols = [col for col in ['Unnamed: 0', 'CustomerID'] if col in df.columns]
    df = df.drop(columns=drop_cols)
    
    # Categorical standardizations
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
    if 'MaritalStatus' in df.columns:
        df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})
        
    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']
    
    # 80-20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save artifacts locally for the GitHub Actions job
    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)
    
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    print("Splits successfully saved to Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

if __name__ == "__main__":
    prepare_data()
