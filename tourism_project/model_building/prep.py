import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    df = pd.read_csv("tourism_project/data/tourism.csv")
    
    drop_cols = [c for c in ['CustomerID', 'Unnamed: 0'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'}).str.strip()
    if 'MaritalStatus' in df.columns:
        df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'}).str.strip()

    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)

if __name__ == "__main__":
    prepare_data()
