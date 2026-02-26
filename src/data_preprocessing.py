import pandas as pd
import os
from sklearn.model_selection import train_test_split

Base_Dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(Base_Dir, "..", "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

def load_data():
    df = pd.read_csv(path)

    # Fix dirty numeric column
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop ID column safely
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    return df

def split_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Save processed train/test splits for reproducibility
    processed_dir = os.path.join(Base_Dir, "..", "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df["Churn"] = y_train
    test_df = X_test.copy()
    test_df["Churn"] = y_test

    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    return X_train, X_test, y_train, y_test