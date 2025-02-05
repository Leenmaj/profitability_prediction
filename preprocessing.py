from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE



def preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Handle missing values (drop rows with missing data)
    df = df.dropna()

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Separate features and labels
    X = df.drop("target", axis=1)
    y = df["target"]

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train and test sets
    n_samples , n_features = X.shape
    n_channel = n_features
    n_length = 1

    X = X.reshape(n_samples, n_channel, n_length)



    return X ,y

def preprocess_data_with_smote(file_path):
    """
    Preprocess the input CSV data for ResNet1D with SMOTE.

    Args:
        file_path (str): Path to the CSV file.
        

    Returns:
        tuple: Processed data (X), labels (y).
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Handle missing values (drop rows with missing data)
    df = df.dropna()

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Separate features and labels
    X = df.drop("target", axis=1).values
    y = df["target"].values

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Reshape data for ResNet1D
    n_samples, n_features = X.shape
    n_channel = n_features
    n_length = 1

    X = X.reshape(n_samples, n_channel, n_length)

    return X, y

