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

    #removed encoding categorial columns 

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


def preprocess_autoencoder_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()

    #separate features (but not labels)
    X = df.drop("target", axis=1)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X



def preprocess_autoencoder_data_with_smote(file_path, test_size=0.2, random_state=42):
  

    
    df = pd.read_csv(file_path)
    df = df.dropna()  

    
    X = df.drop(columns=["target"])  
    y = df["target"]  

    #split data BEFORE applying SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    #fit the scaler ONLY on original X_train (before SMOTE)
    scaler = StandardScaler()
    scaler.fit(X_train)

    #apply SMOTE to X_train (ONLY training data)
    smote = SMOTE(sampling_strategy='auto', random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    #transform both X_train_resampled and X_test using the same scaler
    X_train_resampled = scaler.transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    return X_train_resampled, X_test, y_train_resampled, y_test
