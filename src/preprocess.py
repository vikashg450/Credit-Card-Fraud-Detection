import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def preprocess_data(csv_path="creditcard.csv", output_dir="models"):
    print("--- Phase 1: Preprocessing Data ---")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    print(f"Loading dataset from: {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Check for and drop exact duplicates
    duplicates_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates_count}")
    if duplicates_count > 0:
        df = df.drop_duplicates()
        print(f"Dropped duplicates. New shape: {df.shape}")
        
    # Split into features (X) and label (y)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    # Stratified split (80% train, 20% test)
    # Stratify by y to ensure the 0.17% fraud ratio is identical in both splits
    print("Performing stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train fraud ratio: {y_train.mean():.5f} ({y_train.sum()} fraud / {len(y_train)} total)")
    print(f"Test fraud ratio: {y_test.mean():.5f} ({y_test.sum()} fraud / {len(y_test)} total)")
    
    # Scale Time and Amount columns using RobustScaler (fits on train set only)
    print("Scaling 'Time' and 'Amount' features using RobustScaler...")
    scaler = RobustScaler()
    
    # We must scale Time and Amount. Let's do it by modifying only those columns.
    # To avoid slice warning, we copy
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    cols_to_scale = ["Time", "Amount"]
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # Save the fitted scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved fitted scaler to: {scaler_path}")
    
    # Save splits as compressed numpy file
    split_data_path = os.path.join(output_dir, "split_data.npz")
    np.savez_compressed(
        split_data_path,
        X_train=X_train.values,
        X_test=X_test.values,
        y_train=y_train.values,
        y_test=y_test.values,
        columns=X.columns.values
    )
    print(f"Saved split datasets to: {split_data_path}")
    print("Preprocessing completed successfully!\n")

if __name__ == "__main__":
    preprocess_data()
