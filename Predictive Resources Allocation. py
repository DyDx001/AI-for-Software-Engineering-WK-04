import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def run_predictive_model():
    # --- 1. Load and Preprocess Data ---
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Original target (for Section 2)
    y_binary = data.target 
    y_binary_names = data.target_names # ['malignant', 'benign']

    print("--- Task 3: Predictive Analytics ---")
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    
    
    # --- SECTION 1: Answering the Prompt (Predicting Simulated Priority) ---
    #
    # The Breast Cancer dataset is for binary (benign/malignant) classification.
    # To predict 'priority', we must simulate that data.
    # We will use 'mean radius' as a proxy for 'priority'.
    #
    print("\n--- Section 1: Predicting SIMULATED Issue Priority (Multiclass) ---")
    
    # Create 3 'priority' bins based on the 'mean radius' feature
    X['simulated_priority'] = pd.qcut(X['mean radius'], 3, labels=['low', 'medium', 'high'])
    
    # Define our X and y for this task
    y_multi = X['simulated_priority']
    X_multi = X.drop('simulated_priority', axis=1) # Drop the label from features
    
    # Preprocessing
    scaler = StandardScaler()
    X_multi_scaled = scaler.fit_transform(X_multi)
    
    # Split
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi_scaled, y_multi, test_size=0.3, random_state=42, stratify=y_multi
    )
    
    # Train Random Forest
    rf_multi = RandomForestClassifier(random_state=42)
    rf_multi.fit(X_train_m, y_train_m)
    
    # Evaluate
    y_pred_m = rf_multi.predict(X_test_m)
    acc_m = accuracy_score(y_test_m, y_pred_m)
    f1_m = f1_score(y_test_m, y_pred_m, average='weighted') # 'weighted' for multiclass
    
    print(f"Model: Random Forest (Multiclass)")
    print(f"Accuracy: {acc_m:.4f}")
    print(f"F1-Score (Weighted): {f1_m:.4f}")
    print("\nClassification Report (Simulated Priority):")
    print(classification_report(y_test_m, y_pred_m))
    
    
    # --- SECTION 2: Solving the REAL Dataset Problem (Benign/Malignant) ---
    #
    # This is the correct, real-world use of this dataset.
    #
    print("\n--- Section 2: Predicting ACTUAL Diagnosis (Binary) ---")
    
    # Preprocessing
    X_binary_scaled = scaler.fit_transform(X.drop('simulated_priority', axis=1))
    
    # Split
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_binary_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    
    # Train Random Forest
    rf_binary = RandomForestClassifier(random_state=42)
    rf_binary.fit(X_train_b, y_train_b)
    
    # Evaluate
    y_pred_b = rf_binary.predict(X_test_b)
    acc_b = accuracy_score(y_test_b, y_pred_b)
    f1_b = f1_score(y_test_b, y_pred_b, average='binary') # 'binary' for 0/1
    
    print(f"Model: Random Forest (Binary)")
    print(f"Accuracy: {acc_b:.4f}")
    print(f"F1-Score (Binary): {f1_b:.4f}")
    print("\nClassification Report (Actual Diagnosis):")
    print(classification_report(y_test_b, y_pred_b, target_names=y_binary_names))


if __name__ == "__main__":
    run_predictive_model()
