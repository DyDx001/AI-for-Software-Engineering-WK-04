import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def load_and_prep_data():
    """
    Loads breast cancer data, engineers the 'priority' target,
    and returns processed X and y.
    """
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    
    # --- Target Engineering ---
    # Create the 'priority' (high/medium/low) target based on 'mean area'
    y_priority = pd.qcut(X['mean area'], q=3, labels=['Low', 'Medium', 'High'])
    
    # --- Feature Engineering ---
    # CRITICAL: Drop leakage columns
    features_to_drop = [
        'mean area', 'mean radius', 'mean perimeter', 
        'area error', 'radius error', 'perimeter error',
        'worst area', 'worst radius', 'worst perimeter'
    ]
    # Use .copy() to avoid SettingWithCopyWarning
    X_final = X.drop(columns=features_to_drop).copy()
    
    # Encode target
    encoder = LabelEncoder()
    y_final_encoded = encoder.fit_transform(y_priority)
    
    # Return all the pieces needed by the app
    return X_final, y_final_encoded, encoder, X_final.columns, X.mean()

def train_model(X, y):
    """
    Splits, scales, and trains a Random Forest model.
    Returns the trained model, scaler, and test sets for evaluation.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the feature data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    return model, scaler, X_test_scaled, y_test, y_pred
