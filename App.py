import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- 1. Data Loading (Cached) ---
# Use st.cache_data to load data once and store it in cache
@st.cache_data
def load_data():
    """Loads, preprocesses, and returns the dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Target 1: Original Binary (Benign/Malignant)
    y_binary = data.target 
    y_binary_names = data.target_names # ['malignant', 'benign']

    # Target 2: Simulated Multiclass (Priority)
    # We create 3 'priority' bins based on the 'mean radius' feature
    y_multi_labels = ['low', 'medium', 'high']
    X['simulated_priority'] = pd.qcut(X['mean radius'], 3, labels=y_multi_labels)
    y_multi = X['simulated_priority']
    
    # Clean X for training (remove the simulated label)
    X_trainable = X.drop('simulated_priority', axis=1)
    
    return X_trainable, y_binary, y_multi, y_binary_names, y_multi_labels

# --- 2. Model Training (Cached) ---
# Use st.cache_resource to train the model once and cache it
@st.cache_resource
def train_model(X_data, y_data):
    """Splits data, trains a Random Forest, and returns all objects."""
    
    # Split the data
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Return everything needed for evaluation and prediction
    return model, scaler, X_test_scaled, y_test, X_test_df

# --- 3. Main App UI ---
st.set_page_config(layout="wide")
st.title("🔬 Predictive Analytics Showcase")
st.write("Using the Breast Cancer Dataset to train two different models.")

# Load the data
X, y_binary, y_multi, bin_names, multi_names = load_data()

# --- 4. Sidebar for Navigation ---
st.sidebar.title("Controls")
app_mode = st.sidebar.selectbox(
    "Choose a Model to Analyze:",
    ["Simulated Issue Priority", "Actual Cancer Diagnosis"]
)

# --- 5. Model Logic ---
if app_mode == "Simulated Issue Priority":
    st.header("Model 1: Predicting Simulated Issue Priority (Multiclass)")
    st.write("This model predicts a 'priority' (low/medium/high) based on tumor features.")
    
    # Train the multiclass model
    model, scaler, X_test_scaled, y_test, X_test_df = train_model(X, y_multi)
    target_names = multi_names

    # --- Performance ---
    st.subheader("Model Performance")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("F1-Score (Weighted)", f"{f1:.4f}")
    
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=target_names))

else: # Actual Cancer Diagnosis
    st.header("Model 2: Predicting Actual Cancer Diagnosis (Binary)")
    st.write("This model predicts if a tumor is 'malignant' or 'benign'.")

    # Train the binary model
    model, scaler, X_test_scaled, y_test, X_test_df = train_model(X, y_binary)
    target_names = bin_names

    # --- Performance ---
    st.subheader("Model Performance")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary') # Use 'binary'
    
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("F1-Score (Binary)", f"{f1:.4f}")
    
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=target_names))

# --- 6. Interactive Predictor (in Sidebar) ---
st.sidebar.header("Live Predictor")
st.sidebar.write("Select a sample from the test set to predict.")

# Use the test set (before scaling) to show the user
sample_index = st.sidebar.slider(
    "Select a test sample index:", 0, len(X_test_df) - 1, 0
)

# Get the single row of data
sample_data_df = X_test_df.iloc[[sample_index]]

st.sidebar.subheader("Sample Features:")
st.sidebar.dataframe(sample_data_df)

# --- Prediction Logic ---
if st.sidebar.button("Run Prediction"):
    # Scale the sample data
    sample_data_scaled = scaler.transform(sample_data_df)
    
    # Get prediction and probabilities
    prediction_proba = model.predict_proba(sample_data_scaled)[0]
    prediction = np.argmax(prediction_proba)
    
    # Get the correct label
    predicted_label = target_names[prediction]
    
    # Get the actual label
    actual_label_index = y_test.iloc[sample_index]
    actual_label = target_names[actual_label_index] if app_mode == "Actual Cancer Diagnosis" else actual_label_index

    st.sidebar.subheader("Prediction")
    st.sidebar.write(f"**Actual:** `{actual_label}`")
    st.sidebar.write(f"**Predicted:** `{predicted_label}`")
    
    st.sidebar.write("Prediction Probabilities:")
    st.sidebar.dataframe(pd.Series(prediction_proba, index=model.classes_))
