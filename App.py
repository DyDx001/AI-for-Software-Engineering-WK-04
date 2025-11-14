import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- 1. Data Loading (Cached) ---
@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Binary labels
    y_binary = data.target
    y_binary_names = data.target_names

    # Simulated multiclass
    y_multi_labels = ['low', 'medium', 'high']
    X['simulated_priority'] = pd.qcut(X['mean radius'], 3, labels=y_multi_labels)
    y_multi = X['simulated_priority']
    
    X_trainable = X.drop('simulated_priority', axis=1)
    
    return X_trainable, y_binary, y_multi, y_binary_names, y_multi_labels


# --- 2. Model Training (Cached) ---
@st.cache_resource
def train_model(X_data, y_data):
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test, X_test_df


# --- 3. Main UI ---
st.set_page_config(layout="wide")
st.title("🔬 Predictive Analytics Showcase")
st.write("Using the Breast Cancer Dataset to train two different models.")

# Load data
X, y_binary, y_multi, bin_names, multi_names = load_data()


# --- 4. Sidebar Navigation ---
st.sidebar.title("Controls")
app_mode = st.sidebar.selectbox(
    "Choose a Model to Analyze:",
    ["Simulated Issue Priority", "Actual Cancer Diagnosis"]
)


# --- 5. Model Logic ---
if app_mode == "Simulated Issue Priority":
    st.header("Model 1: Predicting Simulated Issue Priority (Multiclass)")
    st.write("This model predicts a 'priority' (low/medium/high) based on tumor features.")

    model, scaler, X_test_scaled, y_test, X_test_df = train_model(X, y_multi)

    st.subheader("Model Performance")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("F1-Score (Weighted)", f"{f1:.4f}")

    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred))

else:
    st.header("Model 2: Predicting Actual Cancer Diagnosis (Binary)")
    st.write("This model predicts if a tumor is 'malignant' or 'benign'.")

    model, scaler, X_test_scaled, y_test, X_test_df = train_model(X, y_binary)

    st.subheader("Model Performance")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("F1-Score (Binary)", f"{f1:.4f}")

    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=bin_names))


# --- 6. Interactive Predictor ---
st.sidebar.header("Live Predictor")
st.sidebar.write("Select a sample from the test set to predict.")

sample_index = st.sidebar.slider(
    "Select a test sample index:",
    0, len(X_test_df) - 1, 0
)

sample_data_df = X_test_df.iloc[[sample_index]]

st.sidebar.subheader("Sample Features:")
st.sidebar.dataframe(sample_data_df)


# --- Corrected Prediction Logic ---
if st.sidebar.button("Run Prediction"):
    sample_data_scaled = scaler.transform(sample_data_df)
    prediction_proba = model.predict_proba(sample_data_scaled)[0]

    # Correct mapping using model.classes_
    predicted_label = model.classes_[np.argmax(prediction_proba)]

    actual_label = y_test.iloc[sample_index]

    st.sidebar.subheader("Prediction")
    st.sidebar.write(f"**Actual:** `{actual_label}`")
    st.sidebar.write(f"**Predicted:** `{predicted_label}`")

    st.sidebar.write("Prediction Probabilities:")
    st.sidebar.dataframe(pd.Series(prediction_proba, index=model.classes_))
