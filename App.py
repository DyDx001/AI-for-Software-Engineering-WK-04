import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
# Import our new "backend" functions from model.py
from model import load_and_prep_data, train_model

# Set page config
st.set_page_config(page_title="Predictive Priority AI", layout="wide")

# --- 1. Load and Train Model (using cached functions) ---
# We wrap our imported functions in Streamlit's cache decorators
@st.cache_data
def cached_load_data():
    return load_and_prep_data()

@st.cache_resource
def cached_train_model(X, y):
    return train_model(X, y)

# Run the functions
X_final, y_final_encoded, encoder, feature_names, base_data_mean = cached_load_data()
model, scaler, X_test_scaled, y_test, y_pred = cached_train_model(X_final, y_final_encoded)

# --- 2. Streamlit UI ---

# --- Header ---
st.title("Task 3: Predictive Analytics for Resource Allocation")
st.write("This app trains a Random Forest model to predict 'Issue Priority' (Low, Medium, High) based on the Breast Cancer dataset.")

# --- 3. Performance Metrics ---
st.header("Model Performance Metrics")

# Get text labels for reports
y_test_labels = encoder.inverse_transform(y_test)
y_pred_labels = encoder.inverse_transform(y_pred)

# Calculate metrics (use the imported functions)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Display metrics
col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("F1-Score (Weighted)", f"{f1:.4f}")

# --- 4. Visualizations ---
st.subheader("Performance Visualization")

col1, col2 = st.columns(2)

with col1:
    # Plot Confusion Matrix
    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=encoder.classes_)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(fig)

with col2:
    # Classification Report
    st.write("**Classification Report**")
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(2))

# --- 5. Interactive Prediction Sidebar ---
st.sidebar.header("Live Priority Prediction")
st.sidebar.write("Adjust features to predict priority:")

def user_input_features():
    mean_texture = st.sidebar.slider('Mean Texture', 
                                     float(X_final['mean texture'].min()), 
                                     float(X_final['mean texture'].max()), 
                                     float(X_final['mean texture'].mean()))
    
    mean_concavity = st.sidebar.slider('Mean Concavity', 
                                       float(X_final['mean concavity'].min()), 
                                       float(X_final['mean concavity'].max()), 
                                       float(X_final['mean concavity'].mean()))
    
    mean_smoothness = st.sidebar.slider('Mean Smoothness', 
                                        float(X_final['mean smoothness'].min()), 
                                        float(X_final['mean smoothness'].max()), 
                                        float(X_final['mean smoothness'].mean()))
    
    # Create a DataFrame from the inputs
    base_data = base_data_mean.to_dict()
    if 'mean texture' in base_data:
        base_data['mean texture'] = mean_texture
    if 'mean concavity' in base_data:
        base_data['mean concavity'] = mean_concavity
    if 'mean smoothness' in base_data:
        base_data['mean smoothness'] = mean_smoothness
    
    input_df_full = pd.DataFrame([base_data])
    input_df = input_df_full[feature_names] 
    
    return input_df

# Get user input
input_df = user_input_features()

# Predict
if st.sidebar.button("Predict Priority"):
    # Scale the user input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction_encoded = model.predict(input_scaled)[0]
    prediction_label = encoder.inverse_transform([prediction_encoded])[0]
    
    # Display prediction
    st.sidebar.subheader("Prediction Result")
    if prediction_label == 'High':
        st.sidebar.error(f"Predicted Priority: **HIGH**")
    elif prediction_label == 'Medium':
        st.sidebar.warning(f"Predicted Priority: **MEDIUM**")
    else:
        st.sidebar.success(f"Predicted Priority: **LOW**")
