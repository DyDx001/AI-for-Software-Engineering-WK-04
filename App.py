import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Set page config
st.set_page_config(page_title="Predictive Priority AI", layout="wide")

# --- 1. Data Loading and Preprocessing ---
# We cache the data loading to speed up the app.
@st.cache_data
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
    # CRITICAL: We must drop all features directly related to the target 
    # ('mean area', 'mean radius', 'mean perimeter') to prevent data leakage.
    # The model should learn from texture, concavity, etc., not the proxy.
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
    
    return X_final, y_final_encoded, encoder, X_final.columns, X.mean()

# --- 2. Model Training ---
# We cache the trained model to avoid retraining on every interaction.
@st.cache_resource
def train_model(X, y):
    """
    Splits, scales, and trains a Random Forest model.
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

# --- 3. Streamlit UI ---

# Load data and train model
X_final, y_final_encoded, encoder, feature_names, base_data_mean = load_and_prep_data()
model, scaler, X_test_scaled, y_test, y_pred = train_model(X_final, y_final_encoded)

# --- Header ---
st.title("Task 3: Predictive Analytics for Resource Allocation")
st.write("This app trains a Random Forest model to predict 'Issue Priority' (Low, Medium, High) based on the Breast Cancer dataset.")

# --- 4. Performance Metrics ---
st.header("Model Performance Metrics")

# Get text labels for reports
y_test_labels = encoder.inverse_transform(y_test)
y_pred_labels = encoder.inverse_transform(y_pred)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Display metrics
col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("F1-Score (Weighted)", f"{f1:.4f}")

# --- 5. Visualizations ---
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

# --- 6. Interactive Prediction Sidebar ---
st.sidebar.header("Live Priority Prediction")

# Get a few key features for the sliders. Let's pick based on importance.
st.sidebar.write("Adjust features to predict priority:")

def user_input_features():
    # We will use the non-scaled data to get realistic min/max for sliders
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
    # We use a 'base' row (the mean) and overwrite it with slider values
    # This ensures the model gets all features it expects
    base_data = base_data_mean.to_dict() # Get a dict of all features
    
    # Overwrite the dict with our slider values
    # We only need to provide the features our *final model* was trained on
    if 'mean texture' in base_data:
        base_data['mean texture'] = mean_texture
    if 'mean concavity' in base_data:
        base_data['mean concavity'] = mean_concavity
    if 'mean smoothness' in base_data:
        base_data['mean smoothness'] = mean_smoothness
    
    # Create a DataFrame from the dictionary
    input_df_full = pd.DataFrame([base_data])
    
    # Ensure the columns match the model's training columns
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
