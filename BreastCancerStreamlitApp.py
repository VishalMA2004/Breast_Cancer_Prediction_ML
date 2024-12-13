import streamlit as st
import pickle
import pandas as pd

# Load the trained model, scaler, and imputer
with open('finalized_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('finalized_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('finalized_imputer.pkl', 'rb') as imputer_file:
    imputer = pickle.load(imputer_file)

# Page Configuration
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🏦")

# Title and description
st.title("Breast Cancer Prediction App")
st.write("This app predicts whether a tumor is benign or malignant based on user-provided features.")

# Feature input form
st.sidebar.header("Input Features")
def user_input_features():
    radius_mean = st.sidebar.number_input("Radius Mean", min_value=0.0, step=0.1)
    texture_mean = st.sidebar.number_input("Texture Mean", min_value=0.0, step=0.1)
    perimeter_mean = st.sidebar.number_input("Perimeter Mean", min_value=0.0, step=0.1)
    area_mean = st.sidebar.number_input("Area Mean", min_value=0.0, step=0.1)
    smoothness_mean = st.sidebar.number_input("Smoothness Mean", min_value=0.0, step=0.001)
    compactness_mean = st.sidebar.number_input("Compactness Mean", min_value=0.0, step=0.001)
    concavity_mean = st.sidebar.number_input("Concavity Mean", min_value=0.0, step=0.001)
    concave_points_mean = st.sidebar.number_input("Concave Points Mean", min_value=0.0, step=0.001)
    symmetry_mean = st.sidebar.number_input("Symmetry Mean", min_value=0.0, step=0.001)
    fractal_dimension_mean = st.sidebar.number_input("Fractal Dimension Mean", min_value=0.0, step=0.001)

    data = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,  # Adjusted to match feature names used during training
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean
    }
    return pd.DataFrame(data, index=[0])

# Get user inputs
input_df = user_input_features()

# Add a submit button
if st.sidebar.button("Submit"):
    # Ensure all features match the training set columns
    missing_cols = [col for col in scaler.feature_names_in_ if col not in input_df.columns]
    for col in missing_cols:
        input_df[col] = 0  # Fill missing columns with default values (e.g., 0)
    input_df = input_df[scaler.feature_names_in_]

    # Handle missing values
    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

    # Standardize the inputs
    scaled_input = scaler.transform(input_imputed)

    # Make predictions
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Display the results
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.write("**The tumor is predicted to be Malignant (M).**")
    else:
        st.write("**The tumor is predicted to be Benign (B).**")

    st.subheader("Prediction Probability")
    st.write(f"Malignant: {prediction_proba[0][1]:.2f}, Benign: {prediction_proba[0][0]:.2f}")

# Footer
st.write("\n---\n")
st.write("Developed using Streamlit and a Logistic Regression Model.")
