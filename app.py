import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open("xgb_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load selected features
with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

# Streamlit UI
st.title("Heart Disease Prediction")

# Create input fields for user data
user_input = []
for feature in selected_features:
    value = st.number_input(f"Enter {feature}", value=0.0)
    user_input.append(value)

# Prediction button
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease! Consult a Doctor.")
    else:
        st.success("✅ Low Risk of Heart Disease. Maintain a healthy lifestyle!")
