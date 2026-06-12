import streamlit as st
import joblib
import numpy as np

model = joblib.load('parkinsons_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title("Parkinson's Disease Prediction")
st.write("Enter voice measurement values to predict Parkinson's disease.")

input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0, format="%.5f")
    input_data.append(value)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    if prediction == 1:
        st.error(f"Parkinson's Detected (Confidence: {probability*100:.2f}%)")
    else:
        st.success(f"Healthy (Confidence: {(1-probability)*100:.2f}%)")
