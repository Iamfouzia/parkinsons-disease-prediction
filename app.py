import streamlit as st
import joblib
import numpy as np
import pandas as pd

# loading my trained model and scaler
model = joblib.load('parkinsons_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Parkinsons Detector", page_icon="🧠")

st.title("🧠 Parkinson's Disease Prediction")
st.write("This app predicts Parkinson's disease using voice measurements.")
st.write("You can either upload a CSV file or enter values manually.")
st.markdown("---")

tab1, tab2 = st.tabs(["📁 Upload CSV File", "✏️ Enter Manually"])

with tab1:
    st.subheader("Upload your CSV file")
    st.write("Make sure your CSV has these columns:")
    st.code(", ".join(feature_names))
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.write("Your uploaded data:")
        st.dataframe(df)
        
        predict_btn = st.button("Predict Now 🔍")
        
        if predict_btn:
            # check if all columns exist
            missing_cols = [col for col in feature_names 
                          if col not in df.columns]
            
            if len(missing_cols) > 0:
                st.error("These columns are missing: " + 
                        str(missing_cols))
            else:
                input_data = df[feature_names]
                scaled_data = scaler.transform(input_data)
                predictions = model.predict(scaled_data)
                confidence = model.predict_proba(scaled_data)[:, 1]
                
                st.write("### Prediction Results:")
                
                for i in range(len(predictions)):
                    if predictions[i] == 1:
                        st.error(f"Patient {i+1}: "
                                f"Parkinson's Detected "
                                f"({confidence[i]*100:.1f}% confident)")
                    else:
                        st.success(f"Patient {i+1}: Healthy "
                                  f"({(1-confidence[i])*100:.1f}% confident)")
                
                st.markdown("---")
                total = len(predictions)
                sick = sum(predictions)
                st.info(f"Summary → Total: {total} | "
                       f"Healthy: {total-sick} | "
                       f"Parkinson's: {sick}")

with tab2:
    st.subheader("Enter voice measurement values")
    st.write("Fill in the values below and click predict")
    
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(
            label=feature,
            value=0.0,
            format="%.5f"
        )
    
    if st.button("Predict 🔍", key="manual_predict"):
        input_array = np.array([list(user_input.values())])
        scaled_array = scaler.transform(input_array)
        result = model.predict(scaled_array)[0]
        prob = model.predict_proba(scaled_array)[0][1]
        
        st.markdown("---")
        if result == 1:
            st.error(f"⚠️ Parkinson's Disease Detected!\n\n"
                    f"Confidence: {prob*100:.1f}%")
        else:
            st.success(f"✅ No Parkinson's Detected!\n\n"
                      f"Confidence: {(1-prob)*100:.1f}%")
        
        st.warning("⚠️ This is an AI prediction only. "
                  "Please consult a doctor for proper diagnosis.")
