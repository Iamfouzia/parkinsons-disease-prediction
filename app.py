import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

st.set_page_config(
    page_title="Parkinson's AI Detector",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin: 0.5rem 0 0 0;
        color: white;
    }
    .stat-box {
        background: #f0f4f8;
        border-left: 4px solid #2d6a9f;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .result-card-danger {
        background: #fff0f0;
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-card-success {
        background: #f0fff4;
        border: 2px solid #27ae60;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .stButton > button {
        background: #2d6a9f;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: #1e3a5f;
        transform: translateY(-1px);
    }
    .upload-box {
        border: 2px dashed #2d6a9f;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8faff;
    }
    .badge-healthy {
        background: #27ae60;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-sick {
        background: #e74c3c;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .info-banner {
        background: #e8f4f8;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3498db;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load model files
@st.cache_resource
def load_model():
    model = joblib.load('parkinsons_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

model, scaler, feature_names = load_model()

# Sidebar
with st.sidebar:
    st.markdown("### About This App")
    st.markdown("""
    This AI tool uses **voice measurements** to 
    predict Parkinson's disease.
    
    **Model:** Random Forest  
    **Accuracy:** 92.31%  
    **ROC-AUC:** 0.96  
    **Dataset:** UCI Parkinson's Dataset  
    **Features:** 22 voice biomarkers
    """)
    st.markdown("---")
    st.markdown("### How To Use")
    st.markdown("""
    **Option 1 — CSV Upload:**  
    Upload a CSV with patient voice data
    
    **Option 2 — Manual Input:**  
    Enter values one by one
    """)
    st.markdown("---")
    st.caption("Built by Fouzia | AI Healthcare Project")
    st.caption("⚠️ For research purposes only")

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Parkinson's Disease AI Detector</h1>
    <p>Early detection using voice biomarker analysis • 92% Accuracy</p>
</div>
""", unsafe_allow_html=True)

# Stats row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="stat-box">
    <b>92.31%</b><br><small>Model Accuracy</small>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="stat-box">
    <b>0.96</b><br><small>ROC-AUC Score</small>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="stat-box">
    <b>22</b><br><small>Voice Features</small>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["📁 Upload CSV File", "✏️ Manual Input"])

# ── TAB 1: CSV Upload ──
with tab1:
    st.markdown("""<div class="info-banner">
    Upload a CSV file with patient voice measurements. 
    The app will automatically analyze all patients at once.
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your CSV file here",
        type=["csv"],
        help="CSV must contain the 22 voice measurement columns"
    )

    with st.expander("See required CSV columns"):
        st.code(", ".join(feature_names))

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded! Found {len(df)} patient(s)")
        st.dataframe(df, use_container_width=True)

        if st.button("Analyze All Patients"):
            missing_cols = [c for c in feature_names 
                          if c not in df.columns]

            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                with st.spinner("Analyzing voice patterns..."):
                    time.sleep(1)

                input_data = df[feature_names]
                scaled_data = scaler.transform(input_data)
                predictions = model.predict(scaled_data)
                confidence = model.predict_proba(scaled_data)[:, 1]

                st.markdown("### Results")

                for i in range(len(predictions)):
                    col_a, col_b, col_c = st.columns([2, 2, 1])
                    with col_a:
                        st.write(f"**Patient {i+1}**")
                    with col_b:
                        if predictions[i] == 1:
                            st.markdown(
                                '<span class="badge-sick">'
                                "Parkinson's Detected</span>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<span class="badge-healthy">'
                                "Healthy</span>",
                                unsafe_allow_html=True
                            )
                    with col_c:
                        conf_val = (confidence[i] if predictions[i]==1 
                                   else 1-confidence[i])
                        st.write(f"{conf_val*100:.1f}%")

                st.markdown("---")
                total = len(predictions)
                sick = int(sum(predictions))
                healthy = total - sick

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Patients", total)
                c2.metric("Healthy", healthy, 
                         delta=f"{healthy/total*100:.0f}%")
                c3.metric("Parkinson's", sick,
                         delta=f"-{sick/total*100:.0f}%",
                         delta_color="inverse")

                results_df = pd.DataFrame({
                    'Patient': range(1, total+1),
                    'Result': ["Parkinson's" if p==1 
                              else "Healthy" for p in predictions],
                    'Confidence': [f"{(c if p==1 else 1-c)*100:.1f}%"
                                  for p, c in zip(predictions, confidence)]
                })
                csv_out = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_out,
                    file_name="parkinsons_results.csv",
                    mime="text/csv"
                )

# ── TAB 2: Manual Input ──
with tab2:
    st.markdown("""<div class="info-banner">
    Enter voice measurement values manually for a single patient.
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Voice Measurements")

    user_input = {}
    col1, col2 = st.columns(2)

    for idx, feature in enumerate(feature_names):
        if idx % 2 == 0:
            with col1:
                user_input[feature] = st.number_input(
                    feature, value=0.0, format="%.5f", key=feature
                )
        else:
            with col2:
                user_input[feature] = st.number_input(
                    feature, value=0.0, format="%.5f", key=feature
                )

    if st.button("Run Prediction", key="manual_btn"):
        with st.spinner("Analyzing..."):
            time.sleep(0.8)

        input_array = np.array([list(user_input.values())])
        scaled = scaler.transform(input_array)
        result = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]
        confidence_val = prob if result == 1 else 1 - prob

        if result == 1:
            st.markdown(f"""
            <div class="result-card-danger">
                <h2 style="color:#e74c3c; margin:0">
                    Parkinson's Disease Detected
                </h2>
                <p style="font-size:1.3rem; margin:0.5rem 0">
                    Confidence: <b>{confidence_val*100:.1f}%</b>
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-success">
                <h2 style="color:#27ae60; margin:0">
                    No Parkinson's Detected
                </h2>
                <p style="font-size:1.3rem; margin:0.5rem 0">
                    Confidence: <b>{confidence_val*100:.1f}%</b>
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.warning(
            "This is an AI-based prediction for research purposes only. "
            "Always consult a qualified medical professional for diagnosis."
        )
