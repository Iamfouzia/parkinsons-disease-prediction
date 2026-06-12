import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

st.set_page_config(
    page_title="Parkinson's AI Detector",
    layout="centered"
)

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 800px !important;
    }

    .stApp {
        background-color: #0f1117;
    }

    .hero {
        background: #1e2a3a;
        border: 1px solid #2d3748;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        color: #e2e8f0;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
    }
    .hero p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }

    .stat-card {
        border-radius: 12px;
        padding: 1.1rem 1rem;
        text-align: center;
        border: 1px solid;
    }
    .stat-card .number {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
    }
    .stat-card .label {
        font-size: 0.78rem;
        margin: 4px 0 0 0;
    }
    .blue-card {
        background: #0d1f35;
        border-color: #1d4ed8;
    }
    .blue-card .number { color: #60a5fa; }
    .blue-card .label  { color: #3b82f6; }

    .green-card {
        background: #0a1f15;
        border-color: #15803d;
    }
    .green-card .number { color: #4ade80; }
    .green-card .label  { color: #22c55e; }

    .purple-card {
        background: #1a0f2e;
        border-color: #6d28d9;
    }
    .purple-card .number { color: #c084fc; }
    .purple-card .label  { color: #a855f7; }

    .info-box {
        background: #1a2332;
        border-left: 4px solid #3b82f6;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        color: #93c5fd;
        font-size: 0.92rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #2d3748;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #2d6a9f !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"]    { display: none; }

    [data-testid="stFileUploaderDropzone"] {
        background: #1a1f2e !important;
        border: 1px dashed #2d3748 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] span {
        color: #94a3b8 !important;
    }
    [data-testid="stFileUploaderDropzone"] p {
        color: #64748b !important;
    }
    [data-testid="stFileUploader"] label {
        color: #94a3b8 !important;
    }

    .patient-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1.1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        background: #1a1f2e;
        border: 1px solid #2d3748;
    }
    .patient-name { color: #e2e8f0; font-weight: 500; }
    .conf-text    { color: #64748b; font-size: 0.88rem; }

    .pill-sick {
        background: #2d1515;
        color: #f87171;
        padding: 3px 13px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #ef4444;
    }
    .pill-healthy {
        background: #0a1f15;
        color: #4ade80;
        padding: 3px 13px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #22c55e;
    }

    .result-danger {
        background: #1f0f0f;
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-danger h2 { color: #f87171; margin: 0; font-size: 1.5rem; }
    .result-danger p  { color: #fca5a5; margin: 0.4rem 0 0 0; }

    .result-success {
        background: #0a1f15;
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-success h2 { color: #4ade80; margin: 0; font-size: 1.5rem; }
    .result-success p  { color: #86efac; margin: 0.4rem 0 0 0; }

    .stNumberInput input {
        background: #1a1f2e !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 8px !important;
    }
    .stNumberInput label {
        color: #94a3b8 !important;
        font-size: 0.83rem !important;
    }

    .stButton > button {
        background: #2d6a9f !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        font-size: 0.95rem !important;
    }
    .stButton > button:hover {
        background: #1e4d7a !important;
    }

    [data-testid="stMetric"] {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 0.9rem;
        border: 1px solid #2d3748;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; }

    .streamlit-expanderHeader {
        background: #1a1f2e !important;
        color: #94a3b8 !important;
        border-radius: 10px !important;
        border: 1px solid #2d3748 !important;
    }

    .footer-note {
        text-align: center;
        color: #475569;
        font-size: 0.78rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #1e2a3a;
    }
    .stDownloadButton > button {
    background: #2d6a9f !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
   }
   .stDownloadButton > button:hover {
    background: #1e4d7a !important;
   }
   div[data-testid="stFileUploader"] label p {
   color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model    = joblib.load('parkinsons_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

model, scaler, feature_names = load_model()

# hero
st.markdown("""
<div class="hero">
    <h1> Parkinson's Disease AI Detector</h1>
    <p>Early detection through voice biomarker analysis</p>
</div>
""", unsafe_allow_html=True)

# stats
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class="stat-card blue-card">
        <p class="number">92.31%</p>
        <p class="label">Model Accuracy</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="stat-card green-card">
        <p class="number">0.96</p>
        <p class="label">ROC-AUC Score</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="stat-card purple-card">
        <p class="number">22</p>
        <p class="label">Voice Features</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Upload File", "Quick Check"])

with tab1:
    st.markdown("""<div class="info-box">
        Upload a CSV file — all patients analyzed automatically.
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose your CSV file", type=["csv"]
    )

    with st.expander("Required CSV columns"):
        st.code(", ".join(feature_names))

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"{len(df)} patient(s) loaded!")
        st.dataframe(df, use_container_width=True)

        if st.button("Analyze Patients"):
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                with st.spinner("Analyzing voice patterns..."):
                    time.sleep(1)

                X     = scaler.transform(df[feature_names])
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]

                st.markdown("### Results")
                for i, (p, conf) in enumerate(zip(preds, probs)):
                    conf_val = conf if p == 1 else 1 - conf
                    pill = (
                        '<span class="pill-sick">Parkinson\'s</span>'
                        if p == 1 else
                        '<span class="pill-healthy">Healthy</span>'
                    )
                    st.markdown(f"""
                    <div class="patient-row">
                        <span class="patient-name">Patient {i+1}</span>
                        {pill}
                        <span class="conf-text">
                            {conf_val*100:.1f}% confident
                        </span>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                total   = len(preds)
                sick    = int(sum(preds))
                healthy = total - sick

                m1, m2, m3 = st.columns(3)
                m1.metric("Total",       total)
                m2.metric("Healthy",     healthy)
                m3.metric("Parkinson's", sick)

                out_df = pd.DataFrame({
                    "Patient":    range(1, total + 1),
                    "Result":     ["Parkinson's" if p == 1
                                   else "Healthy" for p in preds],
                    "Confidence": [f"{(c if p==1 else 1-c)*100:.1f}%"
                                   for p, c in zip(preds, probs)]
                })
                st.download_button(
                    "⬇ Download Results",
                    out_df.to_csv(index=False),
                    file_name="results.csv",
                    mime="text/csv"
                )

with tab2:
    st.markdown("""<div class="info-box">
        Enter voice measurements for one patient and click Predict.
    </div>""", unsafe_allow_html=True)

    user_input = {}
    col1, col2 = st.columns(2)
    for idx, feat in enumerate(feature_names):
        with (col1 if idx % 2 == 0 else col2):
            user_input[feat] = st.number_input(
                feat, value=0.0, format="%.5f", key=feat
            )

    if st.button("Run Prediction", key="manual_btn"):
        with st.spinner("Analyzing..."):
            time.sleep(0.8)

        arr    = np.array([list(user_input.values())])
        result = model.predict(scaler.transform(arr))[0]
        prob   = model.predict_proba(scaler.transform(arr))[0][1]
        conf   = prob if result == 1 else 1 - prob

        if result == 1:
            st.markdown(f"""
            <div class="result-danger">
                <h2>⚠️ Parkinson's Detected</h2>
                <p>Confidence: <b>{conf*100:.1f}%</b></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-success">
                <h2>✅ No Parkinson's Detected</h2>
                <p>Confidence: <b>{conf*100:.1f}%</b></p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.warning(
            "AI prediction only — not a medical diagnosis. "
            "Please consult a qualified doctor."
        )

st.markdown("""
<div class="footer-note">
    Built by Fouzia &nbsp;|&nbsp; AI Healthcare Project
    &nbsp;|&nbsp; Random Forest · UCI Parkinson's Dataset
</div>
""", unsafe_allow_html=True)
