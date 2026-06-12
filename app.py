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

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    
    .hero {
        background: #1a3a5c;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    .hero p {
        color: #a8c8e8;
        font-size: 1rem;
        margin: 0;
    }
    .hero .badge {
        display: inline-block;
        background: #27ae60;
        color: white;
        font-size: 0.8rem;
        padding: 4px 14px;
        border-radius: 20px;
        margin-top: 0.8rem;
        font-weight: 600;
    }

    .stat-card {
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .stat-card .number {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .stat-card .label {
        font-size: 0.8rem;
        margin: 4px 0 0 0;
        opacity: 0.85;
    }
    .blue-card  { background:#dbeafe; }
    .blue-card  .number { color:#1d4ed8; }
    .blue-card  .label  { color:#3b6fd4; }

    .green-card { background:#dcfce7; }
    .green-card .number { color:#15803d; }
    .green-card .label  { color:#16a34a; }

    .purple-card { background:#ede9fe; }
    .purple-card .number { color:#6d28d9; }
    .purple-card .label  { color:#7c3aed; }

    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
        color: #1e40af;
        font-size: 0.95rem;
    }

    .result-danger {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-danger h2 { color:#dc2626; margin:0; font-size:1.6rem; }
    .result-danger p  { color:#991b1b; margin:0.4rem 0 0 0; font-size:1.1rem; }

    .result-success {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-success h2 { color:#16a34a; margin:0; font-size:1.6rem; }
    .result-success p  { color:#166534; margin:0.4rem 0 0 0; font-size:1.1rem; }

    .patient-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.7rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .pill-sick {
        background:#fef2f2; color:#dc2626;
        padding:4px 14px; border-radius:20px;
        font-size:0.85rem; font-weight:600;
        border: 1px solid #fca5a5;
    }
    .pill-healthy {
        background:#f0fdf4; color:#16a34a;
        padding:4px 14px; border-radius:20px;
        font-size:0.85rem; font-weight:600;
        border: 1px solid #86efac;
    }

    .footer-note {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }

    .stButton > button {
        background: #1a3a5c !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        font-size: 1rem !important;
        transition: background 0.2s !important;
    }
    .stButton > button:hover {
        background: #2d6a9f !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model    = joblib.load('parkinsons_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

model, scaler, feature_names = load_model()

# ── HERO ──
st.markdown("""
<div class="hero">
    <h1>🧠 Parkinson's Disease AI Detector</h1>
    <p>Early detection through voice biomarker analysis</p>
    <span class="badge">✔ 92% Accuracy &nbsp;|&nbsp; ROC-AUC 0.96</span>
</div>
""", unsafe_allow_html=True)

# ── STATS ──
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

# ── TABS ──
tab1, tab2 = st.tabs(["📁 Upload CSV", "✏️ Manual Input"])

# ── TAB 1 ──
with tab1:
    st.markdown("""<div class="info-box">
        Upload a CSV file — the app will analyze 
        all patients automatically in one click.
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose your CSV file", type=["csv"]
    )

    with st.expander("Required CSV columns"):
        st.code(", ".join(feature_names))

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"{len(df)} patient(s) loaded successfully!")
        st.dataframe(df, use_container_width=True)

        if st.button("Analyze Patients"):
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                with st.spinner("Analyzing voice patterns..."):
                    time.sleep(1)

                preds = model.predict(scaler.transform(df[feature_names]))
                probs = model.predict_proba(
                    scaler.transform(df[feature_names]))[:, 1]

                st.markdown("### Results")
                for i, (p, conf) in enumerate(zip(preds, probs)):
                    conf_val = conf if p == 1 else 1 - conf
                    pill = (
                        '<span class="pill-sick">Parkinson\'s Detected</span>'
                        if p == 1 else
                        '<span class="pill-healthy">Healthy</span>'
                    )
                    st.markdown(f"""
                    <div class="patient-row">
                        <span><b>Patient {i+1}</b></span>
                        {pill}
                        <span style="color:#64748b; font-size:0.9rem;">
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

                results_df = pd.DataFrame({
                    "Patient":    range(1, total + 1),
                    "Result":     ["Parkinson's" if p==1
                                   else "Healthy" for p in preds],
                    "Confidence": [f"{(c if p==1 else 1-c)*100:.1f}%"
                                   for p, c in zip(preds, probs)]
                })
                st.download_button(
                    "⬇ Download Results as CSV",
                    results_df.to_csv(index=False),
                    file_name="results.csv",
                    mime="text/csv"
                )

# ── TAB 2 ──
with tab2:
    st.markdown("""<div class="info-box">
        Enter voice measurement values for one patient 
        and click Predict.
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
            "Please consult a doctor."
        )

# ── FOOTER ──
st.markdown("""
<div class="footer-note">
    Built by Fouzia &nbsp;|&nbsp; AI Healthcare Project 
    &nbsp;|&nbsp; Random Forest · UCI Dataset
</div>
""", unsafe_allow_html=True)
