# Parkinson's Disease Prediction

A machine learning web application that predicts Parkinson's Disease by analyzing clinical biomarker data uploaded as a CSV file using a Random Forest classifier trained on the UCI Parkinson's Dataset.

**Live App:**  [Click here to open](https://parkinsons-disease-prediction-kfhxzpzfe6ehsxkctpoytp.streamlit.app)

---

## What This Project Does

This project takes a CSV file containing 22 biomarker features and predicts whether a patient has Parkinson's Disease or not. Users upload their data directly no audio recording or microphone required.

---

## Dataset
**Source:** UCI Machine Learning Repository

**Link:** https://archive.ics.uci.edu/ml/datasets/parkinsons

| Property | Value |
|---|---|
| Total Samples | 195 |
| Total Features | 24 columns |
| Features Used | 22 (after dropping `name` and `status`) |
| Parkinson's Cases | 147 (75.38%) |
| Healthy Cases | 48 (24.62%) |
| Missing Values | 0 |

---

## ML Pipeline

```
Load CSV Data
     ↓
Exploratory Data Analysis (EDA)
     ↓
Drop 'name' column
     ↓
Separate Features (X) and Target (y = status)
     ↓
Train/Test Split — 80% Train, 20% Test (stratified)
     ↓
StandardScaler — fit on train, transform on test
     ↓
Train 4 Models and Compare
     ↓
GridSearchCV Hyperparameter Tuning
     ↓
5-Fold Cross Validation
     ↓
Feature Importance Analysis
     ↓
Save Model + Scaler + Feature Names
     ↓
Deploy via Streamlit
```

---

## Algorithms Compared

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| SVM (RBF kernel) | Non-linear classifier |
| Random Forest | ✅ Selected best stability |
| KNN (k=5) | Distance-based classifier |

**Why Random Forest was selected:**
- Most stable on small datasets
- Provides feature importance rankings
- Less prone to overfitting
- Reliable performance across cross-validation folds

---

## Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 92.31% |
| ROC-AUC Score | 0.96 |
| Cross-Validation | 5-Fold on training set |

**Hyperparameter Tuning via GridSearchCV:**
```
n_estimators:     [50, 100, 200]
max_depth:        [None, 5, 10, 20]
min_samples_split:[2, 5, 10]
```

---

## Features Used (22 Voice Biomarkers)

| Category | Features |
|---|---|
| Frequency | MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz) |
| Jitter | MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP |
| Shimmer | MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA |
| Noise | NHR, HNR |
| Nonlinear | RPDE, DFA, spread1, spread2, D2, PPE |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| ML Library | scikit-learn |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Model Saving | joblib |
| Web App | Streamlit |
| Training | Google Colab |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## Project Structure

```
parkinsons-disease-prediction/
├── app.py                                   # Streamlit web app
├── parkinsons_model.pkl                     # Trained Random Forest model
├── scaler.pkl                               # Fitted StandardScaler
├── feature_names.pkl                        # 22 feature names list
├── requirements.txt                         # Dependencies
├── Parkinson's_Disease_Prediction.ipynb     # Training notebook
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/Iamfouzia/parkinsons-disease-prediction.git
cd parkinsons-disease-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## Requirements

```
streamlit
scikit-learn
joblib
numpy
pandas
```

---

## How to Use the App

**Upload File tab:**
1. Download dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
2. Save as `parkinsons.csv`
3. Upload in app all patients analyzed automatically
4. Download results as CSV

**Quick Check tab:**
Enter 22 feature values manually for a single patient and get instant prediction with confidence score.

---

## Disclaimer

This tool is for research and educational purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified neurologist.
