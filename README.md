# Corzynth — AI Pipeline Corrosion Intelligence Platform

Corzynth is an AI-powered web application that predicts pipeline corrosion rates, assesses risk, forecasts remaining pipeline life, detects geophysical hotspots, optimizes RBI inspection intervals, and calculates corrosion mitigation ROI — enabling data-driven integrity and maintenance decisions.

## 🚀 Live App

👉 [Launch Corzynth](https://corzynth.streamlit.app/)

## ✨ Features

- **Corrosion Prediction** — ML-based corrosion rate prediction from pipeline, soil, and environmental inputs
- **Risk Assessment** — likelihood–consequence risk scoring for pipeline segments
- **Remaining Life Forecast** — estimates years until corrosion reaches the failure threshold
- **Geophysical Hotspot Detection** — identifies high-risk zones from geophysical survey data
- **RBI Optimization** — recommends risk-based inspection intervals
- **Mitigation ROI** — calculates return on investment for corrosion prevention measures
- **Secure Login** — password-protected access to the platform

## 🛠️ Tech Stack

- Python
- Streamlit
- scikit-learn (Machine Learning)
- pandas / numpy
- joblib (model serialization)

## 📦 Run Locally

1. Clone the repository:

```bash
git clone [https://github.com/sughrajaved768-spec/corzynth.git](https://github.com/sughrajaved768-spec/corzynth.git)
cd corzynth
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
app.py                  → Main Streamlit application
corrosion_model.joblib  → Trained ML model
train-model.py          → Model training script
requirements.txt        → Python dependencies
Synthetic-Corrosion-Data-300-Data-Only.csv
Synthetic-Geophysical-Data-300-Rows.csv
```

## 🔐 Security

- Credentials are stored securely in Streamlit Cloud Secrets
- Passwords are hashed using bcrypt
- No secrets or credentials are stored in this repository

## 📊 Data

The platform uses synthetic corrosion and geophysical datasets (300 records each) for demonstration and model training.

## 👤 Author

**sughrajaved768-spec**

- GitHub: [sughrajaved768-spec](https://github.com/sughrajaved768-spec)
- Live App: [corzynth.streamlit.app](https://corzynth.streamlit.app/)