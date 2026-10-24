# Corzynth — AI Pipeline Corrosion Intelligence Platform

Corzynth is an AI-powered web application that predicts pipeline corrosion rates, assesses risk, forecasts remaining pipeline life, detects geophysical hotspots, optimizes RBI inspection intervals, and calculates corrosion mitigation ROI.

## 🚀 Live App

👉 [Launch Corzynth](https://corzynth.streamlit.app)

## ✨ Features

- **Corrosion Prediction** — ML-based corrosion rate prediction from pipeline, soil, and environmental inputs
- **Risk Assessment** — likelihood–consequence risk scoring for pipeline segments
- **Remaining Life Forecast** — estimates years to corrosion failure threshold
- **Geophysical Hotspot Detection** — identifies high-risk zones from survey data
- **RBI Optimization** — recommends risk-based inspection intervals
- **Mitigation ROI** — calculates return on investment for corrosion prevention measures
- **Secure Login** — password-protected access

## 🛠️ Tech Stack

- Python
- Streamlit
- scikit-learn (Machine Learning)
- pandas / numpy
- joblib (model serialization)

## 📦 Installation (Run Locally)

```bash
pip install -r requirements.txt
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
- Passwords are hashed (bcrypt)
- No secrets are stored in the repository

## 👤 Author

**sughrajaved768-spec**