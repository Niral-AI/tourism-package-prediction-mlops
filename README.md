# 🌴 Tourism Package Purchase Prediction - End-to-End MLOps Pipeline

[![CI/CD Pipeline](https://github.com/Niral-AI/tourism-package-prediction-mlops/actions/workflows/pipeline.yml/badge.svg)](https://github.com/Niral-AI/tourism-package-prediction-mlops/actions/workflows/pipeline.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tourism-package-prediction-mlops-grqdujydrspwnucsx3ijm5.streamlit.app/)

## 📌 Business Context
"Visit with Us," a leading travel company, is introducing a new **Wellness Tourism Package**. To optimize marketing operations and customer engagement, this project implements a scalable, automated Machine Learning pipeline to predict which customers are most likely to purchase the package before a sales pitch is made.

This data-driven approach minimizes manual targeting errors, reduces operational costs, and increases campaign conversion rates.

## 🏗️ MLOps Architecture & Tech Stack
This project features a fully automated Continuous Integration and Continuous Delivery (CI/CD) lifecycle.

* **Language:** Python 3.11
* **Machine Learning:** Scikit-Learn, XGBoost, Pandas
* **Experiment Tracking:** MLflow
* **Model Serialization:** Cloudpickle, Joblib
* **CI/CD Orchestration:** GitHub Actions
* **Deployment & UI:** Streamlit Community Cloud

## ⚙️ Pipeline Workflow
The pipeline is orchestrated headlessly via GitHub Actions (`.github/workflows/pipeline.yml`) and consists of three primary stages:

1. **Data Registration & Quality Gate (`data_register.py`):** 
   Validates the incoming raw dataset (`tourism.csv`). It checks for critical target columns, audits missing values, and ensures structural integrity before allowing the pipeline to proceed.
2. **Data Preparation (`prep.py`):** 
   Cleanses survey anomalies, removes non-predictive identifiers, and performs an 80/20 stratified train-test split to preserve target class distribution. Prevents data leakage by deferring scaling and imputation to the model training phase.
3. **Model Training & Tuning (`train.py`):** 
   Encapsulates preprocessing and classification algorithms (Random Forest and XGBoost) inside a Scikit-Learn `Pipeline`. Systematically tunes hyperparameters using `GridSearchCV` (3-fold CV). Metrics and models are logged via **MLflow**, and the absolute best configuration (optimized for F1-Score) is serialized and exported for deployment.

## 🚀 Continuous Delivery & Deployment
The predictive model is deployed via **Streamlit Community Cloud**. 
The architecture guarantees seamless Continuous Delivery: when the GitHub Actions pipeline successfully retrains and commits a new `best_model.joblib` artifact to the `deployment/` directory, the Streamlit app automatically detects the update and serves the newly optimized model.

* **Live Web Application:** [Access the Predictor Here](https://tourism-package-prediction-mlops-grqdujydrspwnucsx3ijm5.streamlit.app/)

## 📂 Repository Structure
```text
├── .github/workflows/
│   └── pipeline.yml          # GitHub Actions CI/CD configuration
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv       # Raw, registered dataset
│   ├── deployment/
│   │   ├── app.py            # Streamlit web application
│   │   ├── best_model.joblib # Serialized winning ML pipeline
│   │   └── requirements.txt  # Cloud deployment dependencies
│   ├── model_building/
│   │   ├── data_register.py  # Quality gate script
│   │   ├── prep.py           # Stratified splitting script
│   │   └── train.py          # Tuning and MLflow tracking script
│   └── requirements.txt      # CI/CD runner dependencies
└── README.md
