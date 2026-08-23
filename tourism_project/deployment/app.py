import streamlit as st
import pandas as pd
import joblib
import os

@st.cache_resource
def load_model():
    for path in ["best_model.joblib", "tourism_project/deployment/best_model.joblib", "deployment/best_model.joblib"]:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("Model not found. Pipeline incomplete.")

model = load_model()
st.title("Tourism Package Purchase Predictor")

with st.form("prediction_form"):
    age = st.number_input("Age", 18, 100, 30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration = st.number_input("Duration of Pitch (Minutes)", 1, value=10)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    persons = st.number_input("Persons Visiting", 1, value=2)
    children = st.number_input("Children Visiting", 0, value=0)
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    passport = st.selectbox("Owns Passport", [0, 1])
    pitch_sat = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    income = st.number_input("Monthly Income", 10000, value=25000)
    submit = st.form_submit_button("Predict Probability")

if submit:
    input_data = pd.DataFrame([{"Age": age, "CityTier": city_tier, "DurationOfPitch": duration, "Occupation": occupation, "Gender": gender, "NumberOfPersonVisiting": persons, "NumberOfChildrenVisiting": children, "MaritalStatus": marital, "Passport": passport, "PitchSatisfactionScore": pitch_sat, "MonthlyIncome": income}])
    pred, prob = model.predict(input_data)[0], model.predict_proba(input_data)[0][1]
    st.success(f"High Likelihood! ({prob:.1%})" if pred == 1 else f"Low Likelihood. ({prob:.1%})")
