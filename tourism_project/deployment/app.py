import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Visit with Us - Wellness Predictor", page_icon="🌴", layout="wide")

@st.cache_resource
def load_model():
    if os.path.exists("best_model.joblib"):
        return joblib.load("best_model.joblib")
    elif os.path.exists("tourism_project/deployment/best_model.joblib"):
        return joblib.load("tourism_project/deployment/best_model.joblib")
    else:
        raise FileNotFoundError("best_model.joblib not found.")

st.title("🌴 Visit with Us: Wellness Tourism Package Predictor")
st.markdown("Predict whether a customer is likely to purchase the new **Wellness Tourism Package** before contacting them.")

try:
    model = load_model()
    
    with st.form("customer_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
            citytier = st.selectbox("City Tier", [1, 2, 3])
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=20, value=2)
            
        with col2:
            followups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=10.0, value=3.0)
            product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
            property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            trips = st.number_input("Number of Trips per Year", min_value=0.0, max_value=30.0, value=2.0)
            passport = st.selectbox("Passport Holder", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
        with col3:
            satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
            own_car = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            children = st.number_input("Children Visiting (<5 yrs)", min_value=0.0, max_value=10.0, value=0.0)
            designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            income = st.number_input("Gross Monthly Income", min_value=1000.0, max_value=500000.0, value=25000.0)
            duration_pitch = st.number_input("Pitch Duration (min)", min_value=1.0, max_value=120.0, value=15.0)
            
        submit = st.form_submit_button("Predict Purchase Likelihood")
        
    if submit:
        input_df = pd.DataFrame([{
            'Age': float(age),
            'TypeofContact': typeofcontact,
            'CityTier': int(citytier),
            'DurationOfPitch': float(duration_pitch),
            'Occupation': occupation,
            'Gender': gender,
            'NumberOfPersonVisiting': int(num_persons),
            'NumberOfFollowups': float(followups),
            'ProductPitched': product_pitched,
            'PreferredPropertyStar': float(property_star),
            'MaritalStatus': marital_status,
            'NumberOfTrips': float(trips),
            'Passport': int(passport),
            'PitchSatisfactionScore': int(satisfaction),
            'OwnCar': int(own_car),
            'NumberOfChildrenVisiting': float(children),
            'Designation': designation,
            'MonthlyIncome': float(income)
        }])
        
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        
        if prediction == 1:
            st.success(f"🎉 **High Purchase Probability: {prob*100:.1f}%** — High-priority prospect.")
        else:
            st.warning(f"⚠️ **Low Purchase Probability: {prob*100:.1f}%** — Prioritize other leads.")
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
