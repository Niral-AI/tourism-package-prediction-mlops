import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------------------------------------------------------
# 1. Model Loading with Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Loads the trained scikit-learn Pipeline containing preprocessing and the 
    best-performing classification model. Caching ensures the model loads only once.
    """
    for path in ["best_model.joblib", "tourism_project/deployment/best_model.joblib", "deployment/best_model.joblib"]:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("Trained model artifact not found. Please verify CI/CD pipeline execution.")

model = load_model()

# -----------------------------------------------------------------------------
# 2. UI Layout & User Input Form
# -----------------------------------------------------------------------------
st.title("🌴 Tourism Package Purchase Predictor")
st.markdown("Predict the likelihood of a customer purchasing the Wellness Tourism Package before pitching.")

with st.form("prediction_form"):
    st.subheader("Customer & Demographic Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, value=25000)
    
    st.subheader("Trip & Pitch Details")
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    duration = st.number_input("Duration of Pitch (Minutes)", min_value=1, value=10)
    number_of_followups = st.slider("Number of Followups", min_value=1, max_value=6, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    
    st.subheader("Traveling Group & Logistics")
    persons = st.number_input("Persons Visiting", min_value=1, value=2)
    children = st.number_input("Children Visiting (Age < 5)", min_value=0, value=0)
    number_of_trips = st.number_input("Number of Trips Ordinarily in a Year", min_value=1, value=2)
    passport = st.selectbox("Owns Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_sat = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    
    submit = st.form_submit_button("Predict Probability")

# -----------------------------------------------------------------------------
# 3. Prediction Execution & Result Display
# -----------------------------------------------------------------------------
if submit:
    # Construct DataFrame with the EXACT feature names expected by the trained pipeline
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_sat,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])
    
    # Generate prediction and purchase conversion probability
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    if pred == 1:
        st.success(f"🎉 **High Likelihood of Purchase!** Conversion Probability: **{prob:.1%}**")
    else:
        st.info(f"💡 **Low Likelihood of Purchase.** Conversion Probability: **{prob:.1%}**")
