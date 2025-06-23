# mental_health_predictor_app.py

import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model and scaler
with open('mental_health_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🧠 Mental Health Disorder Risk Predictor")

st.markdown("""
Enter the required details below and click **Predict** to see if the person is likely to need mental health treatment.
""")

# Input form
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 18, 65, 30)
country = st.selectbox("Country", ["United States", "United Kingdom", "Canada", "Germany", "India", "Nigeria", "Others"])
self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
work_interfere = st.selectbox("How often does mental health interfere with work?", ["Never", "Rarely", "Sometimes", "Often"])
no_employees = st.selectbox("Company Size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
tech_company = st.selectbox("Is it a tech company?", ["Yes", "No"])
benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
care_options = st.selectbox("Do you have mental health care options?", ["Yes", "No", "Not sure"])
wellness_program = st.selectbox("Does your employer have a wellness program?", ["Yes", "No", "Don't know"])
seek_help = st.selectbox("Is there help available for mental health?", ["Yes", "No", "Don't know"])
anonymity = st.selectbox("Is anonymity protected?", ["Yes", "No", "Don't know"])
leave = st.selectbox("How easy is it to take mental health leave?", ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])
mental_health_consequence = st.selectbox("Consequence of mental health disclosure?", ["Yes", "No", "Maybe"])
phys_health_consequence = st.selectbox("Consequence of physical health disclosure?", ["Yes", "No", "Maybe"])
coworkers = st.selectbox("Can you talk to coworkers?", ["Yes", "No", "Some of them"])
supervisor = st.selectbox("Can you talk to your supervisor?", ["Yes", "No", "Some of them"])
mental_health_interview = st.selectbox("Comfortable discussing mental health in interview?", ["Yes", "No", "Maybe"])
phys_health_interview = st.selectbox("Comfortable discussing physical health in interview?", ["Yes", "No", "Maybe"])
mental_vs_physical = st.selectbox("Is mental health as important as physical health?", ["Yes", "No", "Don't know"])
obs_consequence = st.selectbox("Have you observed consequences of mental health issues at work?", ["Yes", "No"])

# Encode inputs
input_data = [gender, country, self_employed, family_history, work_interfere, no_employees, remote_work, tech_company,
              benefits, care_options, wellness_program, seek_help, anonymity, leave, mental_health_consequence,
              phys_health_consequence, coworkers, supervisor, mental_health_interview, phys_health_interview,
              mental_vs_physical, obs_consequence]

# Convert to numerical values using same order as LabelEncoder() earlier
encoders = [
    LabelEncoder().fit(["Male", "Female", "Other"]),
    LabelEncoder().fit(["United States", "United Kingdom", "Canada", "Germany", "India", "Nigeria", "Others"]),
    LabelEncoder().fit(["Yes", "No"]),
    LabelEncoder().fit(["Yes", "No"]),
    LabelEncoder().fit(["Never", "Rarely", "Sometimes", "Often"]),
    LabelEncoder().fit(["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]),
    LabelEncoder().fit(["Yes", "No"]),
    LabelEncoder().fit(["Yes", "No"]),
    LabelEncoder().fit(["Yes", "No", "Don't know"]),
    LabelEncoder().fit(["Yes", "No", "Not sure"]),
    LabelEncoder().fit(["Yes", "No", "Don't know"]),
    LabelEncoder().fit(["Yes", "No", "Don't know"]),
    LabelEncoder().fit(["Yes", "No", "Don't know"]),
    LabelEncoder().fit(["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]),
    LabelEncoder().fit(["Yes", "No", "Maybe"]),
    LabelEncoder().fit(["Yes", "No", "Maybe"]),
    LabelEncoder().fit(["Yes", "No", "Some of them"]),
    LabelEncoder().fit(["Yes", "No", "Some of them"]),
    LabelEncoder().fit(["Yes", "No", "Maybe"]),
    LabelEncoder().fit(["Yes", "No", "Maybe"]),
    LabelEncoder().fit(["Yes", "No", "Don't know"]),
    LabelEncoder().fit(["Yes", "No"])
]

encoded = []
for val, enc in zip(input_data, encoders):
    encoded.append(enc.transform([val])[0])

# Final input vector
final_input = [age] + encoded

# ✅ Debug: Show length of input vector
st.write("🧪 Feature vector length:", len(final_input))  # Should be 23

# Scale the input
final_input_np = scaler.transform([final_input])

# Prediction
if st.button("Predict"):
    prediction = model.predict(final_input_np)
    if prediction[0] == 1:
        st.error("⚠️ Likely needs mental health treatment.")
    else:
        st.success("✅ Unlikely to need mental health treatment.")
