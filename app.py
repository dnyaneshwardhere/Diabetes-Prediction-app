import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model (trained without scaling)
model = joblib.load("diabetes_model.pkl")

# Title
st.title("🩺 Diabetes Risk Predictor")

# Input fields
def user_input():
    st.subheader("Enter Patient Info")
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 100)
    blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 10, 100, 30)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    result_text = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.subheader("Prediction Result")
    st.write(f"**Outcome:** {result_text}")
    st.write(f"**Confidence:** {proba[prediction]:.2f}")

    # Probability bar plot
    st.subheader("Probability Chart")
    fig, ax = plt.subplots()
    sns.barplot(x=["Not Diabetic", "Diabetic"], y=proba, palette="Blues", ax=ax)
    st.pyplot(fig)
