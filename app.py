# app.py

# Import required libraries
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained Random Forest model
model = joblib.load("rf_model.pkl")

# Page title
st.title("Diabetes Prediction using Random Forest")

# Input form for patient data
st.subheader("Enter Patient Details")

with st.form("diabetes_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=72)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)

    with col2:
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.2f")

    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.2f")
        age = st.number_input("Age", min_value=10, max_value=100, value=30)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    input_df = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    st.subheader("Prediction Result")
    st.write(f"Outcome: {result}")
    st.write(f"Confidence: {probabilities[prediction]:.2f}")

    # Probability chart
    st.subheader("Prediction Probability")
    fig, ax = plt.subplots()
    sns.barplot(x=["Not Diabetic", "Diabetic"], y=probabilities, palette="Blues", ax=ax)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # Display entered input
    st.subheader("Entered Data")
    st.dataframe(input_df)
