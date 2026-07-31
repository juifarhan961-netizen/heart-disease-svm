import streamlit as st
import pandas as pd
import joblib

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction")

input_data = {}

with st.form("prediction_form"):

    input_data["Age"] = st.number_input("Age")
    input_data["Sex"] = st.number_input("Sex")
    input_data["Chest pain type"] = st.number_input("Chest pain type")
    input_data["BP"] = st.number_input("BP")
    input_data["Cholesterol"] = st.number_input("Cholesterol")
    input_data["FBS over 120"] = st.number_input("FBS over 120")
    input_data["EKG results"] = st.number_input("EKG results")
    input_data["Max HR"] = st.number_input("Max HR")
    input_data["Exercise angina"] = st.number_input("Exercise angina")
    input_data["ST depression"] = st.number_input("ST depression")
    input_data["Slope of ST"] = st.number_input("Slope of ST")
    input_data["Number of vessels fluro"] = st.number_input("Number of vessels fluro")
    input_data["Thallium"] = st.number_input("Thallium")

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([input_data])
    input_df = scaler.transform(input_df)

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")
