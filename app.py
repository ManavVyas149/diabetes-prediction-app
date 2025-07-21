import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Diabetes Prediction App")

st.write("""
This app predicts whether a person has diabetes based on health input features.
""")

# Sidebar user inputs
st.sidebar.header('Input Parameters')

def user_input_features():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    blood_pressure = st.sidebar.number_input('BloodPressure', min_value=0, max_value=140, value=70)
    skin_thickness = st.sidebar.number_input('SkinThickness', min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=900, value=79)
    bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.sidebar.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=33)

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
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Show user input
st.subheader('User Input parameters')
st.write(input_df)

# Prediction button
if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader('Prediction')
    if prediction == 1:
        st.write("The model predicts: **Diabetic**")
    else:
        st.write("The model predicts: **Non-Diabetic**")

    # Show prediction probabilities
    st.subheader('Prediction Probabilities')
    proba_df = pd.DataFrame({
        'Class': ['Non-Diabetic (0)', 'Diabetic (1)'],
        'Probability': prediction_proba
    }).set_index('Class')

    st.bar_chart(proba_df)