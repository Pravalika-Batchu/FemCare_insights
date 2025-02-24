import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("women_risk_model.pkl")

# Define feature columns
FEATURES = ['Age', 'Systolic BP', 'Diastolic', 'BS', 'Body Temp', 'BMI', 'Previous Complications',
            'Preexisting Diabetes', 'Gestational Diabetes', 'Mental Health', 'Heart Rate']

# Streamlit UI
st.title("Women's Healthcare Risk Prediction")
st.write("Enter the following health details to predict the risk level and get health recommendations.")

# Input fields
age = st.number_input("Age", min_value=10, max_value=100, value=25)
systolic_bp = st.number_input("Systolic BP", min_value=60, max_value=200, value=120)
diastolic = st.number_input("Diastolic", min_value=40, max_value=120, value=80)
blood_sugar = st.number_input("Blood Sugar (BS)", min_value=3.0, max_value=20.0, value=7.0)
body_temp = st.number_input("Body Temperature", min_value=90.0, max_value=110.0, value=98.6)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
prev_complications = st.selectbox("Previous Complications", [0, 1])
preexisting_diabetes = st.selectbox("Preexisting Diabetes", [0, 1])
gestational_diabetes = st.selectbox("Gestational Diabetes", [0, 1])
mental_health = st.selectbox("Mental Health Issues", [0, 1])
heart_rate = st.number_input("Heart Rate", min_value=50, max_value=200, value=75)

# Predict button
if st.button("Predict Risk Level"):
    # Prepare input data as a dataframe
    input_data = pd.DataFrame([[age, systolic_bp, diastolic, blood_sugar, body_temp, bmi, prev_complications,
                                preexisting_diabetes, gestational_diabetes, mental_health, heart_rate]],
                              columns=FEATURES)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    risk_level = "High" if prediction == 1 else "Low"

    # Display the result
    st.subheader(f"Predicted Risk Level: {risk_level}")

    # Provide recommendations based on risk level
    if risk_level == "High":
        st.error("⚠️ **Health Alert: You are at high risk.**")
        st.write("""
        **Possible Health Conditions:**
        - Hypertension (High Blood Pressure)
        - Diabetes or Blood Sugar Irregularities
        - Obesity-related complications
        - Cardiovascular issues (Irregular Heart Rate, High BP)
        - Increased risk of stroke or heart attack
        
        **Health Recommendations:**
        - 🥗 **Diet:** Maintain a balanced diet rich in vegetables, whole grains, and lean protein.
        - 🚶 **Exercise:** Engage in at least **30 minutes of physical activity** daily.
        - 🏥 **Regular Checkups:** Monitor blood pressure, blood sugar, and heart rate frequently.
        - 😴 **Mental Health:** Reduce stress with **yoga, meditation, or counseling**.
        - 💧 **Hydration:** Drink at least **8 glasses of water** daily.
        - ❌ **Avoid:** Processed foods, excessive sugar, and high sodium intake.

        **Next Steps:**
        - Consult a doctor for a **detailed health check-up**.
        - Track symptoms and lifestyle patterns in a **health journal**.
        """)

    else:
        st.success("✅ **Good News: You have a low health risk.**")
        st.write("""
        **Health Insights:**
        - Your vitals are within a safe range.
        - You are at a lower risk of chronic diseases like hypertension and diabetes.
        
        **Tips to Maintain Good Health:**
        - 🍎 **Healthy Diet:** Continue eating balanced meals.
        - 🏃 **Stay Active:** Engage in daily workouts or yoga.
        - 🚰 **Hydration:** Keep drinking plenty of water.
        - 🛌 **Sleep Well:** Ensure 7-9 hours of quality sleep per night.
        - 🧘 **Mental Health:** Practice relaxation techniques.

        **Keep up the great work! 💪**
        """)

st.write("🔹 _Disclaimer: This is a predictive tool and should not replace professional medical advice._") 