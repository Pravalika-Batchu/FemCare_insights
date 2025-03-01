import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(".env.local")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ Gemini API Key is missing. Please check your .env.local file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=api_key)

# Load the trained model & scaler
try:
    risk_model = joblib.load("women_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_features = scaler.feature_names_in_  # Ensure correct feature alignment
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {str(e)}")
    st.stop()

FEATURES = ['Age', 'Systolic BP', 'Diastolic', 'BS (Blood Sugar)', 'Body Temp', 'BMI', 
            'Previous Complications', 'Preexisting Diabetes', 'Gestational Diabetes', 
            'Mental Health', 'PCOS', 'Menopause', 'Menstrual Irregularities', 'Pregnancy Status', 'Heart Rate']

# Function to get AI-generated health recommendations
def get_dynamic_recommendations(user_data):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Based on the following health details, provide personalized health recommendations:
        - Age: {user_data['Age']} years
        - Systolic BP: {user_data['Systolic BP']} mmHg
        - Diastolic BP: {user_data['Diastolic']} mmHg
        - Blood Sugar (BS): {user_data['BS (Blood Sugar)']} mmol/L 
        - Body Temperature: {user_data['Body Temp']} °F
        - BMI: {user_data['BMI']}
        - Previous Complications: {'Yes' if user_data['Previous Complications'] else 'No'}
        - Preexisting Diabetes: {'Yes' if user_data['Preexisting Diabetes'] else 'No'}
        - Gestational Diabetes: {'Yes' if user_data['Gestational Diabetes'] else 'No'}
        - Mental Health Issues: {'Yes' if user_data['Mental Health'] else 'No'}
        - PCOS: {'Yes' if user_data['PCOS'] else 'No'}
        - Menopause: {'Yes' if user_data['Menopause'] else 'No'}
        - Menstrual Irregularities: {'Yes' if user_data['Menstrual Irregularities'] else 'No'}
        - Pregnancy Status: {'Pregnant' if user_data['Pregnancy Status'] else 'Not Pregnant'}
        - Heart Rate: {user_data['Heart Rate']} bpm

        Provide clear and medically sound health suggestions to help improve or maintain good health.
        Summarize the key points in a concise format.
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error generating recommendations: {str(e)}"

st.markdown("""
    <style>
        body {
            background-color: #f4f6f9;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: transparent;
            color: purple;
            font-size: 18px;
            border-radius: 8px;
            padding: 12px 24px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background-color: purple;
            color:white;
        }
        .stTextInput>div>input {
            background-color: #e0f7fa;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px;
        }
        .stNumberInput>div>input {
            background-color: #e0f7fa;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px;
        }
        .stSelectbox>div>div>div {
            font-size: 16px;
        }
        .stSelectbox>div>div>div>label {
            font-size: 16px;
        }
        h1, h2 {
            color: #333;
        }
        .stExpanderHeader {
            font-size: 18px;
            font-weight: bold;
            color: #1E88E5;
        }
        .stExpander {
            border-radius: 8px;
            background-color: #f0f8ff;
            padding: 10px;
        }
        .stAlert {
            padding: 20px;
            font-size: 18px;
            font-weight: bold;
        }
        .stAlert-error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .stAlert-success {
            background-color: #d4edda;
            color: #155724;
        }
        .stMarkdown {
            font-size: 16px;
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

st.title("FemCare Insights: AI for Women's Wellness🏥")
st.write("Enter your health details to predict the risk level and receive personalized recommendations.")

DEFAULT_VALUES = {
    "Age": 30, "Systolic BP": 120, "Diastolic": 80, "BS (Blood Sugar)": 5.5,
    "Body Temp": 98.6, "BMI": 22, "Previous Complications": 0,
    "Preexisting Diabetes": 0, "Gestational Diabetes": 0, "Mental Health": 0,
    "PCOS": 0, "Menopause": 0, "Menstrual Irregularities": 0,
    "Pregnancy Status": 0, "Heart Rate": 75
}

user_data = {}
for feature in FEATURES:
    if feature in ['Previous Complications', 'Preexisting Diabetes', 'Gestational Diabetes', 'Mental Health', 'PCOS', 'Menopause', 'Menstrual Irregularities']:
        user_data[feature] = 1 if st.selectbox(feature, ["No", "Yes"], index=DEFAULT_VALUES[feature]) == "Yes" else 0
    elif feature == 'Pregnancy Status':
        user_data[feature] = 1 if st.selectbox(feature, ["Not Pregnant", "Pregnant"], index=DEFAULT_VALUES[feature]) == "Pregnant" else 0
    else:
        user_data[feature] = st.number_input(feature, min_value=0.0, value=float(DEFAULT_VALUES[feature]))

if st.button("🔍 Predict Risk Level"):
    input_df = pd.DataFrame([user_data], columns=FEATURES)
    input_df = input_df[expected_features]  

    try:
        input_scaled = scaler.transform(input_df)

        prediction = risk_model.predict(input_scaled)[0]
        risk_level = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"
        st.subheader(f"**Predicted Risk Level:** {risk_level}")
        
        # Generate personalized health recommendations
        st.write("🔄 **Generating personalized health recommendations...**")
        recommendations = get_dynamic_recommendations(user_data)
        
        if prediction == 1:
            st.error("⚠️ **High risk detected! Please consult a healthcare professional.**")
        else:
            st.success("✅ **Low health risk detected! Maintain a healthy lifestyle.**")

        summarized_tips = list(set(recommendations.split("\n")))[:5]  # Remove duplicates and limit to 5 points
        st.markdown("### **Summarized Health Recommendations:**")
        for tip in summarized_tips:
            if tip.strip():
                st.write(f"- {tip.strip()}")
        
        with st.expander("🔍 Read Full Health Recommendations"):
            st.write(recommendations)

    except Exception as e:
        st.error(f"⚠️ Error making prediction: {str(e)}")
