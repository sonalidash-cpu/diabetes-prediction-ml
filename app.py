import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Custom CSS
st.markdown("""
<style>
.big-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
.stButton>button {
    background-color: #00bcd4;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 50px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="big-title">🧠 Diabetes Prediction System</div>', unsafe_allow_html=True)
st.write("### Enter Patient Health Details:")

# Input layout
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0, value=1)
    glucose = st.number_input("Glucose Level", value=85.0)
    bp = st.number_input("Blood Pressure", value=70.0)
    skin = st.number_input("Skin Thickness", value=20.0)

with col2:
    insulin = st.number_input("Insulin", value=80.0)
    bmi = st.number_input("BMI", value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", value=0.5)
    age = st.number_input("Age", value=25)

# Prediction
if st.button("🔍 Predict Diabetes Risk"):

    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    result = model.predict(data)
    prob = model.predict_proba(data)

    st.write("## 📋 Prediction Result")

    if result[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    # Confidence
    confidence = max(prob[0]) * 100

    st.write("## 📊 Prediction Confidence")
    st.progress(int(confidence))
    st.info(f"Confidence Level: {confidence:.2f}%")