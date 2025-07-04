import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Asteroid Mineability Detector")
st.title("🚀 Asteroid Mineability Detector")
st.markdown("Enter asteroid characteristics to check if it's mineable.")

# Inputs
albedo = st.number_input("Albedo", min_value=0.0, max_value=1.0, value=0.15)
diameter = st.number_input("Diameter (km)", min_value=0.0, max_value=10.0, value=0.5)
q = st.number_input("Perihelion Distance (AU)", min_value=0.0, max_value=3.0, value=1.2)
e = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.3)
i = st.number_input("i (°)", min_value=0.0, max_value=60.0, value=10.0)

if st.button("🔍 Predict"):
    input_data = np.array([[albedo, q, e, i, diameter]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][1]  # probability of class 1 (mineable)

    if prediction == 1:
        st.success(f"✅ This asteroid is **MINEABLE**")
        st.info(f"Confidence: **{confidence:.2%}**")
    else:
        st.error(f"❌ This asteroid is **NOT mineable**")
        st.info(f"Confidence: **{confidence:.2%}**")
