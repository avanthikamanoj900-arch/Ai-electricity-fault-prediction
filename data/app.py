import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Electrical Fault Predictor",
    page_icon="⚡",
    layout="centered"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("models/electrical_fault_model.pkl")
scaler = joblib.load("models/scaler.pkl")


# ==========================================
# TITLE
# ==========================================

st.title("⚡ AI Electrical Fault Prediction")

st.write(
    "Enter the electrical parameters below and the trained "
    "Machine Learning model will predict the possible fault."
)


# ==========================================
# INPUT PARAMETERS
# ==========================================

st.subheader("Electrical Parameters")


voltage = st.number_input(
    "Voltage (V)",
    min_value=0.0,
    max_value=500.0,
    value=230.0
)


current = st.number_input(
    "Current (A)",
    min_value=0.0,
    max_value=50.0,
    value=5.0
)


frequency = st.number_input(
    "Frequency (Hz)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)


power_factor = st.number_input(
    "Power Factor",
    min_value=0.0,
    max_value=1.0,
    value=0.95
)


temperature = st.number_input(
    "Temperature (°C)",
    min_value=-50.0,
    max_value=150.0,
    value=30.0
)


thd = st.number_input(
    "THD (%)",
    min_value=0.0,
    max_value=100.0,
    value=2.0
)


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔍 PREDICT FAULT"):

    # Create input dataframe

    input_data = pd.DataFrame(
        [[
            voltage,
            current,
            frequency,
            power_factor,
            temperature,
            thd
        ]],
        columns=[
            "Voltage",
            "Current",
            "Frequency",
            "Power_Factor",
            "Temperature",
            "THD"
        ]
    )


    # Scale input

    input_scaled = scaler.transform(input_data)


    # Make prediction

    prediction = model.predict(input_scaled)[0]


    # Prediction probability

    probabilities = model.predict_proba(input_scaled)[0]

    confidence = max(probabilities) * 100


    # ==========================================
    # DISPLAY RESULT
    # ==========================================

    st.subheader("Prediction Result")


    if prediction == "Normal":

        st.success(
            f"✅ NORMAL SYSTEM\n\n"
            f"Confidence: {confidence:.2f}%"
        )

    else:

        st.error(
            f"🚨 FAULT DETECTED\n\n"
            f"Fault Type: {prediction}\n\n"
            f"Confidence: {confidence:.2f}%"
        )


    # ==========================================
    # SHOW INPUT DATA
    # ==========================================

    st.subheader("Input Parameters")

    st.dataframe(input_data)
