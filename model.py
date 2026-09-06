"""
model.py — backend ML logic for ElectroGuard AI.

Loads the trained Random Forest model + StandardScaler (produced by
train_model.py) and exposes a single predict() function that the Streamlit
UI (app.py) calls. Keeping this separate from app.py means the model-loading
and inference logic can be reused, unit-tested, or swapped out (e.g. moved
behind a real API) without touching the UI code.
"""

from typing import Tuple
import numpy as np
import joblib
import streamlit as st

# NOTE: adjust this label map to match the exact class order your model was
# trained/encoded with — check train_model.py for the LabelEncoder / class
# order used at training time.
LABEL_MAP = {
    0: "Normal",
    1: "Overvoltage",
    2: "Undervoltage",
    3: "Overcurrent",
}


@st.cache_resource
def load_model():
    """Load and cache the trained model and scaler from disk."""
    model = joblib.load("electrical_fault_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


def predict(
    voltage: float,
    current: float,
    frequency: float,
    power_factor: float,
    temperature: float,
    thd: float,
) -> Tuple[str, float]:
    """
    Run inference on a single set of electrical readings.

    Returns (label, confidence_percent).
    """
    model, scaler = load_model()

    data = np.array([[voltage, current, frequency, power_factor, temperature, thd]])
    scaled_data = scaler.transform(data)

    raw_prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    confidence = float(np.max(probabilities) * 100)

    try:
        label = LABEL_MAP.get(int(raw_prediction), str(raw_prediction))
    except (TypeError, ValueError):
        label = str(raw_prediction)

    return label, confidence
