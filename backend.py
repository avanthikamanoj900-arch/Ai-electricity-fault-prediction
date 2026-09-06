from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Create FastAPI application
app = FastAPI(
    title="ElectroGuard AI API",
    description="AI-based electrical fault detection backend",
    version="1.0"
)

# Load trained model and scaler
model = joblib.load("electrical_fault_model.pkl")
scaler = joblib.load("scaler.pkl")


# Input data structure
class ElectricalData(BaseModel):

    voltage: float
    current: float
    frequency: float
    power_factor: float
    temperature: float
    thd: float


# Home endpoint
@app.get("/")
def home():

    return {
        "message": "ElectroGuard AI Backend is running",
        "status": "online"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: ElectricalData):

    # Arrange input in the same order used during training
    input_data = np.array([[
        data.voltage,
        data.current,
        data.frequency,
        data.power_factor,
        data.temperature,
        data.thd
    ]])

    # Apply the trained scaler
    input_scaled = scaler.transform(input_data)

    # Predict fault
    prediction = model.predict(input_scaled)[0]

    # Calculate confidence
    probabilities = model.predict_proba(input_scaled)[0]

    confidence = float(np.max(probabilities) * 100)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }
