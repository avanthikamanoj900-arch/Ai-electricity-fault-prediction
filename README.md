# ⚡ AI-Based Electrical Fault Prediction System

An Artificial Intelligence and Machine Learning based system that analyzes electrical parameters and automatically detects different types of electrical faults.

The system uses a trained Machine Learning model to classify the electrical condition as:

- ✅ Normal
- 🚨 Overvoltage
- 🚨 Undervoltage
- 🚨 Overcurrent

The prediction is displayed through an interactive Streamlit web application.

---

## 🎯 Project Objective

Electrical faults can cause equipment damage, power losses, overheating, and safety hazards.

This project aims to provide an intelligent fault detection system that analyzes electrical parameters and identifies abnormal operating conditions at an early stage.

The system takes the following parameters as input:

- Voltage
- Current
- Frequency
- Power Factor
- Temperature
- Total Harmonic Distortion (THD)

Based on these parameters, the trained Machine Learning model predicts the type of electrical condition.

---

## 🧠 Machine Learning Approach

The project follows this pipeline:

```text
Electrical Parameters
        ↓
Data Preprocessing
        ↓
Feature Scaling
        ↓
Train-Test Split
        ↓
Random Forest Classifier
        ↓
Model Evaluation
        ↓
Fault Prediction
        ↓
Streamlit Web Application
