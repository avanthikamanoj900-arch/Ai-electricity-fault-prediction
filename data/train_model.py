import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("data/electrical_fault_data.csv")

print("Dataset loaded successfully!")
print("Number of samples:", len(data))


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = data[
    [
        "Voltage",
        "Current",
        "Frequency",
        "Power_Factor",
        "Temperature",
        "THD"
    ]
]

y = data["Fault"]


# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. SCALE FEATURES
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. TRAIN RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Model training completed!")


# ==========================================
# 6. TEST MODEL
# ==========================================

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 7. CREATE MODELS FOLDER
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# 8. SAVE MODEL AND SCALER
# ==========================================

joblib.dump(model, "models/electrical_fault_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")


print("\nModel saved successfully!")
print("Scaler saved successfully!")
