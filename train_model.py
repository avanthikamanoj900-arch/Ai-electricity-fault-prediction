import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load dataset
data = pd.read_csv("data/electrical_fault_data.csv")

print("Dataset loaded successfully!")
print("Number of samples:", len(data))

# Select input features
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

# Select target
y = data["Fault"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scale the data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(X_train_scaled, y_train)

print("Model training completed!")


# Make predictions
y_pred = model.predict(X_test_scaled)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)


# Save trained model
joblib.dump(model, "models/electrical_fault_model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")


print("\nModel saved successfully!")
print("Scaler saved successfully!")
