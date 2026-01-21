# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# Create Sample Telecom Dataset
# -----------------------------
data = {
    'tenure': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45],
    'MonthlyCharges': [29, 39, 49, 59, 69, 79, 89, 99, 109, 119],
    'TotalCharges': [29, 195, 490, 885, 1380, 1975, 2670, 3465, 4360, 5355],
    'Churn': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

# -----------------------------
# Split Features & Target
# -----------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Random Forest Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
