"""
train_model.py
----------------
Loads the symptom-disease dataset, trains a classifier,
and saves the trained model to disk so app.py can use it later.

Run this once with: python train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load data
data = pd.read_csv("data/symptoms_disease.csv")
print("Data shape:", data.shape)
print(data['disease'].value_counts())

# 2. Split into inputs (X) and target (y)
X = data.drop('disease', axis=1)
y = data['disease']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train a Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)
print("\nDetailed report:\n", classification_report(y_test, predictions))

# 6. Save the trained model + the list of symptom column names
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "symptom_columns.pkl")
print("\nModel saved as model.pkl")
