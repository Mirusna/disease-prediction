"""
app.py
------
A simple Streamlit web app that lets a user check off symptoms
and get the top predicted diseases from the trained model.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the symptom column order it expects
model = joblib.load("model.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")

st.set_page_config(page_title="Disease Prediction", page_icon="🩺")

st.title("🩺 Symptom-Based Disease Predictor")
st.write(
    "Select the symptoms you're experiencing, and the model will show "
    "the most likely conditions. **This is a learning project, not medical advice.**"
)

st.divider()

# Show a checkbox for every symptom, arranged in a few columns
st.subheader("Select your symptoms:")
selected = {}
cols = st.columns(3)
for i, symptom in enumerate(symptom_columns):
    label = symptom.replace("_", " ").title()
    with cols[i % 3]:
        selected[symptom] = st.checkbox(label)

st.divider()

if st.button("Predict Disease", type="primary"):
    # Build a single-row input in the exact column order the model expects
    input_row = pd.DataFrame([[int(selected[s]) for s in symptom_columns]],
                              columns=symptom_columns)

    if input_row.sum(axis=1)[0] == 0:
        st.warning("Please select at least one symptom.")
    else:
        # Get probability for each disease, not just the single best guess
        probabilities = model.predict_proba(input_row)[0]
        disease_names = model.classes_

        results = pd.DataFrame({
            "Disease": disease_names,
            "Confidence": probabilities
        }).sort_values("Confidence", ascending=False).head(3)

        st.subheader("Top predicted conditions:")
        for _, row in results.iterrows():
            st.write(f"**{row['Disease']}** — {row['Confidence']*100:.1f}% confidence")
            st.progress(float(row["Confidence"]))

        st.info(
            "⚠️ This tool is for educational purposes only and is not a "
            "substitute for professional medical advice. Please consult a "
            "doctor for an actual diagnosis."
        )
