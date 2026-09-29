import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@st.cache_resource
def load_and_train_model():
    data = pd.read_csv("symptoms_disease.csv")
    X = data.drop("disease", axis=1)
    y = data["disease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, list(X.columns), test_accuracy


model, symptom_columns, test_accuracy = load_and_train_model()

st.set_page_config(page_title="Disease Prediction", page_icon="🩺")
st.title("🩺 Symptom-Based Disease Predictor")
st.write(
    "Select the symptoms you're experiencing, and the model will show "
    "the most likely conditions. **This is a learning project, not medical advice.**"
)
st.caption(f"Model test accuracy: {test_accuracy*100:.1f}%")
st.divider()

st.subheader("Select your symptoms:")
selected = {}
cols = st.columns(3)
for i, symptom in enumerate(symptom_columns):
    label = symptom.replace("_", " ").title()
    with cols[i % 3]:
        selected[symptom] = st.checkbox(label)

st.divider()

if st.button("Predict Disease", type="primary"):
    input_row = pd.DataFrame(
        [[int(selected[s]) for s in symptom_columns]], columns=symptom_columns
    )
    if input_row.sum(axis=1)[0] == 0:
        st.warning("Please select at least one symptom.")
    else:
        probabilities = model.predict_proba(input_row)[0]
        disease_names = model.classes_
        results = pd.DataFrame(
            {"Disease": disease_names, "Confidence": probabilities}
        ).sort_values("Confidence", ascending=False).head(3)

        st.subheader("Top predicted conditions:")
        for _, row in results.iterrows():
            st.write(f"**{row['Disease']}** — {row['Confidence']*100:.1f}% confidence")
            st.progress(float(row["Confidence"]))

        st.info(
            "⚠️ This tool is for educational purposes only and is not a "
            "substitute for professional medical advice. Please consult a doctor."
        )
