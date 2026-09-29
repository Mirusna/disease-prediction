# Disease Prediction from Symptoms

A machine learning project that predicts the most likely disease based on a
patient's symptoms, built as a hands-on introduction to classification,
model evaluation, and deploying an interactive ML app.

## How it works

1. **Data**: Each row represents a patient case — a set of symptoms (0/1)
   paired with the diagnosed disease.
2. **Model**: A Random Forest classifier is trained on the symptom data to
   learn which symptom combinations are associated with which diseases.
3. **App**: A Streamlit web interface lets a user select symptoms via
   checkboxes and instantly see the top 3 predicted diseases with
   confidence scores.

## Project structure

```
disease-prediction/
├── data/
│   └── symptoms_disease.csv    # training data
├── train_model.py              # trains and saves the model
├── app.py                      # Streamlit web app
├── requirements.txt            # dependencies
└── README.md
```

## Running it locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (creates model.pkl)
python train_model.py

# 3. Launch the web app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Tech stack

- **Python** — core language
- **pandas** — data loading and manipulation
- **scikit-learn** — model training (Random Forest) and evaluation
- **Streamlit** — interactive web app front-end
- **joblib** — saving/loading the trained model

## Results

The Random Forest model achieves strong accuracy on held-out test data
(see console output from `train_model.py` for the exact score and a
per-disease breakdown via `classification_report`).

## Disclaimer

This project is built for educational purposes to demonstrate an
end-to-end machine learning workflow. It is **not** a medical diagnostic
tool and should not be used as a substitute for professional medical
advice.

## Possible next steps

- Add SHAP-based explainability to show which symptoms drove each prediction
- Expand the dataset with more diseases and real-world data
- Add severity flags for symptoms that warrant urgent care
