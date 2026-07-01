import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
model = joblib.load("models/heart_disease_prediction_model.pkl")

categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
feature_columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

recommendations = {
    "yes": "The model predicts a higher risk of heart disease. Please consult a medical professional for a detailed evaluation and consider adopting a heart-healthy lifestyle.",
    "no": "The model predicts low risk for heart disease. Maintain a balanced diet, regular exercise, and regular health checkups to keep your heart healthy.",
}


def build_input_dataframe(form_data):
    values = []
    for feature in feature_columns:
        raw_value = form_data.get(feature, "")
        if raw_value == "":
            values.append(np.nan)
        else:
            values.append(float(raw_value))
    return pd.DataFrame([values], columns=feature_columns)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    probability_score = None
    recommendation = None

    if request.method == "POST":
        input_df = build_input_dataframe(request.form)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        prediction_text = "Yes" if prediction == 1 else "No"
        probability_score = f"{probability * 100:.2f}%"
        recommendation = recommendations["yes" if prediction == 1 else "no"]

    return render_template(
        "index.html",
        prediction=prediction_text,
        probability=probability_score,
        recommendation=recommendation,
    )


if __name__ == "__main__":
    app.run(debug=True)
