# Heart Disease Prediction Using Machine Learning

## Project Overview
This end-to-end data science project focuses on predicting heart disease using a real-world dataset from the UCI Machine Learning Repository. The project includes data preprocessing, exploratory data analysis, model training, evaluation, and deployment through a Flask web app.

## Project Structure
```
HeartDiseasePrediction/
│
├── dataset/
│   └── heart.csv
├── notebooks/
│   └── Heart_Disease_Analysis.ipynb
├── models/
│   └── heart_disease_prediction_model.pkl
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── requirements.txt
├── README.md
└── report.pdf
```

## Setup Instructions
1. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\\Scripts\\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset:
   ```bash
   python download_dataset.py
   ```
4. Train the models and save the best pipeline:
   ```bash
   python train_model.py
   ```
5. Run the Flask app:
   ```bash
   python app.py
   ```

## Usage
- Open the Flask app in your browser at `http://127.0.0.1:5000`
- Enter patient values and submit the form
- The app returns a prediction, probability score, and a basic recommendation

## Future Scope
- Add cross-validation and hyperparameter tuning
- Use additional heart disease datasets
- Deploy to cloud services like Heroku or Azure
- Improve UI and add patient history analysis
