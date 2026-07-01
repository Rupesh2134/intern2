import joblib
import numpy as np
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load dataset from dataset/heart.csv
file_path = "dataset/heart.csv"
df = pd.read_csv(file_path)

# Replace empty strings with NaN and convert numeric columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Create a binary target: 0 = no heart disease, 1 = heart disease
# The UCI dataset uses values 0 and 1-4 for different disease severity levels.
df["target"] = np.where(df["target"] > 0, 1, 0)

# Remove duplicate records
initial_rows = len(df)
df = df.drop_duplicates()
removed_duplicates = initial_rows - len(df)
print(f"Dropped {removed_duplicates} duplicate rows.")

# Handle missing values using median imputation for numeric columns
missing_counts = df.isna().sum()
print("Missing values by column:\n", missing_counts)
df = df.fillna(df.median(numeric_only=True))

# Define features and target
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
X = df[feature_columns]
y = df["target"]

# Use a consistent preprocessing pipeline for numeric and categorical columns
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

numeric_transformer = Pipeline([("scaler", StandardScaler())])
cat_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features),
    ]
)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results.append(
        {
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pipeline": pipeline,
        }
    )
    print(f"{name}: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, ROC-AUC={roc_auc:.3f}")

# Select the best model by ROC-AUC score
best_result = max(results, key=lambda r: r["roc_auc"])
best_pipeline = best_result["pipeline"]

print(f"\nBest model: {best_result['model']} with ROC-AUC {best_result['roc_auc']:.3f}")

# Save the best model pipeline
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "heart_disease_prediction_model.pkl")
joblib.dump(best_pipeline, model_path)
print(f"Saved best model pipeline to {model_path}")

# Print a short classification report for the best model
best_y_pred = best_pipeline.predict(X_test)
print("\nClassification report for best model:\n")
print(classification_report(y_test, best_y_pred, digits=4))

# Save feature names and importance for tree-based models if available
if hasattr(best_pipeline.named_steps["classifier"], "feature_importances_"):
    feature_names = numeric_features + list(
        best_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps[
            "encoder"
        ].get_feature_names_out(categorical_features)
    )
    importances = best_pipeline.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)
    print("Top feature importances:\n", importance_df.head(10))
