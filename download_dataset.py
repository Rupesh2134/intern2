import csv
import os
import urllib.request

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
output_path = os.path.join("dataset", "heart.csv")

print("Downloading dataset...")
raw = urllib.request.urlopen(url).read().decode("utf-8").strip().splitlines()
columns = [
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
    "target",
]
rows = [row.split(",") for row in raw]

for row in rows:
    for i, value in enumerate(row):
        if value == "?":
            row[i] = ""

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(rows)

print(f"Saved heart disease dataset to {output_path}")
