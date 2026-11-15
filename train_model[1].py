
"""Train a Student Performance Prediction Model using synthetic data."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. FILE PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "student_performance_model.joblib"

DATA_PATH = ROOT / "student_performance_sample.csv"
METRICS_PATH = ROOT / "model_metrics.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

NUMERIC_FEATURES = [
    "hours_studied",
    "attendance",
    "previous_scores",
    "sleep_hours",
    "practice_papers"
]

CATEGORICAL_FEATURES = [
    "extracurricular",
    "parental_support"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "exam_score"


# ==========================================
# 3. CREATE SYNTHETIC DATASET
# ==========================================

def create_sample_dataset(n_samples=1200, random_state=42):

    rng = np.random.default_rng(random_state)

    hours = rng.uniform(0, 10, n_samples)
    attendance = rng.uniform(50, 100, n_samples)
    previous = rng.uniform(35, 100, n_samples)
    sleep = rng.uniform(4, 10, n_samples)
    papers = rng.integers(0, 11, n_samples)

    extracurricular = rng.choice(
        ["Yes", "No"], n_samples
    )

    support = rng.choice(
        ["Low", "Medium", "High"], n_samples
    )

    support_effect = pd.Series(support).map({
        "Low": -2,
        "Medium": 0,
        "High": 3
    }).to_numpy()

    extra_effect = np.where(
        extracurricular == "Yes", 1, 0
    )

    score = (
        18
        + 2.0 * hours
        + 0.22 * attendance
        + 0.28 * previous
        + 1.25 * papers
        - 0.9 * np.abs(sleep - 8)
        + support_effect
        + extra_effect
        + rng.normal(0, 5, n_samples)
    )

    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        "hours_studied": hours.round(1),
        "attendance": attendance.round(1),
        "previous_scores": previous.round(1),
        "sleep_hours": sleep.round(1),
        "practice_papers": papers,
        "extracurricular": extracurricular,
        "parental_support": support,
        "exam_score": score.round(1)
    })

    return df


# ==========================================
# 4. TRAIN MODEL
# ==========================================

def train_and_save():

    print("Creating synthetic student dataset...")

    data = create_sample_dataset()

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Numerical preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    # Random Forest Regressor
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])

    print("Training machine learning model...")

    model.fit(X_train, y_train)

    # ==========================================
    # 5. MODEL EVALUATION
    # ==========================================

    predictions = model.predict(X_test)

    metrics = {
        "mae": round(
            float(mean_absolute_error(y_test, predictions)), 2
        ),

        "rmse": round(
            float(np.sqrt(mean_squared_error(y_test, predictions))), 2
        ),

        "r2": round(
            float(r2_score(y_test, predictions)), 4
        ),

        "train_records": int(len(X_train)),

        "test_records": int(len(X_test)),

        "data_note": "Synthetic demonstration data only."
    }

    # ==========================================
    # 6. SAVE MODEL AND FILES
    # ==========================================

    joblib.dump(model, MODEL_PATH)

    data.to_csv(DATA_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # ==========================================
    # 7. DISPLAY RESULTS
    # ==========================================

    print("\n===================================")
    print("MODEL TRAINING COMPLETED")
    print("===================================")

    print("MAE:", metrics["mae"])
    print("RMSE:", metrics["rmse"])
    print("R2 Score:", metrics["r2"])

    print("Training records:", metrics["train_records"])
    print("Testing records:", metrics["test_records"])

    print("\nSaved files:")
    print(MODEL_PATH)
    print(DATA_PATH)
    print(METRICS_PATH)

    # IMPORTANT: Return THREE values for app.py
    return model, data, metrics


# ==========================================
# 8. RUN TRAINING
# ==========================================

if __name__ == "__main__":
    train_and_save()
