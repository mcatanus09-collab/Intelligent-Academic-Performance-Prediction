"""
Train a student performance prediction model.

This project uses a reproducible synthetic dataset so it can run immediately.
For real-world use, replace the generated data with a properly collected,
consented, representative student dataset.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "student_performance_model.joblib"
DATA_PATH = ROOT / "student_performance_sample.csv"

FEATURES = [
    "hours_studied", "attendance", "previous_scores", "sleep_hours",
    "practice_papers", "extracurricular", "parental_support"
]
NUMERIC_FEATURES = [
    "hours_studied", "attendance", "previous_scores", "sleep_hours",
    "practice_papers"
]
CATEGORICAL_FEATURES = ["extracurricular", "parental_support"]
TARGET = "exam_score"


def create_sample_dataset(n_samples=1200, random_state=42):
    """Create sample data for demonstration; not real student records."""
    rng = np.random.default_rng(random_state)
    hours = rng.uniform(0, 10, n_samples)
    attendance = rng.uniform(50, 100, n_samples)
    previous = rng.uniform(35, 100, n_samples)
    sleep = rng.uniform(4, 10, n_samples)
    papers = rng.integers(0, 11, n_samples)
    extracurricular = rng.choice(["Yes", "No"], n_samples, p=[0.45, 0.55])
    support = rng.choice(["Low", "Medium", "High"], n_samples, p=[0.2, 0.5, 0.3])

    support_effect = pd.Series(support).map(
        {"Low": -2.0, "Medium": 0.0, "High": 3.0}
    ).to_numpy()
    extra_effect = np.where(extracurricular == "Yes", 1.0, 0.0)

    score = (
        18 + 2.0 * hours + 0.22 * attendance + 0.28 * previous
        + 1.25 * papers - 0.9 * np.abs(sleep - 8)
        + support_effect + extra_effect + rng.normal(0, 5, n_samples)
    )
    score = np.clip(score, 0, 100)

    return pd.DataFrame({
        "hours_studied": hours.round(1),
        "attendance": attendance.round(1),
        "previous_scores": previous.round(1),
        "sleep_hours": sleep.round(1),
        "practice_papers": papers,
        "extracurricular": extracurricular,
        "parental_support": support,
        "exam_score": score.round(1),
    })


def main():
    df = create_sample_dataset()
    df.to_csv(DATA_PATH, index=False)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=250, random_state=42, min_samples_leaf=2,
            n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Training completed successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Sample dataset saved to: {DATA_PATH}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R² Score: {r2_score(y_test, predictions):.3f}")
    print("Note: metrics reflect synthetic demonstration data, not real-world accuracy.")


if __name__ == "__main__":
    main()
