
import os
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "student_performance_model.joblib"
DATA_PATH = BASE_DIR / "student_data.csv"
METRICS_PATH = BASE_DIR / "model_metrics.json"


# ==========================================
# 2. INPUT FEATURES
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

TARGET = "predicted_exam_score"


# ==========================================
# 3. GENERATE SYNTHETIC DATASET
# ==========================================

def load_dataset():

    np.random.seed(42)

    number_of_students = 1500

    data = pd.DataFrame({

        "hours_studied": np.random.uniform(
            1, 10, number_of_students
        ),

        "attendance": np.random.uniform(
            50, 100, number_of_students
        ),

        "previous_scores": np.random.uniform(
            30, 100, number_of_students
        ),

        "sleep_hours": np.random.uniform(
            4, 10, number_of_students
        ),

        "practice_papers": np.random.randint(
            0, 11, number_of_students
        ),

        "extracurricular": np.random.choice(
            ["Yes", "No"],
            number_of_students
        ),

        "parental_support": np.random.choice(
            ["Low", "Medium", "High"],
            number_of_students
        )

    })

    # Generate synthetic exam scores
    score = (
        data["hours_studied"] * 3.0
        + data["attendance"] * 0.20
        + data["previous_scores"] * 0.25
        + data["sleep_hours"] * 1.5
        + data["practice_papers"] * 1.5
        + data["extracurricular"].map({
            "Yes": 2,
            "No": 0
        })
        + data["parental_support"].map({
            "Low": 0,
            "Medium": 3,
            "High": 6
        })
        + np.random.normal(0, 5, number_of_students)
    )

    data[TARGET] = np.clip(score, 0, 100)

    # Save generated dataset
    data.to_csv(DATA_PATH, index=False)

    return data


# ==========================================
# 4. TRAIN MACHINE LEARNING MODEL
# ==========================================

def train_and_save():

    print("Loading student dataset...")

    data = load_dataset()

    X = data[FEATURES]
    y = data[TARGET]

    # Split dataset: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Numerical preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer,
             CATEGORICAL_FEATURES)
        ]
    )

    # Random Forest Regressor
    model = Pipeline(steps=[
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

    print("Training model...")

    model.fit(X_train, y_train)

    # ==========================================
    # 5. MODEL EVALUATION
    # ==========================================

    predictions = model.predict(X_test)

    accuracy_metrics = {
        "mae": round(
            float(mean_absolute_error(y_test, predictions)), 2
        ),

        "rmse": round(
            float(np.sqrt(mean_squared_error(y_test, predictions))), 2
        ),

        "r2_score": round(
            float(r2_score(y_test, predictions)), 4
        )
    }

    # ==========================================
    # 6. SAVE TRAINED MODEL
    # ==========================================

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as file:
        json.dump(accuracy_metrics, file, indent=4)

    print("\n================================")
    print("MODEL TRAINING COMPLETED")
    print("================================")

    print(f"MAE: {accuracy_metrics['mae']}")
    print(f"RMSE: {accuracy_metrics['rmse']}")
    print(f"R2 Score: {accuracy_metrics['r2_score']}")

    print(f"\nModel saved at: {MODEL_PATH}")
    print(f"Dataset saved at: {DATA_PATH}")
    print(f"Metrics saved at: {METRICS_PATH}")

    # Return both values for the application
    return model, accuracy_metrics


# ==========================================
# 7. RUN TRAINING
# ==========================================

if __name__ == "__main__":
    train_and_save()


    
