
    if df.empty:
        raise ValueError("The UCI dataset contains no usable records.")

    return df


def build_pipeline(X):
    """Create preprocessing and Random Forest classification pipeline."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train_and_save():
    """Fetch data, train, evaluate, save model and supporting artifacts."""
    df = fetch_dataset()

    # Target: 1 = AT RISK; 0 = NOT AT RISK.
    y = (df[TARGET_COLUMN] < 10).astype(int)

    # G3 must never be included in model inputs.
    X = df.drop(columns=[TARGET_COLUMN]).copy()

    if X.empty:
        raise ValueError("No predictor columns were found after removing G3.")

    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError(
            "The dataset needs at least two examples in each target class "
            "to perform a stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_pipeline(X_train)
    print("Training Random Forest Classifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    labels = [0, 1]
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=["Not at risk", "At risk"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "dataset": "UCI Student Performance (ID 320)",
        "records": int(len(df)),
        "features": list(X.columns),
        "target_rule": "G3 < 10 = AT RISK; G3 >= 10 = NOT AT RISK",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_at_risk": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall_at_risk": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1_at_risk": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "test_records": int(len(y_test)),
        "train_records": int(len(y_train)),
    }

    # Store original dataset including G3 for overview and historical summaries.
    # The app removes G3 before making predictions.
    df.to_csv(DATA_PATH, index=False)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nTraining completed.")
    print(f"Dataset: {DATA_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision (At risk): {metrics['precision_at_risk']:.3f}")
    print(f"Recall (At risk): {metrics['recall_at_risk']:.3f}")
    print(f"F1-score (At risk): {metrics['f1_at_risk']:.3f}")
    print("\nImportant: these results are based on the UCI historical dataset "
          "and one train/test split. They do not guarantee performance at another school.")

    return model, df, metrics


if __name__ == "__main__":
    train_and_save()





