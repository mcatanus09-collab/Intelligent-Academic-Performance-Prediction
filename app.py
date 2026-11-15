  # Evaluate model
    predictions = model.predict(X_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "data_note": "Synthetic demonstration data only."
    }

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Save dataset
    df.to_csv(DATA_PATH, index=False)

    # Save metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Model training completed!")
    print(metrics)

    # Return THREE values required by app.py
    return model, df, metrics


if __name__ == "__main__":
    train_and_save()
