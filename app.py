        mime="text/csv",
    )

    uploaded = st.file_uploader("Choose student CSV", type=["csv"])
    if uploaded is not None:
        try:
            new_data = pd.read_csv(uploaded)
            missing = [column for column in feature_columns if column not in new_data.columns]

            if new_data.empty:
                st.warning("The uploaded CSV contains no records.")
            elif missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                predictions = model.predict(new_data[feature_columns])
                results = new_data.copy()
                results["Prediction"] = [
                    RISK_LABELS[int(value)] for value in predictions
                ]

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(new_data[feature_columns])
                    results["At_Risk_Probability_%"] = (
                        probabilities[:, 1] * 100
                    ).round(1)

                st.success("Predictions completed.")
                show_prediction_result(results)
                st.dataframe(results, use_container_width=True)

                st.download_button(
                    "Download prediction results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="student_predictions.csv",
                    mime="text/csv",
                )
        except Exception as error:
            st.error(f"CSV processing error: {error}")

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    st.write("Evaluation results from the held-out test set (20% of the dataset).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("Precision (At risk)", f"{metrics['precision_at_risk'] * 100:.2f}%")
    c3.metric("Recall (At risk)", f"{metrics['recall_at_risk'] * 100:.2f}%")
    c4.metric("F1-score (At risk)", f"{metrics['f1_at_risk'] * 100:.2f}%")

    st.subheader("Classification report")
    report = metrics.get("classification_report", {})
    report_rows = []
    for label in ["Not at risk", "At risk"]:
        if label in report:
            row = report[label]
            report_rows.append({
                "Class": label,
                "Precision": row.get("precision", 0),
                "Recall": row.get("recall", 0),
                "F1-score": row.get("f1-score", 0),
                "Support": row.get("support", 0),
            })
    if report_rows:
        st.dataframe(pd.DataFrame(report_rows).set_index("Class"), use_container_width=True)

    st.subheader("Confusion matrix")
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Not at risk", "Actual: At risk"],
        columns=["Predicted: Not at risk", "Predicted: At risk"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.caption(
        "Recall for the At risk class indicates the proportion of actual at-risk "
        "students correctly identified by the model. Review precision and recall "
        "together; accuracy alone is not sufficient."
    )

elif page == "About the System":
    st.header("About the Student Performance Prediction System")
    st.write(
        "This classroom prototype uses historical student information to "
        "estimate whether a record meets the academic-risk rule defined in "
        "the laboratory exercise."
    )
    st.subheader("Dataset")
    st.write("UCI Machine Learning Repository — Student Performance (dataset ID 320).")
    st.write("https://archive.ics.uci.edu/dataset/320/student+performance")

    st.subheader("Prediction rule")
    st.markdown("- **AT RISK:** G3 < 10\n- **NOT AT RISK:** G3 ≥ 10")
    st.subheader("Methodology")
    st.markdown(
        "1. Retrieve and inspect the dataset.\n"
        "2. Remove duplicate records and prepare the target.\n"
        "3. Exclude G3 from predictor features.\n"
        "4. Split the data into 80% training and 20% testing sets.\n"
        "5. Impute missing values, scale numeric variables, and encode categorical variables.\n"
        "6. Train a Random Forest Classifier.\n"
        "7. Evaluate using accuracy, precision, recall, F1-score, and a confusion matrix."
    )
    st.subheader("Limitations and responsible use")
    st.warning(
        "This is a classroom demonstration using historical public data from "
        "Portuguese secondary-school students. Results may not generalize to "
        "another school or population. Predictions are not diagnoses and must "
        "not be the sole basis for decisions about students. An at-risk result "
        "should prompt supportive human review, not punishment or automatic action."
    )

st.divider()
st.caption(
    "Student Performance Prediction System | Python • Scikit-learn • Streamlit | "
    "Educational prototype"
)





