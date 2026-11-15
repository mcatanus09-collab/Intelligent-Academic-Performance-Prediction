"""Student Performance Prediction System - Streamlit application."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_model import (
    FEATURES, MODEL_PATH, DATA_PATH, METRICS_PATH, train_and_save,
)

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Prediction System")
st.write(
    "Estimate a student's exam score using study habits and learning-related information."
)
st.warning(
    "Educational demonstration only. This project uses synthetic data, "
    "not real student records, and must not be used for actual grading or high-stakes decisions."
)                                                                                                                                                       


@st.cache_resource
def get_model_and_artifacts():
    """Load saved model and artifacts, or train them if they are not present."""
    if MODEL_PATH.exists() and DATA_PATH.exists() and METRICS_PATH.exists():
        model = joblib.load(MODEL_PATH)
        data = pd.read_csv(DATA_PATH)
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        return model, data, metrics
    return train_and_save()


try:
    with st.spinner("Loading prediction model..."):
        model, data, metrics = get_model_and_artifacts()
    st.success("Prediction model is ready!")
except Exception as exc:
    st.error(f"Could not prepare the model: {exc}")
    st.info("Check requirements.txt and the Streamlit deployment logs.")
    st.stop()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Individual Prediction", "Batch Prediction", "Model Evaluation", "About"],
)

if page == "Dashboard":
    st.header("Dashboard Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample student records", f"{len(data):,}")
    c2.metric("Model MAE", f"{metrics['mae']:.2f}")
    c3.metric("Model R²", f"{metrics['r2']:.3f}")

    st.subheader("Sample exam score distribution")
    st.bar_chart(data["exam_score"].value_counts(bins=10, sort=False))
    st.subheader("Sample dataset preview")
    st.dataframe(data.head(10), use_container_width=True)
    st.download_button(
        "Download sample dataset (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="student_performance_sample.csv",
        mime="text/csv",
    )

elif page == "Individual Prediction":
    st.header("Individual Student Prediction")
    st.write("Enter the student's information below.")
    with st.form("individual_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            hours_studied = st.slider("Hours studied per day", 0.0, 10.0, 4.0, 0.5)
            attendance = st.slider("Attendance (%)", 50.0, 100.0, 85.0, 1.0)
            previous_scores = st.slider("Previous scores (%)", 35.0, 100.0, 70.0, 1.0)
            sleep_hours = st.slider("Sleep hours per night", 4.0, 10.0, 7.0, 0.5)
        with col2:
            practice_papers = st.slider("Practice papers completed", 0, 10, 3)
            extracurricular = st.selectbox("Extracurricular activities", ["Yes", "No"])
            parental_support = st.selectbox("Parental support level", ["Low", "Medium", "High"])
        submitted = st.form_submit_button("Predict Performance", type="primary")

    if submitted:
        student = pd.DataFrame([{
            "hours_studied": hours_studied,
            "attendance": attendance,
            "previous_scores": previous_scores,
            "sleep_hours": sleep_hours,
            "practice_papers": practice_papers,
            "extracurricular": extracurricular,
            "parental_support": parental_support,
        }], columns=FEATURES)
        try:
            predicted_score = float(np.clip(model.predict(student)[0], 0, 100))
            st.subheader("Prediction Result")
            st.metric("Estimated Exam Score", f"{predicted_score:.1f} / 100")
            if predicted_score >= 85:
                st.success("Estimated performance: Excellent")
            elif predicted_score >= 75:
                st.info("Estimated performance: Good")
            elif predicted_score >= 60:
                st.warning("Estimated performance: Satisfactory")
            else:
                st.error("Estimated performance: Needs Improvement")
            st.caption("This is an estimate from synthetic demonstration data, not an actual grade.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Batch Prediction":
    st.header("Batch Student Prediction")
    st.write("Upload a CSV containing all seven predictor columns. Do not include exam_score.")
    template = data[FEATURES].head(5)
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="student_prediction_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload student CSV", type=["csv"])
    if uploaded is not None:
        try:
            new_data = pd.read_csv(uploaded)
            missing = [column for column in FEATURES if column not in new_data.columns]
            if new_data.empty:
                st.warning("The uploaded CSV contains no student records.")
            elif missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                predictions = np.clip(model.predict(new_data[FEATURES]), 0, 100)
                results = new_data.copy()
                results["predicted_exam_score"] = np.round(predictions, 1)
                st.success(f"Predictions completed for {len(results)} student record(s).")
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download prediction results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="student_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"CSV processing error: {exc}")

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    st.write("Evaluation on a held-out test set from the synthetic demonstration dataset.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.2f}")
    c2.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']:.2f}")
    c3.metric("R² Score", f"{metrics['r2']:.3f}")
    st.write(f"Training records: **{metrics['train_records']}**")
    st.write(f"Testing records: **{metrics['test_records']}**")
    st.caption(metrics["data_note"])

else:
    st.header("About the System")
    st.markdown(
        """
        **Purpose:** Demonstrate how machine learning can estimate an exam score
        from study-related information.

        **Inputs:** Hours studied, attendance, previous scores, sleep hours,
        practice papers completed, extracurricular activities, and parental support.

        **Process:** Data preparation → preprocessing → Random Forest regression → prediction.

        **Outputs:** Estimated exam score, performance category, batch CSV predictions,
        and evaluation metrics.

        **Limitation:** The model is trained on synthetic demonstration data. Its
        metrics do not establish real-world accuracy or fairness. Use a properly
        collected, consented, representative dataset before any real-world study.
        """
    )

st.divider()
st.caption("Student Performance Prediction System | Python • Scikit-learn • Streamlit")

