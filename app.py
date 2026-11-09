"""Student Performance Prediction System - Streamlit App"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_model import train_and_save, FEATURES, MODEL_PATH

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction System")

st.write(
    "Enter the student's learning-related information "
    "to estimate an exam score."
)

st.info(
    "Educational demonstration only. The model uses synthetic "
    "data and is not intended for actual student grading."
)


@st.cache_resource
def get_model():
    """Load existing model or automatically train a new one."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            st.warning("Saved model could not be loaded. Retraining...")

    return train_and_save()


try:
    with st.spinner("Preparing prediction model..."):
        model = get_model()

    st.success("Prediction model is ready!")

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


st.subheader("Student Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        hours_studied = st.slider(
            "Hours studied per day", 0.0, 10.0, 4.0, 0.5
        )

        attendance = st.slider(
            "Attendance (%)", 50.0, 100.0, 85.0, 1.0
        )

        previous_scores = st.slider(
            "Previous scores (%)", 0.0, 100.0, 70.0, 1.0
        )

    with col2:
        sleep_hours = st.slider(
            "Sleep hours per night", 4.0, 10.0, 7.0, 0.5
        )

        practice_papers = st.slider(
            "Practice papers completed", 0, 10, 3
        )

        extracurricular = st.selectbox(
            "Extracurricular activities", ["Yes", "No"]
        )

        parental_support = st.selectbox(
            "Parental support level", ["Low", "Medium", "High"]
        )

    submitted = st.form_submit_button(
        "Predict Performance",
        type="primary"
    )


if submitted:

    student = pd.DataFrame([{
        "hours_studied": hours_studied,
        "attendance": attendance,
        "previous_scores": previous_scores,
        "sleep_hours": sleep_hours,
        "practice_papers": practice_papers,
        "extracurricular": extracurricular,
        "parental_support": parental_support
    }])

    try:
        prediction = float(model.predict(student)[0])
        prediction = float(np.clip(prediction, 0, 100))

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            "Estimated Exam Score",
            f"{prediction:.1f} / 100"
        )

        if prediction >= 85:
            st.success("Estimated Performance: Excellent")

        elif prediction >= 75:
            st.info("Estimated Performance: Good")

        elif prediction >= 60:
            st.warning("Estimated Performance: Satisfactory")

        else:
            st.error("Estimated Performance: Needs Improvement")

    except Exception as error:
        st.error(f"Prediction error: {error}")


st.divider()

st.subheader("Batch Student Prediction")

st.write("Upload a CSV file containing student information.")

uploaded = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded is not None:

    try:
        data = pd.read_csv(uploaded)

        missing = [
            column for column in FEATURES
            if column not in data.columns
        ]

        if missing:
            st.error(
                "Missing columns: " + ", ".join(missing)
            )

        elif data.empty:
            st.warning("The CSV file contains no records.")

        else:
            predictions = model.predict(data[FEATURES])

            result = data.copy()

            result["predicted_exam_score"] = np.clip(
                predictions, 0, 100
            ).round(1)

            st.success("Predictions completed!")

            st.dataframe(
                result,
                use_container_width=True
            )

            st.download_button(
                "Download Prediction Results",
                result.to_csv(index=False).encode("utf-8"),
                file_name="student_predictions.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(f"CSV processing error: {error}")


st.divider()

st.caption(
    "Student Performance Prediction System | "
    "Python, Machine Learning and Streamlit"
)
