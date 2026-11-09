"""Streamlit interface for the Student Performance Prediction System."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_performance_model.joblib"

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Prediction System")
st.write(
    "Enter the student's learning-related information to estimate an exam score. "
    "This is an educational demonstration, not a formal assessment."
)

if not MODEL_PATH.exists():
    st.warning("The trained model was not found. Train it first using: python train_model.py")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

with st.form("prediction_form"):
    st.subheader("Student Information")
    col1, col2 = st.columns(2)
    with col1:
        hours_studied = st.slider("Hours studied per day", 0.0, 10.0, 4.0, 0.5)
        attendance = st.slider("Attendance (%)", 50.0, 100.0, 85.0, 1.0)
        previous_scores = st.slider("Previous scores (%)", 0.0, 100.0, 70.0, 1.0)
    with col2:
        sleep_hours = st.slider("Sleep hours per night", 4.0, 10.0, 7.0, 0.5)
        practice_papers = st.slider("Practice papers completed", 0, 10, 3)
        extracurricular = st.selectbox("Participates in extracurricular activities?", ["Yes", "No"])
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
    }])
    prediction = float(model.predict(student)[0])
    prediction = max(0.0, min(100.0, prediction))

    st.divider()
    st.subheader("Prediction Result")
    st.metric("Estimated exam score", f"{prediction:.1f} / 100")
    if prediction >= 85:
        st.success("Estimated performance: Excellent")
    elif prediction >= 75:
        st.info("Estimated performance: Good")
    elif prediction >= 60:
        st.warning("Estimated performance: Satisfactory")
    else:
        st.error("Estimated performance: Needs Improvement")

    st.caption(
        "The estimate is generated from a synthetic training dataset. "
        "It should not be used to grade, label, or make high-impact decisions about a student."
    )

st.divider()
st.subheader("Optional: Predict from a CSV file")
st.write("Upload a CSV containing the seven input columns used above.")
uploaded = st.file_uploader("Choose CSV", type=["csv"])
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        required = [
            "hours_studied", "attendance", "previous_scores", "sleep_hours",
            "practice_papers", "extracurricular", "parental_support"
        ]
        missing = [column for column in required if column not in data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            predictions = model.predict(data[required])
            result = data.copy()
            result["predicted_exam_score"] = predictions.clip(0, 100).round(1)
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Download predictions as CSV",
                result.to_csv(index=False).encode("utf-8"),
                file_name="student_performance_predictions.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Could not process the uploaded CSV: {exc}")
