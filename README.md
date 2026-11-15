
# 🎓 Student Performance Prediction System

## 1. Project Overview

The Student Performance Prediction System is a machine learning-based application designed to estimate students' examination scores using different academic and personal learning-related factors.

The system uses a Random Forest Regressor algorithm to analyze student information and predict their estimated examination performance.

The application is developed using Python and Streamlit, providing a user-friendly interface for individual and batch student predictions.

**Note:** This project uses a synthetic dataset for educational and demonstration purposes. Predictions should not be used for actual student grading or high-stakes academic decisions.

## 2. Project Objectives

The main objectives of this project are:

1. To develop a machine learning model that predicts student examination scores.
2. To identify learning-related factors that may influence academic performance.
3. To provide an individual student performance prediction feature.
4. To allow batch prediction through CSV file uploads.
5. To evaluate the performance of the trained machine learning model.
6. To present prediction results through a user-friendly Streamlit application.

## 3. System Features

### A. Dashboard Overview
- Displays the total number of student records.
- Presents academic performance statistics.
- Shows sample dataset information.
- Provides an overview of model evaluation results.

### B. Individual Student Prediction
- Accepts student information.
- Predicts the estimated examination score.
- Displays the predicted score out of 100.
- Classifies performance into:
  - Excellent
  - Good
  - Satisfactory
  - Needs Improvement

### C. Batch Student Prediction
- Allows users to upload CSV files.
- Processes multiple student records.
- Generates predicted examination scores.
- Displays prediction results in a table.
- Allows users to download prediction results.

### D. Model Evaluation
- Displays Mean Absolute Error (MAE).
- Displays R² Score.
- Presents model evaluation results.

## 4. Input Variables

The system uses the following input variables:

| Variable | Description |
|---|---|
| hours_studied | Hours studied per day |
| attendance | Student attendance percentage |
| previous_scores | Previous examination scores |
| sleep_hours | Hours of sleep per night |
| practice_papers | Number of practice papers completed |
| extracurricular | Participation in extracurricular activities |
| parental_support | Level of parental support |

**Target Variable:** `exam_score`

## 5. Machine Learning Algorithm

The project uses the **Random Forest Regressor** algorithm.

Random Forest is a machine learning algorithm that combines multiple decision trees to generate predictions.

It is used in this project to estimate students' examination scores based on the provided input variables.

### Model Evaluation Metrics

**Mean Absolute Error (MAE)**

Measures the average difference between actual and predicted examination scores. Lower values indicate smaller prediction errors.

**R² Score**

Measures how well the model explains variations in examination scores. A value closer to 1 generally indicates a better fit.

## 6. Dataset Description

This project generates a synthetic dataset containing 1,200 sample student records.

The dataset includes:
- Study hours
- Attendance
- Previous scores
- Sleep hours
- Practice papers
- Extracurricular activities
- Parental support
- Examination scores

The dataset is generated automatically using Python and NumPy.

It does not contain actual student records.

## 7. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Random Forest | Prediction algorithm |
| Joblib | Model saving and loading |
| Streamlit | Web application interface |
| GitHub | Source code repository |
| Streamlit Cloud | Application deployment |

## 8. Project Folder Structure

```text
Student-Performance-Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── student_performance_sample.csv
│
└── models/
    └── student_performance_model.joblib
```

Generated files are created after successfully running the training script.

## 9. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

```bash
cd Student-Performance-Prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Machine Learning Model

```bash
python train_model.py
```

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

## 10. CSV Upload Format

For batch prediction, upload a CSV file containing the following columns:

```csv
hours_studied,attendance,previous_scores,sleep_hours,practice_papers,extracurricular,parental_support
4,85,70,7,3,Yes,Medium
6,92,80,8,5,No,High
2,65,55,6,1,Yes,Low
```

The `exam_score` column is not required when uploading records for prediction.

## 11. System Limitations

1. The model uses synthetic demonstration data rather than actual student records.
2. Prediction results may not accurately represent real student performance.
3. Model evaluation results only reflect the generated sample dataset.
4. The system cannot consider every factor affecting academic performance.
5. Predictions should not replace teachers' professional judgment.

## 12. Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment steps:

1. Upload the project files to GitHub.
2. Open Streamlit Community Cloud.
3. Connect the GitHub repository.
4. Select `app.py` as the main application file.
5. Deploy the application.

## 13. Conclusion

The Student Performance Prediction System demonstrates how machine learning can be applied to estimate student examination performance.

Through the use of Random Forest Regression and Streamlit, the application provides individual predictions, batch processing, and model evaluation.

This project serves as an educational demonstration of machine learning and data-driven academic analysis.

---

**Project Title:** Student Performance Prediction System

**Algorithm:** Random Forest Regressor

**Framework:** Streamlit

**Language:** Python

**Project Type:** Machine Learning / Educational Application
