Student Performance Prediction System
A Streamlit classroom prototype that uses the UCI Student Performance dataset to classify whether a student record is AT RISK or NOT AT RISK.
Prediction rule
- G3 < 10 → AT RISK
- G3 >= 10 → NOT AT RISK
Important: G3 is the target and is never included in the input features.
Dataset
UCI Machine Learning Repository — Student Performance, dataset ID 320.

The laboratory handout describes using the Mathematics records. The provided ucimlrepo retrieval code combines the repository's returned features and targets. Inspect the downloaded rows and columns and confirm the Mathematics-only selection expected by your instructor before submitting.
Features
- Dashboard overview and historical dataset summary
- Individual student risk prediction
- Batch CSV upload and downloadable predictions
- Accuracy, precision, recall, F1-score, classification report, and confusion matrix
- Dataset preview and sample CSV download
- Responsible-use and limitation notes
Setup
Python 3.10 or later is recommended.
pip install -r requirements.txt
python train_model.py
streamlit run app.py
Training downloads the UCI dataset and creates:
- student_data.csv
- student_model.pkl
- model_metrics.json
For Streamlit Community Cloud, commit these generated files to the repository so the deployed app can load them without needing to retrain on every start.
GitHub files
.
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── student_data.csv
├── student_model.pkl
└── model_metrics.json
Limitations and responsible use
This is a classroom demonstration using historical public data. The source population is Portuguese secondary-school students, so model performance may not generalize to another school or population. Predictions are not diagnoses and must not be the sole basis for decisions about students. An at-risk prediction should lead to supportive human review.
