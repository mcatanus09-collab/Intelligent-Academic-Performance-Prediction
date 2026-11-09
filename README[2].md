# Complete Student Performance Prediction System

A beginner-friendly Python project that trains a machine-learning regression model and provides a Streamlit web app for estimating student exam scores.

## Project files

- `train_model.py` — creates a demonstration dataset, trains a Random Forest model, evaluates it, and saves the model.
- `app.py` — Streamlit application for individual and CSV-based predictions.
- `requirements.txt` — Python dependencies.
- `SAD_mini_analysis.md` — mini System Analysis and Design analysis.
- `.gitignore` — excludes generated files and local environments.
- `models/` — created automatically when the model is trained.
- `student_performance_sample.csv` — created automatically by the training script.

## Requirements

- Python 3.9 or newer
- Internet connection for installing packages

## Run locally

Open a terminal in this project folder and run:

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Start the web app:

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

## CSV prediction format

For batch predictions, upload a CSV with these columns:

```csv
hours_studied,attendance,previous_scores,sleep_hours,practice_papers,extracurricular,parental_support
4,85,70,7,3,Yes,Medium
6,92,80,8,6,No,High
```

The app adds a `predicted_exam_score` column and lets you download the results.

## Google Colab

Upload/extract the project ZIP in Colab, change to the project folder, then run:

```python
%pip install -r requirements.txt
!python train_model.py
```

To launch Streamlit in Colab, you need a tunnel or a notebook-specific Streamlit integration. The simplest option is to run the training script in Colab and run the Streamlit app locally.

## Important limitation

The training script generates synthetic demonstration data. Its metrics and predictions are **not evidence of real-world accuracy** and must not be used for actual student grading, admissions, discipline, or other high-impact decisions. For a meaningful model, use an authorized, representative, privacy-protected dataset; document consent and data handling; validate performance on unseen real data; and check for bias.

## GitHub upload

1. Create a new repository at https://github.com/new.
2. Name it `Complete-Student-Performance-Prediction-System`.
3. Extract this ZIP file.
4. Upload the project files and folders to the new repository, or use Git:

```bash
git init
git add .
git commit -m "Initial student performance prediction system"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Complete-Student-Performance-Prediction-System.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.
