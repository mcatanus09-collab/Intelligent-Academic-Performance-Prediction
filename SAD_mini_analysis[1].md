# SAD Mini-Analysis: Student Performance Prediction System

## 1. Problem
Teachers and students may find it difficult to summarize several learning-related factors when discussing possible academic performance. This project demonstrates how a machine-learning model can estimate an exam score from selected input variables.

## 2. Users
- **Student:** enters study and learning information and views an estimated score.
- **Teacher or instructor:** may use the demonstration to understand how input factors can be analyzed.
- **Developer or system administrator:** installs dependencies, trains the model, and maintains the application.

## 3. Inputs
The application accepts:
- Hours studied per day
- Attendance percentage
- Previous scores
- Sleep hours per night
- Number of practice papers completed
- Extracurricular participation (Yes/No)
- Parental support level (Low/Medium/High)

For batch prediction, the same fields are accepted in a CSV file.

## 4. Process
1. The training script generates a reproducible synthetic sample dataset.
2. It separates the input features from the target exam score.
3. It splits the data into training and test sets.
4. It preprocesses numeric and categorical values.
5. It trains a Random Forest regression model.
6. It evaluates the model using Mean Absolute Error and R².
7. It saves the trained pipeline as a Joblib file.
8. The Streamlit app loads the model and predicts an estimated score from user inputs.

## 5. Outputs
- Estimated exam score from 0 to 100
- A simple performance description
- A downloadable CSV containing predictions for uploaded records
- Training evaluation metrics printed in the terminal

## 6. Limitations
- The sample dataset is synthetic and does not represent actual students.
- Model metrics from synthetic data do not establish real-world predictive accuracy.
- Predictions may be inaccurate and can reflect assumptions built into the generated data.
- Important factors such as teaching quality, learning differences, access to resources, and personal circumstances are not fully represented.
- The application is a prototype and should not be used for grading or other high-impact decisions.

## 7. Conclusion
The project demonstrates a basic machine-learning workflow, from data preparation and model training to prediction through a web interface. A real deployment would require authorized real data, privacy safeguards, fairness testing, expert review, and ongoing validation.
