# Diabetes Prediction Streamlit App

This repository contains a Streamlit web application that predicts whether a person has diabetes based on health-related input features. The prediction is made using a trained Random Forest machine learning model on the PIMA diabetes dataset.

---

## Project Overview

- **Model:** Random Forest Classifier
- **Dataset:** PIMA Indians Diabetes Dataset
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** Outcome (0 = Non-Diabetic, 1 = Diabetic)
- **App:** Interactive Streamlit app to input features and get diabetes prediction along with probability visualization

---

## How to Run the App Locally

### Prerequisites

- Python 3.7 or higher
- 'pip' package manager

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/ManavVyas149/diabetes-prediction-app.git
cd diabetes-prediction-app
```
2. **Create and activate a virtual environment (recommended)**

- On Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

- On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run app.py
```
5. **Access the app**

Open your browser to http://localhost:8501 if it doesn’t open automatically.

## How to Deploy the App
You can deploy this app easily on Streamlit Community Cloud:

1. Push your code to a public GitHub repository (this repo).

2. Go to Streamlit Cloud and sign in.

3. Click New app, connect your GitHub repo, select the main branch, and set app.py as the entrypoint.

4. Deploy the app and share the URL!

## Files in this Repository
- app.py: Streamlit app source code.

- train_model.py: Script to train and save the machine learning model.

- model.pkl: Trained Random Forest model saved with pickle.

- sample_data.csv: Dataset used for training.

- requirements.txt: Python dependencies list.

- README.md: This file.

## Notes
- The model was trained using scikit-learn's RandomForestClassifier.

- The app provides prediction probabilities for more informed decisions.

- All inputs must be numerical and correspond to the dataset's feature ranges.

## Contact
For questions or feedback, feel free to reach out!
