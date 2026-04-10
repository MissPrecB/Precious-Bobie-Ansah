# Breast Cancer Prediction System

## 1. Project Overview

In this project, you will build a Breast Cancer Prediction System using the Wisconsin Breast Cancer Dataset. The goal is to learn how machine learning is applied in healthcare to classify tumors as benign or malignant.

You will perform:
- Data loading and exploration
- Data cleaning
- Feature scaling
- Building machine learning models
- Evaluating model performance
- Deploying a Streamlit app for predictions

## 2. Dataset Description

Dataset: Breast Cancer Wisconsin (Diagnostic)

Source:
- `sklearn.datasets.load_breast_cancer()`
- or Kaggle version of the dataset

Common features include:
- radius_mean
- texture_mean
- smoothness_mean
- concavity_mean
- symmetry_mean
- fractal_dimension_mean
- plus other mean, standard error, and worst features (30 total)

Target values:
- `0 = Malignant` (harmful)
- `1 = Benign` (not harmful)

## 3. Tools & Technologies You Must Use

Required:
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit
- Joblib or Pickle

Optional:
- SHAP for explainability
- GridSearchCV for hyperparameter tuning

## 4. Project Tasks (Step-by-Step)

### Task 1: Load & Understand the Dataset

1. Load the dataset from scikit-learn or from a CSV file.
2. Display the first 5 rows of data.
3. Print the dataset shape.
4. Check the target distribution and class balance.
5. Show summary statistics for the features.

#### Example code outline:
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target
print(df.head())
print(df.shape)
print(df['target'].value_counts())
print(df.describe())
```

### Task 2: Data Preprocessing

You must:
1. Handle missing values.
2. Check for duplicate rows.
3. Encode the target if text labels are used.
4. Scale features using `StandardScaler`.
5. Split the dataset into train/test sets (80/20).

#### Preprocessing checklist:
- `df.isnull().sum()`
- `df.duplicated().sum()`
- `LabelEncoder` if needed
- `StandardScaler` for feature scaling
- `train_test_split` for splitting

#### Example code outline:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

### Task 3: Build Machine Learning Models

Train at least three models. Suggested models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

For each model:
- Train on the training data
- Predict on the test data
- Evaluate with accuracy, precision, recall, F1 score, and confusion matrix

#### Example evaluation:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
```

### Task 4: Choose the Best Model

1. Compare evaluation metrics for all models.
2. Select the best-performing model.
3. Explain why you chose that model.
4. Save the chosen model using joblib or pickle.

#### Example save command:
```python
import joblib
joblib.dump(best_model, 'breast_cancer_model.pkl')
```

### Task 5: Build the Streamlit App

Create `app.py` with a working UI.

Required UI elements:
- App title: `Breast Cancer Prediction System`
- Input fields for the feature values
- A Predict button
- Display prediction results as:
  - `Malignant — High Risk` in red
  - `Benign — Low Risk` in green

Optional UI enhancements:
- Sidebar with dataset description
- Show model accuracy
- Link to dataset source
- Short "How to Use" guide

#### Streamlit app flow:
1. Load the saved model.
2. Collect user inputs.
3. Create a feature vector.
4. Make a prediction.
5. Display the result with a colored message.

### Example Streamlit structure:
```python
import streamlit as st
import joblib
import numpy as np

st.title('Breast Cancer Prediction System')

model = joblib.load('breast_cancer_model.pkl')

radius_mean = st.number_input('radius_mean', value=14.0)
texture_mean = st.number_input('texture_mean', value=20.0)
# add other inputs...

if st.button('Predict'):
    features = np.array([[radius_mean, texture_mean, ...]])
    prediction = model.predict(features)[0]
    if prediction == 0:
        st.error('Malignant — High Risk')
    else:
        st.success('Benign — Low Risk')
```

## 5. Project Deliverables

Students must submit:
- Machine Learning notebook with:
  - EDA
  - Preprocessing
  - Model training
  - Evaluation scores
  - Confusion matrix
  - Final saved model file (`.pkl`)
- Streamlit app (`app.py`)
- Screenshots of the app interface:
  - Homepage
  - Prediction sample
- A short report summarizing:
  - Dataset used
  - Best model chosen and reason
  - Accuracy score
  - Challenges faced
  - What you learned

## 6. Suggested File Structure

- `breast_cancer_prediction.ipynb`
- `app.py`
- `breast_cancer_model.pkl`
- `requirements.txt`
- `README.md`

## 7. Extra Tips

- Use `stratify=y` in `train_test_split` to preserve target balance.
- Visualize class counts with a bar plot.
- Plot a confusion matrix heatmap.
- Save both the scaler and model if input scaling is needed in the app.
- Keep the Streamlit UI simple and clear.
