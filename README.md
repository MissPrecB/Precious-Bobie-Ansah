# Breast Cancer Prediction System

## Files

- `breast_cancer_prediction.py`: trains machine learning models and saves the best model plus scaler.
- `app.py`: Streamlit application for interactive prediction.
- `requirements.txt`: required Python packages.
- `Breast_Cancer_Prediction_Guide.md`: step-by-step student project guide.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Train the model

Run:
```bash
python breast_cancer_prediction.py
```

This will create:
- `breast_cancer_model.pkl`
- `scaler.pkl`

## Run the Streamlit app

Run:
```bash
streamlit run app.py
```

Then use the web UI to input tumor feature values and predict whether the tumor is malignant or benign.
