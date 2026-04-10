import joblib
import numpy as np
import streamlit as st

MODEL_PATH = 'breast_cancer_model.pkl'
SCALER_PATH = 'scaler.pkl'

FEATURE_NAMES = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
    'mean compactness',
    'mean concavity',
    'mean concave points',
    'mean symmetry',
    'mean fractal dimension',
    'radius error',
    'texture error',
    'perimeter error',
    'area error',
    'smoothness error',
    'compactness error',
    'concavity error',
    'concave points error',
    'symmetry error',
    'fractal dimension error',
    'worst radius',
    'worst texture',
    'worst perimeter',
    'worst area',
    'worst smoothness',
    'worst compactness',
    'worst concavity',
    'worst concave points',
    'worst symmetry',
    'worst fractal dimension',
]

DEFAULT_VALUES = [14.0, 20.0, 90.0, 600.0, 0.1, 0.1, 0.1, 0.05, 0.18, 0.06,
                  0.5, 1.0, 6.0, 40.0, 0.005, 0.02, 0.03, 0.01, 0.02, 0.003,
                  25.0, 25.0, 100.0, 900.0, 0.15, 0.3, 0.4, 0.1, 0.2, 0.07,
                  0.3, 25.0]


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def build_inputs():
    st.sidebar.header('How to Use')
    st.sidebar.write(
        'Enter numeric values for the tumor features, then click the Predict button.'
    )
    st.sidebar.markdown('**Target:** 0 = Malignant, 1 = Benign')
    st.sidebar.markdown(
        '[Dataset source](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)'
    )

    st.header('Breast Cancer Tumor Feature Inputs')

    values = []
    for name, default in zip(FEATURE_NAMES, DEFAULT_VALUES):
        feature_key = name.replace(' ', '_')
        value = st.number_input(name.title(), value=float(default), format='%.4f')
        values.append(value)

    return np.array(values).reshape(1, -1)


def main():
    st.title('Breast Cancer Prediction System')
    st.write('Predict whether a tumor is Benign or Malignant using a trained model.')

    if st.button('Load Model and Show Summary'):
        model, scaler = load_model()
        st.write('Model loaded successfully.')
        st.write('Model type:', type(model).__name__)

    user_input = build_inputs()

    if st.button('Predict'):
        try:
            model, scaler = load_model()
            scaled_input = scaler.transform(user_input)
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0]
            score = max(probability)

            if prediction == 0:
                st.error(f'Malignant — High Risk (Confidence: {score:.2f})')
            else:
                st.success(f'Benign — Low Risk (Confidence: {score:.2f})')

            st.write('Prediction value:', int(prediction))
            st.write('Malignant = 0, Benign = 1')
        except FileNotFoundError:
            st.error('Model or scaler file not found. Run training first to generate `breast_cancer_model.pkl` and `scaler.pkl`.')


if __name__ == '__main__':
    main()
