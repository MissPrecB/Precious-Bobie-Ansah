import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_breast_cancer

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

RADAR_FEATURES = [
    'mean radius',
    'mean texture',
    'mean smoothness',
    'mean concavity',
    'mean symmetry',
    'mean fractal dimension',
]


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def set_page_style():
    st.set_page_config(
        page_title='Breast Cancer Prediction System',
        page_icon='🩺',
        layout='wide',
    )

    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(180deg, #f4f7fc 0%, #ffffff 100%);
        }
        .stApp {
            color: #091C4B;
        }
        .css-18e3th9 {
            padding-top: 1rem;
        }
        .stButton>button {
            background-color: #0b4d82;
            color: white;
        }
        .st-bf {
            background-color: #e8f2ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_sidebar():
    st.sidebar.title('Breast Cancer Prediction')
    st.sidebar.markdown(
        'Use the inputs to provide tumor feature values and click **Predict** to see the result.'
    )
    st.sidebar.markdown('---')
    st.sidebar.subheader('Dataset Info')
    st.sidebar.write('Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.')
    st.sidebar.write('- 30 numerical input features')
    st.sidebar.write('- Target: 0 = Malignant, 1 = Benign')
    st.sidebar.write('- Useful for binary classification tasks')
    st.sidebar.markdown('---')
    st.sidebar.subheader('Quick Guide')
    st.sidebar.write(
        '1. Enter values for selected tumor features.\n'
        '2. Click Predict to evaluate the model.\n'
        '3. Review the prediction and radar chart.'
    )
    st.sidebar.markdown('---')
    st.sidebar.markdown(
        '[Dataset source](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)'
    )


def build_inputs():
    values = []
    for name, default in zip(FEATURE_NAMES, DEFAULT_VALUES):
        value = st.number_input(name.title(), value=float(default), format='%.4f')
        values.append(value)

    return np.array(values).reshape(1, -1)


def radar_chart(values):
    categories = RADAR_FEATURES + [RADAR_FEATURES[0]]
    data = [values[FEATURE_NAMES.index(f)] for f in RADAR_FEATURES]
    data = data + [data[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=data,
                theta=categories,
                fill='toself',
                name='Tumor profile',
                marker=dict(color='#0b4d82'),
            )
        ]
    )
    fig.update_layout(
        polar=dict(
            bgcolor='#f7fbff',
            radialaxis=dict(visible=True, range=[0, max(data) * 1.2]),
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def build_insights(user_input):
    cancer = load_dataset_info()
    mean_vals = np.mean(cancer.data, axis=0)
    values = user_input.flatten()
    insights = []

    for feature in RADAR_FEATURES:
        idx = FEATURE_NAMES.index(feature)
        current = values[idx]
        average = mean_vals[idx]
        relation = 'higher' if current > average else 'lower'
        insights.append(
            f'`{feature}` is {relation} than the dataset average ({current:.2f} vs {average:.2f}).'
        )

    return insights


def feature_comparison_chart(values):
    cancer = load_dataset_info()
    mean_vals = np.mean(cancer.data, axis=0)
    selected_values = [values[FEATURE_NAMES.index(f)] for f in RADAR_FEATURES]
    average_values = [mean_vals[FEATURE_NAMES.index(f)] for f in RADAR_FEATURES]

    fig = go.Figure(
        data=[
            go.Bar(
                name='Selected',
                x=RADAR_FEATURES,
                y=selected_values,
                marker_color='#0b4d82',
            ),
            go.Bar(
                name='Dataset average',
                x=RADAR_FEATURES,
                y=average_values,
                marker_color='#7DB3E6',
            ),
        ]
    )
    fig.update_layout(
        barmode='group',
        title='Selected values vs dataset average',
        xaxis_title='Feature',
        yaxis_title='Value',
        margin=dict(l=20, r=20, t=40, b=80),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def trend_chart(values):
    cancer = load_dataset_info()
    mean_vals = np.mean(cancer.data, axis=0)
    selected_values = [values[FEATURE_NAMES.index(f)] for f in RADAR_FEATURES]
    average_values = [mean_vals[FEATURE_NAMES.index(f)] for f in RADAR_FEATURES]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=RADAR_FEATURES,
                y=selected_values,
                mode='lines+markers',
                name='Selected trend',
                line=dict(color='#0b4d82', width=3),
            ),
            go.Scatter(
                x=RADAR_FEATURES,
                y=average_values,
                mode='lines+markers',
                name='Dataset average trend',
                line=dict(color='#7db3e6', dash='dash', width=2),
            ),
        ]
    )
    fig.update_layout(
        title='Feature trend across selected tumor values',
        xaxis_title='Features',
        yaxis_title='Value',
        margin=dict(l=20, r=20, t=40, b=80),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def probability_gauge(score, prediction):
    label = 'Benign' if prediction == 1 else 'Malignant'
    color = '#1f9d55' if prediction == 1 else '#c0392b'
    fig = go.Figure(
        go.Indicator(
            mode='gauge+number+delta',
            value=score * 100,
            delta={'reference': 50, 'increasing': {'color': '#0b4d82'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#091C4B'},
                'bar': {'color': color},
                'bgcolor': '#f4f7fc',
                'steps': [
                    {'range': [0, 50], 'color': '#d7e9fb'},
                    {'range': [50, 100], 'color': '#a6c8ff'},
                ],
            },
            title={'text': f'Prediction confidence ({label})', 'font': {'size': 16}},
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
    return fig


def feature_distribution_chart(values):
    cancer = load_dataset_info()
    features = ['mean radius', 'mean texture', 'mean area']
    fig = go.Figure()

    for feature in features:
        idx = FEATURE_NAMES.index(feature)
        fig.add_trace(
            go.Histogram(
                x=cancer.data[:, idx],
                name=feature,
                opacity=0.6,
                nbinsx=30,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[values[idx], values[idx]],
                y=[0, 25],
                mode='lines',
                name=f'{feature} input',
                line=dict(width=3),
            )
        )

    fig.update_layout(
        barmode='overlay',
        title='Selected values against dataset distributions',
        xaxis_title='Value',
        yaxis_title='Count',
        margin=dict(l=20, r=20, t=40, b=40),
    )
    return fig


def load_dataset_info():
    cancer = load_breast_cancer()
    return cancer


def display_prediction(model, scaler, user_input):
    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]
    score = max(probability)

    if prediction == 0:
        result_text = f'Malignant — High Risk (Confidence: {score:.2f})'
        st.error(result_text)
    else:
        result_text = f'Benign — Low Risk (Confidence: {score:.2f})'
        st.success(result_text)

    st.markdown('**Prediction value:** ' + str(int(prediction)))
    st.markdown('**Legend:** 0 = Malignant, 1 = Benign')

    return prediction, score, probability


def main():
    set_page_style()
    build_sidebar()

    st.title('Breast Cancer Prediction System')
    st.markdown(
        'A simple interface to predict whether a breast tumor is benign or malignant using a trained model.'
    )

    with st.container():
        left_col, right_col = st.columns([2, 3])

        with left_col:
            st.subheader('Input tumor feature values')
            user_input = build_inputs()
            if st.button('Predict'):
                try:
                    model, scaler = load_model()
                    prediction, score, probability = display_prediction(model, scaler, user_input)

                    chart = radar_chart(user_input.flatten())
                    comparison = feature_comparison_chart(user_input.flatten())
                    trend = trend_chart(user_input.flatten())
                    gauge = probability_gauge(score, prediction)
                    distribution = feature_distribution_chart(user_input.flatten())

                    tabs = right_col.tabs(['Radar', 'Comparison', 'Trend', 'Confidence', 'Distribution'])
                    tabs[0].plotly_chart(chart, use_container_width=True)
                    tabs[1].plotly_chart(comparison, use_container_width=True)
                    tabs[2].plotly_chart(trend, use_container_width=True)
                    tabs[3].plotly_chart(gauge, use_container_width=True)
                    tabs[4].plotly_chart(distribution, use_container_width=True)

                    insights = build_insights(user_input)
                    with right_col.expander('Model Insight'):
                        st.markdown(
                            '### Feature comparison with dataset average'
                        )
                        for insight in insights:
                            st.write(insight)
                        st.markdown(
                            '**Insight:** values that are higher than average for `mean radius`, `mean concavity`, and `mean symmetry` often increase model risk toward malignant classification.'
                        )
                except FileNotFoundError:
                    st.error(
                        'Model files not found. Run the training script to create `breast_cancer_model.pkl` and `scaler.pkl`.'
                    )

        with right_col:
            st.subheader('Prediction Summary')
            st.write('Enter tumor feature values on the left and click Predict to visualize the risk profile.')
            try:
                model, scaler = load_model()
                cancer = load_dataset_info()
                X = scaler.transform(cancer.data)
                accuracy = model.score(X, cancer.target)
                st.metric('Model score (approx.)', f'{accuracy:.2%}')
            except FileNotFoundError:
                st.info('Model not loaded yet. Run training first to enable prediction.')

    st.markdown('---')
    st.markdown(
        '**Note:** This app is for educational purposes and should not replace medical diagnosis.'
    )


if __name__ == '__main__':
    main()
