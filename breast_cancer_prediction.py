import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    # Load dataset
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target

    print('--- Dataset Overview ---')
    print('First 5 rows:')
    print(df.head())
    print('\nShape:', df.shape)
    print('\nTarget distribution:')
    print(df['target'].value_counts())
    print('\nSummary statistics:')
    print(df.describe().T)

    # Data quality checks
    print('\nMissing values:')
    print(df.isnull().sum())
    print('\nDuplicate rows:')
    print(df.duplicated().sum())

    # Preprocessing
    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results.append(
            {
                'model': name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'trained_model': model,
            }
        )

        print(f'--- {name} ---')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 score: {f1:.4f}')
        print('Confusion matrix:')
        print(cm)
        print()

    best = max(results, key=lambda x: x['f1_score'])
    print('--- Best Model ---')
    print(f"Selected model: {best['model']}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"F1 Score: {best['f1_score']:.4f}")

    joblib.dump(best['trained_model'], 'breast_cancer_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print('\nSaved model to breast_cancer_model.pkl')
    print('Saved scaler to scaler.pkl')


if __name__ == '__main__':
    main()
