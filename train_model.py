# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# Ensure output directory exists
os.makedirs("trained_data", exist_ok=True)

# Load student data
csv_path = 'csv/Student-scores.csv'
df = pd.read_csv(csv_path)

# Create labels for dropout risk and support need
df['At Risk Dropout'] = ((df['absence_days'] > 10) & (df['weekly_self_study_hours'] < 5)).astype(int)
df['Needs Support'] = ((df[['english_score', 'math_score', 'physics_score', 'chemistry_score', 'biology_score']].mean(axis=1) < 70) | (df['absence_days'] > 7)).astype(int)


# Select features
X = df[['english_score', 'physics_score', 'chemistry_score', 'biology_score', 'math_score', 'absence_days', 'weekly_self_study_hours']]

# Define helper function to train and save binary classifier
def train_and_save_model(X, y, filename):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(32,),
            activation='logistic',  # sigmoid
            max_iter=2000,
            early_stopping=True,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f'trained_data/{filename}')

# Train models
train_and_save_model(X, df['AtRiskDropout'], 'model_dropout.pkl')
train_and_save_model(X, df['NeedsSupport'], 'model_support.pkl')

# Save feature list
joblib.dump(X.columns.tolist(), 'trained_data/model_features.pkl')

print("✅ Models trained and saved: model_dropout.pkl, model_support.pkl")
