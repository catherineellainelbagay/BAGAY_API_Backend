import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# Output directory exists
os.makedirs("trained_data", exist_ok=True)

# Load student data
csv_path = 'csv/Student-scores.csv'
df = pd.read_csv(csv_path)

# Labels for dropout risk and support need
df['AtRiskDropout'] = ((df['absence_days'] > 10) & (df['weekly_self_study_hours'] < 5)).astype(int)
df['NeedsSupport'] = (df[['english_score', 'math_score', 'physics_score', 'chemistry_score', 'biology_score']].mean(axis=1) < 70).astype(int)

# Select features
X = df[['english_score', 'physics_score', 'chemistry_score', 'biology_score', 'math_score', 'absence_days', 'weekly_self_study_hours']]

# Define helper function to train and save binary classifier
def train_and_save_model(X, y, filename):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(
            random_state=45
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
