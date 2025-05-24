import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# Ensure save directory exists
os.makedirs("trained_data", exist_ok=True)

# Load your student dataset
df = pd.read_csv('csv/Student_Performance.csv')  # Change to your actual file name

# Create 'Result' column: Pass/Fail
df['Result'] = df['Grade'].apply(lambda x: 'Pass' if x in ['A', 'B', 'C'] else 'Fail')

# Select only needed columns
X = df[['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Participation_Score', 'Attendance (%)']]
y = df['Result']

# Encode output labels (Pass = 1, Fail = 0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the encoder
joblib.dump(label_encoder, 'trained_data/label_encoder.pkl')

# Save feature names
joblib.dump(X.columns.tolist(), 'trained_data/model_features.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(32,),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        random_state=42
    ))
])

# Train and save model
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'trained_data/model_cls.pkl')

print("✅ Model training complete. Saved to 'trained_data/'")
