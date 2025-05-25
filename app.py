# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load models and feature list
dropout_model = joblib.load('trained_data/model_dropout.pkl')
support_model = joblib.load('trained_data/model_support.pkl')
features = joblib.load('trained_data/model_features.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    input_encoded = input_df.reindex(columns=features, fill_value=0)

    # Make predictions
    dropout_pred = dropout_model.predict(input_encoded)[0]
    support_pred = support_model.predict(input_encoded)[0]

    # Optional logic override for debugging
    if dropout_pred and support_pred:
        status = "🚨 High Risk: Needs urgent academic support"
    elif dropout_pred and not support_pred:
        status = "⚠️ At Risk of Dropout: Monitor closely"
    elif not dropout_pred and support_pred:
        status = "📘 Needs Academic Support"
    else:
        status = "✅ Stable: No immediate risk detected"

    response = {
        "AtRiskOfDropout": bool(dropout_pred),
        "NeedsAcademicSupport": bool(support_pred),
        "Status": status
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
    