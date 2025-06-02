from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app) 

# Load models and features
dropout_model = joblib.load('trained_data/model_dropout.pkl')
support_model = joblib.load('trained_data/model_support.pkl')
features = joblib.load('trained_data/model_features.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received JSON:", data)
        input_df = pd.DataFrame([data])
        input_encoded = input_df.reindex(columns=features, fill_value=0)

        dropout_pred = dropout_model.predict(input_encoded)[0]
        support_pred = support_model.predict(input_encoded)[0]

        if dropout_pred and support_pred:
            status = "🚨 High Risk: Needs urgent academic support"
        elif dropout_pred and not support_pred:
            status = "⚠️ At Risk of Dropout: Monitor closely"
        elif not dropout_pred and support_pred:
            status = "📘 Needs Academic Support"
        else:
            status = "✅ Stable: No immediate risk detected"

        return jsonify({
            "AtRiskOfDropout": bool(dropout_pred),
            "NeedsAcademicSupport": bool(support_pred),
            "Status": status
        })
    except Exception as e:
        print("❌ Error in /predict:", e)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
