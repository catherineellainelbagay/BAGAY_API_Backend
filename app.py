from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and metadata
model = joblib.load('trained_data/model_cls.pkl')
features = joblib.load('trained_data/model_features.pkl')
label_encoder = joblib.load('trained_data/label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Prepare input
    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Predict result
    pred = model.predict(input_df)[0]
    result = label_encoder.inverse_transform([pred])[0]

    return jsonify({"Result": result})

if __name__ == '__main__':
    app.run(debug=True)
