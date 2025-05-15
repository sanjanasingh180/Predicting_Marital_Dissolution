from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load("../models/xgboost.pkl")  # Update path if needed
    expected_features = 54  # Adjust based on your dataset
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return jsonify({"message": "Divorce Prediction API is Running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON with 'application/json' header"}), 415

        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "'features' key is missing from request"}), 400

        input_features = np.array(data["features"])

        # Validate feature length
        if input_features.shape[0] != expected_features:
            return jsonify({"error": f"Feature shape mismatch, expected: {expected_features}, got {input_features.shape[0]}"}), 400

        input_features = input_features.reshape(1, -1)

        # Debugging: Print input features
        print("Received Features:", input_features.tolist())

        # Make prediction
        probability = model.predict_proba(input_features)[0, 1]
        
        # Adjust threshold (Try 0.6 if model is predicting too many divorces)
        threshold = 0.6
        prediction = 1 if probability > threshold else 0

        # Debugging: Print prediction details
        print(f"Prediction: {prediction}, Probability: {probability}")

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
