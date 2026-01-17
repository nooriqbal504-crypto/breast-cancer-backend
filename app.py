from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# -------------------------------
# APP SETUP
# -------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------
# LOAD FILES
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def home():
    return "Breast Cancer Survival Prediction API is running"

# -------------------------------
# PREDICTION ROUTE
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Check missing fields
        missing = [f for f in features if f not in data]
        if missing:
            return jsonify({
                "error": "Missing fields in request",
                "missing_fields": missing
            }), 400

        # Arrange input in correct order
        input_data = [float(data[f]) for f in features]
        input_array = np.array(input_data).reshape(1, -1)

        # Scale
        input_scaled = scaler.transform(input_array)

        # Predict
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        result = "Alive" if int(pred) == 1 else "Deceased"

        return jsonify({
            "prediction": result,
            "survival_probability_percent": round(prob * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
