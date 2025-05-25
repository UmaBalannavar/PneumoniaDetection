from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (important for React frontend)

# Load the trained ML model
model = joblib.load("diabetes_model.pkl")

@app.route('/')
def home():
    return "ML Model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receive JSON from frontend
    print("Received data:", data)

    # Extract input features in the correct order
    try:
        features = [
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]  # 0 or 1
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=7000)
