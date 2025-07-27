from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Root endpoint to verify the API is running
@app.route('/')
def home():
    return "✅ Flask ML API is running!"

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Ensure 'data' key is present
        if 'data' not in data:
            return jsonify({'error': "'data' key not found in request"}), 400

        features = np.array(data['data']).reshape(1, -1)
        prediction = model.predict(features)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
