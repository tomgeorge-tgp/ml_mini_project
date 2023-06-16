import os
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.validation import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_H5_PATH = './model/model.h5'
TOKENIZER_PATH = './model/tokenizer.bin'

# Load pre-trained model
model = keras.models.load_model(MODEL_H5_PATH)
# Load tokenizer
tokenizer = joblib.load(TOKENIZER_PATH)

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get input text from request
        text = request.json['text']
        #    print(text);
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([text])
        # Pad sequence
        padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    
        prediction = model.predict(padded)[0]
    
        response = {
            'toxic': float(prediction[0]),
            'severe_toxic': float(prediction[1]),
            'obscene': float(prediction[2]),
            'threat': float(prediction[3]),
            'insult': float(prediction[4]),
            'identity_hate': float(prediction[5])
        }    
    except Exception as e:
        return jsonify({"error": e}), 400
    else:
        return jsonify(response), 200 

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
