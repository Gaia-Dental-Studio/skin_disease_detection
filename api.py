# skin_cancer_api.py
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('model/skin-cancer_VGG-model.h5')

# Class labels
CLASS_LABELS = [
    'actinic keratosis', 
    'basal cell carcinoma', 
    'dermatofibroma', 
    'melanoma', 
    'nevus', 
    'pigmented benign keratosis', 
    'seborrheic keratosis', 
    'squamous cell carcinoma', 
    'vascular lesion'
]

# Function to preprocess the image
def preprocess_image(img):
    # Resize image to match model input
    img = img.resize((48, 48))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1] range
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Open and preprocess the image
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img_array = preprocess_image(img)

    # Run prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = CLASS_LABELS[predicted_index]
    predicted_score = float(predictions[0][predicted_index])

    # Prepare confidence scores for each class
    confidence_scores = {CLASS_LABELS[i]: float(predictions[0][i]) for i in range(len(CLASS_LABELS))}

    # Return prediction and confidence scores
    return jsonify({
        "predicted_class": predicted_label,
        "confidence_score": predicted_score,
        "confidence_scores": confidence_scores
    })

if __name__ == '__main__':
    app.run(debug=True)
