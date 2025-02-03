from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model_path = './model/skin_disease_model.h5'  # Update path to your model file
model = load_model(model_path)

# Define class labels
class_labels = ['BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus',
                'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the input image for prediction."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read the image at path: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict skin disease from an uploaded image."""
    try:
        # Get the file from the request
        file = request.files['image']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Save the uploaded image to a temporary file
        temp_path = 'temp.jpg'
        file.save(temp_path)

        # Preprocess and predict
        preprocessed_img = preprocess_image(temp_path)
        predictions = model.predict(preprocessed_img)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_idx]
        
        # Use the following for decimal result format  
        # confidence = round(predictions[predicted_class_idx], 2)
        # class_confidences = {class_labels[i]: round(float(predictions[i]), 2) for i in range(len(class_labels))}

        # Use the following for percentage result format
        confidence = round(predictions[predicted_class_idx] * 100, 2)
        
        # Create and sort class confidence scores in descending order
        class_confidences = {class_labels[i]: round(float(predictions[i]) * 100, 2) for i in range(len(class_labels))}
        sorted_confidences = dict(sorted(class_confidences.items(), key=lambda item: item[1], reverse=True))

        # Filter scores above threshold (50%)
        filtered_confidences = {k: f"{v}%" for k, v in sorted_confidences.items() if v >= 50}

        # Clean up temporary file
        os.remove(temp_path)

        # Return JSON response for decimal format
        # return jsonify({
        #     'predicted_class': predicted_class,
        #     'confidence': float(confidence),
        #     'class_confidences': class_confidences
        # })

        # Return JSON response for percentage format
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': f"{str(confidence)}%",  # Format as a percentage string
            'class_confidences': filtered_confidences
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
