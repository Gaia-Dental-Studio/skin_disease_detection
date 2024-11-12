# skin_cancer_app.py
import streamlit as st
import requests
from PIL import Image

# Streamlit app title and description
st.title("Skin Cancer Classification")
st.write("Upload an image of a skin lesion to get a prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display the image and make predictions if an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Send the image to the Flask API
    if st.button("Predict"):
        # Prepare image for API request
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        files = {'file': open(uploaded_file.name, 'rb')}
        
        # Send request to the Flask API
        response = requests.post("http://127.0.0.1:5000/predict", files=files)
        data = response.json()

        if response.status_code == 200:
            st.write(f"### Predicted Class: {data['predicted_class']}")
            st.write(f"Confidence Score: {data['confidence_score']:.4f}")
            
            # Display confidence scores for each class
            st.write("### Confidence Scores for All Classes:")
            for label, score in data['confidence_scores'].items():
                st.write(f"{label}: {score:.4f}")
        else:
            st.write("Error:", data["error"])
