import streamlit as st
import requests
from PIL import Image

# Define the backend API URL
API_URL = "http://127.0.0.1:5000/predict"

st.title("Skin Disease Detection")
st.write("Upload an image to detect the skin disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send the image to the backend API
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            try:
                # Save the image to a temporary file
                temp_path = "temp_uploaded.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Send a POST request to the API
                with open(temp_path, "rb") as f:
                    response = requests.post(API_URL, files={"image": f})

                # Process the response
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Predicted Class: {result['predicted_class']}")
                    st.write(f"Confidence: {result['confidence']:.2f}")
                    st.write("Confidence Scores for All Classes:")
                    st.json(result['class_confidences'])
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
