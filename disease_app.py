import streamlit as st
import requests
from PIL import Image
import os
import tempfile

# Define the backend API URL
API_URL = "http://127.0.0.1:5000/predict"

st.title("Skin Disease Detection")
st.write("Upload an image of a skin condition to detect the possible disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=500)

    # Send the image to the backend API
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            try:
                # Save the image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                    temp.write(uploaded_file.getbuffer())
                    temp_path = temp.name

                # Send a POST request to the API
                with open(temp_path, "rb") as f:
                    response = requests.post(API_URL, files={"image": f})

                # Clean up the temp file
                os.remove(temp_path)

                # Process the response
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success(f"✅ Prediction: **{result['predicted_class']['en']}** ({result['predicted_class']['id']})")
                    st.write(f"🔍 Confidence: **{result['confidence']}**")
                    
                    st.subheader("📝 Description")
                    st.markdown(f"- **EN**: {result['description']['en']}")
                    st.markdown(f"- **ID**: {result['description']['id']}")

                    st.subheader("📚 Condition Explanation")
                    st.markdown(f"- **EN**: {result['condition_explanation']['en']}")
                    st.markdown(f"- **ID**: {result['condition_explanation']['id']}")
                else:
                    st.error(f"❌ API Error: {response.json().get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"❌ Internal Error: {str(e)}")
