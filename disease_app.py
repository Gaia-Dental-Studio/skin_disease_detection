import streamlit as st
import requests
from PIL import Image
import os
import plotly.express as px
import pandas as pd
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

                    # Expandable: Show all class confidences
                    with st.expander("📊 Show All Class Confidences"):
                        st.write("Confidence scores for all predicted classes:")
                        # Sort again just to ensure descending order
                        # sorted_conf = sorted(result['all_confidence'].items(), key=lambda item: item[1], reverse=True)

                        # for label, score in sorted_conf:
                        #     st.write(f"- **{label}**: {score}%")
                        
                        # Convert and sort the confidence dict
                        sorted_conf = sorted(result['all_confidence'].items(), key=lambda item: item[1], reverse=True)
                        labels, scores = zip(*sorted_conf)
                        conf_df = pd.DataFrame({'Class': labels, 'Confidence (%)': scores})

                        # Plot horizontal bar chart
                        # Create bar chart with value labels
                        fig = px.bar(
                            conf_df,
                            x="Confidence (%)",
                            y="Class",
                            orientation="h",
                            title="Class Confidence Scores",
                            text="Confidence (%)"  # Show values
                        )

                        fig.update_traces(
                            marker_color='steelblue',     # Single solid color
                            textposition='outside'        # Put values at the end of the bars
                        )

                        fig.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            plot_bgcolor='white',
                            xaxis_range=[0, 100],  # Optional: always show full scale
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    
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
