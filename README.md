# Skin Disease Detection
This repository consists of the helper code for two models of skin-related diseases differentiated into Skin Disease and Skin Cancer.

## Skin Disease
Skin Disease model aims to detect skin infections from a human skin picture. The image picture can be taken using a smartphone camera or other camera where the entire skin surface of the infection can be clearly visible. 

The model will classify the input image based on the pathogen category:
- **Bacterial Infections (BA)**: cellulitis, impetigo.
- **Fungal Infections (FU)**: athlete-foot, nail-fungus, ringworm.
- **Parasitic Infections (PA)**: cutaneous-larva-migrans.
- **Viral skin infections (VI)**: chickenpox, shingles. 

The model was trained by fine-tuning EfficienNetB0 model with [Skin Disease Dataset](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset) from Kaggle. The final model accuracy is 96%, where each class f1-score ranges from 0.88 - 1.0. The model is available on this [Link](https://drive.google.com/file/d/1--kRS_2wGzAT_lKJFwtvk9bbZqOSPv3j/view?usp=drive_link)

API endpoint for this model is disease_api.py, built using flask for the backend, which receives the image, processes it, and runs the model until it gets the output, and the result is sent as the JSON format output to the frontend (disease_app.py), built using Streamlit.

## Skin Cancer (under development)
This model aims to predict skin cancer types based on dermatoscopy images. Unlike the skin disease model, where the image taken could be from the phone camera, this model only works for pictures taken from dermatoscopy cameras. 





