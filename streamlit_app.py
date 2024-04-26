import streamlit as st
from streamlit_image_select import image_select

import torch
import numpy as np

from PIL import Image

import os, random
from alzheimersdetection import AlzheimerModel

def predict(data):

    
    model = AlzheimerModel.load(fr'models\best_ad_model.pt')
    result = AlzheimerModel.predict(model, data)

    return result

def label_mapping(result):
    labels = ["Not Demented",
              "Very Mild Demented",
              "Mild Demented",
              "Moderate Demented"]
    
    return labels[result]

img1 = np.array(Image.open(random.choice(os.listdir(fr'assets\Kaggle\alzheimer_mri_preprocessed_dataset\raw\Non_Demented'))))
img2 = np.array(Image.open(random.choice(os.listdir(fr'assets\Kaggle\alzheimer_mri_preprocessed_dataset\raw\Very_Mild_Demented'))))
img3 = np.array(Image.open(random.choice(os.listdir(fr'assets\Kaggle\alzheimer_mri_preprocessed_dataset\raw\Mild_Demented'))))
img4 = np.array(Image.open(random.choice(os.listdir(fr'assets\Kaggle\alzheimer_mri_preprocessed_dataset\raw\Moderate_Demented'))))

st.title("Classifying Alzheimer's Disease")
st.write("""
Alzheimer's Disease is classified in terms of dementia severity.
Pick a random image below to model the dementia rating.
""")

img = image_select(
    label="Select an MRI scan of the brain",
    images=[img1, img2, img3, img4]
)

if st.button("Predict Dementia Severity"):
    result = None
    with st.spinner("Predicting..."):
        result = predict(img)

    st.write(label_mapping(result))
    st.image(img)
