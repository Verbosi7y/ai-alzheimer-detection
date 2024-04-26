import streamlit as st
from streamlit_image_select import image_select

import torch
import numpy as np

import skimage

import os, random
from alzheimersdetection import AlzheimerModel
from alzheimersdetection.AlzheimerModel import AlzheimerCNN

def predict(data):
    model = AlzheimerCNN()
    checkpoint = torch.load(fr'models/best_ad_model.pt', map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint)

    result = AlzheimerModel.predict(model, data)

    return result

def label_mapping(result):
    labels = ["Not Demented",
              "Very Mild Demented",
              "Mild Demented",
              "Moderate Demented"]
    
    return labels[result]

parent_dir = fr'assets/Kaggle/alzheimer_mri_preprocessed_dataset/raw'

img1 = random.choice(os.listdir(fr'{parent_dir}/Non_Demented'))
img2 = random.choice(os.listdir(fr'{parent_dir}/Very_Mild_Demented'))
img3 = random.choice(os.listdir(fr'{parent_dir}/Mild_Demented'))
img4 = random.choice(os.listdir(fr'{parent_dir}/Moderate_Demented'))

img1 = skimage.io.imread(fr'{parent_dir}/Non_Demented/{img1}', as_gray=True)
img2 = skimage.io.imread(fr'{parent_dir}/Very_Mild_Demented/{img2}', as_gray=True)
img3 = skimage.io.imread(fr'{parent_dir}/Mild_Demented/{img3}', as_gray=True)
img4 = skimage.io.imread(fr'{parent_dir}/Moderate_Demented/{img4}', as_gray=True)

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
