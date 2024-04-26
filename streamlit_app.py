import streamlit as st
from streamlit_image_select import image_select

import torch

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

if 'img1' not in st.session_state:
    st.session_state.img1 = random.choice(os.listdir(fr'{parent_dir}/Non_Demented'))
    st.session_state.img2 = random.choice(os.listdir(fr'{parent_dir}/Very_Mild_Demented'))
    st.session_state.img3 = random.choice(os.listdir(fr'{parent_dir}/Mild_Demented'))
    st.session_state.img4 = random.choice(os.listdir(fr'{parent_dir}/Moderate_Demented'))

    st.session_state.img1 = skimage.io.imread(fr'{parent_dir}/Non_Demented/{st.session_state.img1}', as_gray=True)
    st.session_state.img2 = skimage.io.imread(fr'{parent_dir}/Very_Mild_Demented/{st.session_state.img2}', as_gray=True)
    st.session_state.img3 = skimage.io.imread(fr'{parent_dir}/Mild_Demented/{st.session_state.img3}', as_gray=True)
    st.session_state.img4 = skimage.io.imread(fr'{parent_dir}/Moderate_Demented/{st.session_state.img4}', as_gray=True)

st.title("Classifying Alzheimer's Disease")
st.write("""
Alzheimer's Disease is classified in terms of dementia severity.
Pick a random image below to model the dementia rating.
""")

img = image_select(
    label="Select an MRI scan of the brain",
    images=[st.session_state.img1, st.session_state.img2, st.session_state.img3, st.session_state.img4]
)

if st.button("Predict Dementia Severity"):
    result = None
    with st.spinner("Predicting..."):
        result = predict(img)

    st.write(label_mapping(result))
    st.image(img)
