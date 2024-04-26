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
        *Alzheimer's Disease* is classified in terms of dementia severity.\n
        Pick a random image below to model the dementia rating.\n
        **Note:** The model does not know the actual severity.
        Only ***YOU*** know the classification labeled on the image captions. The model will try to predict the caption/label/classification.
        """)

img = image_select(
    label="Select an MRI scan of the brain",
    images=[st.session_state.img1, st.session_state.img2, st.session_state.img3, st.session_state.img4],
    captions=["Non Demented", "Very Mild Demented", "Mild Demented" , "Moderate Demented"]
)

if st.button("Predict Dementia Severity"):
    result = None
    with st.spinner("Predicting..."):
        result = predict(img)

    st.write("Prediction: ", label_mapping(result))
    st.image(img)


st.write("""
        ### Evaluation Metrics based on model used:\n
        Non_Demented(0) - Accuracy: 0.9421875, Precision: 0.9435736677115988, Recall: 0.940625
        Very_Mild_Demented(1) - Accuracy: 0.9359375, Precision: 0.9103139013452914, Recall: 0.90625
        Mild_Demented(2) - Accuracy: 0.9671875, Precision: 0.8743169398907104, Recall: 0.8938547486033519
        Moderate_Demented(3) - Accuracy: 1.0, Precision: 1.0, Recall: 1.0
        """)