import streamlit as st
from alzheimersdetection import AlzheimerModel

def predict(data):
    labels = ["Non_Demented",
              "Very_Mild_Demented",
              "Mild_Demented",
              "Moderate_Demented"]
    
    clf = AlzheimerModel.load(fr'models\best_ad_model.pt')