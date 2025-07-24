# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

model = YOLO('C:/Users/Rahila/Desktop/project/eda.ipynb/military_detection_project/yolov8n_military_fast/weights/best.pt')


st.title("🔍 Military Threat Detection")
uploaded = st.file_uploader("Upload image", type=['jpg', 'png'])

if uploaded:
    image = Image.open(uploaded)
    image_np = np.array(image)
    results = model(image_np)[0]
    output = results.plot()
    st.image(output, caption="Detection Result", use_column_width=True)
