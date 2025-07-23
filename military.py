import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("🎯 Military Object Detection")

model = YOLO('runs/detect/train/weights/best.pt')

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    results = model(img_array)[0]
    annotated = results.plot()

    st.image(annotated, caption="Detection Result", use_column_width=True)

    st.write("### Detected Objects")
    for box in results.boxes:
        cls = int(box.cls[0])
        st.write(f"- **{model.names[cls]}** ({box.conf[0]:.2f})")
