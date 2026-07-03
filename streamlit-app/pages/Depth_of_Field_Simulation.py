import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import show_page_info

st.set_page_config(page_title="Depth of Field Simulation", page_icon="🔭")

st.title("Simulating Depth of Field")
show_page_info("Depth_of_Field_Simulation")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Original Image")

    # Simulate depth of field
    # TODO: Improve depth of field simulation with actual depth maps. Refer to BasicImageOperations.ipynb for advanced depth-of-field effects.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_bgr, (15, 15), 0)
    mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.GaussianBlur(mask.astype(float), (51, 51), 0)
    mask = (mask / 255.0).astype(np.float32)

    result = cv2.addWeighted(img_bgr.astype(np.float32), mask[:, :, np.newaxis], blur.astype(np.float32), 1 - mask[:, :, np.newaxis], 0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    st.image(result, channels='BGR', caption="Depth of Field Simulation")

else:
    st.info("Please upload an image for depth of field simulation.")