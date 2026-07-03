import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from utils import show_page_info

st.set_page_config(page_title="OCR - Optical Character Recognition", page_icon="📝")

st.title("Optical Character Recognition (OCR)")
show_page_info("OCR")

uploaded_file = st.file_uploader("Upload an image with text", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    st.image(image, caption="Uploaded Image")

    # Preprocessing options
    st.subheader("Preprocessing")
    apply_threshold = st.checkbox("Apply Threshold", value=True)
    apply_blur = st.checkbox("Apply Blur", value=False)
    blur_kernel = st.slider("Blur Kernel Size", 1, 9, 3, step=2) if apply_blur else 3

    processed_img = img_gray.copy()

    if apply_blur:
        processed_img = cv2.GaussianBlur(processed_img, (blur_kernel, blur_kernel), 0)

    if apply_threshold:
        _, processed_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    st.subheader("Processed Image")
    st.image(processed_img, caption="Processed for OCR")

    # Extract text
    if st.button("Extract Text"):
        text = pytesseract.image_to_string(processed_img)
        st.subheader("Extracted Text")
        st.text_area("Text", text, height=200)

else:
    st.info("Please upload an image containing text to perform OCR.")