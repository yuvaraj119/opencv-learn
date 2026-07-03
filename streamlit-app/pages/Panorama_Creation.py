import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import show_page_info

st.set_page_config(page_title="Panorama Creation", page_icon="🌄")

st.title("Panorama Creation by Image Stitching")
show_page_info("Panorama_Creation")

uploaded_files = st.file_uploader("Upload multiple images for stitching", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) >= 2:
    images = []
    for file in uploaded_files:
        image = Image.open(file)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        images.append(img_bgr)

    st.write(f"Uploaded {len(images)} images.")

    if st.button("Create Panorama"):
        try:
            stitcher = cv2.Stitcher_create()
            status, panorama = stitcher.stitch(images)

            if status == cv2.Stitcher_OK:
                st.success("Panorama created successfully!")
                st.image(panorama, channels='BGR', caption="Panorama")
            else:
                st.error("Stitching failed. Try with overlapping images.")
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("Please upload at least 2 images with overlapping regions to create a panorama.")