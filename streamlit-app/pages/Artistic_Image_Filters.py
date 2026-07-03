import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import show_page_info

st.set_page_config(page_title="Artistic Image Filters", page_icon="🎨")

st.title("Artistic Image Filters")
show_page_info("Artistic_Image_Filters")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Original Image")

    filter_options = [
        "Original", "Black & White", "Sepia/Vintage", "Vignette", "Edge Detection",
        "Embossed Edges", "Exposure Improvement", "Outline", "Pencil Sketch", "Stylization"
    ]

    selected_filter = st.selectbox("Select Filter", filter_options)

    if selected_filter == "Original":
        result = img_bgr
    elif selected_filter == "Black & White":
        result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif selected_filter == "Sepia/Vintage":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        result = cv2.transform(img_bgr, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
    elif selected_filter == "Vignette":
        rows, cols = img_bgr.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/3)
        kernel_y = cv2.getGaussianKernel(rows, rows/3)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        result = np.copy(img_bgr)
        for i in range(3):
            result[:, :, i] = result[:, :, i] * mask
    elif selected_filter == "Edge Detection":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray, 100, 200)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif selected_filter == "Embossed Edges":
        kernel = np.array([[ -2, -1, 0],
                           [ -1,  1, 1],
                           [  0,  1, 2]])
        result = cv2.filter2D(img_bgr, -1, kernel) + 128
        result = np.clip(result, 0, 255).astype(np.uint8)
    elif selected_filter == "Exposure Improvement":
        result = cv2.convertScaleAbs(img_bgr, alpha=1.2, beta=10)
    elif selected_filter == "Outline":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        result = cv2.bitwise_not(thresh)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif selected_filter == "Pencil Sketch":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        inv_blur = cv2.bitwise_not(blur)
        result = cv2.divide(gray, inv_blur, scale=256.0)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif selected_filter == "Stylization":
        result = cv2.stylization(img_bgr, sigma_s=60, sigma_r=0.45)

    st.image(result, channels='BGR', caption=f"Filtered Image: {selected_filter}")

else:
    st.info("Please upload an image to apply artistic filters.")