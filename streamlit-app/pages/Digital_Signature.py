import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
import base64
from utils import show_page_info

st.set_page_config(page_title="Creating a Digital Signature", page_icon="✍️")

st.title("Creating a Digital Signature")
show_page_info("Digital_Signature")

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

uploaded_file = st.file_uploader("Upload Signature Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Reset applied state if new file is uploaded
    if "prev_sig" not in st.session_state or st.session_state.prev_sig != uploaded_file.name:
        st.session_state.applied_sig = False
        st.session_state.prev_sig = uploaded_file.name

    # Read image using PIL to handle EXIF orientation
    image_pil = Image.open(uploaded_file)
    image_pil = ImageOps.exif_transpose(image_pil)
    
    # Original dimensions
    width_orig, height_orig = image_pil.size

    st.subheader("Adjustments")
    
    # Resize slider
    resize_factor = st.slider("Resize Factor (%)", 10, 200, 100)
    new_width = int(width_orig * (resize_factor / 100))
    new_height = int(height_orig * (resize_factor / 100))
    
    # Color picker
    sig_color = st.color_picker("Pick Signature Color", "#0000FF") # Default Blue
    
    # Threshold slider
    threshold_val = st.slider("Threshold Value", 0, 255, 150)
    st.info("Adjust threshold to separate the signature from the background.")

    apply_button = st.button("Apply and Extract Signature")
    
    if apply_button:
        st.session_state.applied_sig = True

    # Process Image
    # 1. Resize
    image_resized = image_pil.resize((new_width, new_height), Image.LANCZOS)
    img_cv = cv2.cvtColor(np.array(image_resized.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # 2. Convert to Grayscale for thresholding
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 3. Create Alpha Mask (Inverted threshold)
    # Background is usually white (higher values), signature is dark (lower values)
    # We want signature to be white in the mask (255) and background black (0)
    _, alpha_mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    # 4. Color Tinting
    # Convert hex color to BGR
    hex_color = sig_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    
    # Create a solid color image
    colored_img = np.full(img_cv.shape, bgr_color, dtype=np.uint8)
    
    # Combine color and alpha
    b, g, r = cv2.split(colored_img)
    final_png = cv2.merge([b, g, r, alpha_mask])
    
    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_resized, caption="Uploaded Image (Resized)", use_container_width=True)
    
    if st.session_state.get('applied_sig', False):
        with col2:
            # For display, we show it over a checkboard or just as is
            # PIL handles RGBA display well
            final_pil = Image.fromarray(cv2.cvtColor(final_png, cv2.COLOR_BGRA2RGBA))
            st.image(final_pil, caption="Extracted Signature", use_container_width=True)
            
        # Download
        st.markdown(get_image_download_link(final_pil, "digital_signature.png", "Download Digital Signature (Transparent PNG)"), unsafe_allow_html=True)
    else:
        with col2:
            st.info("Adjust settings and click 'Apply and Extract Signature' to see the result.")

else:
    st.info("Please upload a signature image to proceed.")
