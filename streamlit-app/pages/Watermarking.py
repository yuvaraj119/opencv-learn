import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
import base64
from utils import show_page_info

st.set_page_config(page_title="Creating Watermarks", page_icon="🖼️")

st.title("Creating Watermarks")
show_page_info("Watermarking")

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

col_up1, col_up2 = st.columns(2)
with col_up1:
    base_image_file = st.file_uploader("Upload Base Image", type=['jpg', 'jpeg', 'png'], key="base")
with col_up2:
    logo_image_file = st.file_uploader("Upload Logo Image", type=['jpg', 'jpeg', 'png', 'svg'], key="logo")

# Reset applied state if new files are uploaded
if "prev_base" not in st.session_state or st.session_state.prev_base != (base_image_file.name if base_image_file else None) or \
   "prev_logo" not in st.session_state or st.session_state.prev_logo != (logo_image_file.name if logo_image_file else None):
    st.session_state.applied = False
    st.session_state.prev_base = base_image_file.name if base_image_file else None
    st.session_state.prev_logo = logo_image_file.name if logo_image_file else None

if base_image_file and logo_image_file:
    # Read base image using PIL to handle EXIF orientation
    base_pil = Image.open(base_image_file)
    base_pil = ImageOps.exif_transpose(base_pil)
    base_img = cv2.cvtColor(np.array(base_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # Read logo image using PIL to handle EXIF orientation
    logo_pil = Image.open(logo_image_file)
    logo_pil = ImageOps.exif_transpose(logo_pil)
    
    # Convert to OpenCV format (preserving alpha if present)
    if logo_pil.mode == 'RGBA':
        logo_img = cv2.cvtColor(np.array(logo_pil), cv2.COLOR_RGBA2BGRA)
    else:
        logo_img = cv2.cvtColor(np.array(logo_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    if logo_img is None:
        st.error("Could not decode logo image.")
    else:
        # Initialize session state for applied state
        if 'applied' not in st.session_state:
            st.session_state.applied = False

        # If logo doesn't have 4 channels, add an alpha channel
        if logo_img.shape[2] == 3:
            logo_img = cv2.cvtColor(logo_img, cv2.COLOR_BGR2BGRA)
            logo_img[:, :, 3] = 255 # fully opaque by default

        img_h, img_w, _ = base_img.shape
        logo_h_orig, logo_w_orig, _ = logo_img.shape

        st.subheader("ROI & Scale")
        scale = st.slider("Logo Scale (%)", 1, 100, 20)
        
        # Resize logo
        new_logo_w = int(logo_w_orig * (scale / 100))
        new_logo_h = int(logo_h_orig * (scale / 100))
        
        # Ensure logo is not bigger than base image
        if new_logo_w > img_w or new_logo_h > img_h:
            st.warning("Logo is too large for the base image at this scale. Resizing to fit.")
            scale_factor = min(img_w / logo_w_orig, img_h / logo_h_orig)
            new_logo_w = int(logo_w_orig * scale_factor)
            new_logo_h = int(logo_h_orig * scale_factor)

        logo_resized = cv2.resize(logo_img, (new_logo_w, new_logo_h), interpolation=cv2.INTER_LINEAR)
        
        # ROI Position
        max_x = img_w - new_logo_w
        max_y = img_h - new_logo_h
        
        tlc_x = st.slider("ROI X Position", 0, max_x, int(max_x / 2))
        tlc_y = st.slider("ROI Y Position", 0, max_y, int(max_y / 2))
        
        st.subheader("Transparency")
        alpha_val = st.slider("Alpha (Transparency)", 0.0, 1.0, 1.0, 0.05)
        use_alpha_channel = st.checkbox("Use Logo's Alpha Channel", value=True)

        apply_button = st.button("Apply Watermark")
        if apply_button:
            st.session_state.applied = True

        # Process Watermark
        brc_x = tlc_x + new_logo_w
        brc_y = tlc_y + new_logo_h
        
        roi = base_img[tlc_y:brc_y, tlc_x:brc_x].copy()
        
        logo_bgr = logo_resized[:, :, 0:3]
        logo_alpha = logo_resized[:, :, 3]
        
        if use_alpha_channel:
            # Normalize alpha channel to 0-1 and multiply by global alpha_val
            mask = (logo_alpha / 255.0) * alpha_val
            mask_3ch = cv2.merge([mask, mask, mask])
            
            # Blending: roi * (1 - mask) + logo_bgr * mask
            blended_roi = (roi.astype(float) * (1.0 - mask_3ch) + logo_bgr.astype(float) * mask_3ch).astype(np.uint8)
        else:
            # Simple addWeighted blending
            blended_roi = cv2.addWeighted(roi, 1.0, logo_bgr, alpha_val, 0)
            
        output_img = base_img.copy()
        output_img[tlc_y:brc_y, tlc_x:brc_x] = blended_roi
        
        # Display
        col1, col2 = st.columns(2)
        with col1:
            st.image(base_img, channels="BGR", caption="Original Image", use_container_width=True)
        
        if st.session_state.applied:
            with col2:
                st.image(output_img, channels="BGR", caption="Watermarked Image", use_container_width=True)
                
            # Download
            out_pil = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
            st.markdown(get_image_download_link(out_pil, "watermarked_image.png", "Download Watermarked Image"), unsafe_allow_html=True)
        else:
            with col2:
                st.info("Adjust settings and click 'Apply Watermark' to see the result.")

elif not base_image_file or not logo_image_file:
    st.info("Please upload both a base image and a logo image to proceed.")
