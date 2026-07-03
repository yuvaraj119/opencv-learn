import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import glob
from utils import show_page_info

st.set_page_config(page_title="Deforestation using Color Segmentation", page_icon="🌳")

st.title("Deforestation using Color Segmentation")
show_page_info("Deforestation_Detection")

def detect_green_BGR(img):
    """Detect and return a mask for the green area of an image using BGR segmentation."""
    lower_BGR_values = np.array([0, 50, 0], dtype = 'uint8')
    upper_BGR_values = np.array([255, 100, 255], dtype = 'uint8')
    mask_BGR = cv2.inRange(img, lower_BGR_values, upper_BGR_values)
    return mask_BGR

def detect_green_HSV(img):
    """Detect and return a mask for the green area of an image using HSV segmentation."""
    HSV_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_HSV_values = np.array([36, 0, 50], dtype = 'uint8')
    upper_HSV_values = np.array([86, 250, 100], dtype = 'uint8')
    mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values)
    return mask_HSV

def percent_forest(gray_img):
    """Return the percentage of the image detected to be forested."""
    c = cv2.countNonZero(gray_img)
    t = gray_img.shape[0] * gray_img.shape[1]
    return round((c / t) * 100, 2)

uploaded_files = st.file_uploader("Upload images (up to 5)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Please upload up to 5 images only. Only the first 5 will be processed.")
        uploaded_files = uploaded_files[:5]

    analysis_type = st.selectbox("Analysis Type", ["Forest Detection (Green)", "Single Channel Analysis"])
    segmentation_mode = st.selectbox("Color Space", ["HSV", "BGR"])
    
    selected_channel = None
    min_val = 0
    max_val = 255
    
    if analysis_type == "Single Channel Analysis":
        if segmentation_mode == "HSV":
            selected_channel = st.selectbox("Select Channel", ["Hue", "Saturation", "Value"])
            max_limit = 179 if selected_channel == "Hue" else 255
            min_val = st.slider(f"Min {selected_channel}", 0, max_limit, 0)
            max_val = st.slider(f"Max {selected_channel}", 0, max_limit, max_limit)
        else:
            selected_channel = st.selectbox("Select Channel", ["Blue", "Green", "Red"])
            min_val = st.slider(f"Min {selected_channel}", 0, 255, 0)
            max_val = st.slider(f"Max {selected_channel}", 0, 255, 255)

    # Reset applied state if settings change
    if "prev_analysis_type" not in st.session_state or st.session_state.prev_analysis_type != analysis_type or \
       "prev_segmentation_mode" not in st.session_state or st.session_state.prev_segmentation_mode != segmentation_mode or \
       "prev_selected_channel" not in st.session_state or st.session_state.prev_selected_channel != selected_channel or \
       "prev_min_val" not in st.session_state or st.session_state.prev_min_val != min_val or \
       "prev_max_val" not in st.session_state or st.session_state.prev_max_val != max_val:
        st.session_state.applied = False
        st.session_state.prev_analysis_type = analysis_type
        st.session_state.prev_segmentation_mode = segmentation_mode
        st.session_state.prev_selected_channel = selected_channel
        st.session_state.prev_min_val = min_val
        st.session_state.prev_max_val = max_val

    apply_btn = st.button("Apply Analysis")
    
    if "applied" not in st.session_state:
        st.session_state.applied = False
    
    if apply_btn:
        st.session_state.applied = True

    results = []

    if st.session_state.applied:
        for uploaded_file in uploaded_files:
            # Load image
            pil_img = Image.open(uploaded_file)
            pil_img = ImageOps.exif_transpose(pil_img)
            img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            
            if analysis_type == "Forest Detection (Green)":
                if segmentation_mode == "HSV":
                    mask = detect_green_HSV(img_bgr)
                else:
                    mask = detect_green_BGR(img_bgr)
                caption = f"Forest Mask ({segmentation_mode})"
            else:
                # Single Channel Analysis
                if segmentation_mode == "HSV":
                    img_conv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                    channel_idx = ["Hue", "Saturation", "Value"].index(selected_channel)
                else:
                    img_conv = img_bgr
                    channel_idx = ["Blue", "Green", "Red"].index(selected_channel)
                
                channel_data = img_conv[:, :, channel_idx]
                mask = cv2.inRange(channel_data, min_val, max_val)
                caption = f"{selected_channel} Mask [{min_val}, {max_val}]"
            
            perc = percent_forest(mask)
            results.append({
                "name": uploaded_file.name,
                "original": img_bgr,
                "mask": mask,
                "percentage": perc,
                "caption": caption
            })

    # Display results
    if results:
        for i, res in enumerate(results):
            st.subheader(f"Image {i+1}: {res['name']}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(res['original'], channels="BGR", caption="Original", use_container_width=True)
            
            with col2:
                st.image(res['mask'], caption=res['caption'], use_container_width=True)
                
            with col3:
                st.metric("Detection Area (%)", f"{res['percentage']}%")
                if i > 0:
                    change = round(res['percentage'] - results[i-1]['percentage'], 2)
                    st.metric("Change from Previous", f"{change}%", delta=change, delta_color="inverse")

        if len(results) > 1:
            st.divider()
            st.header("Comparative Analysis")
            # Summary table
            summary_data = {
                "Image": [r['name'] for r in results],
                "Detection Coverage (%)": [r['percentage'] for r in results]
            }
            st.table(summary_data)
    elif st.session_state.applied == False:
        st.info("Adjust settings and click 'Apply Analysis' to process images.")
else:
    st.info("Please upload images to begin the analysis.")
