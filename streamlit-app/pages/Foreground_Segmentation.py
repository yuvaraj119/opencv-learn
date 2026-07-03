import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from io import BytesIO
import base64
from utils import show_page_info, ensure_model

st.set_page_config(page_title="Foreground Segmentation", page_icon="✂️")
st.title("Foreground Segmentation & Background Effects")
show_page_info("Foreground_Segmentation")
st.write("Separate the foreground subject from the background using MediaPipe Image Segmentation.")


def get_download_link(img_bgr, filename):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Result</a>'

@st.cache_resource()
def load_segmenter():
    base_options = python.BaseOptions(model_asset_path=ensure_model("person_segmenter.tflite"))
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    return vision.ImageSegmenter.create_from_options(options)

uploaded_file = st.file_uploader("Upload an image (works best with people / selfies)", type=['jpg', 'jpeg', 'png'])
bg_file = st.file_uploader("Optional: Upload a custom background image (for Replace Background mode)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image     = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    effect = st.selectbox("Background Effect", [
        "Blur Background",
        "Grayscale Background",
        "Grayscale + Blur Background",
        "Replace Background",
    ])

    threshold    = st.slider("Segmentation Threshold (higher = tighter mask around subject)", 0.1, 0.95, 0.5, 0.05)
    blur_strength = 35
    if "Blur" in effect:
        blur_strength = st.slider("Blur Strength", 5, 101, 35, step=2)

    bg_img = None
    if effect == "Replace Background":
        if bg_file is not None:
            bg_pil = Image.open(bg_file).convert("RGB")
            bg_img = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
            bg_img = cv2.resize(bg_img, (img_bgr.shape[1], img_bgr.shape[0]))
        else:
            st.warning("Upload a background image above to use the Replace Background effect.")

    with st.spinner("Segmenting foreground..."):
        segmenter = load_segmenter()
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        seg_result = segmenter.segment(mp_image)
        seg_mask   = seg_result.confidence_masks[0].numpy_view()  # float32 in [0,1], 1=foreground

    binary_mask = (seg_mask > threshold).astype(np.uint8)
    mask3 = np.stack([binary_mask] * 3, axis=-1)

    k = blur_strength | 1  # ensure odd
    if effect == "Blur Background":
        blurred = cv2.GaussianBlur(img_bgr, (k, k), 0)
        output  = np.where(mask3 == 1, img_bgr, blurred)
    elif effect == "Grayscale Background":
        gray   = cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        output = np.where(mask3 == 1, img_bgr, gray)
    elif effect == "Grayscale + Blur Background":
        gray         = cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        blurred_gray = cv2.GaussianBlur(gray, (k, k), 0)
        output       = np.where(mask3 == 1, img_bgr, blurred_gray)
    elif effect == "Replace Background":
        output = np.where(mask3 == 1, img_bgr, bg_img) if bg_img is not None else img_bgr.copy()
    else:
        output = img_bgr.copy()

    col1, col2 = st.columns(2)
    col1.image(image,  caption="Original",   use_container_width=True)
    col2.image(output, channels='BGR', caption=effect, use_container_width=True)

    st.image(seg_mask, caption="Segmentation Confidence Mask (brighter = more foreground)",
             use_container_width=True, clamp=True)
    st.markdown(get_download_link(output, "segmented.png"), unsafe_allow_html=True)
else:
    st.info("Upload an image to apply background effects.")
    st.write("""
    **Best results with:**
    - Portrait / selfie photos with a clear subject
    - Good lighting and a relatively uncluttered background
    - Adjust the segmentation threshold if the mask leaks into the background or cuts into the subject
    """)
