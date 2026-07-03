import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import os
from utils import show_page_info

st.set_page_config(page_title="Super Resolution", page_icon="🔍")
st.title("Image Super Resolution")
show_page_info("Super_Resolution")
st.write("Enhance image resolution using deep learning-based super resolution models (FSRCNN, ESPCN, LapSRN).")

MODELS = {
    "FSRCNN ×2 (fast, good quality)":  ("models/FSRCNN_x2.pb", 2),
    "FSRCNN ×3":                        ("models/FSRCNN_x3.pb", 3),
    "FSRCNN ×4":                        ("models/FSRCNN_x4.pb", 4),
    "ESPCN ×2 (efficient, sharp)":      ("models/ESPCN_x2.pb",  2),
    "ESPCN ×3":                         ("models/ESPCN_x3.pb",  3),
    "ESPCN ×4":                         ("models/ESPCN_x4.pb",  4),
    "LapSRN ×2 (best perceptual)":      ("models/LapSRN_x2.pb", 2),
    "LapSRN ×4":                        ("models/LapSRN_x4.pb", 4),
    "LapSRN ×8":                        ("models/LapSRN_x8.pb", 8),
}

@st.cache_resource()
def load_sr_model(model_path, scale):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    algo = os.path.basename(model_path).split("_")[0].upper()
    sr.readModel(model_path)
    sr.setModel(algo.lower(), scale)
    return sr

def get_download_link(img_bgr, filename):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Result</a>'

uploaded_file = st.file_uploader("Upload a low-resolution image", type=['jpg', 'jpeg', 'png'])
model_choice  = st.selectbox("Super Resolution Model", list(MODELS.keys()))

if uploaded_file is not None:
    image   = Image.open(uploaded_file).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    oh, ow  = img_bgr.shape[:2]

    # For large inputs, optionally downscale first so SR is meaningful
    max_input_dim = st.slider("Max input dimension (px) — downscale before SR if needed", 64, 512, 256, 32)
    if max(oh, ow) > max_input_dim:
        scale_down = max_input_dim / max(oh, ow)
        img_bgr = cv2.resize(img_bgr, (int(ow * scale_down), int(oh * scale_down)))

    h, w = img_bgr.shape[:2]
    st.caption(f"Input size for SR: {w} × {h} px")

    model_path, scale = MODELS[model_choice]

    with st.spinner(f"Running {model_choice}..."):
        try:
            sr     = load_sr_model(model_path, scale)
            result = sr.upsample(img_bgr)
        except Exception as e:
            st.error(f"Super resolution failed: {e}")
            st.stop()

    # Bicubic reference at the same scale
    bicubic = cv2.resize(img_bgr, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_CUBIC)

    rh, rw = result.shape[:2]
    st.subheader("Comparison")
    col1, col2, col3 = st.columns(3)
    col1.image(img_bgr, channels='BGR', caption=f"Input ({w}×{h})", use_container_width=True)
    col2.image(bicubic, channels='BGR', caption=f"Bicubic ×{scale} ({rw}×{rh})", use_container_width=True)
    col3.image(result,  channels='BGR', caption=f"DL SR ×{scale} ({rw}×{rh})", use_container_width=True)

    st.markdown(get_download_link(result, "super_resolution.png"), unsafe_allow_html=True)

    with st.expander("Model comparison guide"):
        st.write("""
        | Model | Speed | Quality | Notes |
        |-------|-------|---------|-------|
        | **FSRCNN** | Fast | Good | Best for real-time or batch use |
        | **ESPCN** | Fast | Good | Sub-pixel convolution — efficient |
        | **LapSRN** | Slower | Best | Laplacian pyramid — best perceptual quality |

        Tip: Use a small input image (≤256 px) for fastest results. The model upscales by the chosen factor.
        """)
else:
    st.info("Upload a low-resolution image to enhance it with deep learning super resolution.")
