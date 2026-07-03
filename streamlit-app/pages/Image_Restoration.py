import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from utils import show_page_info

st.set_page_config(page_title="Image Restoration", page_icon="🛠️")
st.title("Image Restoration & Noise Reduction")
show_page_info("Image_Restoration")
st.write("Remove noise, smooth skin, and enhance photo quality using various OpenCV filtering techniques.")

def get_download_link(img_bgr, filename):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Result</a>'

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Original Image", use_container_width=True)

    mode = st.selectbox("Restoration Mode", [
        "Fast Non-Local Means Denoising  (best for general photos)",
        "Median Filter  (salt & pepper noise)",
        "Bilateral Filter  (skin smoothing, edge-preserving)",
        "Gaussian Blur  (Gaussian / uniform noise)",
    ])

    if mode.startswith("Fast"):
        h_val         = st.slider("Filter Strength (h)", 1, 30, 10)
        template_size = st.slider("Template Window Size", 3, 15, 7, step=2)
        search_size   = st.slider("Search Window Size", 15, 35, 21, step=2)
        result = cv2.fastNlMeansDenoisingColored(img_bgr, None, h_val, h_val, template_size, search_size)

    elif mode.startswith("Median"):
        k      = st.slider("Kernel Size", 3, 15, 5, step=2)
        result = cv2.medianBlur(img_bgr, k)

    elif mode.startswith("Bilateral"):
        d           = st.slider("Diameter", 5, 50, 15)
        sigma_color = st.slider("Sigma Color  (higher = more color blending)", 10, 250, 75)
        sigma_space = st.slider("Sigma Space  (higher = farther pixels influence)", 10, 250, 75)
        result = cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)

    else:  # Gaussian
        k      = st.slider("Kernel Size", 3, 31, 5, step=2)
        result = cv2.GaussianBlur(img_bgr, (k, k), 0)

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", use_container_width=True)
    col2.image(result, channels='BGR', caption="Restored", use_container_width=True)

    st.markdown(get_download_link(result, "restored.png"), unsafe_allow_html=True)

    with st.expander("Mode guide"):
        st.write("""
        | Mode | Best for |
        |------|----------|
        | **Fast Non-Local Means** | General photo denoising — best quality, slower |
        | **Median Filter** | Salt & pepper noise (random black/white dots) |
        | **Bilateral Filter** | Skin smoothing, portrait retouching — blurs noise while keeping edges sharp |
        | **Gaussian Blur** | Soft overall blur, uniform noise reduction |
        """)
else:
    st.info("Upload an image to start restoring it.")
