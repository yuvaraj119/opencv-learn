import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from utils import show_page_info

st.set_page_config(page_title="Color Pop Effect", page_icon="🌈")
st.title("Color Pop Effect")
show_page_info("Color_Pop")
st.write("Keep one color vibrant while converting the rest to grayscale. Great for emphasizing a subject's color.")

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
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    st.image(image, caption="Original Image", use_container_width=True)

    st.subheader("Color Range to Keep")
    st.write("Adjust the HSV range to select which color stays vibrant.")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Lower Bound")
        h_min = st.slider("Hue Min (0=Red, 60=Green, 120=Blue)", 0, 179, 0)
        s_min = st.slider("Saturation Min", 0, 255, 50)
        v_min = st.slider("Value Min", 0, 255, 50)
    with col2:
        st.caption("Upper Bound")
        h_max = st.slider("Hue Max", 0, 179, 15)
        s_max = st.slider("Saturation Max", 0, 255, 255)
        v_max = st.slider("Value Max", 0, 255, 255)

    # Handle hue wrap-around for red (hue near 0/179)
    if h_min <= h_max:
        mask = cv2.inRange(img_hsv,
                           np.array([h_min, s_min, v_min]),
                           np.array([h_max, s_max, v_max]))
    else:
        mask1 = cv2.inRange(img_hsv, np.array([h_min, s_min, v_min]), np.array([179, s_max, v_max]))
        mask2 = cv2.inRange(img_hsv, np.array([0, s_min, v_min]), np.array([h_max, s_max, v_max]))
        mask = cv2.bitwise_or(mask1, mask2)

    # Dilate mask slightly for cleaner edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Grayscale version of the full image
    gray_bgr = cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # Composite: keep color where mask is white, gray elsewhere
    result = np.where(mask[:, :, np.newaxis] > 0, img_bgr, gray_bgr).astype(np.uint8)

    col_a, col_b = st.columns(2)
    col_a.image(mask, caption="Selected Color Mask", use_container_width=True)
    col_b.image(result, channels='BGR', caption="Color Pop Result", use_container_width=True)

    st.markdown(get_download_link(result, "color_pop.png"), unsafe_allow_html=True)
else:
    st.info("Upload an image to apply the Color Pop effect.")
    st.write("""
    **Hue quick reference:**
    - 0–10 and 165–179 = Red
    - 20–35 = Orange / Yellow
    - 35–85 = Green
    - 100–130 = Blue
    - 130–165 = Purple / Magenta

    Raise Saturation Min to avoid selecting muted/grey pixels.
    """)
