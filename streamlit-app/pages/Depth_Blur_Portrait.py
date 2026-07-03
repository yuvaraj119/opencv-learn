import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from utils import show_page_info, ensure_model

st.set_page_config(page_title="Portrait Mode / Depth Blur", page_icon="📸")
st.title("Portrait Mode / Depth Blur")
show_page_info("Depth_Blur_Portrait")
st.write("Simulate smartphone portrait mode by detecting the face and blurring the background.")

@st.cache_resource()
def load_face_model():
    model_path = ensure_model("face_detection_yunet_2023mar.onnx")
    return cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold=0.1)

def get_download_link(img_bgr, filename):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Result</a>'

uploaded_file = st.file_uploader("Upload a portrait image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        conf_threshold = st.slider("Face Detection Confidence", 0.1, 1.0, 0.5, 0.05)
    with col_s2:
        blur_strength = st.slider("Background Blur Strength", 5, 101, 35, step=2)
    with col_s3:
        expand_factor = st.slider("Face Region Size", 1.0, 3.0, 1.5, 0.1,
                                   help="How far beyond the detected face box to keep sharp")

    detector = load_face_model()
    detector.setInputSize((w, h))
    detector.setScoreThreshold(conf_threshold)
    _, faces = detector.detect(img_bgr)

    face_mask = np.zeros((h, w), dtype=np.uint8)
    face_count = 0
    if faces is not None:
        for face in faces:
            x, y, fw_det, fh_det = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            x2, y2 = x + fw_det, y + fh_det
            cx, cy = (x + x2) // 2, (y + y2) // 2
            fw = int(fw_det * expand_factor / 2)
            fh = int(fh_det * expand_factor / 2)
            cv2.ellipse(face_mask, (cx, cy), (max(1, fw), max(1, fh)), 0, 0, 360, 255, -1)
            face_count += 1

    if face_count > 0:
        # Smooth mask edges for natural blending
        face_mask_smooth = cv2.GaussianBlur(face_mask.astype(float), (51, 51), 0) / 255.0
        face_mask_3ch = np.stack([face_mask_smooth] * 3, axis=-1)

        k = blur_strength | 1  # ensure odd
        blurred = cv2.GaussianBlur(img_bgr, (k, k), 0)
        output = (img_bgr.astype(float) * face_mask_3ch +
                  blurred.astype(float) * (1 - face_mask_3ch)).astype(np.uint8)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_container_width=True)
        col2.image(output, channels='BGR', caption="Portrait Mode", use_container_width=True)

        st.success(f"Detected {face_count} face(s). Background blurred.")
        st.markdown(get_download_link(output, "portrait_mode.png"), unsafe_allow_html=True)
    else:
        st.warning("No face detected. Try lowering the confidence threshold or use a clearer frontal face photo.")
        st.image(image, caption="Original Image", use_container_width=True)
else:
    st.info("Upload a portrait photo to apply the depth blur effect.")
