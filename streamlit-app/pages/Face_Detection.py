import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from utils import show_page_info, ensure_model

st.title("Face Detection")
show_page_info("Face_Detection")
img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

if 'file_uploaded_name' not in st.session_state:
    st.session_state.file_uploaded_name = None
if 'detections' not in st.session_state:
    st.session_state.detections = None


@st.cache_resource()
def load_model():
    model_path = ensure_model("face_detection_yunet_2023mar.onnx")
    # Low score threshold so we cache all candidates; post-filter in draw_detections()
    return cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold=0.1)


def run_detection(detector, image):
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    return faces  # None or ndarray (N, 15): [x, y, w, h, ...landmarks..., score]


def draw_detections(frame, faces, conf_threshold):
    bboxes = []
    if faces is None:
        return frame, bboxes
    thickness = max(1, int(round(frame.shape[0] / 200)))
    for face in faces:
        score = float(face[14])
        if score < conf_threshold:
            continue
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        x2, y2 = x + w, y + h
        bboxes.append([x, y, x2, y2])
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), thickness, cv2.LINE_8)
    return frame, bboxes


def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


detector = load_model()

if img_file_buffer is not None:
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    file_name = img_file_buffer.name

    placeholders = st.columns(2)
    placeholders[0].image(image, channels='BGR')
    placeholders[0].text("Input Image")

    conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)

    if file_name != st.session_state.file_uploaded_name:
        st.session_state.file_uploaded_name = file_name
        st.session_state.detections = run_detection(detector, image)
        st.write("New image uploaded, calling the face detection model.")
    else:
        st.write("Same image used, processing with the previous detections.")

    out_image, _ = draw_detections(image.copy(), st.session_state.detections, conf_threshold)

    placeholders[1].image(out_image, channels='BGR')
    placeholders[1].text("Output Image")

    out_image_pil = Image.fromarray(out_image[:, :, ::-1])
    st.markdown(get_image_download_link(out_image_pil, "face_output.jpg", 'Download Output Image'),
                unsafe_allow_html=True)
