import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from PIL import Image
from utils import show_page_info, ensure_model

st.set_page_config(page_title="Blink Detection", page_icon="👁️")
st.title("Blink Detection using Facial Landmarks")
show_page_info("Blink_Detection")
st.write("Uses MediaPipe FaceLandmarker to detect eye blendshape scores — a direct measure of how open or closed each eye is.")


# EAR landmark indices (same topology as old FaceMesh 468 landmarks)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C > 0 else 0.0

@st.cache_resource()
def load_face_landmarker():
    base_options = python.BaseOptions(model_asset_path=ensure_model("face_landmarker.task"))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=5,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)

uploaded_file = st.file_uploader("Upload an image with a face", type=['jpg', 'jpeg', 'png'])

blink_threshold = st.slider(
    "Blink Score Threshold (eye considered closed if blink score exceeds this)",
    0.1, 0.9, 0.4, 0.05,
    help="Blink score 0 = fully open, 1 = fully closed. Typical threshold: 0.3–0.5",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr  = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    with st.spinner("Detecting facial landmarks..."):
        landmarker = load_face_landmarker()
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        result     = landmarker.detect(mp_image)

    if result.face_landmarks:
        annotated = img_bgr.copy()
        summary   = []

        for face_idx, face_landmarks in enumerate(result.face_landmarks):
            lms = face_landmarks

            # EAR-based calculation
            left_ear  = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0

            # Blendshape-based blink scores (more direct and accurate)
            left_blink_score  = 0.0
            right_blink_score = 0.0
            if result.face_blendshapes and face_idx < len(result.face_blendshapes):
                for bs in result.face_blendshapes[face_idx]:
                    if bs.category_name == "eyeBlinkLeft":
                        left_blink_score = bs.score
                    elif bs.category_name == "eyeBlinkRight":
                        right_blink_score = bs.score

            avg_blink  = (left_blink_score + right_blink_score) / 2.0
            is_blinking = avg_blink > blink_threshold
            color  = (0, 0, 255) if is_blinking else (0, 200, 0)
            status = "BLINK" if is_blinking else "OPEN"

            # Draw eye landmark points
            for idx in LEFT_EYE + RIGHT_EYE:
                x_pt = int(lms[idx].x * w)
                y_pt = int(lms[idx].y * h)
                cv2.circle(annotated, (x_pt, y_pt), 3, color, -1)

            label = f"Face {face_idx + 1}: blink={avg_blink:.2f} [{status}]"
            cv2.putText(annotated, label, (10, 30 + face_idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            summary.append({
                "face": face_idx + 1,
                "left_blink":  left_blink_score,
                "right_blink": right_blink_score,
                "avg_blink":   avg_blink,
                "left_ear":    left_ear,
                "right_ear":   right_ear,
                "avg_ear":     avg_ear,
                "is_blinking": is_blinking,
            })

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_container_width=True)
        col2.image(annotated, channels='BGR', caption="Blink Detection Result", use_container_width=True)

        st.subheader("Results")
        for s in summary:
            status_str = "BLINK Detected" if s["is_blinking"] else "Eyes Open"
            with st.expander(f"Face {s['face']} — {status_str}", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Left Blink Score",  f"{s['left_blink']:.3f}")
                c2.metric("Right Blink Score", f"{s['right_blink']:.3f}")
                c3.metric("Avg Blink Score",   f"{s['avg_blink']:.3f}", delta=status_str)
                st.caption(f"EAR (geometric): left={s['left_ear']:.3f}, right={s['right_ear']:.3f}, avg={s['avg_ear']:.3f}")
    else:
        st.warning("No face detected. Try a clearer, well-lit frontal face image.")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    st.info("**Blink Score:** from MediaPipe face blendshapes — 0 = fully open, 1 = fully closed. "
            "More reliable than EAR for partial blinks and varied lighting.")
else:
    st.info("Upload an image with a face to detect blinks.")
