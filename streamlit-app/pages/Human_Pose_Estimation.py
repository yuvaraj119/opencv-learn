import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from PIL import Image
from utils import show_page_info

st.set_page_config(page_title="Human Pose Estimation", page_icon="🧘")
st.title("Human Pose Estimation using MediaPipe")
show_page_info("Human_Pose_Estimation")
st.write("Detect and visualize 33 body landmarks using MediaPipe PoseLandmarker.")

MODEL_PATH = "models/pose_landmarker_heavy.task"

@st.cache_resource()
def load_pose_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=4,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated = np.copy(rgb_image)
    landmark_style = drawing_styles.get_default_pose_landmarks_style()
    for pose_landmarks in detection_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=landmark_style,
        )
    return annotated

def angle_between(p1, p2, p3):
    """Angle at p2 formed by vectors p2->p1 and p2->p3 (degrees)."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1, 1))))

def get_xy(lm_list, idx, w, h):
    lm = lm_list[idx]
    return (int(lm.x * w), int(lm.y * h))

L = vision.PoseLandmark
JOINT_TRIPLETS = {
    "Left Elbow":  (L.LEFT_SHOULDER,  L.LEFT_ELBOW,  L.LEFT_WRIST),
    "Right Elbow": (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST),
    "Left Knee":   (L.LEFT_HIP,       L.LEFT_KNEE,   L.LEFT_ANKLE),
    "Right Knee":  (L.RIGHT_HIP,      L.RIGHT_KNEE,  L.RIGHT_ANKLE),
    "Left Hip":    (L.LEFT_SHOULDER,  L.LEFT_HIP,    L.LEFT_KNEE),
    "Right Hip":   (L.RIGHT_SHOULDER, L.RIGHT_HIP,   L.RIGHT_KNEE),
}

uploaded_file = st.file_uploader("Upload an image with a person", type=['jpg', 'jpeg', 'png'])
show_angles   = st.checkbox("Overlay joint angles on the image", value=True)

if uploaded_file is not None:
    image     = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    h, w      = img_array.shape[:2]

    with st.spinner("Estimating pose..."):
        landmarker = load_pose_landmarker()
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        result     = landmarker.detect(mp_image)

    if result.pose_landmarks:
        annotated = draw_landmarks_on_image(img_array, result)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        if show_angles:
            for lms in result.pose_landmarks:
                for joint_name, (a, b, c) in JOINT_TRIPLETS.items():
                    try:
                        p1 = get_xy(lms, a, w, h)
                        p2 = get_xy(lms, b, w, h)
                        p3 = get_xy(lms, c, w, h)
                        ang = angle_between(p1, p2, p3)
                        cv2.putText(annotated_bgr, f"{int(ang)}°", p2,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                    except Exception:
                        pass

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_container_width=True)
        col2.image(annotated_bgr, channels='BGR', caption="Pose Estimation", use_container_width=True)

        st.subheader("Joint Angles (first person)")
        lms = result.pose_landmarks[0]
        cols = st.columns(3)
        for i, (joint_name, (a, b, c)) in enumerate(JOINT_TRIPLETS.items()):
            try:
                p1 = get_xy(lms, a, w, h)
                p2 = get_xy(lms, b, w, h)
                p3 = get_xy(lms, c, w, h)
                ang = angle_between(p1, p2, p3)
                cols[i % 3].metric(joint_name, f"{ang:.1f}°")
            except Exception:
                cols[i % 3].metric(joint_name, "N/A")

        st.caption(f"Detected {len(result.pose_landmarks)} person(s) in the image.")
    else:
        st.warning("No person detected. Try a clearer image with a full or half body visible.")
        st.image(image, caption="Uploaded Image", use_container_width=True)
else:
    st.info("Upload an image with a person to estimate their pose.")
    st.write("""
    **Model:** MediaPipe PoseLandmarker Heavy — detects 33 body landmarks including shoulders,
    elbows, wrists, hips, knees, and ankles. Joint angles (elbow, knee, hip) are computed from
    triplets of adjacent landmarks.
    """)
