import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from PIL import Image
from utils import show_page_info

st.set_page_config(page_title="Golf Swing Analysis", page_icon="⛳")
st.title("Golf Swing Analysis and Training")
show_page_info("Golf_Swing_Analysis")
st.write("""
Analyze body posture during a golf swing using MediaPipe PoseLandmarker.
Key measurements: spine angle, arm bend, hip–shoulder alignment, and knee flex.
""")

MODEL_PATH = "models/pose_landmarker_heavy.task"

@st.cache_resource()
def load_pose_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
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

def get_pt(lms, idx, w, h):
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h])

def vec_angle(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1, 1))))

def joint_angle(p1, pivot, p3):
    return vec_angle(p1 - pivot, p3 - pivot)

L = vision.PoseLandmark

def analyze_swing(lms, w, h, img_bgr):
    out = img_bgr.copy()
    metrics = {}

    left_shoulder  = get_pt(lms, L.LEFT_SHOULDER,  w, h)
    right_shoulder = get_pt(lms, L.RIGHT_SHOULDER, w, h)
    left_hip       = get_pt(lms, L.LEFT_HIP,       w, h)
    right_hip      = get_pt(lms, L.RIGHT_HIP,      w, h)
    left_knee      = get_pt(lms, L.LEFT_KNEE,      w, h)
    right_knee     = get_pt(lms, L.RIGHT_KNEE,     w, h)
    left_ankle     = get_pt(lms, L.LEFT_ANKLE,     w, h)
    right_ankle    = get_pt(lms, L.RIGHT_ANKLE,    w, h)
    left_elbow     = get_pt(lms, L.LEFT_ELBOW,     w, h)
    right_elbow    = get_pt(lms, L.RIGHT_ELBOW,    w, h)
    left_wrist     = get_pt(lms, L.LEFT_WRIST,     w, h)
    right_wrist    = get_pt(lms, L.RIGHT_WRIST,    w, h)
    nose           = get_pt(lms, L.NOSE,            w, h)

    mid_shoulder = (left_shoulder + right_shoulder) / 2
    mid_hip      = (left_hip + right_hip) / 2

    # Spine angle from vertical
    spine_vec   = mid_shoulder - mid_hip
    spine_angle = vec_angle(spine_vec, np.array([0.0, -1.0]))
    metrics["Spine Angle from Vertical"] = f"{spine_angle:.1f}°"
    cv2.line(out, tuple(mid_hip.astype(int)), tuple(mid_shoulder.astype(int)), (0, 200, 255), 3)
    cv2.putText(out, f"Spine {spine_angle:.0f}°",
                (int(mid_hip[0]) + 8, int(mid_hip[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    # Shoulder–hip tilt
    tilt = vec_angle(right_shoulder - left_shoulder, right_hip - left_hip)
    metrics["Shoulder–Hip Tilt"] = f"{tilt:.1f}°"
    cv2.line(out, tuple(left_shoulder.astype(int)),  tuple(right_shoulder.astype(int)), (255, 200, 0), 2)
    cv2.line(out, tuple(left_hip.astype(int)),       tuple(right_hip.astype(int)),      (255, 200, 0), 2)

    # Lead arm elbow (left for right-handed golfer)
    lead_elbow = joint_angle(left_shoulder, left_elbow, left_wrist)
    metrics["Lead Arm (L Elbow)"] = f"{lead_elbow:.1f}°"

    # Trail arm elbow
    trail_elbow = joint_angle(right_shoulder, right_elbow, right_wrist)
    metrics["Trail Arm (R Elbow)"] = f"{trail_elbow:.1f}°"

    # Knee flex
    lead_knee  = joint_angle(left_hip,  left_knee,  left_ankle)
    trail_knee = joint_angle(right_hip, right_knee, right_ankle)
    metrics["Lead Knee Flex"]  = f"{lead_knee:.1f}°"
    metrics["Trail Knee Flex"] = f"{trail_knee:.1f}°"

    # Head offset
    head_dx = int(nose[0] - mid_shoulder[0])
    metrics["Head Lateral Offset"] = f"{head_dx:+d} px"
    cv2.circle(out, tuple(nose.astype(int)), 8, (255, 0, 255), -1)

    return out, metrics

def coaching_notes(metrics):
    notes = []
    spine = float(metrics["Spine Angle from Vertical"].rstrip("°"))
    lead_e = float(metrics["Lead Arm (L Elbow)"].rstrip("°"))
    lead_k = float(metrics["Lead Knee Flex"].rstrip("°"))

    if spine < 20:
        notes.append(("info", "Spine nearly vertical — try more forward tilt for a better swing plane."))
    elif spine > 45:
        notes.append(("warn", "Excessive spine tilt — may cause back strain over time."))
    else:
        notes.append(("good", f"Spine angle {spine:.0f}° looks solid."))

    if lead_e >= 160:
        notes.append(("good", "Lead arm is straight — great extension at address."))
    else:
        notes.append(("info", f"Lead arm bent {180 - lead_e:.0f}° — try to keep it straighter."))

    if 140 <= lead_k <= 168:
        notes.append(("good", "Lead knee flex looks good — stable base."))
    elif lead_k > 168:
        notes.append(("info", "Lead knee is very straight — a slight flex improves stability."))
    else:
        notes.append(("warn", "Lead knee heavily flexed — check stance width."))

    return notes

uploaded_file = st.file_uploader(
    "Upload a golf swing image (face-on or down-the-line view with full body visible)",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    image     = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w      = img_array.shape[:2]

    with st.spinner("Analyzing golf swing pose..."):
        landmarker = load_pose_landmarker()
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        result     = landmarker.detect(mp_image)

    if result.pose_landmarks:
        annotated_rgb = draw_landmarks_on_image(img_array, result)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        analyzed_bgr, metrics = analyze_swing(result.pose_landmarks[0], w, h, annotated_bgr)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_container_width=True)
        col2.image(analyzed_bgr, channels='BGR', caption="Swing Analysis", use_container_width=True)

        st.subheader("Biomechanical Measurements")
        cols = st.columns(3)
        for i, (k, v) in enumerate(metrics.items()):
            cols[i % 3].metric(k, v)

        st.subheader("Coaching Feedback")
        for kind, note in coaching_notes(metrics):
            if kind == "good":
                st.success(note)
            elif kind == "warn":
                st.warning(note)
            else:
                st.info(note)
    else:
        st.warning("No person detected. Try a clearer image showing the full body of the golfer.")
        st.image(image, caption="Uploaded Image", use_container_width=True)
else:
    st.info("Upload a golf swing image to get started.")
    st.write("""
    **What is analyzed:**
    - Spine angle from vertical (forward tilt)
    - Shoulder–hip lateral alignment
    - Lead & trail arm bend (elbow angles)
    - Lead & trail knee flex
    - Head position relative to shoulder center

    Works best with **face-on** (front view) or **down-the-line** (side view) swing photos where
    the full body is clearly visible.
    """)
