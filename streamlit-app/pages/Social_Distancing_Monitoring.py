import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from utils import show_page_info, ensure_model

st.set_page_config(page_title="Social Distancing Monitoring", page_icon="📏")
st.title("Social Distancing Monitoring")
show_page_info("Social_Distancing_Monitoring")
st.write("Detect people in an image and flag pairs that are too close together using MobileNet-SSD.")

MODEL_CFG  = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

@st.cache_resource()
def load_model():
    return cv2.dnn.readNet(ensure_model("ssd_mobilenet_frozen_inference_graph.pb"), MODEL_CFG)

def detect_people(frame, net, conf_thresh=0.5):
    """Return list of (confidence, (x1,y1,x2,y2), (cx,cy)) for every detected person."""
    results = []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    output = net.forward()
    for i in np.arange(0, output.shape[2]):
        class_id   = output[0, 0, i, 1]
        confidence = output[0, 0, i, 2]
        if confidence > conf_thresh and int(class_id) == 1:  # class 1 = person (COCO)
            box = (output[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            cx  = int((box[0] + box[2]) / 2)
            cy  = int((box[1] + box[3]) / 2)
            results.append((float(confidence), tuple(box), (cx, cy)))
    return results

def euclidean_dist(A, B):
    p1 = np.sum(A**2, axis=1)[:, np.newaxis]
    p2 = np.sum(B**2, axis=1)
    p3 = -2 * np.dot(A, B.T)
    return np.round(np.sqrt(np.maximum(p1 + p2 + p3, 0)), 2)

def detect_violations(results, fac=1.2):
    violations = set()
    if len(results) < 2:
        return violations
    widths    = np.array([abs(r[1][2] - r[1][0]) for r in results])
    centroids = np.array([r[2] for r in results], dtype=float)
    dist_mat  = euclidean_dist(centroids, centroids)
    for row in range(len(results)):
        for col in range(row + 1, len(results)):
            ref_dist = fac * min(widths[row], widths[col])
            if dist_mat[row, col] < ref_dist:
                violations.add(row)
                violations.add(col)
    return violations

def get_download_link(img_bgr, filename):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Result</a>'

uploaded_file = st.file_uploader("Upload an image with people", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        conf_thresh = st.slider("Detection Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    with col_s2:
        dist_factor = st.slider("Distance Violation Factor (× person width)", 0.5, 3.0, 1.2, 0.1,
                                 help="Two people flagged unsafe if centroid distance < factor × min(widths)")

    with st.spinner("Detecting people..."):
        try:
            net     = load_model()
            results = detect_people(img_bgr, net, conf_thresh)
        except Exception as e:
            st.error(f"Model loading error: {e}")
            st.stop()

    result_img = img_bgr.copy()

    if results:
        violations = detect_violations(results, fac=dist_factor)

        for idx, (prob, (x1, y1, x2, y2), centroid) in enumerate(results):
            color = (0, 0, 255) if idx in violations else (0, 200, 0)
            label = "UNSAFE" if idx in violations else "SAFE"
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(result_img, centroid, 5, color, -1)
            cv2.putText(result_img, f"{label} {prob:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Draw red lines between violating pairs
        for row in range(len(results)):
            for col in range(row + 1, len(results)):
                if row in violations and col in violations:
                    cv2.line(result_img, results[row][2], results[col][2], (0, 0, 255), 1)

        col1, col2 = st.columns(2)
        col1.image(image,      caption="Original",               use_container_width=True)
        col2.image(result_img, channels='BGR', caption="Analysis", use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("People Detected", len(results))
        m2.metric("Violations",      len(violations),      delta="unsafe" if violations else None)
        m3.metric("Safe",            len(results) - len(violations))

        st.markdown(get_download_link(result_img, "social_distancing.png"), unsafe_allow_html=True)
    else:
        st.warning("No people detected. Try lowering the confidence threshold or upload a clearer image.")
        st.image(image, caption="Original Image", use_container_width=True)
else:
    st.info("Upload an image with people to monitor social distancing.")
