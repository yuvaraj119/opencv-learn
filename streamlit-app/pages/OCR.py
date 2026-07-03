import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from utils import show_page_info, ensure_model

st.set_page_config(page_title="OCR - Optical Character Recognition", page_icon="📝")
st.title("Optical Character Recognition (OCR)")
show_page_info("OCR")

MODE_DOC   = "Document / Form"
MODE_SCENE = "Signboard / Scene Text"

mode = st.radio(
    "Select OCR approach based on your use case:",
    [MODE_DOC, MODE_SCENE],
    horizontal=True,
)

uploaded_file = st.file_uploader("Upload an image with text", type=['jpg', 'jpeg', 'png'])

if uploaded_file is None:
    col1, col2 = st.columns(2)
    col1.info(
        "**Document / Form**\n\n"
        "Best for:\n- Scanned documents\n- Invoices & forms\n- Book / article pages\n- Printed text\n\n"
        "*Engine: Tesseract*"
    )
    col2.info(
        "**Signboard / Scene Text**\n\n"
        "Best for:\n- Street & traffic signs\n- Billboards & posters\n- Product labels\n- ID cards\n\n"
        "*Engine: OpenCV DB detector + CRNN recognizer*"
    )
    st.stop()

image    = Image.open(uploaded_file).convert("RGB")
img_arr  = np.array(image)
img_bgr  = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

# ── MODE 1: Document / Form (Tesseract) ──────────────────────────────────────
if mode == MODE_DOC:
    try:
        import pytesseract
    except ImportError:
        st.error(
            "**pytesseract not installed.**\n\n"
            "Run: `pip install pytesseract`\n\n"
            "Also install the Tesseract binary: `brew install tesseract` (macOS) "
            "or `apt install tesseract-ocr` (Linux)."
        )
        st.stop()

    PSM_OPTIONS = {
        "Auto — full page (default)":      3,
        "Uniform text block / paragraph":  6,
        "Single text line":                7,
        "Single word":                     8,
        "Sparse / scattered text":         11,
    }

    col_ctrl, col_preview = st.columns([1, 2])

    with col_ctrl:
        psm_label    = st.selectbox("Page layout / text structure", list(PSM_OPTIONS.keys()))
        apply_blur   = st.checkbox("Denoise (Gaussian blur)", value=False)
        blur_k       = st.slider("Blur kernel size", 1, 9, 3, step=2) if apply_blur else 3
        apply_thresh = st.checkbox("Binarize (Otsu threshold)", value=True)

    img_gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    processed = img_gray.copy()
    if apply_blur:
        processed = cv2.GaussianBlur(processed, (blur_k, blur_k), 0)
    if apply_thresh:
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    with col_preview:
        t1, t2 = st.tabs(["Original", "Preprocessed"])
        t1.image(image, use_container_width=True)
        t2.image(processed, use_container_width=True)

    if st.button("Run OCR", type="primary"):
        psm    = PSM_OPTIONS[psm_label]
        config = f"--oem 3 --psm {psm}"
        with st.spinner("Extracting text…"):
            try:
                text = pytesseract.image_to_string(processed, config=config)
                data = pytesseract.image_to_data(
                    processed, config=config, output_type=pytesseract.Output.DICT
                )
            except pytesseract.TesseractNotFoundError:
                st.error(
                    "Tesseract binary not found. Install it:\n\n"
                    "`brew install tesseract` (macOS) or `apt install tesseract-ocr` (Linux)"
                )
                st.stop()

        # Draw word bounding boxes on original image
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for i, word in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if conf > 30 and word.strip():
                x, y, w, h = (
                    data["left"][i], data["top"][i],
                    data["width"][i], data["height"][i],
                )
                draw.rectangle([x, y, x + w, y + h], outline="#00cc44", width=2)

        col_ann, col_txt = st.columns(2)
        col_ann.image(annotated, caption="Detected words (green boxes)", use_container_width=True)
        col_txt.text_area("Extracted Text", text.strip() or "(no text found)", height=300)

        if text.strip():
            st.download_button(
                "Download extracted text (.txt)",
                text.encode(),
                "ocr_document.txt",
                "text/plain",
            )


# ── MODE 2: Signboard / Scene Text (DB detector + CRNN recognizer) ───────────
elif mode == MODE_SCENE:

    def four_points_transform(frame, vertices):
        vertices = np.asarray(vertices, dtype=np.float32)
        out_size = (100, 32)
        target   = np.array(
            [[0, out_size[1]-1], [0, 0],
             [out_size[0]-1, 0], [out_size[0]-1, out_size[1]-1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(vertices, target)
        return cv2.warpPerspective(frame, M, out_size)

    @st.cache_resource()
    def load_scene_models():
        det_path = ensure_model("text_detection_DB_IC15_resnet18_2021sep.onnx")
        rec_path = ensure_model("text_recognition_CRNN_EN_2021sep.onnx")

        detector = cv2.dnn_TextDetectionModel_DB(det_path)
        detector.setBinaryThreshold(0.3).setPolygonThreshold(0.5)
        detector.setInputParams(
            1.0 / 255, (640, 640),
            (122.67891434, 116.66876762, 104.00698793), True,
        )

        recognizer = cv2.dnn_TextRecognitionModel(rec_path)
        recognizer.setDecodeType("CTC-greedy")
        recognizer.setVocabulary(list("0123456789abcdefghijklmnopqrstuvwxyz"))
        recognizer.setInputParams(1 / 127.5, (100, 32), (127.5, 127.5, 127.5), True)

        return detector, recognizer

    col_ctrl, col_img = st.columns([1, 3])
    with col_ctrl:
        bin_thresh  = st.slider("Binary threshold",  0.1, 0.9, 0.3, 0.05,
                                help="Controls text/background separation sensitivity")
        poly_thresh = st.slider("Polygon threshold", 0.1, 0.9, 0.5, 0.05,
                                help="Controls how tightly boxes wrap the text")
        show_canvas = st.checkbox("Show recognized text canvas", value=True)
    col_img.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Detect & Recognize Text", type="primary"):
        with st.spinner("Loading models (first run downloads ~22 MB)…"):
            try:
                detector, recognizer = load_scene_models()
            except Exception as e:
                st.error(f"Failed to load scene text models: {e}")
                st.stop()

        detector.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)

        with st.spinner("Detecting text regions…"):
            boxes, confs = detector.detect(img_bgr)

        if boxes is None or len(boxes) == 0:
            st.warning("No text detected. Try lowering the thresholds.")
            st.image(image, use_container_width=True)
            st.stop()

        words          = []
        annotated      = img_bgr.copy()
        output_canvas  = np.full(img_bgr.shape, 255, dtype=np.uint8)

        for box in boxes:
            roi  = four_points_transform(img_bgr, box)
            word = recognizer.recognize(roi)

            cv2.polylines(annotated, [box.astype(np.int32)], True, (255, 0, 255), 2)

            if word.strip():
                words.append(word)
                if show_canvas:
                    box_h = max(int(abs(box[0, 1] - box[1, 1])), 8)
                    fs    = cv2.getFontScaleFromHeight(
                        cv2.FONT_HERSHEY_SIMPLEX, max(box_h - 8, 5), 1
                    )
                    pos = (int(box[0, 0]), int(box[0, 1]))
                    cv2.putText(output_canvas, word, pos,
                                cv2.FONT_HERSHEY_SIMPLEX, fs, (180, 0, 0), 1, cv2.LINE_AA)

        col1, col2 = st.columns(2)
        col1.image(annotated, channels="BGR",
                   caption=f"Detected regions ({len(boxes)} boxes)", use_container_width=True)
        if show_canvas:
            col2.image(output_canvas, channels="BGR",
                       caption="Recognized text", use_container_width=True)

        st.subheader(f"Recognized words — {len(words)} found")
        full_text = " ".join(words)
        st.text_area("Output", full_text or "(no text recognized)", height=120)

        if words:
            st.download_button(
                "Download recognized text (.txt)",
                full_text.encode(),
                "scene_ocr_output.txt",
                "text/plain",
            )
