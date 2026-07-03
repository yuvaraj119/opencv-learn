import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import show_page_info

st.set_page_config(page_title="QR Code Detection", page_icon="📱")

st.title("QR Code Detection")
show_page_info("QR_Code_Detection")

uploaded_file = st.file_uploader("Upload an image with QR code", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image, dtype=np.uint8)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image")

    # Detect QR code
    qr_detector = cv2.QRCodeDetector()
    data, bbox, _ = qr_detector.detectAndDecode(img_bgr)

    if bbox is not None:
        st.success("QR Code detected!")
        st.write(f"Decoded data: {data}")

        # Draw bounding box
        img_with_box = img_bgr.copy()
        bbox = bbox.astype(int)
        cv2.polylines(img_with_box, [bbox], True, (0, 255, 0), 2)

        st.image(img_with_box, channels='BGR', caption="Detected QR Code")
    else:
        st.warning("No QR code detected in the image.")

else:
    st.info("Please upload an image containing a QR code.")