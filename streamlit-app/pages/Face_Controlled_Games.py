import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import show_page_info

st.set_page_config(page_title="Face-Controlled Games", page_icon="🎮")

st.title("Playing Online Games with Faces")
show_page_info("Face_Controlled_Games")

uploaded_file = st.file_uploader("Upload an image with a face", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Original Image")

    # Placeholder for game control
    # TODO: Implement real-time face tracking for game control (e.g., using webcam). Refer to BasicImageOperations.ipynb for integration with games.

    # Example: Detect face position to control game
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        center_x = x + w // 2
        center_y = y + h // 2
        st.write(f"Face center: ({center_x}, {center_y}) - Use this to control game elements.")
    else:
        st.write("No face detected.")

else:
    st.info("Please upload an image with a face to control games.")