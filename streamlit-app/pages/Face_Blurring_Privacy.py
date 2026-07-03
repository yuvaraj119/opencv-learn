import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import show_page_info

st.set_page_config(page_title="Face Blurring for Privacy", page_icon="🙈")

st.title("Face Blurring for Privacy Preservation")
show_page_info("Face_Blurring_Privacy")

uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]

    if file_type == 'image':
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Original Image")

        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        result = img_bgr.copy()
        for (x, y, w, h) in faces:
            face_roi = result[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            result[y:y+h, x:x+w] = blurred_face

        st.image(result, channels='BGR', caption="Blurred Faces")

    elif file_type == 'video':
        st.info("Video face blurring is not yet implemented. Please upload an image.")

else:
    st.info("Please upload an image to blur faces.")