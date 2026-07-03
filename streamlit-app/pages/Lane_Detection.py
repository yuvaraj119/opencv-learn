import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import show_page_info

st.set_page_config(page_title="Lane Detection", page_icon="🛣️")

st.title("Lane Detection using Hough Transform")
show_page_info("Lane_Detection")

uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]

    if file_type == 'image':
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Original Image")

        # Lane detection function
        def detect_lanes(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            # Define region of interest
            height, width = img.shape[:2]
            mask = np.zeros_like(edges)
            polygon = np.array([[
                (0, height),
                (width, height),
                (width, height//2),
                (0, height//2)
            ]], np.int32)
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)

            # Hough transform
            lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)

            line_img = np.zeros_like(img)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)

            result = cv2.addWeighted(img, 0.8, line_img, 1, 0)
            return result

        result = detect_lanes(frame)
        st.image(result, channels='BGR', caption="Lane Detection Result")

    elif file_type == 'video':
        st.info("Video processing for lane detection is not yet implemented. Please upload an image.")

else:
    st.info("Please upload an image to detect lanes.")