import streamlit as st

st.set_page_config(
    page_title="OpenCV Streamlit App",
    page_icon="👋",
)

st.title("Welcome to the OpenCV Streamlit App!")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This app showcases various OpenCV-based image and video processing applications.
    Select any application from the sidebar to get started.

    ---

    ### 🎨 Creative & Artistic
    - **Artistic Image Filters** — Apply artistic styles: pencil sketch, oil painting, watercolor, cartoon, and more.
    - **Color Pop Effect** — Keep a selected color range vivid while converting the rest of the image to grayscale.
    - **Watermarking** — Overlay a logo or watermark onto an image with adjustable position, scale, and transparency.
    - **Virtual Billboard** — Replace any flat surface in a photo with your own image using perspective homography.

    ### 🧠 AI & Deep Learning
    - **Face Detection** — Detect faces using a deep learning SSD model with ResNet-10 backbone.
    - **Object Detection** — Detect 80+ everyday objects using SSD MobileNet V2 trained on COCO.
    - **Image Classification** — Classify images into 1000 ImageNet categories using DenseNet-121.
    - **Super Resolution** — Upscale low-resolution images with FSRCNN, ESPCN, or LapSRN deep learning models.
    - **Depth Aware Processing** — Apply depth-aware effects using a DNN-estimated depth map.

    ### 🧍 Human Analysis
    - **Human Pose Estimation** — Detect and visualize 33 body landmarks using MediaPipe PoseLandmarker.
    - **Golf Swing Analysis** — Analyze golf swing posture: spine angle, arm bend, knee flex, and alignment.
    - **Blink Detection** — Detect eye blinks using MediaPipe FaceLandmarker blendshape scores.
    - **Face Blurring for Privacy** — Automatically detect and blur faces to protect privacy.
    - **Social Distancing Monitoring** — Detect people and flag pairs violating a minimum distance threshold.
    - **Face Controlled Games** — Use your face position to control online browser games.

    ### 🌿 Segmentation & Background
    - **Foreground Segmentation** — Separate the subject from the background and apply blur, grayscale, or custom backgrounds using MediaPipe.
    - **Portrait Mode / Depth Blur** — Simulate DSLR-style background blur by detecting and isolating faces.
    - **Depth of Field Simulation** — Simulate shallow depth of field with a blurred background gradient.
    - **Deforestation Detection** — Analyze forest coverage using HSV/BGR color segmentation across multiple images.

    ### 📡 Specialized Vision
    - **Augmented Reality (ArUco)** — Overlay images onto ArUco markers for augmented reality effects.
    - **Lane Detection** — Detect road lanes in images using Hough line transform and Canny edge detection.
    - **Object Tracking** — Track moving objects across video frames using OpenCV tracking algorithms.
    - **Panorama Creation** — Stitch multiple overlapping photos into a seamless panorama.
    - **Satellite Imagery Analysis** — Analyze GeoTIFF satellite imagery, compute NDVI, and visualize vegetation coverage.

    ### 🛠️ Utility Tools
    - **OpenCV Basic Operation Tools** — Interactive image editor: resize, crop, rotate, draw, blur, threshold, detect edges and contours — with undo/reset.
    - **Image Restoration** — Reduce noise using fast non-local means, median, bilateral, or Gaussian filtering.
    - **OCR** — Extract text from images using Tesseract with threshold and blur preprocessing options.
    - **QR Code Generation** — Generate QR codes from any text or URL and download them as PNG.
    - **QR Code Detection** — Scan and decode QR codes from uploaded images.
    - **Digital Signature** — Extract a clean transparent-background digital signature from a photo with custom color.
    - **Intruder Detection** — Detect motion and intruders in video using KNN background subtraction.
    """
)