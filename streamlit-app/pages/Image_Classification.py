import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import show_page_info, ensure_model

st.set_page_config(page_title="Image Classification", page_icon="🖼️")
st.title("Deep Learning Image Classification")
show_page_info("Image_Classification")
st.write("Classify images into 1000 ImageNet categories using a DenseNet-121 model via OpenCV DNN.")

MODEL_FILE  = None  # loaded lazily via ensure_model()
CONFIG_FILE = "models/DenseNet_121.prototxt"
LABELS_FILE = "models/classification_classes_ILSVRC2012.txt"

@st.cache_resource()
def load_model():
    with open(LABELS_FILE) as f:
        image_net_names = f.read().split("\n")
    class_names = [name.split(",")[0] for name in image_net_names if name.strip()]
    net = cv2.dnn.readNet(model=ensure_model("DenseNet_121.caffemodel"), config=CONFIG_FILE, framework="Caffe")
    return net, class_names

def classify(net, img_bgr, class_names, top_k=5):
    if img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=0.017, size=(224, 224), mean=(104, 117, 123))
    net.setInput(blob)
    outputs = net.forward()
    scores = outputs[0].flatten()
    probs  = np.exp(scores) / np.sum(np.exp(scores))
    top_indices = probs.argsort()[::-1][:top_k]
    return [(class_names[i], float(probs[i]) * 100) for i in top_indices]

uploaded_file = st.file_uploader("Upload an image to classify", type=['jpg', 'jpeg', 'png'])
top_k = st.slider("Number of top predictions to show", 1, 10, 5)

if uploaded_file is not None:
    image   = Image.open(uploaded_file).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        try:
            net, class_names = load_model()
            predictions = classify(net, img_bgr, class_names, top_k=top_k)
        except Exception as e:
            st.error(f"Error loading or running model: {e}")
            st.stop()

    st.subheader("Top Predictions")
    for rank, (name, confidence) in enumerate(predictions, 1):
        col1, col2 = st.columns([3, 1])
        col1.write(f"**{rank}. {name}**")
        col2.write(f"{confidence:.2f}%")
        st.progress(min(confidence / 100, 1.0))

    best_name, best_conf = predictions[0]
    st.success(f"Most likely: **{best_name}** ({best_conf:.1f}% confidence)")
else:
    st.info("Upload an image to classify it into one of 1000 ImageNet categories.")
    st.write("""
    **About the model:**
    - Architecture: DenseNet-121 trained on ImageNet
    - 1000 object categories including animals, vehicles, food, everyday objects, and more
    """)
