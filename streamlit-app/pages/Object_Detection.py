import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from utils import show_page_info, ensure_model

#---------------------------------------------------------------------------------------
# The session_state function allows us to initialize and save variables across for across
# session states. This is a valuable feature that enables us to take different actions
# depending on the state of selected variables in the code. If this is not done then
# all variables are reset any time the application state changes (e.g., when a user
# interacts with a widget). For example, the confidence threshold of the slider has
# changed, but we are still working with the same image, we can detect that by
# comparing the current file_uploaded_id (img_file_buffer.id) with the
# previous value (ss.file_uploaded_id) and if they are the same then we know we
# don't need to call the object detection model again. We just simply need to process
# the previous set of detections.
#---------------------------------------------------------------------------------------

# Create application title and file uploader widget.
st.title("Object Detection")
show_page_info("Object_Detection")
img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

# Initialize session state variables
if 'file_uploaded_name' not in st.session_state:
    st.session_state.file_uploaded_name = None
if 'detections' not in st.session_state:
    st.session_state.detections = None

# Function for detecting objects in an image.
def detectObjectsOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob as input to the model.
    net.setInput(blob)
    # Get Detections.
    detections = net.forward()
    return detections

# Function for annotating the image with bounding boxes for each detected object.
def process_detections(frame, detections, class_names, conf_threshold=0.5):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    bb_line_thickness = max(1, int(round(frame_h / 200)))
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            class_id = int(detections[0, 0, i, 1])
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            cv2.putText(frame, f"{label} {confidence:.2f}",
                        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame, bboxes

# Function to load the DNN model.
@st.cache_resource()
def load_model():
    modelFile  = ensure_model("ssd_mobilenet_frozen_inference_graph.pb")
    configFile = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    with open("models/coco_class_labels.txt") as f:
        class_names = f.read().strip().split("\n")
    net = cv2.dnn.readNet(modelFile, configFile)
    return net, class_names

# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

net, class_names = load_model()

if img_file_buffer is not None:
    # Read the file and convert it to OpenCV Image.
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # Loads image in a BGR channel order.
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    file_name = img_file_buffer.name

    # Create placeholders to display input and output images.
    placeholders = st.columns(2)
    # Display Input image in the first placeholder.
    placeholders[0].image(image, channels='BGR')
    placeholders[0].text("Input Image")

    # Create a Slider and get the threshold from the slider.
    conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)

    # Check if the loaded image is "new", if so call the object detection model function.
    if file_name != st.session_state.file_uploaded_name:
        # Set the file_uploaded_name equal to the name of the file that was just uploaded.
        st.session_state.file_uploaded_name = file_name
        # Save the detections in the session-state for future use with the current loaded image.
        st.session_state.detections = detectObjectsOpenCVDnn(net, image)

        st.write("New image uploaded, calling the object detection model.")
    else:
        st.write("Same image used, processing with the previous detections.")

    # Process the detections based on the current confidence threshold.
    out_image, _ = process_detections(image, st.session_state.detections, class_names, conf_threshold=conf_threshold)

    # Display Detected objects.
    placeholders[1].image(out_image, channels='BGR')
    placeholders[1].text("Output Image")

    # Convert OpenCV image to PIL.
    out_image = Image.fromarray(out_image[:, :, ::-1])
    # Create a link for downloading the output file.
    st.markdown(get_image_download_link(out_image, "object_output.jpg", 'Download Output Image'), unsafe_allow_html=True)