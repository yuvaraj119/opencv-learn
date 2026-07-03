import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
import base64
from utils import show_page_info

st.set_page_config(page_title="OpenCV Basic Operation Tools", page_icon="🛠️", layout="wide")

# Custom CSS to reduce font sizes and component padding
st.markdown("""
    <style>
    /* Reduce subheader size */
    h3 {
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    /* Reduce general text size */
    p, label {
        font-size: 0.85rem !important;
    }
    /* Shrink the size of various inputs */
    div[data-baseweb="select"] > div {
        height: 30px !important;
        min-height: 30px !important;
    }
    div[data-testid="stSlider"] {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    div[data-testid="stNumberInput"] > div {
        height: 30px !important;
    }
    /* Reduce tab padding */
    button[data-baseweb="tab"] {
        padding: 5px 10px !important;
        font-size: 0.85rem !important;
    }
    /* Reduce vertical space between elements */
    .stElementContainer {
        margin-bottom: 0.2rem !important;
    }
    /* Adjust button size */
    div.stButton > button {
        padding: 2px 10px !important;
        font-size: 0.8rem !important;
        height: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)

# st.title("OpenCV Basic Operation Tools (Image Editor)")
show_page_info("OpenCV_Basic_Operation_Tools")

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    if isinstance(img, np.ndarray):
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pass
    else:
        return ""
    
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Initialize Session State
if 'working_img' not in st.session_state:
    st.session_state.working_img = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'orig_img' not in st.session_state:
    st.session_state.orig_img = None

uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Check if new file uploaded
    file_bytes = uploaded_file.getvalue()
    if st.session_state.get('last_uploaded_file') != file_bytes:
        pil_img = Image.open(uploaded_file)
        pil_img = ImageOps.exif_transpose(pil_img)
        img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        st.session_state.working_img = img_bgr.copy()
        st.session_state.orig_img = img_bgr.copy()
        st.session_state.history = [img_bgr.copy()]
        st.session_state.last_uploaded_file = file_bytes

    img = st.session_state.working_img
    h, w = img.shape[:2]

    st.divider()
    
    # Tab-based toolbar at the top
    tabs = st.tabs([
        "Original", "Resize", "Gray", "Shapes", "Text", 
        "Crop", "Trans", "Blur", "Edges", "Morph", "Thresh", "Contour"
    ])

    # Operations Logic
    with tabs[0]: # 1. Preview Original
        # st.subheader("Preview Original")
        st.info("Original image.")
        if st.button("Reset to Original", key="btn_reset_orig"):
            st.session_state.working_img = st.session_state.orig_img.copy()
            st.session_state.history.append(st.session_state.working_img.copy())
            st.rerun()

    with tabs[1]: # 2. Resize
        # st.subheader("Resize Image")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            res_opt = st.radio("Resize by", ["Scale", "Dims"], key="res_opt", horizontal=True)
        with col_res2:
            if res_opt == "Scale":
                pct = st.slider("Scale %", 10, 200, 100, key="res_pct")
                nw, nh = int(w * pct / 100), int(h * pct / 100)
            else:
                nw = st.number_input("W", 1, 5000, w, key="res_w")
                nh = st.number_input("H", 1, 5000, h, key="res_h")
        
        col_res3, col_res4 = st.columns(2)
        with col_res3:
            interp = st.selectbox("Interp", ["INTER_LINEAR", "INTER_NEAREST", "INTER_CUBIC"], key="res_interp")
        with col_res4:
            if st.button("Apply Resize", key="btn_resize"):
                resized = cv2.resize(img, (nw, nh), interpolation=getattr(cv2, interp))
                st.session_state.working_img = resized
                st.session_state.history.append(resized.copy())
                st.rerun()

    with tabs[2]: # 3. Grayscale
        # st.subheader("Grayscale")
        st.write("Convert to grayscale.")
        if st.button("Apply Grayscale", key="btn_gray"):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.session_state.working_img = gray
                st.session_state.history.append(gray.copy())
                st.rerun()
            else:
                st.warning("Image is already grayscale.")

    with tabs[3]: # 4. Drawing Shapes
        # st.subheader("Drawing Shapes")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            shape = st.selectbox("Shape", ["Line", "Circle", "Rect"], key="shape_sel")
        with col_s2:
            color = st.color_picker("Color", "#00FF00", key="shape_color")
        with col_s3:
            thick = st.slider("Thick", 1, 20, 2, key="shape_thick")
        
        rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])
        
        if shape == "Line":
            c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
            x1 = c1.number_input("X1", 0, w, min(0, w), key="line_x1")
            y1 = c2.number_input("Y1", 0, h, min(0, h), key="line_y1")
            x2 = c3.number_input("X2", 0, w, w, key="line_x2")
            y2 = c4.number_input("Y2", 0, h, h, key="line_y2")
            if c5.button("Draw", key="btn_line"):
                temp = img.copy()
                cv2.line(temp, (x1, y1), (x2, y2), bgr, thick)
                st.session_state.working_img = temp
                st.session_state.history.append(temp.copy())
                st.rerun()
        elif shape == "Circle":
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            cx = c1.number_input("CX", 0, w, w//2, key="circ_cx")
            cy = c2.number_input("CY", 0, h, h//2, key="circ_cy")
            r = c3.number_input("R", 1, max(1, min(w, h)), min(50, min(w, h)), key="circ_r")
            if c4.button("Draw", key="btn_circ"):
                temp = img.copy()
                cv2.circle(temp, (cx, cy), r, bgr, thick)
                st.session_state.working_img = temp
                st.session_state.history.append(temp.copy())
                st.rerun()
        else: # Rect
            c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
            rx1 = c1.number_input("X1", 0, w, min(10, w), key="rect_x1")
            ry1 = c2.number_input("Y1", 0, h, min(10, h), key="rect_y1")
            rx2 = c3.number_input("X2", 0, w, max(0, w-10), key="rect_x2")
            ry2 = c4.number_input("Y2", 0, h, max(0, h-10), key="rect_y2")
            if c5.button("Draw", key="btn_rect"):
                temp = img.copy()
                cv2.rectangle(temp, (rx1, ry1), (rx2, ry2), bgr, thick)
                st.session_state.working_img = temp
                st.session_state.history.append(temp.copy())
                st.rerun()

    with tabs[4]: # 5. Putting Text
        # st.subheader("Putting Text")
        c1, c2, c3 = st.columns([2, 1, 1])
        txt_val = c1.text_input("Text", "Hello OpenCV", key="txt_val")
        t_color = c2.color_picker("Color", "#FFFFFF", key="txt_color")
        t_thick = c3.slider("Thick", 1, 10, 2, key="txt_thick")
        
        c4, c5, c6, c7 = st.columns([1, 1, 1, 1])
        tx = c4.number_input("X", 0, w, min(50, w), key="txt_x")
        ty = c5.number_input("Y", 0, h, min(50, h), key="txt_y")
        t_scale = c6.slider("Scale", 0.1, 5.0, 1.0, key="txt_scale")
        
        rgb = tuple(int(t_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])
        if c7.button("Apply Text", key="btn_text"):
            temp = img.copy()
            cv2.putText(temp, txt_val, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, t_scale, bgr, t_thick)
            st.session_state.working_img = temp
            st.session_state.history.append(temp.copy())
            st.rerun()

    with tabs[5]: # 6. Cropping
        # st.subheader("Cropping")
        c1, c2, c3, c4, c5 = st.columns(5)
        cx1 = c1.slider("X", 0, max(0, w-1), 0, key="crop_x")
        cy1 = c2.slider("Y", 0, max(0, h-1), 0, key="crop_y")
        cw = c3.slider("W", 1, max(1, w - cx1), max(1, w - cx1), key="crop_w")
        ch = c4.slider("H", 1, max(1, h - cy1), max(1, h - cy1), key="crop_h")
        if c5.button("Apply Crop", key="btn_crop"):
            cropped = img[cy1:cy1+ch, cx1:cx1+cw]
            st.session_state.working_img = cropped
            st.session_state.history.append(cropped.copy())
            st.rerun()

    with tabs[6]: # 7. Translation & Rotation
        # st.subheader("Translation & Rotation")
        c1, c2, c3, c4, c5 = st.columns(5)
        angle = c1.slider("Angle", -180, 180, 0, key="rot_angle")
        tx_val = c2.slider("Trans X", -w, w, 0, key="trans_x")
        ty_val = c3.slider("Trans Y", -h, h, 0, key="trans_y")
        scale_val = c4.slider("Scale", 0.1, 2.0, 1.0, key="trans_scale")
        if c5.button("Apply", key="btn_trans"):
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale_val)
            M[0, 2] += tx_val
            M[1, 2] += ty_val
            transformed = cv2.warpAffine(img, M, (w, h))
            st.session_state.working_img = transformed
            st.session_state.history.append(transformed.copy())
            st.rerun()

    with tabs[7]: # 8. Image Blurring
        # st.subheader("Image Blurring")
        c1, c2, c3 = st.columns(3)
        b_type = c1.selectbox("Blur Mode", ["Gaussian", "Median", "Bilateral"], key="blur_mode")
        k_val = c2.slider("K Size", 1, 31, 5, step=2, key="blur_k")
        if c3.button("Apply Blur", key="btn_blur"):
            if b_type == "Gaussian":
                res = cv2.GaussianBlur(img, (k_val, k_val), 0)
            elif b_type == "Median":
                res = cv2.medianBlur(img, k_val)
            else:
                res = cv2.bilateralFilter(img, 9, 75, 75)
            st.session_state.working_img = res
            st.session_state.history.append(res.copy())
            st.rerun()

    with tabs[8]: # 9. Edge Detection
        # st.subheader("Edge Detection")
        c1, c2, c3 = st.columns(3)
        t1_val = c1.slider("T1", 0, 255, 100, key="edge_t1")
        t2_val = c2.slider("T2", 0, 255, 200, key="edge_t2")
        if c3.button("Apply Canny", key="btn_edge"):
            edges = cv2.Canny(img, t1_val, t2_val)
            st.session_state.working_img = edges
            st.session_state.history.append(edges.copy())
            st.rerun()

    with tabs[9]: # 10. Dilation & Erosion
        # st.subheader("Dilation & Erosion")
        c1, c2, c3, c4 = st.columns(4)
        m_op_val = c1.radio("Mode", ["Dilation", "Erosion"], key="morph_mode", horizontal=True)
        mk_val = c2.slider("K Size", 1, 15, 3, step=2, key="morph_k")
        mi_val = c3.slider("Iter", 1, 10, 1, key="morph_i")
        if c4.button("Apply Morph", key="btn_morph"):
            kernel = np.ones((mk_val, mk_val), np.uint8)
            if m_op_val == "Dilation":
                res = cv2.dilate(img, kernel, iterations=mi_val)
            else:
                res = cv2.erode(img, kernel, iterations=mi_val)
            st.session_state.working_img = res
            st.session_state.history.append(res.copy())
            st.rerun()

    with tabs[10]: # 11. Thresholding
        # st.subheader("Thresholding")
        c1, c2, c3, c4 = st.columns(4)
        thresh_type = c1.selectbox("Type", ["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_TOZERO_INV"], key="thresh_type")
        thresh_val = c2.slider("Thresh", 0, 255, 127, key="thresh_val")
        max_val = c3.slider("Max Val", 0, 255, 255, key="thresh_max")
        if c4.button("Apply Threshold", key="btn_thresh"):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            _, thresh = cv2.threshold(gray, thresh_val, max_val, getattr(cv2, thresh_type))
            st.session_state.working_img = thresh
            st.session_state.history.append(thresh.copy())
            st.rerun()

    with tabs[11]: # 12. Contour Detection
        # st.subheader("Contour Detection")
        c1, c2, c3 = st.columns(3)
        mode = c1.selectbox("Mode", ["RETR_EXTERNAL", "RETR_LIST", "RETR_CCOMP", "RETR_TREE"], key="contour_mode")
        method = c2.selectbox("Method", ["CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_KCOS"], key="contour_method")
        if c3.button("Find Contours", key="btn_contour"):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, getattr(cv2, mode), getattr(cv2, method))
            contour_img = img.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            st.session_state.working_img = contour_img
            st.session_state.history.append(contour_img.copy())
            st.rerun()

    st.divider()
    
    col1, col2 = st.columns([1, 1])

    with col1:
        # st.subheader("Preview")
        if len(img.shape) == 2: # Grayscale/Edges
            st.image(img, caption="Current State", use_container_width=True)
        else:
            st.image(img, channels="BGR", caption="Current State", use_container_width=True)
        
        c_dl, c_inf = st.columns([1, 1])
        with c_dl:
            st.markdown(get_image_download_link(img, "edited_image.png", "Download Image"), unsafe_allow_html=True)
        with c_inf:
            st.write(f"Size: {img.shape[1]}x{img.shape[0]}")

    with col2:
        # st.subheader("Controls")
        if st.button("Undo Last", use_container_width=True) and len(st.session_state.history) > 1:
            st.session_state.history.pop()
            st.session_state.working_img = st.session_state.history[-1]
            st.rerun()
        if st.button("Reset All", use_container_width=True):
            st.session_state.working_img = st.session_state.orig_img.copy()
            st.session_state.history = [st.session_state.orig_img.copy()]
            st.rerun()

else:
    st.info("Upload an image to start using the basic operation tools.")