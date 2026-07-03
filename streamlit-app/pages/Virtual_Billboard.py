import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from utils import show_page_info

st.set_page_config(page_title="Virtual Billboard", page_icon="📺")
st.title("Virtual Billboard using Homography")
show_page_info("Virtual_Billboard")
st.write("Replace any flat surface (billboard, screen, canvas) in a photo with your own image "
         "using perspective transformation.")

def get_download_link(img_bgr, filename):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Result</a>'

scene_file   = st.file_uploader("1. Upload scene image (e.g., photo with a billboard, wall, or screen)",
                                 type=['jpg', 'jpeg', 'png'])
replace_file = st.file_uploader("2. Upload replacement image (what to put on the surface)",
                                 type=['jpg', 'jpeg', 'png'])

if scene_file is not None:
    scene_pil = Image.open(scene_file).convert("RGB")
    scene_bgr = cv2.cvtColor(np.array(scene_pil), cv2.COLOR_RGB2BGR)
    h, w = scene_bgr.shape[:2]

    st.image(scene_pil, caption=f"Scene Image ({w} × {h} px)", use_container_width=True)

    st.subheader("3. Define the target region")
    st.write("Enter the four corner coordinates of the region to replace, going **clockwise from the top-left**.")

    # Sensible defaults: a centred rectangle at 20–80 % of the image
    defaults = [
        (int(w * 0.20), int(h * 0.20)),
        (int(w * 0.80), int(h * 0.20)),
        (int(w * 0.80), int(h * 0.80)),
        (int(w * 0.20), int(h * 0.80)),
    ]
    labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    pts_dst = []
    cols = st.columns(4)
    for i, (col, label, (dx, dy)) in enumerate(zip(cols, labels, defaults)):
        with col:
            st.caption(label)
            x = st.number_input(f"X{i+1}", 0, w, dx, key=f"vb_x{i}")
            y = st.number_input(f"Y{i+1}", 0, h, dy, key=f"vb_y{i}")
            pts_dst.append([x, y])

    # Live preview with the selection drawn on the scene
    pts_array   = np.array(pts_dst, dtype=np.int32)
    preview_img = scene_bgr.copy()
    cv2.polylines(preview_img, [pts_array], isClosed=True, color=(0, 255, 255), thickness=2)
    for i, pt in enumerate(pts_dst):
        cv2.circle(preview_img, tuple(pt), 8, (0, 200, 0), -1)
        cv2.putText(preview_img, labels[i][:2], (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    st.image(preview_img, channels='BGR', caption="Selected region (yellow outline)", use_container_width=True)

    if replace_file is not None:
        if st.button("Apply Billboard Replacement", type="primary"):
            replace_pil = Image.open(replace_file).convert("RGB")
            replace_bgr = cv2.cvtColor(np.array(replace_pil), cv2.COLOR_RGB2BGR)
            rh, rw = replace_bgr.shape[:2]

            # Map the four corners of the replacement image to the selected region
            pts_src = np.array([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]], dtype=np.float32)
            pts_dst_f = np.array(pts_dst, dtype=np.float32)

            H, _ = cv2.findHomography(pts_src, pts_dst_f, cv2.RANSAC)
            warped = cv2.warpPerspective(replace_bgr, H, (w, h))

            # Black out the destination polygon, then add the warped image
            result = scene_bgr.copy()
            cv2.fillConvexPoly(result, pts_array, 0)
            result = cv2.add(result, warped)

            col1, col2 = st.columns(2)
            col1.image(scene_pil,   caption="Original Scene",          use_container_width=True)
            col2.image(result, channels='BGR', caption="Billboard Applied", use_container_width=True)
            st.markdown(get_download_link(result, "virtual_billboard.png"), unsafe_allow_html=True)
    else:
        st.info("Upload a replacement image (step 2) to apply the billboard effect.")
else:
    st.info("Upload a scene image to get started.")
    st.write("""
    **Example use cases:**
    - Replace a Times Square billboard with your own ad
    - Put your artwork on a gallery wall
    - Swap a TV screen with a custom display
    - Replace a poster on a wall

    **Tips:** Choose 4 corners in clockwise order. The replacement image will be perspective-warped to fit exactly.
    """)
