import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import base64
from streamlit_image_coordinates import streamlit_image_coordinates
from utils import show_page_info

st.set_page_config(page_title="Augmented Reality", page_icon="🔍", layout="wide")
st.title("Augmented Reality — Click to Place ROI")
show_page_info("Augmented_Reality_ArUco")

# ── constants ─────────────────────────────────────────────────────────────────

DISPLAY_WIDTH = 700   # image is resized to this width in the browser
CORNER_LABELS = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
CORNER_COLORS = ["#00cc44", "#ff8800", "#0055ff", "#cc00cc"]
DOT_RADIUS    = 8
DEFAULT_IDS   = [23, 25, 30, 33]

DICT_OPTIONS = {
    "DICT_4X4_50":              cv2.aruco.DICT_4X4_50,
    "DICT_4X4_250":             cv2.aruco.DICT_4X4_250,
    "DICT_5X5_100":             cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250":             cv2.aruco.DICT_5X5_250,
    "DICT_6X6_250 (recommended)": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_250":             cv2.aruco.DICT_7X7_250,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def img_to_download_link(img_bgr, filename, label):
    result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(result_rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">{label}</a>'

def draw_roi_preview(pil_img: Image.Image, pts_display: list) -> Image.Image:
    """Draw dots and polygon on the display-size image."""
    preview = pil_img.copy().convert("RGBA")
    draw    = ImageDraw.Draw(preview, "RGBA")

    # Fill polygon when all 4 corners are set
    if len(pts_display) == 4:
        draw.polygon(pts_display, fill=(100, 200, 100, 60), outline=None)
        draw.line(pts_display + [pts_display[0]], fill="#00cc44", width=2)

    for i, (x, y) in enumerate(pts_display):
        color = CORNER_COLORS[i]
        draw.ellipse(
            [(x - DOT_RADIUS, y - DOT_RADIUS), (x + DOT_RADIUS, y + DOT_RADIUS)],
            fill=color, outline="white",
        )
        draw.text((x + DOT_RADIUS + 4, y - DOT_RADIUS), CORNER_LABELS[i], fill=color)

    return preview.convert("RGB")

def display_to_original(dx, dy, orig_w, orig_h):
    """Convert display-space coordinates to original image pixel coordinates."""
    scale = orig_w / DISPLAY_WIDTH
    return int(dx * scale), int(dy * scale)

def place_markers_on_image(bg_bgr, roi_pts, dictionary, marker_ids, marker_size_px):
    """Stamp a marker centered on each clicked corner point."""
    result  = bg_bgr.copy()
    img_h, img_w = result.shape[:2]
    half = marker_size_px // 2

    for (cx, cy), mid in zip(roi_pts, marker_ids):
        marker     = cv2.aruco.generateImageMarker(dictionary, mid, marker_size_px)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

        # Destination rect on canvas
        dst_x1 = max(0, cx - half)
        dst_y1 = max(0, cy - half)
        dst_x2 = min(img_w, cx - half + marker_size_px)
        dst_y2 = min(img_h, cy - half + marker_size_px)

        # Matching source rect (handles clipping at edges)
        src_x1 = dst_x1 - (cx - half)
        src_y1 = dst_y1 - (cy - half)
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            result[dst_y1:dst_y2, dst_x1:dst_x2] = marker_bgr[src_y1:src_y2, src_x1:src_x2]

    return result

# ── session state ─────────────────────────────────────────────────────────────

for k, v in {
    "bg_pil":        None,   # PIL image (original size)
    "roi_pts":       [],     # list of (x, y) in ORIGINAL image space
    "last_click":    None,   # last click dict from streamlit_image_coordinates
    "marked_bg_bgr": None,   # background with ArUco markers stamped on it
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── tabs ──────────────────────────────────────────────────────────────────────

step1, step2 = st.tabs([
    "Step 1 — Upload & Mark Corners",
    "Step 2 — Apply Overlay",
])

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload background + click 4 corners
# ══════════════════════════════════════════════════════════════════════════════
with step1:
    st.subheader("Upload your background image and click the 4 corners of your ROI")
    st.write(
        "Click the **4 corners** of the rectangular area where you want the overlay to appear — "
        "in this order: **Top-Left → Top-Right → Bottom-Right → Bottom-Left**."
    )

    bg_file = st.file_uploader(
        "Upload background image", type=["jpg", "jpeg", "png"], key="bg_upload"
    )

    if bg_file:
        new_img = Image.open(bg_file).convert("RGB")
        # Reset corners when a new image is loaded
        if st.session_state.bg_pil is None or \
                new_img.size != st.session_state.bg_pil.size:
            st.session_state.bg_pil     = new_img
            st.session_state.roi_pts    = []
            st.session_state.last_click = None
        else:
            st.session_state.bg_pil = new_img

    if st.session_state.bg_pil is not None:
        pil_img  = st.session_state.bg_pil
        orig_w, orig_h = pil_img.size
        n_placed = len(st.session_state.roi_pts)

        # Build display-size preview with already-placed dots
        pts_display = []
        scale = DISPLAY_WIDTH / orig_w
        for (ox, oy) in st.session_state.roi_pts:
            pts_display.append((int(ox * scale), int(oy * scale)))

        display_img = pil_img.resize(
            (DISPLAY_WIDTH, int(orig_h * DISPLAY_WIDTH / orig_w)), Image.LANCZOS
        )
        annotated = draw_roi_preview(display_img, pts_display)

        # Status bar
        col_status, col_reset = st.columns([4, 1])
        with col_status:
            if n_placed < 4:
                next_label = CORNER_LABELS[n_placed]
                color      = CORNER_COLORS[n_placed]
                st.markdown(
                    f"**Click {n_placed + 1}/4 —** "
                    f"<span style='color:{color}; font-weight:bold'>{next_label}</span> corner",
                    unsafe_allow_html=True,
                )
            else:
                st.success("All 4 corners placed! Proceed to **Step 2**.")

        with col_reset:
            if st.button("Reset corners"):
                st.session_state.roi_pts    = []
                st.session_state.last_click = None
                st.rerun()

        # Clickable image — coordinates returned in DISPLAY space
        click = streamlit_image_coordinates(
            annotated,
            key="roi_click",
            width=DISPLAY_WIDTH,
        )

        # Accumulate clicks (only process genuinely new ones)
        if click is not None and click != st.session_state.last_click:
            st.session_state.last_click = click
            if n_placed < 4:
                ox, oy = display_to_original(click["x"], click["y"], orig_w, orig_h)
                st.session_state.roi_pts.append((ox, oy))
                st.rerun()

        # Corner summary table
        if st.session_state.roi_pts:
            st.markdown("**Placed corners**")
            cols = st.columns(4)
            for i, col in enumerate(cols):
                if i < len(st.session_state.roi_pts):
                    ox, oy = st.session_state.roi_pts[i]
                    col.markdown(
                        f"<span style='color:{CORNER_COLORS[i]}'>"
                        f"**{CORNER_LABELS[i]}**</span><br>({ox}, {oy})",
                        unsafe_allow_html=True,
                    )
                else:
                    col.markdown(
                        f"<span style='color:{CORNER_COLORS[i]}; opacity:0.4'>"
                        f"**{CORNER_LABELS[i]}**</span><br>—",
                        unsafe_allow_html=True,
                    )

        # ── ArUco marker generation (only after all 4 corners placed) ─────────
        if n_placed == 4:
            st.divider()
            st.markdown("#### Add ArUco Markers to Image")
            st.write(
                "Generate the 4 ArUco markers and stamp them onto the image at your "
                "corner positions. This creates the scene you can use for AR detection."
            )

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                marker_size_pct = st.slider(
                    "Marker size (% of image width)", 3, 20, 7,
                    help="Adjust until the markers look right relative to your image.",
                )
                marker_size_px = max(50, int(orig_w * marker_size_pct / 100))
            with m_col2:
                dict_name = st.selectbox(
                    "ArUco dictionary",
                    list(DICT_OPTIONS.keys()),
                    index=list(DICT_OPTIONS.keys()).index("DICT_6X6_250 (recommended)"),
                    key="marker_dict",
                )

            if st.button("Add Markers to Image", type="primary", key="add_markers_btn"):
                dictionary = cv2.aruco.getPredefinedDictionary(DICT_OPTIONS[dict_name])
                bg_bgr = cv2.cvtColor(np.array(st.session_state.bg_pil), cv2.COLOR_RGB2BGR)
                st.session_state.marked_bg_bgr = place_markers_on_image(
                    bg_bgr, st.session_state.roi_pts, dictionary, DEFAULT_IDS, marker_size_px
                )

            if st.session_state.marked_bg_bgr is not None:
                st.image(
                    st.session_state.marked_bg_bgr,
                    channels="BGR",
                    caption="Background with ArUco markers placed at corners",
                    use_container_width=True,
                )
                st.markdown(
                    img_to_download_link(
                        st.session_state.marked_bg_bgr,
                        "aruco_marked_scene.png",
                        "⬇ Download marked image",
                    ),
                    unsafe_allow_html=True,
                )
                st.info("Proceed to **Step 2** to apply your overlay image.")
    else:
        st.info("Upload a background image above to get started.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Upload overlay and apply AR
# ══════════════════════════════════════════════════════════════════════════════
with step2:
    st.subheader("Upload your overlay image and apply the AR effect")

    ready = (
        st.session_state.bg_pil is not None
        and len(st.session_state.roi_pts) == 4
    )

    if not ready:
        st.warning("Place all 4 corners in Step 1 first.")
    else:
        pil_img  = st.session_state.bg_pil
        bg_bgr   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        roi_pts  = st.session_state.roi_pts
        orig_w, orig_h = pil_img.size

        # Show a small reminder of the ROI
        scale       = DISPLAY_WIDTH / orig_w
        pts_display = [(int(x * scale), int(y * scale)) for x, y in roi_pts]
        display_img = pil_img.resize(
            (DISPLAY_WIDTH, int(orig_h * DISPLAY_WIDTH / orig_w)), Image.LANCZOS
        )
        roi_preview = draw_roi_preview(display_img, pts_display)
        st.image(roi_preview, caption="Your defined ROI (green area)", use_container_width=False, width=400)

        overlay_file = st.file_uploader(
            "Upload the overlay image (will be warped into the green ROI)",
            type=["jpg", "jpeg", "png"],
            key="overlay_upload",
        )

        with st.expander("Fine-tune ROI edges (optional)", expanded=False):
            pad = st.slider(
                "Inset / outset (pixels, negative = expand outward)",
                -50, 50, 0,
                help="Shrink or expand the ROI boundary before warping.",
            )

        if overlay_file:
            overlay_bgr = cv2.cvtColor(
                np.array(Image.open(overlay_file).convert("RGB")), cv2.COLOR_RGB2BGR
            )
            oh, ow = overlay_bgr.shape[:2]

            # Apply padding to destination corners
            if pad != 0:
                cx = sum(p[0] for p in roi_pts) / 4
                cy = sum(p[1] for p in roi_pts) / 4
                dst_pts = []
                for (x, y) in roi_pts:
                    dx = cx - x
                    dy = cy - y
                    dist = max((dx**2 + dy**2) ** 0.5, 1e-6)
                    dst_pts.append((x - int(pad * dx / dist), y - int(pad * dy / dist)))
            else:
                dst_pts = roi_pts

            pts_src = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], dtype=np.float32)
            pts_dst = np.array(dst_pts, dtype=np.float32)

            H, _   = cv2.findHomography(pts_src, pts_dst)
            h_b, w_b = bg_bgr.shape[:2]
            warped = cv2.warpPerspective(overlay_bgr, H, (w_b, h_b))

            mask   = np.zeros_like(bg_bgr)
            cv2.fillPoly(mask, [pts_dst.astype(int)], (255, 255, 255))
            result = cv2.add(
                cv2.bitwise_and(bg_bgr, cv2.bitwise_not(mask)),
                cv2.bitwise_and(warped, mask),
            )

            col1, col2 = st.columns(2)
            col1.image(bg_bgr,  channels="BGR", caption="Original",  use_container_width=True)
            col2.image(result,  channels="BGR", caption="AR Result",  use_container_width=True)

            st.markdown(
                img_to_download_link(result, "ar_result.png", "⬇ Download AR Result"),
                unsafe_allow_html=True,
            )
        else:
            st.info("Upload your overlay image above.")
