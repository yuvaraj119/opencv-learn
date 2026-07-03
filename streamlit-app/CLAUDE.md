# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run streamlit_app_ss.py

# Deploy (Heroku-style) — runs setup.sh then launches app
sh setup.sh && streamlit run streamlit_app_ss.py
```

There are no tests or linting configured in this project.

## Architecture

This is a **Streamlit multi-page app** for OpenCV-based image and video processing.

- **Entry point**: `streamlit_app_ss.py` — renders the home/welcome page and sidebar navigation.
- **Feature pages**: `pages/` — each `.py` file is a self-contained Streamlit page. Streamlit discovers them automatically and lists them in the sidebar.
- **Model files**:
  - Root: `deploy.prototxt` + `res10_300x300_ssd_iter_140000_fp16.caffemodel` — SSD ResNet-10 face detection, loaded via relative path.
  - `models/` directory: `MobileNetSSD_deploy.*` (person detection), `DenseNet_121.*` + `classification_classes_ILSVRC2012.txt` (image classification), `FSRCNN/ESPCN/LapSRN_x*.pb` (super resolution).

## Page Conventions

Each page in `pages/` follows the same pattern:

1. **Image input**: `st.file_uploader(..., type=['jpg', 'jpeg', 'png'])`, decoded with `cv2.imdecode` or `PIL.Image.open`.
2. **Color space**: OpenCV works in **BGR**. PIL/NumPy arrays are RGB. Always convert on the boundary:
   - PIL → OpenCV: `cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)`
   - Display BGR with Streamlit: `st.image(img, channels='BGR')`
   - Save/download: convert BGR → RGB before creating a PIL image.
3. **Model caching**: use `@st.cache_resource()` for loading heavy models so they load once per session.
4. **State across widget interactions**: use `st.session_state` to cache expensive computation results (e.g., model detections) and compare uploaded file names/hashes to avoid re-running inference on every slider change.
5. **Download links**: generated as base64-encoded `data:` URIs rendered via `st.markdown(..., unsafe_allow_html=True)`.
6. **Undo/history** (used in `OpenCV_Basic_Operation_Tools.py`): maintain a `st.session_state.history` list of image copies; "Undo" pops the last entry.

## Key Dependencies

| Package | Purpose |
|---|---|
| `opencv-python-headless` | Core image processing (no GUI) |
| `mediapipe` | Pose estimation, facial landmarks |
| `pytesseract` | OCR (requires Tesseract binary installed separately) |
| `GDAL` | GeoTIFF/satellite imagery processing |
| `qrcode[pil]` | QR code generation |
| `streamlit` | Web UI framework |