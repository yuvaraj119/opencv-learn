# OpenCV Vision Lab — Technical Reference

## Architecture Overview

```
streamlit_app_ss.py          ← Home/welcome page, sidebar nav
utils.py                     ← ensure_model(), show_page_info()
pages/                       ← 31 self-contained feature pages
models/                      ← Config files (checked in) + binaries (gitignored)
```

Streamlit discovers every `.py` file in `pages/` automatically and adds it to the sidebar. Each page is fully self-contained — it imports from `utils.py` only for shared helpers.

---

## Model Download System (`utils.ensure_model`)

All heavy model binaries are downloaded on first use and cached in `models/`. The lookup order:

1. **File already exists** — return path immediately, no network request.
2. **`_DIRECT_URLS` dict** — authoritative upstream source (opencv_zoo, ONNX model zoo, Google Storage).
3. **GitHub Release fallback** — `https://github.com/yuvaraj119/opencv-learn/releases/download/v1.0-models/<filename>`.

Add new models to `_DIRECT_URLS` in `utils.py` before the release exists; the release acts as a mirror once it's created.

---

## Deep Learning Pipelines

### Face Detection — YuNet ONNX

```python
detector = cv2.FaceDetectorYN.create(path, "", (320, 320), score_threshold=0.1)
detector.setInputSize((w, h))
_, faces = detector.detect(img_bgr)
# faces shape: (N, 15)  — face[0:4] = x,y,w,h  |  face[14] = confidence
```

Used by: **Face Detection**, **Depth Blur / Portrait Mode**.

### Object / Person Detection — TF SSD MobileNet V2 (COCO)

```python
net = cv2.dnn.readNet("ssd_mobilenet_frozen_inference_graph.pb",
                       "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
net.setInput(blob)
detections = net.forward()
# Person class ID = 1 (COCO)
```

Used by: **Object Detection**, **Social Distancing Monitoring**.

### Image Classification — GoogLeNet-9 ONNX

```python
net = cv2.dnn.readNet("googlenet-9.onnx")
blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (104, 117, 123))
net.setInput(blob)
scores = net.forward().flatten()
scores -= scores.max()
probs = np.exp(scores) / np.sum(np.exp(scores))   # stable softmax
```

Labels from `classification_classes_ILSVRC2012.txt` (1000 ImageNet classes).

### OCR — Document Mode (Tesseract)

Pre-processing pipeline before `pytesseract.image_to_string`:

```
PIL RGB → BGR → Grayscale → [Gaussian blur] → [Otsu threshold]
```

PSM options exposed to user: Auto (3), Uniform block (6), Single line (7), Single word (8), Sparse (11).

### OCR — Scene Text Mode (DB detector + CRNN)

```
DB detector (DB_TD500_resnet18.onnx)
  → polygon boxes around each text region
  → fourPointsTransform: perspective-warp each box to 100×32
  → CRNN recognizer (text_recognition_CRNN_EN_2021sep.onnx)
  → CTC-greedy decode → word string
```

Vocabulary: `0-9 a-z` (36 chars). Thresholds (binary + polygon) are exposed as sliders.

### Super Resolution

Three model families, each at 2×/3×/4× (or 2×/4×/8× for LapSRN):

| Model | Architecture | Speed | Quality |
|-------|-------------|-------|---------|
| FSRCNN | Compact CNN | Fast | Good |
| ESPCN | Sub-pixel conv | Fast | Good |
| LapSRN | Laplacian pyramid | Slow | Best |

---

## Page Conventions

Every page follows the same structure:

1. `st.set_page_config(...)` — must be the first Streamlit call.
2. `st.title(...)` then `show_page_info("Page_Stem")` — renders the "About" info dialog.
3. `st.file_uploader(type=['jpg','jpeg','png'])` — primary input.
4. Heavy models loaded with `@st.cache_resource()` at **module level** (not inside conditionals) so the cache key is stable across reruns.
5. Expensive inference results stored in `st.session_state` keyed by filename hash to avoid re-running on every widget interaction.

### Colour Space Cheat-Sheet

| From | To | Code |
|------|----|------|
| PIL (RGB) | OpenCV (BGR) | `cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)` |
| OpenCV (BGR) | Streamlit display | `st.image(img, channels='BGR')` |
| OpenCV (BGR) | PIL for download | `Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))` |
| Any PIL | Safe uint8 array | `np.array(img.convert("RGB"), dtype=np.uint8)` |

The last row matters for 1-bit PNGs: PIL returns a bool array which OpenCV 5 rejects (CV_Bool depth error).

---

## MediaPipe Tasks API

`mp.solutions.*` was removed in MediaPipe 0.10.x. All pages use the Tasks API:

```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
)
landmarker = vision.FaceLandmarker.create_from_options(options)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
result = landmarker.detect(mp_image)
```

---

## OpenCV 5 Migration Notes

OpenCV 5.0 removed the Caffe importer entirely. Pages migrated:

| Old (OpenCV 4) | New (OpenCV 5) |
|----------------|----------------|
| `cv2.dnn.readNetFromCaffe(prototxt, caffemodel)` | `cv2.FaceDetectorYN.create(onnx, ...)` |
| `DenseNet_121.caffemodel` | `googlenet-9.onnx` via `cv2.dnn.readNet()` |
| `MobileNetSSD_deploy.caffemodel` (VOC, person=15) | `ssd_mobilenet_frozen_inference_graph.pb` (COCO, person=1) |

`cv2.dnn.readNet(modelFile, configFile)` is the universal loader for ONNX and TF frozen graphs. Note the argument order: model first, config second (opposite of the old `readNetFromCaffe` convention).

---

## Deployment

### Local

```bash
cd streamlit-app
python -m streamlit run streamlit_app_ss.py --server.port 8501
```

### Heroku / Cloud

```bash
sh setup.sh && streamlit run streamlit_app_ss.py
```

`setup.sh` installs system dependencies (Tesseract, etc.). The `Procfile` calls this same sequence.

### System Dependencies (macOS)

```bash
brew install tesseract          # OCR binary
brew install gdal               # Satellite imagery
pip install GDAL==$(gdal-config --version)   # Python bindings must match C library version
```

---

## Model Release

Models are hosted on the [v1.0-models GitHub Release](https://github.com/yuvaraj119/opencv-learn/releases/tag/v1.0-models). To add a new model:

1. Add its direct URL to `_DIRECT_URLS` in `utils.py`.
2. Download it locally (it will auto-download on first page load).
3. Upload to the release: `gh release upload v1.0-models <file> --repo yuvaraj119/opencv-learn --clobber`.
