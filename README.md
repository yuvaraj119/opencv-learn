# OpenCV Vision Lab — Streamlit App

An interactive web application built with **Streamlit** and **OpenCV** that lets you explore 31 computer vision techniques directly in your browser — no code required.

## Tech Stack

| Layer | Library |
|-------|---------|
| UI framework | [Streamlit](https://streamlit.io) |
| Computer vision | [OpenCV](https://opencv.org) 5.x |
| Deep learning inference | OpenCV DNN (ONNX, TF), [MediaPipe](https://mediapipe.dev) Tasks API |
| Image I/O | [Pillow](https://pillow.readthedocs.io) |
| Satellite imagery | [GDAL](https://gdal.org) |
| OCR | [Tesseract](https://github.com/tesseract-ocr/tesseract) via pytesseract + OpenCV DB/CRNN |
| QR codes | [qrcode](https://github.com/lincolnloop/python-qrcode) |

---

## Features

### Creative & Artistic
| Page | What it does |
|------|-------------|
| **Artistic Image Filters** | Black & white, sepia, vignette, pencil sketch, watercolor, oil paint, pixelate, glitch, and more |
| **Color Pop** | Keep one hue, desaturate everything else — isolate a subject by colour |
| **Depth Blur / Portrait Mode** | Face-aware background blur using YuNet ONNX face detector |
| **Depth of Field Simulation** | Radial / lens-style blur with adjustable aperture and focus point |
| **Virtual Billboard** | Replace a planar surface with any image using homography |

### AI & Deep Learning
| Page | What it does |
|------|-------------|
| **Face Detection** | YuNet ONNX face detector with confidence slider and bounding-box overlay |
| **Object Detection** | SSD MobileNet V2 (TF, COCO) — detects 80 object categories |
| **Image Classification** | GoogLeNet-9 ONNX top-5 predictions with confidence bars (1000 ImageNet classes) |
| **Super Resolution** | Upscale images 2×/3×/4× with FSRCNN, ESPCN, or LapSRN models |
| **Image Restoration** | Denoise with Gaussian, median, bilateral, or NL-Means filtering |

### Human Analysis
| Page | What it does |
|------|-------------|
| **Human Pose Estimation** | MediaPipe PoseLandmarker — 33 body landmarks with joint-angle overlays |
| **Golf Swing Analysis** | Biomechanical feedback: spine angle, elbow flex, knee bend, shoulder tilt |
| **Blink Detection** | MediaPipe FaceLandmarker blendshape scores for real-time blink counting |
| **Face Blurring for Privacy** | Auto-detect and pixelate/blur all faces in an image |
| **Face-Controlled Games** | Head-movement game controller using facial landmark tracking |

### Segmentation & Background
| Page | What it does |
|------|-------------|
| **Foreground Segmentation** | MediaPipe ImageSegmenter — clean person/background separation |
| **Depth Aware Processing** | Depth-guided sharpening, contrast, and colour adjustment |
| **Object Tracking** | Frame-by-frame tracker with CSRT / KCF / MOSSE algorithms |
| **Social Distancing Monitoring** | TF SSD MobileNet V2 person detector with distance-violation alerts |

### Specialised Vision
| Page | What it does |
|------|-------------|
| **Augmented Reality (ArUco)** | Click 4 corners on any image to define an ROI, place ArUco markers, then warp an overlay into it |
| **Lane Detection** | Canny + Hough transform lane-line overlay for road images |
| **Deforestation Detection** | HSV colour segmentation to highlight vegetation loss between two images |
| **Satellite Imagery Analysis** | GeoTIFF ingestion, NDVI calculation, vegetation-mask overlay |
| **Intruder Detection** | Background subtraction (MOG2) to flag moving objects in a video feed |

### Utility Tools
| Page | What it does |
|------|-------------|
| **OpenCV Basic Operations** | Full image-editing pipeline: crop, rotate, flip, resize, brightness/contrast, blur, sharpen, colour channels, histograms — with undo history |
| **Watermarking** | Overlay a logo/watermark with position, scale, and alpha controls |
| **Digital Signature** | Extract a handwritten signature and export it as a transparent PNG |
| **OCR** | Dual-mode: Document/Form text via Tesseract; Signboard/Scene text via OpenCV DB detector + CRNN recognizer |
| **QR Code Detection** | Decode any QR code and display its payload |
| **QR Code Generation** | Generate styled QR codes with custom colours and embedded logos |
| **Panorama Creation** | ORB feature matching + RANSAC homography image stitching |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Tesseract OCR binary: `brew install tesseract` (macOS) or `apt install tesseract-ocr` (Linux) — only needed for the OCR page
- GDAL: `brew install gdal` (macOS), then `pip install GDAL==$(gdal-config --version)` — only needed for the Satellite Imagery page

### Install

```bash
cd streamlit-app
pip install -r requirements.txt
```

> **GDAL note:** The Python GDAL bindings must match the system library version exactly. Use `pip install GDAL==$(gdal-config --version)` rather than plain `pip install GDAL`.

### Model Files

Large model binaries are excluded from git and downloaded automatically on first use via `utils.ensure_model()`. They are cached in `streamlit-app/models/` and fetched from the [v1.0-models GitHub release](https://github.com/yuvaraj119/opencv-learn/releases/tag/v1.0-models).

| File | Used by |
|------|---------|
| `face_detection_yunet_2023mar.onnx` | Face Detection, Depth Blur / Portrait |
| `googlenet-9.onnx` | Image Classification |
| `DB_TD500_resnet18.onnx` | OCR — Scene Text Detection |
| `text_recognition_CRNN_EN_2021sep.onnx` | OCR — Scene Text Recognition |
| `ssd_mobilenet_frozen_inference_graph.pb` | Object Detection, Social Distancing |
| `face_landmarker.task` | Blink Detection |
| `person_segmenter.tflite` | Foreground Segmentation |
| `pose_landmarker_heavy.task` | Pose Estimation, Golf Swing |
| `LapSRN_x2/4/8.pb`, `FSRCNN_x2/3/4.pb`, `ESPCN_x2/3/4.pb` | Super Resolution |

### Run

```bash
# From inside streamlit-app/
python -m streamlit run streamlit_app_ss.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
streamlit-app/
├── streamlit_app_ss.py      # Home page and sidebar navigation
├── utils.py                 # ensure_model() download helper + show_page_info() dialog
├── pages/                   # One .py file per feature page (31 total)
├── models/                  # Model configs + auto-downloaded binaries (binaries gitignored)
│   ├── deploy.prototxt
│   ├── MobileNetSSD_deploy.prototxt
│   ├── ssd_mobilenet_v2_coco_2018_03_29.pbtxt
│   ├── coco_class_labels.txt
│   ├── alphabet_94.txt              # CRNN vocabulary for scene OCR
│   └── classification_classes_ILSVRC2012.txt
├── docs/
│   └── docs.md              # Architecture and model pipeline reference
├── requirements.txt
├── setup.sh                 # Heroku deployment setup
└── Procfile
```

---

## OpenCV Concepts Covered

| Concept | Pages |
|---------|-------|
| DNN inference (ONNX, TensorFlow) | Face Detection, Object Detection, Image Classification, Social Distancing, OCR |
| Feature detection & matching (ORB, SIFT) | Panorama Creation |
| Homography & perspective transform | Virtual Billboard, Augmented Reality |
| Hough transforms | Lane Detection |
| Background subtraction (MOG2) | Intruder Detection |
| Colour space operations (HSV, LAB, BGR) | Deforestation Detection, Color Pop, Artistic Filters |
| Morphological operations | OCR pre-processing, Foreground Segmentation |
| Image blending & masking | Watermarking, Depth Blur |
| Frequency domain (DFT) | Image Restoration |
| ArUco marker generation & detection | Augmented Reality |
| Text detection & recognition (DB + CRNN) | OCR — Signboard / Scene Text mode |
| MediaPipe Tasks API | Pose Estimation, Blink Detection, Foreground Segmentation |

---

## References

- [Mastering OpenCV with Python](https://opencv.org/university/courses/mastering-opencv-with-python/?utm_source=ocv&utm_medium=midblog&utm_campaign=AdvancecareerOpenCV) — OpenCV University course this project is based on
- [Certificate of Completion](https://courses.opencv.org/certificates/f26b4185996d412fb9a5a2d8454d4442) — Yuvaraj Yadav

## License

Apache 2.0 — see [LICENSE](LICENSE).
