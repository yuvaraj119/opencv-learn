# OpenCV Vision Lab — Streamlit App

An interactive web application built with **Streamlit** and **OpenCV** that lets you explore 31 computer vision techniques directly in your browser — no code required.

## Tech Stack

| Layer | Library |
|-------|---------|
| UI framework | [Streamlit](https://streamlit.io) |
| Computer vision | [OpenCV](https://opencv.org) 4.x |
| Deep learning inference | OpenCV DNN, [MediaPipe](https://mediapipe.dev) Tasks API |
| Image I/O | [Pillow](https://pillow.readthedocs.io) |
| Satellite imagery | [GDAL](https://gdal.org) |
| OCR | [Tesseract](https://github.com/tesseract-ocr/tesseract) via pytesseract |
| QR codes | [qrcode](https://github.com/lincolnloop/python-qrcode) |

---

## Features

### Creative & Artistic
| Page | What it does |
|------|-------------|
| **Artistic Image Filters** | Black & white, sepia, vignette, pencil sketch, watercolor, oil paint, pixelate, glitch, and more |
| **Color Pop** | Keep one hue, desaturate everything else — isolate a subject by colour |
| **Depth Blur / Portrait Mode** | Face-aware background blur using SSD ResNet-10 depth estimation |
| **Depth of Field Simulation** | Radial / lens-style blur with adjustable aperture and focus point |
| **Virtual Billboard** | Replace a planar surface with any image using homography |

### AI & Deep Learning
| Page | What it does |
|------|-------------|
| **Face Detection** | SSD ResNet-10 face detector with confidence slider and bounding-box overlay |
| **Object Detection** | SSD MobileNet V2 trained on COCO — detects 80 object categories |
| **Image Classification** | DenseNet-121 top-5 predictions with confidence bars (1000 ImageNet classes) |
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
| **Social Distancing Monitoring** | MobileNet SSD people detector with distance-violation alerts |

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
| **OCR** | Tesseract text extraction with threshold and blur pre-processing |
| **QR Code Detection** | Decode any QR code and display its payload |
| **QR Code Generation** | Generate styled QR codes with custom colours and embedded logos |
| **Panorama Creation** | ORB feature matching + RANSAC homography image stitching |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Tesseract OCR binary: `brew install tesseract` (macOS) or `apt install tesseract-ocr` (Linux)
- GDAL: `brew install gdal` (macOS) — only needed for the Satellite Imagery page

### Install

```bash
cd streamlit-app
pip install -r requirements.txt
```

### Download Model Files

Large model binaries are excluded from git. Place them in `streamlit-app/models/`:

| File | Used by |
|------|---------|
| `res10_300x300_ssd_iter_140000_fp16.caffemodel` | Face Detection, Depth Blur |
| `MobileNetSSD_deploy.caffemodel` | Object Detection, Social Distancing |
| `DenseNet_121.caffemodel` | Image Classification |
| `ssd_mobilenet_frozen_inference_graph.pb` | Object Detection |
| `face_landmarker.task` | Blink Detection |
| `person_segmenter.tflite` | Foreground Segmentation |
| `pose_landmarker_heavy.task` | Pose Estimation, Golf Swing |
| `LapSRN_x2.pb`, `LapSRN_x4.pb`, `LapSRN_x8.pb` | Super Resolution |
| `FSRCNN_x2.pb`, `FSRCNN_x3.pb`, `FSRCNN_x4.pb` | Super Resolution |
| `ESPCN_x2.pb`, `ESPCN_x3.pb`, `ESPCN_x4.pb` | Super Resolution |

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
├── utils.py                 # Shared show_page_info() dialog helper
├── pages/                   # One .py file per feature page (31 total)
├── models/                  # Model config files (binaries gitignored)
│   ├── deploy.prototxt
│   ├── MobileNetSSD_deploy.prototxt
│   ├── DenseNet_121.prototxt
│   ├── classification_classes_ILSVRC2012.txt
│   ├── coco_class_labels.txt
│   └── ssd_mobilenet_v2_coco_2018_03_29.pbtxt
├── requirements.txt
├── setup.sh                 # Heroku deployment setup
└── Procfile
```

---

## OpenCV Concepts Covered

| Concept | Pages |
|---------|-------|
| DNN inference (Caffe, TensorFlow) | Face Detection, Object Detection, Image Classification, Social Distancing |
| Feature detection & matching (ORB, SIFT) | Panorama Creation |
| Homography & perspective transform | Virtual Billboard, Augmented Reality |
| Hough transforms | Lane Detection |
| Background subtraction (MOG2) | Intruder Detection |
| Colour space operations (HSV, LAB, BGR) | Deforestation Detection, Color Pop, Artistic Filters |
| Morphological operations | OCR pre-processing, Foreground Segmentation |
| Image blending & masking | Watermarking, Depth Blur |
| Frequency domain (DFT) | Image Restoration |
| ArUco marker generation & detection | Augmented Reality |
| MediaPipe Tasks API | Pose Estimation, Blink Detection, Foreground Segmentation |

---

## References

- [Mastering OpenCV with Python](https://opencv.org/university/courses/mastering-opencv-with-python/?utm_source=ocv&utm_medium=midblog&utm_campaign=AdvancecareerOpenCV) — OpenCV University course this project is based on
- [Certificate of Completion](https://courses.opencv.org/certificates/f26b4185996d412fb9a5a2d8454d4442) — Yuvaraj Yadav

## License

Apache 2.0 — see [LICENSE](LICENSE).
