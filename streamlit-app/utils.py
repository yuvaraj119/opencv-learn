import os
import requests
import streamlit as st

_PAGE_INFO = {
    "artistic_filters": {
        "about": "Apply classic artistic transformations to any photo — pencil sketch, oil painting, watercolor, cartoon, sepia, and more — entirely with OpenCV image processing.",
        "use_cases": [
            "Turn a portrait photo into a pencil sketch or cartoon",
            "Generate stylized versions of product or nature photos",
            "Create artistic content for social media or presentations",
        ],
        "input": "Any JPG or PNG image. Works best with photos that have clear subjects and good lighting.",
        "output": "Side-by-side view of the original and stylized image, with a download link for the result.",
    },
    "augmented_reality_aruco": {
        "about": "Overlay any image onto physical ArUco markers using homography, creating an augmented reality effect visible in photos.",
        "use_cases": [
            "Overlay product labels or artwork onto printed markers",
            "Create AR demos for education or presentations",
            "Test AR marker tracking with custom overlays",
        ],
        "input": "A scene photo containing one or more printed ArUco markers, plus the overlay image you want to place on them.",
        "output": "The scene photo with your overlay image perspective-warped onto each detected ArUco marker.",
    },
    "blink_detection": {
        "about": "Detect whether eyes are open or closed using MediaPipe FaceLandmarker blendshape scores — more reliable than geometric EAR for varied lighting and partial blinks.",
        "use_cases": [
            "Driver drowsiness alerting systems",
            "Accessibility tools triggered by blinks",
            "Attention monitoring in e-learning",
        ],
        "input": "A JPG or PNG image with a clearly visible face (frontal or near-frontal). Works with multiple faces.",
        "output": "Annotated image with eye landmark dots colored green (open) or red (blink), plus per-face blink score metrics.",
    },
    "color_pop": {
        "about": "Keep one selected color range vivid while turning everything else grayscale — the classic 'color pop' effect seen in professional photography.",
        "use_cases": [
            "Highlight a red dress or flower in a portrait",
            "Emphasize product color against a grey background",
            "Create striking photography for social media",
        ],
        "input": "Any JPG or PNG image. Use the HSV sliders to target the color you want to keep (H = hue, S = saturation, V = brightness).",
        "output": "The image with only the selected color range in full color; all other areas converted to grayscale.",
    },
    "deforestation_detection": {
        "about": "Quantify and visualize forest or vegetation coverage in aerial/satellite photos using HSV or BGR color segmentation.",
        "use_cases": [
            "Compare forest coverage between historical aerial photos",
            "Estimate vegetation loss after a wildfire or storm",
            "Quick greenery analysis without GIS tools",
        ],
        "input": "Up to 5 JPG or PNG aerial/satellite images. Images with clear green vegetation work best.",
        "output": "For each image: original, vegetation mask, and percentage coverage. Multi-image: a comparison table and change metrics.",
    },
    "depth_aware_processing": {
        "about": "Estimate a depth map from a single photo using a DNN model, then apply depth-driven effects like selective blur or colorization.",
        "use_cases": [
            "Apply stronger blur to far-away areas of a scene",
            "Visualize scene depth for robotics or autonomous vehicle demos",
            "Create artistic depth-tinted images",
        ],
        "input": "Any JPG or PNG image. Outdoor scenes with clear foreground/background separation give the most interesting results.",
        "output": "The estimated depth map and the processed image with the depth-aware effect applied.",
    },
    "depth_blur_portrait": {
        "about": "Simulate DSLR-style portrait mode by detecting the face, creating a smooth face mask, and blurring the background behind it.",
        "use_cases": [
            "Add portrait-mode blur to photos taken without a depth camera",
            "Create professional-looking headshots from standard photos",
            "Reduce distracting backgrounds in product or event photos",
        ],
        "input": "A JPG or PNG portrait photo with a clearly visible face. Works best with a single dominant face.",
        "output": "Portrait-mode output with the face area sharp and the background softly blurred. Adjustable blur strength.",
    },
    "depth_of_field": {
        "about": "Simulate shallow depth of field by applying a gradient blur that increases from a chosen focal plane toward the edges of the image.",
        "use_cases": [
            "Add cinematic depth to flat product or landscape photos",
            "Simulate a film camera aesthetic on phone photos",
            "Create eye-guiding focus effects for presentations",
        ],
        "input": "Any JPG or PNG image. Use the focal point and blur strength sliders to control where the sharp zone falls.",
        "output": "The image with a smooth blur gradient applied — sharp at the focal plane, increasingly blurred away from it.",
    },
    "digital_signature": {
        "about": "Extract a clean digital signature from a photo of a handwritten signature, producing a transparent-background PNG you can embed in documents.",
        "use_cases": [
            "Digitize a physical signature for contracts and PDFs",
            "Create a reusable signature asset for email or letter templates",
            "Convert any dark-on-white handwriting to a transparent PNG",
        ],
        "input": "A JPG or PNG photo of a signature on a light/white background. Good contrast between pen and paper gives the cleanest result.",
        "output": "A transparent-background PNG with the signature in your chosen color, ready to download and embed.",
    },
    "face_blurring": {
        "about": "Automatically detect all faces in a photo and apply a pixelation or Gaussian blur to anonymize them, protecting individual privacy.",
        "use_cases": [
            "Anonymize crowd photos before publishing on social media",
            "Blur bystander faces in street photography",
            "GDPR-compliant image anonymization pipelines",
        ],
        "input": "Any JPG or PNG image. The SSD face detector works best on images where faces are at least ~30px wide.",
        "output": "The image with all detected faces blurred or pixelated. Adjust the confidence threshold to catch more or fewer faces.",
    },
    "face_controlled_games": {
        "about": "Use your face position and movements as a game controller — the app tracks your face via webcam and maps it to keyboard inputs for browser games.",
        "use_cases": [
            "Hands-free gaming as an accessibility aid",
            "Fun demo of real-time face tracking",
            "Educational tool for human-computer interaction",
        ],
        "input": "A working webcam. Point your face at the camera and follow the on-screen calibration instructions.",
        "output": "Real-time face tracking overlay with instructions on which face movement maps to which game control.",
    },
    "face_detection": {
        "about": "Detect human faces in images using OpenCV's DNN module with a SSD ResNet-10 model trained specifically for face detection.",
        "use_cases": [
            "Count people in crowd photos",
            "Pre-screen images before feeding to a face recognition pipeline",
            "Verify face presence in ID or passport photo validation",
        ],
        "input": "Any JPG or PNG image. Frontal faces are detected most reliably. Adjust the confidence threshold slider.",
        "output": "Bounding boxes drawn around each detected face with confidence score.",
    },
    "foreground_segmentation": {
        "about": "Use MediaPipe's person segmentation model to separate the foreground subject from the background, then apply blur, grayscale, or a custom background image.",
        "use_cases": [
            "Add virtual backgrounds to portrait photos (like Zoom/Teams)",
            "Create dramatic grayscale-background portraits",
            "Remove distracting backgrounds for product or headshot photos",
        ],
        "input": "A JPG or PNG portrait or selfie photo. Optionally a second image to use as the replacement background.",
        "output": "The photo with the chosen background effect (blur / grayscale / custom background) applied behind the detected person.",
    },
    "golf_swing_analysis": {
        "about": "Analyze golf swing biomechanics using MediaPipe PoseLandmarker — measures spine angle, arm bend, knee flex, and shoulder-hip alignment with coaching feedback.",
        "use_cases": [
            "Self-analysis tool for amateur golfers to improve form",
            "Coaching aid to quickly flag common posture mistakes",
            "Compare address positions across different practice sessions",
        ],
        "input": "A JPG or PNG golf swing photo — face-on (front view) or down-the-line (side view) with the full body visible.",
        "output": "Annotated pose skeleton with spine angle line, joint angle measurements, and coaching notes for each key position.",
    },
    "human_pose_estimation": {
        "about": "Detect and visualize 33 full-body landmarks using MediaPipe PoseLandmarker Heavy, and compute joint angles at elbows, knees, and hips.",
        "use_cases": [
            "Fitness form analysis — check squat depth, push-up alignment",
            "Dance or yoga pose verification",
            "Sports performance and biomechanics research",
        ],
        "input": "A JPG or PNG image with a clearly visible person. Full or half-body shots work best; the model handles up to 4 people.",
        "output": "Pose skeleton overlay on the image plus a metrics panel with joint angles (elbow, knee, hip) for the first detected person.",
    },
    "image_classification": {
        "about": "Classify any image into one of 1000 ImageNet categories using DenseNet-121 run entirely through OpenCV's DNN module — no cloud API needed.",
        "use_cases": [
            "Quickly identify what object or animal is in a photo",
            "Verify image content for automated tagging pipelines",
            "Teach students about CNN-based image classification",
        ],
        "input": "Any JPG or PNG image. Works best for the 1000 standard ImageNet categories (animals, vehicles, food, household objects, etc.).",
        "output": "Top-K predicted class names with confidence percentages and progress bars. Adjust K with the slider.",
    },
    "image_restoration": {
        "about": "Reduce noise and restore clarity to degraded images using four classic denoising algorithms: fast non-local means, median, bilateral, and Gaussian.",
        "use_cases": [
            "Clean up low-light or high-ISO camera photos",
            "Remove salt-and-pepper noise from scanned documents",
            "Pre-process images before feeding them to a detection model",
        ],
        "input": "Any JPG or PNG image — ideally one with visible noise, graininess, or compression artifacts.",
        "output": "Side-by-side comparison of the original and denoised image with a download link.",
    },
    "intruder_detection": {
        "about": "Detect motion and intruders in video footage using KNN background subtraction, contour analysis, and bounding box annotation.",
        "use_cases": [
            "Analyze security camera footage for unauthorized movement",
            "Detect animals entering a restricted outdoor area",
            "Count moving objects in a traffic or wildlife video",
        ],
        "input": "An MP4, MOV, or AVI video file. Works best with a mostly static camera and a plain background.",
        "output": "Processed video with bounding boxes around moving regions plus a quad-view showing raw mask, eroded mask, and annotated frames.",
    },
    "lane_detection": {
        "about": "Detect road lane markings in images using Canny edge detection and Hough line transform, then overlay the detected lanes on the original image.",
        "use_cases": [
            "Prototype lane-keeping assist algorithms",
            "Analyze dashcam footage for lane markings",
            "Teach autonomous vehicle computer vision concepts",
        ],
        "input": "A JPG or PNG road/dashcam image. Clear lane markings on a reasonably flat road give the best results.",
        "output": "The original image with detected lane lines overlaid in color.",
    },
    "ocr": {
        "about": "Extract printed or handwritten text from images using Tesseract OCR with optional preprocessing (thresholding and blur) to improve accuracy.",
        "use_cases": [
            "Digitize text from printed documents or book pages",
            "Extract data from receipts, invoices, or forms",
            "Read text from screenshots or signage photos",
        ],
        "input": "A JPG or PNG image containing text. High-contrast, clearly focused text gives the most accurate results.",
        "output": "The preprocessed image used for OCR and the extracted text in a copyable text area.",
    },
    "object_detection": {
        "about": "Detect 80+ everyday objects in images using SSD MobileNet V2 trained on the COCO dataset, running via OpenCV's DNN module.",
        "use_cases": [
            "Inventory counting from shelf or warehouse photos",
            "Safety checks — detect people, vehicles, or equipment in a scene",
            "Pre-labeling images for a custom ML training dataset",
        ],
        "input": "Any JPG or PNG image. The model detects 80 COCO categories including people, animals, vehicles, and common objects.",
        "output": "Image with bounding boxes and class labels for each detected object. Adjust the confidence threshold slider.",
    },
    "object_tracking": {
        "about": "Track a selected object across video frames using OpenCV's built-in tracking algorithms (CSRT, KCF, MOSSE, etc.).",
        "use_cases": [
            "Track a ball or player in sports footage",
            "Monitor a vehicle or person across security camera frames",
            "Prototype tracking pipelines for robotics or drones",
        ],
        "input": "An MP4, MOV, or AVI video file. Draw a bounding box around the object to track in the first frame.",
        "output": "Video with a bounding box following the tracked object across all frames.",
    },
    "basic_operations": {
        "about": "An interactive image editor built on OpenCV — apply resize, crop, rotate, draw shapes, add text, blur, detect edges, morphological ops, thresholding, and contours — all with undo/reset.",
        "use_cases": [
            "Quick non-destructive edits without installing desktop software",
            "Teach students individual OpenCV operations interactively",
            "Chain multiple operations to prepare images for a pipeline",
        ],
        "input": "Any JPG or PNG image. Upload once and apply as many sequential operations as you like using the tabs.",
        "output": "Live preview of the current image state after each operation, with undo and full reset controls, plus a download link.",
    },
    "panorama_creation": {
        "about": "Stitch multiple overlapping photos into a single seamless wide-angle panorama using OpenCV's feature-based image stitching pipeline.",
        "use_cases": [
            "Stitch landscape or cityscape photos taken by panning a phone",
            "Create wide-field product or real estate shots",
            "Combine microscope or aerial tiles into a single view",
        ],
        "input": "2 or more JPG or PNG images of the same scene with 30–50% overlap between adjacent frames. Upload in left-to-right order.",
        "output": "A single stitched panorama image with download link.",
    },
    "qr_detection": {
        "about": "Scan and decode QR codes from uploaded images using OpenCV's built-in QR code detector.",
        "use_cases": [
            "Verify QR codes printed in documents or on packaging",
            "Batch-decode QR codes from event or product photos",
            "Test QR code readability before printing",
        ],
        "input": "A JPG or PNG image containing one or more QR codes. Ensure the QR code is sharp and well-lit.",
        "output": "The decoded text or URL for each detected QR code, with the QR region highlighted on the image.",
    },
    "qr_generation": {
        "about": "Generate QR codes from any text, URL, or data string with adjustable size and error correction level.",
        "use_cases": [
            "Generate a QR code linking to your website or portfolio",
            "Create QR codes for business cards or event flyers",
            "Encode contact info, Wi-Fi credentials, or short messages",
        ],
        "input": "Type any text or URL into the input field. Choose error correction level (L/M/Q/H) and QR size.",
        "output": "A downloadable PNG QR code image ready to embed in documents or print.",
    },
    "satellite_imagery": {
        "about": "Load multi-band GeoTIFF satellite imagery, compute NDVI vegetation index, generate vegetation masks, and visualize individual spectral bands.",
        "use_cases": [
            "Monitor seasonal vegetation change with Landsat data",
            "Assess crop health or reforestation progress",
            "Calculate green coverage for urban planning or environmental reports",
        ],
        "input": "One or more GeoTIFF (.tif) files with at least 4 bands (Red band 3 + NIR band 4). Landsat 5/7/8 exports work well.",
        "output": "NDVI index map (colorized), vegetation binary mask with coverage percentage, and optional individual band visualizations.",
    },
    "social_distancing": {
        "about": "Detect people in images using MobileNetSSD and flag pairs standing closer than a configurable minimum distance threshold.",
        "use_cases": [
            "Analyze crowd photos for social distancing compliance",
            "Retail or office safety audits from overhead camera snapshots",
            "Prototype proximity alert systems",
        ],
        "input": "A JPG or PNG image with people visible. Overhead or slight-angle camera views give the most accurate distance estimates.",
        "output": "Annotated image with green bounding boxes for all detected people and red lines connecting pairs that violate the distance threshold.",
    },
    "super_resolution": {
        "about": "Upscale low-resolution images using deep learning super resolution models (FSRCNN, ESPCN, LapSRN) at 2×, 3×, or 4×/8× scale — far sharper than bicubic interpolation.",
        "use_cases": [
            "Enhance old or low-res photos before printing",
            "Upscale thumbnails or compressed images for analysis",
            "Compare DL super resolution quality across model architectures",
        ],
        "input": "Any JPG or PNG image. For best results keep the input under 256px on the longest side (use the downscale slider), then let SR upscale it.",
        "output": "Three-column comparison: original input, bicubic upscale, and deep learning SR result — all at the same output resolution.",
    },
    "virtual_billboard": {
        "about": "Replace any flat surface (billboard, canvas, screen, wall) in a photo with your own image using perspective homography transformation.",
        "use_cases": [
            "Mockup your ad creative on a Times Square billboard",
            "Preview artwork hanging on a gallery wall",
            "Swap a TV/monitor screen with a custom display",
        ],
        "input": "A scene photo containing a flat surface, plus your replacement image. Define the four corners of the target surface.",
        "output": "The scene photo with your replacement image perspective-warped to exactly fit the selected surface region.",
    },
    "watermarking": {
        "about": "Overlay a logo or watermark image onto a base photo with full control over position, scale, and transparency (supports PNG alpha channels).",
        "use_cases": [
            "Brand your photography portfolio with a logo watermark",
            "Add a copyright or attribution to images before sharing",
            "Create consistent branded thumbnails for social media",
        ],
        "input": "A base image (JPG or PNG) and a logo/watermark image (PNG with transparency recommended).",
        "output": "The base image with the logo composited onto it at your chosen position and opacity. Download as PNG.",
    },
}

_FILE_TO_KEY = {
    "Artistic_Image_Filters":        "artistic_filters",
    "Augmented_Reality_ArUco":       "augmented_reality_aruco",
    "Blink_Detection":               "blink_detection",
    "Color_Pop":                     "color_pop",
    "Deforestation_Detection":       "deforestation_detection",
    "Depth_Aware_Processing":        "depth_aware_processing",
    "Depth_Blur_Portrait":           "depth_blur_portrait",
    "Depth_of_Field_Simulation":     "depth_of_field",
    "Digital_Signature":             "digital_signature",
    "Face_Blurring_Privacy":         "face_blurring",
    "Face_Controlled_Games":         "face_controlled_games",
    "Face_Detection":                "face_detection",
    "Foreground_Segmentation":       "foreground_segmentation",
    "Golf_Swing_Analysis":           "golf_swing_analysis",
    "Human_Pose_Estimation":         "human_pose_estimation",
    "Image_Classification":          "image_classification",
    "Image_Restoration":             "image_restoration",
    "Intruder_Detection":            "intruder_detection",
    "Lane_Detection":                "lane_detection",
    "OCR":                           "ocr",
    "Object_Detection":              "object_detection",
    "Object_Tracking":               "object_tracking",
    "OpenCV_Basic_Operation_Tools":  "basic_operations",
    "Panorama_Creation":             "panorama_creation",
    "QR_Code_Detection":             "qr_detection",
    "QR_Code_Generation":            "qr_generation",
    "Satellite_Imagery_Analysis":    "satellite_imagery",
    "Social_Distancing_Monitoring":  "social_distancing",
    "Super_Resolution":              "super_resolution",
    "Virtual_Billboard":             "virtual_billboard",
    "Watermarking":                  "watermarking",
}


def show_page_info(page_key: str) -> None:
    """Show an 'About this tool' button that opens a modal dialog.

    Args:
        page_key: Either a _PAGE_INFO key (e.g. "face_detection") or the page
                  filename stem (e.g. "Face_Detection") — both are accepted.
    """
    key = _FILE_TO_KEY.get(page_key, page_key)
    info = _PAGE_INFO.get(key)
    if info is None:
        return

    @st.dialog("ℹ️ About this tool")
    def _show_dialog():
        st.markdown("#### What it does")
        st.write(info["about"])

        st.markdown("#### Use cases")
        for uc in info["use_cases"]:
            st.write(f"• {uc}")

        st.markdown("#### What you need to input")
        st.info(info["input"])

        st.markdown("#### Expected output")
        st.success(info["output"])

    if st.button("ℹ️ About this tool", key=f"about_{key}"):
        _show_dialog()


# ── Model auto-downloader ─────────────────────────────────────────────────────

_RELEASE_BASE = (
    "https://github.com/yuvaraj119/opencv-learn/releases/download/v1.0-models"
)

# Models with a stable authoritative URL — downloaded directly, not from GitHub Releases
_DIRECT_URLS: dict[str, str] = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet"
        "/face_detection_yunet_2023mar.onnx"
    ),
    "googlenet-9.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/classification"
        "/inception_and_googlenet/googlenet/model/googlenet-9.onnx"
    ),
    "FSRCNN_x2.pb": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
    "FSRCNN_x3.pb": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb",
    "FSRCNN_x4.pb": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
    "ESPCN_x2.pb": "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb",
    "ESPCN_x3.pb": "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb",
    "ESPCN_x4.pb": "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb",
    "LapSRN_x2.pb": "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x2.pb",
    "LapSRN_x4.pb": "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb",
    "LapSRN_x8.pb": "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x8.pb",
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker"
        "/face_landmarker/float16/1/face_landmarker.task"
    ),
    "person_segmenter.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter"
        "/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    ),
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
        "/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    ),
    "ssd_mobilenet_frozen_inference_graph.pb": (
        "https://opencv-courses.s3.us-west-2.amazonaws.com/ssd_mobilenet_frozen_inference_graph.pb"
    ),
}

# Maps filename → approximate size in bytes (for progress display)
_MODEL_REGISTRY: dict[str, int] = {
    "face_detection_yunet_2023mar.onnx":             388_000,
    "googlenet-9.onnx":                              26_000_000,
    "ssd_mobilenet_frozen_inference_graph.pb":       66_000_000,
    "face_landmarker.task":                           3_600_000,
    "person_segmenter.tflite":                        2_700_000,
    "pose_landmarker_heavy.task":                    29_000_000,
    "LapSRN_x2.pb":                                   1_300_000,
    "LapSRN_x4.pb":                                   2_600_000,
    "LapSRN_x8.pb":                                   3_900_000,
    "FSRCNN_x2.pb":                                      38_000,
    "FSRCNN_x3.pb":                                      39_000,
    "FSRCNN_x4.pb":                                      41_000,
    "ESPCN_x2.pb":                                       84_000,
    "ESPCN_x3.pb":                                       90_000,
    "ESPCN_x4.pb":                                       98_000,
}

_MODELS_DIR = "models"


def ensure_model(filename: str) -> str:
    """Return the local path to a model file, downloading it from GitHub
    Releases if it is not already present on disk.

    Shows a Streamlit progress indicator while downloading. Safe to call
    inside @st.cache_resource loaders — the download only runs once per
    missing file per app restart.
    """
    os.makedirs(_MODELS_DIR, exist_ok=True)
    local_path = os.path.join(_MODELS_DIR, filename)

    if os.path.exists(local_path):
        return local_path

    url = _DIRECT_URLS.get(filename) or f"{_RELEASE_BASE}/{filename}"
    approx_size = _MODEL_REGISTRY.get(filename, 0)
    size_mb = f"{approx_size / 1_000_000:.0f} MB" if approx_size else "?"

    with st.status(f"Downloading model: {filename} ({size_mb})", expanded=True) as status:
        st.write(f"Source: {url}")
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            total = int(response.headers.get("content-length", approx_size or 1))
            bar = st.progress(0, text="Starting download…")
            downloaded = 0
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65_536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = min(downloaded / total, 1.0)
                    bar.progress(pct, text=f"{downloaded / 1_000_000:.1f} / {total / 1_000_000:.1f} MB")
            status.update(label=f"✅ {filename} ready", state="complete", expanded=False)
        except Exception as exc:
            status.update(label=f"❌ Failed to download {filename}", state="error")
            st.error(str(exc))
            raise

    return local_path
