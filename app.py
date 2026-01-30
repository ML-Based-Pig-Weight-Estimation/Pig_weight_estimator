import streamlit as st
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import json
import cv2 
import os
from pathlib import Path
from ultralytics import YOLO
import rawpy
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Pig Weight Estimator",
    page_icon="🐷",
    layout="wide"


)
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #A47DAB;
        
    
    }

    /* Reduce side padding so it truly feels full width */
    .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🐷 Pig Weight Estimator")
st.markdown("---")

st.subheader("Upload or capture a pig image")
st.write("You can upload an image or take a photo using your device camera.")
st.write("Please make sure the image clearly shows the entire body parts of the pig for accurate weight estimation.")

# Hide Streamlit UI elements (optional)
# st.markdown(
#     """
#     <style>
#     header{visibility: hidden;}
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     ._terminalButton_rix23_138{visibility: hidde;}
#     </style>


#     """,
#     unsafe_allow_html=True
# )

# --- Sidebar ---
st.sidebar.subheader("About the App")
st.sidebar.markdown(
    """
    - 🐖 Upload or capture a pig image  
    - ⚡ Instant weight estimation  
    - 🤖 ML-powered prediction
    """
)


st.sidebar.markdown(
    """
    ---

    <style>
    /* Make sidebar full height and flex */
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }

    /* Footer styling */
    .sidebar-footer {
        margin-top: auto;       /* Push to bottom */
        width: 100%;
        background-color: #f8f9fa;
        color: #555;
        text-align: center;
        padding: 5px ,5px;
        font-size: 14px;
        border-top: 1px solid #e0e0e0;
    }
    </style>

    <div class="sidebar-footer">
        👨‍💻 Developed by <strong>Leonard Niyitegeka</strong>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Image input section ---
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])

with tab1:
    uploaded_img = st.file_uploader(
        "Upload an image of a pig",
        type=["jpg", "jpeg", "png", "dng"]
    )

with tab2:
    camera_img = st.camera_input("Capture a pig image")
    

st.markdown(
    """
    <style>
    /* Tab container */
    div[data-baseweb="tab-list"] {
        gap: 8px;
    }

    /* Individual tabs */
    button[data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }

    /* Hover effect for all tabs */
    button[data-baseweb="tab"]:hover {
        color: green;
    }

    /* Active tab (default state) */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: green;
    }

    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"]:hover,
    button[data-baseweb="tab"][aria-selected="true"]:focus,
    button[data-baseweb="tab"][aria-selected="true"]:active {
        color: green !important;
    }

    /* Active tab indicator (the line underneath) */
    div[data-baseweb="tab-highlight"] {
        background-color: green;
        height: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .camera-box {
        max-width: 500px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if uploaded_img is not None:
    pig_img = uploaded_img
elif camera_img is not None:
    pig_img = camera_img
else:
    pig_img =None

# --- Display image ---
if pig_img is not None:
    print(pig_img.name)
    st.success("Image received successfully!")
    st.text("Here is the image you provided:")
    st.image(pig_img, caption="Pig Image", use_container_width=True)

    st.markdown("---")

    # --- Results section ---

    col1, col2 = st.columns(2)

    with col1:
        st.write("Weight Estimation Progress")

    with col2:
        main_bar=st.progress(0)
        status_text=st.empty()
    # loading the weights of the trained model weights and metadata

    MODEL_PATH = "catboost_weight_model.cbm"
    META_PATH = "model_meta.json"

    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    FEATURES = meta["features"]

    # Force CPU only
    device = "cpu"

    yolo_detect = YOLO("yolo11x.pt")
    yolo_detect.to(device)

    yolo_seg = YOLO("yolo11x-seg.pt")
    yolo_seg.to(device)

    def dng_to_jpg(input_path):
        folder = input_path.split("/")[2]
        img_file = input_path.split("/")[3].split(".")[0]
        input_path = Path(input_path)
        output_dir = Path(f'jpg_images/{folder}/')
        if output_dir.exists() and not output_dir.is_dir():
            os.remove(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{img_file}.jpg"
        with rawpy.imread(str(input_path)) as raw:
            rgb = raw.postprocess()
        Image.fromarray(rgb).save(output_file, "JPEG")
        return str(output_file)
    def load_image_from_streamlit(uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    def extract_features(mask):
    
        if mask is None:
            return []

        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        A = np.sum(mask_bin == 255)
        SR = A / (mask.shape[0] * mask.shape[1])

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return []

        cnt = contours[0]

        # ─── New Feature: Length of the contour ───────────────────────────
        contour_length = len(cnt)

        P = cv2.arcLength(cnt, True)

        rect = cv2.minAreaRect(cnt)
        (_, _), (width, height), _ = rect
        BL = max(width, height)
        BW = min(width, height)
        aspect_ratio = width / height if height > 0 else 0

        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (_, axes, _) = ellipse
            majoraxis = max(axes)
            minoraxis = min(axes)
            E = np.sqrt(1 - (minoraxis / majoraxis) ** 2)
        else:
            E = 0

        hull = cv2.convexHull(cnt)
        A_hull = cv2.contourArea(hull)
        solidity = A / A_hull if A_hull > 0 else 0

        x2, y2, w2, h2 = cv2.boundingRect(cnt)
        rect_area = w2 * h2
        extent = A / rect_area if rect_area > 0 else 0

        compactness = (P ** 2) / (4 * np.pi * A) if A > 0 else 0
        circularity = (4 * np.pi * A) / (P ** 2) if P > 0 else 0
        elongation = BL / BW if BW > 0 else 0

        hu = cv2.HuMoments(cv2.moments(cnt)).flatten()

        return [
            SR, A, P, BL, BW, E,
            A_hull, solidity, extent,
            compactness, circularity, elongation,
            aspect_ratio,
            *hu, contour_length
        ]


    img=load_image_from_streamlit(pig_img)
    main_bar.progress(10)
    status_text.text("10%")
    final_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    detect_results = yolo_detect.predict(
        img,
            conf=0.3,
            verbose=False,
            device=device
        )

    for result in detect_results:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        for bbox in boxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            crop = img[y1:y2, x1:x2]

            seg_results = yolo_seg.predict(
                crop,
                conf=0.3,
                verbose=False,
                device=device
            )
            main_bar.progress(50)
            status_text.text("50%")
            for seg_res in seg_results:
                if seg_res.masks is not None:
                    for mask in seg_res.masks.data:
                        mask_array = mask.cpu().numpy().astype(np.uint8) * 255
                        mask_resized = cv2.resize(mask_array, (x2 - x1, y2 - y1))

                        final_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                            final_mask[y1:y2, x1:x2],
                            mask_resized
                        )

                        pig_only = cv2.bitwise_and(crop, crop, mask=mask_resized)
                        
    main_bar.progress(75)
    status_text.text("75%")

    FEATURE_NAMES = [
        "SR", "A", "P", "BL", "BW", "E",
        "A_hull", "solidity", "extent",
        "compactness", "circularity", "elongation",
        "aspect_ratio",
        "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7", "countour length"
    ]

    results = {name: [] for name in ["pig_id", "weight", "img_file"] + FEATURE_NAMES}
    count = 0

    features = extract_features(final_mask)
    if not features:
        st.error("Failed to extract features from the image. Please try another image.")
    else:
        main_bar.progress(90)
        status_text.text("90%")

        feature_dict = {name: features[i] for i, name in enumerate(FEATURE_NAMES)}
        feature_df = pd.DataFrame([feature_dict])
        predicted_weight = model.predict(feature_df)[0]
        main_bar.progress(100)
        status_text.text("100%")
        st.success(f"Predicted weight: **{predicted_weight:.2f} kg**")