import cv2
import numpy as np
import pandas as pd
import json
import onnxruntime as ort
from catboost import CatBoostRegressor


# loading catboost model

MODEL_PATH = "catboost_weight_model.cbm"
META_PATH = "model_meta.json"

print("Loading CatBoost model...")
regressor = CatBoostRegressor()
regressor.load_model(MODEL_PATH)

model_feature_names = regressor.feature_names_
print("Model feature names:", model_feature_names)

with open(META_PATH, "r") as f:
    meta = json.load(f)

FEATURE_NAMES = meta["features"]
print("These are the features names: ", FEATURE_NAMES)
print("CatBoost loaded.\n")


# loading yolo onnx models

print("Loading YOLO ONNX models...")
ort_session_detect = ort.InferenceSession("yolo11x.onnx", providers=['CPUExecutionProvider'])
ort_session_seg    = ort.InferenceSession("yolo11x-seg.onnx", providers=['CPUExecutionProvider'])
print("ONNX models loaded.\n")


#  feature extraction
def extract_features(mask):
    
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    A = np.sum(mask_bin == 255)
    SR = A / (mask.shape[0] * mask.shape[1])

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    cnt = contours[0]
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
        *hu,
        contour_length
    ]

# 4. helper function for onnx
def preprocess_image(img, size=(640, 640)):
    """Resize and normalize image for YOLO ONNX input."""
    img_resized = cv2.resize(img, size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))  # CHW
    img_batch = np.expand_dims(img_transposed, axis=0)
    return img_batch


def run_onnx_yolo(session, img):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})
    # outputs[0] = bounding boxes / predictions
    # Depending on export, you may need post-processing
    return outputs[0]


# main function prediction
def predict_pig_weight(image_path):
    FEATURE_NAMES = meta["features"]
    img = cv2.imread(image_path)
    if img is None:
        print("ERROR: Image not found.")
        return None

    final_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # 1️⃣ Run detection
    img_input = preprocess_image(img)
    detect_out = run_onnx_yolo(ort_session_detect, img_input)

    # 2️⃣ Process detection output (simplified)

    for bbox in detect_out:
        bbox = np.array(bbox).flatten()
        x1, y1, x2, y2 = map(int, bbox[:4])
    
        crop = img[y1:y2, x1:x2]

        crop_input = preprocess_image(crop)
        seg_out = run_onnx_yolo(ort_session_seg, crop_input)

        # Assuming segmentation mask output is [1,1,H,W]
        mask = seg_out[0][0]  # 0th batch, 0th channel
        mask = (mask * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))

        final_mask[y1:y2, x1:x2] = cv2.bitwise_or(
            final_mask[y1:y2, x1:x2],
            mask_resized
        )

    features = extract_features(final_mask)
    selected_features = [features[FEATURE_NAMES.index(name)] for name in model_feature_names]
    print("length of Extracted features:", len(selected_features))
    if features is None:
        print("ERROR: Feature extraction failed.")
        return None
    #FEATURE_NAMES=["SR", "A", "P", "BL", "BW", "E","A_hull", "solidity", "extent", "compactness", "circularity"," elongation",
    #    "aspect_ratio","hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7","contour_length"]
    FEATURE_NAMES= ['SR', 'elongation', 'BL', 'BW', 'P', 'hu1', 'hu2', 'hu3', 'hu6', 'compactness']
    feature_dict = {FEATURE_NAMES[i]: features[i] for i in range(len(selected_features))}
    feature_df = pd.DataFrame([feature_dict])

    # 4️⃣ Predict weight
    predicted_weight = regressor.predict(feature_df)[0]
    return predicted_weight

#testing the pipeline 

if __name__ == "__main__":

    test_image = 'original_image.png'  
    weight = predict_pig_weight(test_image)

    if weight:
        print("\n===============================")
        print(f"Predicted Pig Weight: {weight:.2f} kg")
        print("===============================")
