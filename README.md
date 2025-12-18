# 🐷 Pig Weight Estimator

A **Streamlit-based web application** that estimates the weight of pigs from images using **computer vision and machine learning**. It leverages **YOLO for pig detection and segmentation** and **CatBoost for weight prediction** based on extracted features.

---

## 🚀 Features

* Upload an image or capture a pig photo using your device camera.
* Automatic pig detection and segmentation using YOLO.
* Extracts morphological features (e.g., body length, width, contour-based features).
* Predicts pig weight using a trained **CatBoost Regressor** model.
* Lightweight, CPU-only inference — deployable on Streamlit Cloud.
* Full-width responsive layout with a clean user interface.

---

## 📸 Screenshots

*(Add screenshots here after running the app)*

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/leonard250/pig-weight-estimator.git
cd pig-weight-estimator
```

### 2. Create and activate a virtual environment

```bash
python -m venv app_venv
# Windows
app_venv\Scripts\activate
# macOS/Linux
source app_venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ⚡ Running the App Locally

```bash
streamlit run app.py
```

Then open the provided URL in your browser (usually `http://localhost:8501`).

---

## 🗂 File Structure

```
Pig_Weight_Estimation/
├─ app.py                 # Main Streamlit app
├─ requirements.txt       # Python dependencies
├─ model_meta.json        # Model metadata
├─ catboost_weight_model.cbm  # Trained CatBoost model
├─ README.md
└─ .gitignore
```

**Note:** YOLO weights (`.pt`) are downloaded automatically on first run. No need to include them in the repository.

---

## 🧩 How it Works

1. **Upload Image / Take Photo**: User provides an image of a pig.
2. **Pig Detection & Segmentation**: YOLO detects the pig and segments it from the background.
3. **Feature Extraction**: The app calculates features such as area, length, width, compactness, circularity, elongation, and Hu moments.
4. **Weight Prediction**: Features are fed into a **CatBoost Regressor** to estimate the pig's weight.
5. **Display**: Segmented pig image and predicted weight are shown in the app.

---

## 💻 Dependencies

* Python 3.10
* Streamlit
* NumPy
* Pandas
* OpenCV (opencv-python-headless)
* CatBoost
* Ultralytics (YOLO)
* Pillow
* rawpy
* PyTorch

---

## 🌐 Deployment

You can deploy the app easily on **Streamlit Cloud**:

1. Push the code to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Create a new app and link your GitHub repository.
4. Select `app.py` as the main file and deploy.

---

## 📦 Notes

* The app is designed for **CPU-only inference** for compatibility with free hosting.
* Model weights and YOLO weights are **not included in the repository** to keep size small. They will download automatically.
* Recommended Python version: **3.10** (PyTorch support).

---

## 👨‍💻 Author

**Leonard Niyitegeka**
[GitHub](https://github.com/leonard250)

