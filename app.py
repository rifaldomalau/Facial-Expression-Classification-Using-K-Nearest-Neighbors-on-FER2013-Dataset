import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
from skimage.feature import hog

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Facial Expression Recognition (KNN)",
    page_icon="😀",
    layout="centered"
)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_artifacts():
    knn = joblib.load("model_knn.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    return knn, scaler, pca

knn_model, scaler, pca = load_artifacts()

# =========================
# EMOTION LABELS
# =========================
EMOTIONS = {
    0: "Angry 😠",
    1: "Disgust 🤢",
    2: "Fear 😨",
    3: "Happy 😄",
    4: "Sad 😢",
    5: "Surprise 😲",
    6: "Neutral 😐"
}

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(image):
    image = cv2.resize(image, (48, 48))
    image = image / 255.0

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1)

# =========================
# UI
# =========================
st.title("Facial Expression Recognition (KNN)")
st.markdown(
    """
    ### 📌 Tentang Aplikasi
    - Dataset: **FER2013**
    - Algoritma: **K-Nearest Neighbors (KNN)**
    - Feature Extraction: **HOG**
    - Dimensionality Reduction: **PCA**

    Upload **gambar wajah**, lalu sistem akan memprediksi **emosi**.
    """
)

st.divider()

uploaded_file = st.file_uploader(
    "📤 Upload gambar wajah (jpg / png)",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", width=200)

    with st.spinner("🔍 Menganalisis ekspresi..."):
        features = extract_features(image_np)
        features = scaler.transform(features)
        features = pca.transform(features)
        prediction = knn_model.predict(features)[0]

    st.success(f"### Ekspresi Terdeteksi: **{EMOTIONS[prediction]}**")

else:
    st.info("⬆️ Silakan upload gambar wajah untuk mulai prediksi.")
