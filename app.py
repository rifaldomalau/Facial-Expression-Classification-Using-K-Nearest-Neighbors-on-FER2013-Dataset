import streamlit as st
import numpy as np
import cv2
import os
import joblib
from skimage.feature import hog
from PIL import Image

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Facial Expression Recognition (KNN)", layout="centered")

st.title("😃 Facial Expression Recognition")
st.write("KNN + HOG + PCA (FER2013 Dataset)")

# ======================================================
# PATH HANDLING (Streamlit Cloud Safe)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models():
    try:
        model = joblib.load(os.path.join(BASE_DIR, "model_knn.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        pca = joblib.load(os.path.join(BASE_DIR, "pca.pkl"))
        return model, scaler, pca
    except Exception as e:
        st.error("❌ Failed to load model files")
        st.exception(e)
        st.stop()

model, scaler, pca = load_models()

# ======================================================
# CLASS LABELS (FER2013)
# ======================================================
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ======================================================
# IMAGE PREPROCESSING
# ======================================================
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1)

# ======================================================
# STREAMLIT UI
# ======================================================
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    features = preprocess_image(img_np)

    # Scale + PCA
    features = scaler.transform(features)
    features = pca.transform(features)

    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    st.subheader("Prediction:")
    st.success(f"**{emotion_labels[prediction]}**")

    st.subheader("Confidence:")
    for i, prob in enumerate(probabilities):
        st.write(f"{emotion_labels[i]} : {prob:.3f}")

else:
    st.info("👆 Upload a face image to get prediction.")
