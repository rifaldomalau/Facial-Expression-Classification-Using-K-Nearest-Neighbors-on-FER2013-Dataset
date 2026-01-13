import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Facial Expression Recognition (KNN)",
    page_icon="😃",
    layout="centered"
)

# ===============================
# Load model & preprocessors
# ===============================
model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
labels = np.load("labels.npy", allow_pickle=True)

# ===============================
# HOG extractor
# ===============================
def extract_hog(image):
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

# ===============================
# Sidebar (Project Info)
# ===============================
st.sidebar.title("🧠 Project Info")
st.sidebar.write("""
**Facial Expression Recognition**  
Using **K-Nearest Neighbors (KNN)**  

This system classifies facial expressions into:

- 😡 Angry  
- 🤢 Disgust  
- 😨 Fear  
- 😄 Happy  
- 😐 Neutral  
- 😢 Sad  
- 😲 Surprise  

**Dataset:** FER2013 (Kaggle)  
**Features:** HOG  
**Classifier:** KNN  
""")

# ===============================
# Main Title
# ===============================
st.title("😃 Facial Expression Recognition (KNN)")
st.write(
    "Upload a face image and this system will predict "
    "the **emotion** using a **KNN-based machine learning model**."
)

st.markdown("---")

# ===============================
# Instructions
# ===============================
st.subheader("📸 How to use")
st.write("""
1. Upload a face image (JPG / PNG).  
2. The system will convert it to grayscale and extract facial features.  
3. KNN will compare your face with thousands of samples in the FER dataset.  
4. The predicted emotion will be displayed.
""")

# ===============================
# Upload
# ===============================
uploaded = st.file_uploader("📤 Upload Face Image", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Uploaded Image")
        st.image(image, use_column_width=True)

    # ===============================
    # Preprocess
    # ===============================
    image = cv2.resize(image, (48,48))
    image = image / 255.0
    features = extract_hog(image)

    features = features.reshape(1, -1)
    features = scaler.transform(features)
    features = pca.transform(features)

    # ===============================
    # Prediction
    # ===============================
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    confidence = probs[pred] * 100

    with col2:
        st.subheader("🤖 Prediction")
        st.success(f"**{labels[pred]}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.subheader("📊 Class Probabilities")
        for i, label in enumerate(labels):
            st.progress(float(probs[i]))
            st.write(f"{label}: {probs[i]*100:.2f}%")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown(
    "<center>📌 Built with KNN, HOG & FER2013 Dataset</center>",
    unsafe_allow_html=True
)
