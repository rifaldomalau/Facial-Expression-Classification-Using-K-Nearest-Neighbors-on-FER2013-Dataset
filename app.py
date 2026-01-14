import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Ekspresi Wajah", layout="centered")

# 1. Load Model dan Helper (PCA, Scaler)
@st.cache_resource
def load_models():
    model = joblib.load("model_knn.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    return model, scaler, pca

try:
    knn_model, scaler, pca = load_models()
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file .pkl tersedia. Error: {e}")

# 2. Fungsi Ekstraksi Fitur (Sama dengan di Notebook)
def extract_features(image):
    # Resize ke 48x48 sesuai training
    img_resized = cv2.resize(image, (48, 48))
    # Normalisasi
    img_norm = img_resized / 255.0
    # Fitur HOG
    features = hog(
        img_norm, 
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        block_norm='L2-Hys'
    )
    return features.reshape(1, -1)

# UI Streamlit
st.title("😊 Klasifikasi Ekspresi Wajah")
st.write("Unggah foto wajah untuk mendeteksi emosinya.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Konversi file unggahan ke OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        # Tampilkan Gambar
        st.image(image, caption='Gambar yang diunggah', use_column_width=True, channels="GRAY")
        
        if st.button("Prediksi Ekspresi"):
            with st.spinner('Menganalisis...'):
                # 1. Ekstraksi HOG
                features = extract_features(image)
                
                # 2. Scaling
                scaled_features = scaler.transform(features)
                
                # 3. PCA
                pca_features = pca.transform(scaled_features)
                
                # 4. Prediksi
                prediction = knn_model.predict(pca_features)
                prediction_proba = knn_model.predict_proba(pca_features)
                
                result = labels[prediction[0]]
                
                # Tampilkan Hasil
                st.success(f"Hasil Prediksi: **{result.upper()}**")
                
                # Tampilkan Probabilitas
                st.write("### Probabilitas:")
                for i, label in enumerate(labels):
                    st.write(f"- {label}: {prediction_proba[0][i]*100:.2f}%")
    else:
        st.error("Gagal membaca gambar.")