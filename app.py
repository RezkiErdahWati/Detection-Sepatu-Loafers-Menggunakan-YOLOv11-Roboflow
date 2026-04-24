import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# ======================
# CONFIG APLIKASI
# ======================
st.set_page_config(page_title="Deteksi Sepatu Loafers YOLOv11")

st.title("👞 Deteksi Sepatu Loafers Menggunakan YOLOv11")
st.write("Tugas Machine Learning - Rezki Erdah Wati | 231001046")

# ======================
# LOAD MODEL (FIXED & AMAN)
# ======================
@st.cache_resource
def load_model():

    # Cek beberapa kemungkinan lokasi model
    model_paths = [
        "model/best.pt",
        "best.pt",
        "runs/detect/train/weights/best.pt",
        "runs/detect/train2/weights/best.pt"
    ]

    for path in model_paths:
        if os.path.exists(path):
            return YOLO(path)

    st.error("Model best.pt tidak ditemukan. Pastikan file ada di folder project.")
    st.stop()

model = load_model()

# ======================
# UPLOAD GAMBAR
# ======================
uploaded_file = st.file_uploader(
    "Upload gambar sepatu Loafers",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Tampilkan gambar asli
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Original", use_container_width=True)

    # Tombol deteksi
    if st.button("Mulai Deteksi"):

        with st.spinner("AI sedang mendeteksi..."):

            # Convert gambar ke format YOLO
            image_np = np.array(image)

            # Prediksi
            results = model(image_np, conf=0.5)

            # Ambil hasil gambar
            result_img = results[0].plot()

            # Tampilkan hasil
            st.image(result_img, caption="Hasil Deteksi YOLOv11", use_container_width=True)

            st.success("Deteksi selesai!")