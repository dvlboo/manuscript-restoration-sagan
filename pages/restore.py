import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.spectral_norm import SpectralNormalization
from utils.self_attention import SelfAttention
from utils.normalization import normalize_data
from utils.metrics import calculate_psnr, calculate_ssim
import os

# Fungsi untuk mengambil daftar model dari folder 'models'
def get_model_options():
  model_options = {}
  for root, _, files in os.walk("./models"):
    for file in files:
      if file.endswith(".h5"):
        full_path = os.path.join(root, file)
        display_name = f"{os.path.basename(root).capitalize()} | {file}"
        model_options[display_name] = full_path
  return model_options

# Fungsi load model berdasarkan path
@st.cache_resource
def load_generator_model(model_path):
  return load_model(
    model_path,
    custom_objects={
      "SpectralNormalization": SpectralNormalization,
      "SelfAttention": SelfAttention
    }
  )

def run():
  st.set_page_config(page_title="Restorasi Manuskrip", layout="wide")
  st.title("🧠 Restorasi Gambar Manuskrip Rusak")

  uploaded_file = st.file_uploader("Unggah gambar manuskrip yang rusak", type=["png", "jpg", "jpeg"])

  if uploaded_file:
    # Konversi dan tampilkan gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Gambar Rusak")

    # Ambil model yang tersedia
    model_dict = get_model_options()
    model_names = list(model_dict.keys())

    # Pilih model
    st.markdown("#### Pilih Learning Rate Model")
    selected_model_name = st.selectbox("LR Generator", model_names)
    selected_model_path = model_dict[selected_model_name]

    # Tombol Proses
    if st.button("🔧 Proses Restorasi"):
      with st.spinner("Melakukan restorasi gambar..."):

        # Normalisasi input
        input_tensor = normalize_data(image_rgb)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Load model dan prediksi
        generator = load_generator_model(selected_model_path)
        generated_img = generator.predict(input_tensor)[0]
        generated_img = np.clip(generated_img * 127.5 + 127.5, 0, 255).astype(np.uint8)

        # Evaluasi metrik
        orig_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        restored_gray = generated_img.squeeze()
        psnr = calculate_psnr(orig_gray, restored_gray)
        ssim = calculate_ssim(orig_gray, restored_gray)

        # Tampilkan hasil
        st.markdown("### Hasil Restorasi")
        col1, col2 = st.columns(2)
        with col1:
          st.image(image_rgb, caption="Gambar Asli (Rusak)")
          st.markdown(f"**LR:** `{selected_model_name}`  \n**PSNR:** {psnr:.2f} dB  \n**SSIM:** {ssim:.4f}")
        with col2:
          st.image(generated_img, caption="Hasil Restorasi")

      st.success("✅ Proses restorasi selesai.")

run()
