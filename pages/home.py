import streamlit as st
from dotenv import load_dotenv
import os


load_dotenv()

def run():
  st.set_page_config(page_title="Restorasi Manuskrip", layout="wide")

  st.markdown(f"""
    <link href="https://cdn.tailwindcss.com" rel="stylesheet">
    <div class="flex items-center justify-center min-h-screen bg-gray-50 p-10">
      <div class="max-w-4xl w-full bg-white rounded-xl shadow-lg p-8">
        <h1 class="text-3xl font-bold text-blue-700 mb-4 text-center">Restorasi Manuskrip Digital Dengan SAGAN</h1>
        <p class="text-gray-700 text-lg mb-6 text-center">
          Proyek ini bertujuan untuk melakukan restorasi terhadap manuskrip digital yang mengalami kerusakan visual menggunakan <strong>SAGAN (Self-Attention GAN)</strong>.
        </p>
      </div>
    </div>

    <div class="bg-blue-50 border-l-4 border-blue-400 p-4 mb-6">
      <p class="text-blue-900 font-medium">Kerusakan yang disimulasikan:</p>
      <ul class="list-disc ml-6 text-gray-800">
        <li>Bercak</li>
        <li>Berlubang</li>
        <li>Tinta Tembus</li>
        <li>Teks Pudar</li>
      </ul>
    </div>

    <hr class="my-6">

    <div>
      <h2 class="text-xl font-semibold text-gray-800 mb-3">🚀 Alur Penggunaan</h2>
      <ol class="list-decimal ml-6 text-gray-700 space-y-2">
        <li>Masuk ke halaman <strong>Restore</strong> dan unggah gambar manuskrip yang rusak.</li>
        <li>Pilih Learning Rate model yang telah disediakan.</li>
        <li>Klik <strong>Proses</strong> untuk melihat hasil generate restorasi.</li>
        <li>Evaluasi hasil menggunakan metrik PSNR dan SSIM.</li>
      </ol>

      <div class="mt-6">
        <h3 class="text-lg font-medium text-gray-800">Contoh Hasil:</h3>
        <div class="grid grid-cols-3 gap-4 text-center mt-3">
          <img src="{os.getenv("DAMAGED_IMG")}" alt="Gambar Rusak" class="w-full h-auto rounded">
          <img src="{os.getenv("GENERATED_IMG")}" alt="Gambar Hasil" class="w-full h-auto rounded">
        </div>
        <p class="mt-2 text-sm text-gray-600">PSNR: 27,98 | SSIM: 0,9876 </p>
      </div>
    </div>
  """, unsafe_allow_html=True)

run()