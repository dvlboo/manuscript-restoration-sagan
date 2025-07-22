import cv2
import numpy as np

def detect_damage_and_percentage(original_img, restored_img, diff_thresh=30, area_thresh=100):
    """
    Mendeteksi area kerusakan pada gambar berdasarkan perbedaan antara original dan hasil restorasi.
    Menyediakan bounding box dan persentase kerusakan.
    """
    # Resize agar ukuran sama
    h, w = 256, 256
    original_resized = cv2.resize(original_img, (w, h))
    restored_resized = cv2.resize(restored_img, (w, h))

    # Pastikan kedua gambar dalam format RGB (3 channel)
    if len(original_resized.shape) == 2:
        original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2RGB)
    if len(restored_resized.shape) == 2 or restored_resized.shape[-1] == 1:
        restored_resized = cv2.cvtColor(restored_resized, cv2.COLOR_GRAY2RGB)

    # Hitung perbedaan absolut
    diff = cv2.absdiff(original_resized, restored_resized)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # Threshold area rusak
    _, damage_mask = cv2.threshold(diff_gray, diff_thresh, 255, cv2.THRESH_BINARY)

    # Persentase kerusakan
    damage_pixels = np.sum(damage_mask > 0)
    total_pixels = damage_mask.size
    damage_percentage = round((damage_pixels / total_pixels) * 100, 2)

    # Temukan kontur area kerusakan
    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxed_img = original_resized.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_thresh:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cv2.rectangle(boxed_img, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)  # biru

    return damage_percentage, boxed_img
