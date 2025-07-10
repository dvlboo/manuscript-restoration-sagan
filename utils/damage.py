import cv2, random
import numpy as np

# Fungsi untuk menambahkan efek BERLUBANG (1 lubang besar)
def add_hole_damage(image):
	damaged = image.copy()
	h, w = damaged.shape[:2]

	# Tentukan titik dan radius lubang
	x, y = random.randint(50, w-50), random.randint(50, h-50)
	radius = random.randint(15, 30)

	# Warna abu-abu untuk lubang
	grey_color = (128, 128, 128)

	# Buat lubang dengan warna abu-abu
	cv2.circle(damaged, (x, y), radius, grey_color, -1)
	return damaged

# Fungsi untuk menambahkan efek Tinta Tembus
def add_ink_bleed(image):
	damaged = image.copy()
	h, w = damaged.shape[:2]
	x_offset = random.randint(10, 20)
	y_offset = random.randint(10, 20)
	mask = np.zeros_like(damaged)

	# Pindahkan aksara agar terlihat tembus
	mask[y_offset:h, x_offset:w] = damaged[0:h-y_offset, 0:w-x_offset]

	# Gabungkan dengan gambar asli (dengan opacity rendah)
	damaged = cv2.addWeighted(damaged, 0.8, mask, 0.2, 0)
	return damaged

# Fungsi untuk menambahkan efek Bercak
def add_stain_damage(image):
	damaged = image.copy()
	h, w = damaged.shape[:2]

	for _ in range(random.randint(1, 3)):  # Tambahkan 1-3 bercak
		x, y = random.randint(20, w-50), random.randint(20, h-50)
		size = random.randint(20, 50)

		# Tentukan warna bercak abu-abu acak
		gray_value = random.randint(100, 200)  # Nilai 100-200 untuk abu-abu

		# Buat bercak berwarna abu-abu (sama untuk R, G, B)
		stain = np.full((size, size, 3), gray_value, dtype=np.uint8)

		# Tambahkan Gaussian Blur agar bercak tampak lebih alami
		stain = cv2.GaussianBlur(stain, (5, 5), 0)

		# Pastikan ukuran bercak sesuai dengan area gambar
		if damaged[y:y+size, x:x+size].shape == stain.shape:
			# Gabungkan bercak dengan gambar asli
			blended = cv2.addWeighted(damaged[y:y+size, x:x+size], 0.3, stain, 0.7, 0)
			damaged[y:y+size, x:x+size] = blended

	return damaged

# Fungsi untuk menambahkan efek TEKS HILANG (Blurring area tertentu)
def add_missing_text(image):
	damaged = image.copy()
	h, w, = damaged.shape[:2]

	for _ in range(random.randint(2, 4)):  # Hilangkan 2-4 area teks
		x, y = random.randint(20, w-160), random.randint(20, h-60)  # Area blur diperbesar
		width, height = random.randint(80, 160), random.randint(30, 60)  # Ukuran blur lebih besar

		blur_region = damaged[y:y+height, x:x+width]

		blur_region = cv2.GaussianBlur(blur_region, (19, 19), 0)  # Blur lebih intens, kernel size diganti menjadi (19, 19)

		damaged[y:y+height, x:x+width] = blur_region

	return damaged