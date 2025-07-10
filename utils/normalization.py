import cv2
import numpy as np

def normalize_data(img):
	# Mengilangkan channel warna jika perlu
	if img.shape[-1] == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# dtype float32
	img = img.astype('float32')
	img = cv2.resize(img, (256, 256))

	# Normalisasi data Tanh
	img = (img - 127.5) / 127.5

	# Ubah channel ke 1
	img = np.expand_dims(img, axis=-1)

	return img