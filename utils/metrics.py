from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(original, restored):
  return peak_signal_noise_ratio(original, restored, data_range=255)

def calculate_ssim(original, restored):
  return structural_similarity(original, restored, multichannel=True, data_range=255)