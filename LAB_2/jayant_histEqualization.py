import cv2
import numpy as np
import matplotlib.pyplot as plt

def jayant_histEqualization(imgg):
    I = cv2.imread(imgg)

    if len(I.shape) == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    M, N = I.shape
    num_pixels = M * N
    I_flat = I.flatten()

    histogram = np.zeros(256, dtype=int)
    for pixel_val in I_flat:
        histogram[pixel_val] += 1

    cdf = np.cumsum(histogram)
    cdf_min = np.min(cdf[cdf > 0])

    equalized = np.round((cdf - cdf_min) / (num_pixels - cdf_min) * 255).astype(np.uint8)
    I_eq_flat = np.array([equalized[p] for p in I_flat], dtype=np.uint8)
    I_eq = I_eq_flat.reshape(M, N)

    eq_hist = np.zeros(256, dtype=int)
    for p in I_eq_flat:
        eq_hist[p] += 1

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.bar(range(256), histogram, width=1)
    plt.xlim([0, 255])
    plt.title('Histogram of Original Image')

    plt.subplot(2, 2, 3)
    plt.imshow(I_eq, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.bar(range(256), eq_hist, width=1)
    plt.xlim([0, 255])
    plt.title('Histogram of Equalized Image')

    plt.tight_layout()
    plt.show()


# ✅ Call the function correctly
jayant_histEqualization("Input_Image_Grayscale.jpg")