import cv2
import numpy as np
import matplotlib.pyplot as plt

def bit_slicing(imgg):
    # 1. Read image
    img = cv2.imread(imgg)

    # 2. Convert to grayscale if RGB
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows, cols = img.shape

    # 3. Initialize 8 bit planes
    bit_planes = np.zeros((rows, cols, 8), dtype=np.uint8)

    # 4. Extract each bit plane (MSB to LSB)
    for bit in range(8):
        bit_planes[:, :, bit] = (img >> (7 - bit)) & 1  # MSB first

    # 5. Display bit planes
    plt.figure(figsize=(10, 6))
    for k in range(8):
        plt.subplot(2, 4, k + 1)
        plt.imshow(bit_planes[:, :, k], cmap='gray')
        plt.title(f'Bit Plane {8 - k}')  # Match MATLAB's numbering
        plt.axis('off')

    plt.tight_layout()
    plt.show()


bit_slicing("Input_Image_Grayscale.jpg")  # Replace with your image path