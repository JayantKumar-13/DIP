import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def jayant_RGBtoGray_RGBPlanes_RGBtoBW(imgg, choice, gray_logic=1):
    """
    Function to process image based on choice:
    choice:
        1 - Convert RGB to Grayscale (3 logic versions)
        2 - Show Red Plane
        3 - Show Green Plane
        4 - Show Blue Plane
        5 - Convert Grayscale to Black and White

    gray_logic (only for choice=1):
        1 - Weighted sum method
        2 - Average method
        3 - Pixel-wise weighted sum using loops
    """

    # Resolve absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, imgg)

    # Read the image (BGR in OpenCV)
    I = cv2.imread(img_path)
    if I is None:
        print("Error: Could not read the image.")
        return

    # Convert to RGB for calculations
    I_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    M, N, _ = I_rgb.shape
    OutputImage = None

    if choice == 1:
        if gray_logic == 1:
            # Weighted sum method
            R = I_rgb[:, :, 0].astype(float)
            G = I_rgb[:, :, 1].astype(float)
            B = I_rgb[:, :, 2].astype(float)
            Gray = 0.298936 * R + 0.587043 * G + 0.114021 * B
            OutputImage = Gray.astype(np.uint8)
        elif gray_logic == 2:
            # Average method
            OutputImage = np.mean(I_rgb, axis=2).astype(np.uint8)
        elif gray_logic == 3:
            # Loop method
            OutputImage = np.zeros((M, N), dtype=np.uint8)
            for i in range(M):
                for j in range(N):
                    R = float(I_rgb[i, j, 0])
                    G = float(I_rgb[i, j, 1])
                    B = float(I_rgb[i, j, 2])
                    OutputImage[i, j] = int(0.298936 * R + 0.587043 * G + 0.114021 * B)

    elif choice == 2:
        Ired = I_rgb.copy()
        Ired[:, :, 1] = 0
        Ired[:, :, 2] = 0
        OutputImage = Ired

    elif choice == 3:
        Ig = I_rgb.copy()
        Ig[:, :, 0] = 0
        Ig[:, :, 2] = 0
        OutputImage = Ig

    elif choice == 4:
        Ib = I_rgb.copy()
        Ib[:, :, 0] = 0
        Ib[:, :, 1] = 0
        OutputImage = Ib

    elif choice == 5:
        R = I_rgb[:, :, 0].astype(float)
        G = I_rgb[:, :, 1].astype(float)
        B = I_rgb[:, :, 2].astype(float)
        Gray = 0.298936 * R + 0.587043 * G + 0.114021 * B
        Gray = Gray.astype(np.uint8)
        BW = np.where(Gray <= 127, 0, 255).astype(np.uint8)
        OutputImage = BW

    else:
        print("Invalid choice.")
        return

    # Show result with matplotlib
    titles = ["Grayscale Image", "Red Plane", "Green Plane", "Blue Plane", "Black and White Image"]
    plt.figure(figsize=(6, 6))
    if choice in [1, 5]:
        plt.imshow(OutputImage, cmap='gray')
    else:
        plt.imshow(OutputImage)
    plt.title(titles[choice - 1])
    plt.axis('off')
    plt.show()


# Example usage
jayant_RGBtoGray_RGBPlanes_RGBtoBW("node.png", choice=1, gray_logic=1)

