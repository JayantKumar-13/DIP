import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def jayant_RGBtoGray_RGBPlanes_RGBtoBW(imgg, output_dir, choice, gray_logic=1):
    """
    Processes an image based on the choice:
        1 - RGB to Grayscale (3 logic variants)
        2 - Red Plane
        3 - Green Plane
        4 - Blue Plane
        5 - RGB to Black and White

    If the input image has fewer than 5 unique colors, it creates and saves
    a synthetic image with 5 colors + white.

    Parameters:
    - imgg: path to the input image
    - output_dir: directory to save the output
    - choice: which operation to perform (1–5)
    - gray_logic: logic for grayscale (only when choice=1)
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Check image path
    if not os.path.exists(imgg):
        print(f"Error: Image path '{imgg}' not found.")
        return

    # Read image and convert to RGB
    I = cv2.imread(imgg)
    if I is None:
        print("Error: Failed to load image.")
        return

    I_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    M, N, _ = I_rgb.shape
    OutputImage = None

    # Check for distinct colors
    unique_colors = np.unique(I_rgb.reshape(-1, 3), axis=0)
    if unique_colors.shape[0] < 5:
        print("Image does not have enough distinct colors. Creating synthetic image.")

        color_blocks = [
            [255, 0, 0],     # Red
            [0, 255, 0],     # Green
            [0, 0, 255],     # Blue
            [255, 255, 0],   # Yellow
            [255, 0, 255],   # Magenta
            [255, 255, 255]  # White
        ]
        block_height = 100
        OutputImage = np.zeros((block_height, block_height * 6, 3), dtype=np.uint8)
        for i, color in enumerate(color_blocks):
            OutputImage[:, i * block_height:(i + 1) * block_height, :] = color

        save_path = os.path.join(output_dir, "synthetic_colored_image.png")
        cv2.imwrite(save_path, cv2.cvtColor(OutputImage, cv2.COLOR_RGB2BGR))
        print(f"Synthetic image saved at: {save_path}")

        plt.imshow(OutputImage)
        plt.title("Synthetic Image with 5 Colors + White")
        plt.axis('off')
        plt.show()
        return

    # Image Processing
    if choice == 1:
        if gray_logic == 1:
            R = I_rgb[:, :, 0].astype(float)
            G = I_rgb[:, :, 1].astype(float)
            B = I_rgb[:, :, 2].astype(float)
            Gray = 0.298936 * R + 0.587043 * G + 0.114021 * B
            OutputImage = Gray.astype(np.uint8)
        elif gray_logic == 2:
            OutputImage = np.mean(I_rgb, axis=2).astype(np.uint8)
        elif gray_logic == 3:
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
        BW = np.where(Gray <= 127, 0, 255).astype(np.uint8)
        OutputImage = BW.astype(np.uint8)

    else:
        print("Invalid choice.")
        return

    # Save output
    filename_map = {
        1: f"gray_logic_{gray_logic}.png",
        2: "red_plane.png",
        3: "green_plane.png",
        4: "blue_plane.png",
        5: "bw_image.png"
    }
    save_path = os.path.join(output_dir, filename_map.get(choice, "output.png"))

    if choice in [1, 5]:
        cv2.imwrite(save_path, OutputImage)
    else:
        cv2.imwrite(save_path, cv2.cvtColor(OutputImage, cv2.COLOR_RGB2BGR))

    print(f"Processed image saved at: {save_path}")

    plt.figure(figsize=(6, 6))
    if choice in [1, 5]:
        plt.imshow(OutputImage, cmap='gray')
    else:
        plt.imshow(OutputImage)
    plt.title(filename_map.get(choice, "Output"))
    plt.axis('off')
    plt.show()
