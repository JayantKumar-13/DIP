# LAB 1: RGB to Grayscale, RGB Planes, and Black & White Image Processing
This repository contains a Python script for basic digital image processing tasks using OpenCV and Matplotlib. The script allows you to:

Convert an RGB image to Grayscale (with three different logic options)
Extract individual Red, Green, and Blue color planes
Convert a Grayscale image to a Black and White (binary) image
File Structure
LAB_1/
    jayant_RGBtoGray_RGBPlanes_RGBtoBW.py
    node.png
    output_imgs/
        balck_and_white.png
        blue_plane.png
        grayscale_img.png
        green_plane.png
        red_planee.png
Requirements
Python 3.x
OpenCV (opencv-python)
NumPy
Matplotlib
Install dependencies with:

pip install opencv-python numpy matplotlib
Usage
Run the script with your image and desired operation. Example:

jayant_RGBtoGray_RGBPlanes_RGBtoBW("node.png", choice=1, gray_logic=1)
Function Arguments
imgg: Path to the input image file (relative or absolute)
choice: Operation to perform:
1: Convert RGB to Grayscale
2: Show Red Plane
3: Show Green Plane
4: Show Blue Plane
5: Convert Grayscale to Black and White
gray_logic: (Only for choice=1) Grayscale conversion logic:
1: Weighted sum method
2: Average method
3: Pixel-wise weighted sum using loops
Example Calls
# Weighted Grayscale
jayant_RGBtoGray_RGBPlanes_RGBtoBW("node.png", choice=1, gray_logic=1)

# Red Plane
jayant_RGBtoGray_RGBPlanes_RGBtoBW("node.png", choice=2)

# Black and White
jayant_RGBtoGray_RGBPlanes_RGBtoBW("node.png", choice=5)
Output
The processed images are displayed using Matplotlib. Example output images are saved in the output_imgs/ folder.

Author
Jayant Kumar

Feel free to fork, use, and modify for your digital image processing experiments!