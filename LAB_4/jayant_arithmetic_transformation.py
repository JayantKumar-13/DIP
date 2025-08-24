import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('SampleInput_lena.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert standard bgr to rgb
# 1. Add (Brighten)
added = cv2.add(image, np.array([50.0]))

# 2. Subtract (Darken)
subtracted = cv2.subtract(image, np.array([50.0]))

# 3. Multiply (Increase contrast)
multiplied = cv2.multiply(image, np.array([1.5]))

# 4. Divide (Reduce contrast)
divided = cv2.divide(image, np.array([1.5]))

titles = ['Original', 'Brightened (+50)', 'Darkened (-50)', 
          'Increase Contrast (X1.5)', 'Decrease Contrast (/1.5)']
images = [image, added, subtracted, multiplied, divided]

plt.figure(figsize=(15, 6))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()