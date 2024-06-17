import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Read the image
image_path = '3.jpeg'  # Replace with your image path
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_mask = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)

# Find contours of the white region
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Identify the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Draw over the largest contour to remove it (fill it with black)
cv2.drawContours(gray_image, [largest_contour], -1, (0, 0, 0), thickness=cv2.FILLED)

# Step 2: Convert the image to grayscale


# Step 3: Apply a binary threshold to create a mask
median_value = np.mean(gray_image)
_, mask = cv2.threshold(gray_image, median_value, 255, cv2.THRESH_BINARY)

# Step 4: Use the mask to isolate the relevant parts of the image
masked_image = cv2.bitwise_and(gray_image, mask)

# Step 5: Enhance contrast using CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(masked_image)

# Step 6: Combine the CLAHE-enhanced regions with the original image
final_image = eroded_image = enhanced_image

# Step 7: Display the original and final enhanced images
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# CLAHE Enhanced Image with Mask
plt.subplot(1, 2, 2)
plt.title('CLAHE Enhanced Image with Mask')
plt.imshow(final_image, cmap='gray')
plt.axis('off')

plt.show()
