import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = '3.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(15, 10))
plt.subplot(3, 4, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

# Threshold to create a binary mask
_, binary_mask = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)
plt.subplot(3, 4, 2)
plt.title("White space on edge of screen")
plt.imshow(binary_mask, cmap='gray')

# Find contours of the white region
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Identify the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Draw over the largest contour to remove it (fill it with black)
cv2.drawContours(image, [largest_contour], -1, (0, 0, 0), thickness=cv2.FILLED)
plt.subplot(3, 4, 3)
plt.title("Contour Removed")
plt.imshow(image, cmap='gray')

# Apply Non-Local Means Denoising
nlm_filtered = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
plt.subplot(3, 4, 4)
plt.title("NLM Filtered")
plt.imshow(nlm_filtered, cmap='gray')

# Calculate median value and create mask
median_value = np.mean(nlm_filtered)
_, mask = cv2.threshold(nlm_filtered, median_value, 255, cv2.THRESH_BINARY)
plt.subplot(3, 4, 5)
plt.title("Mask from NLM Filtered")
plt.imshow(mask, cmap='gray')

# Find contours on the mask
contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Identify the largest contour from the mask
largest_contour2 = max(contours2, key=cv2.contourArea)

black_image = image.copy()
black_image[:, :] = 0

# Draw the largest contour on the original image (fill it with white)
cv2.drawContours(black_image, [largest_contour2], -1, (255, 255, 255), thickness=cv2.FILLED)


# Cut out the region
cut_out = cv2.bitwise_and(image, black_image)
plt.subplot(3, 4, 6)
plt.title("Cut Out")
plt.imshow(black_image, cmap='gray')

# Final image
final_image = cut_out
plt.subplot(3, 4, 7)
plt.title("Final cut out")
plt.imshow(final_image, cmap='gray')





plt.tight_layout()
plt.show()
