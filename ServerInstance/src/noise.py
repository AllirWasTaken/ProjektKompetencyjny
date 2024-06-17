import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load the image
image_path = '1.jpeg'
original_image = cv2.imread(image_path)

# Check if the image was successfully loaded
if original_image is None:
    raise FileNotFoundError(f"The image file {image_path} could not be loaded. Please check the file path.")

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary mask of the white region
_, binary_mask = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)

# Find contours of the white region
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Identify the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Draw over the largest contour to remove it (fill it with black)
cv2.drawContours(gray_image, [largest_contour], -1, (0, 0, 0), thickness=cv2.FILLED)


# Apply threshold
_, thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

# Apply the threshold mask to the original image
thresholded_color_image = cv2.bitwise_and(original_image, original_image, mask=thresholded_image)

# Apply erosion
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded_image = cv2.morphologyEx(thresholded_color_image, cv2.MORPH_OPEN, kernel)

# Apply Gaussian Blur to the eroded image
blurred_image = cv2.GaussianBlur(eroded_image, (3, 3), 0)

sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])

# Apply the sharpening kernel to the image
sharpened_image = cv2.filter2D(blurred_image, -1, sharpening_kernel)

# Convert BGR to RGB for displaying with matplotlib
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
thresholded_color_image_rgb = cv2.cvtColor(thresholded_color_image, cv2.COLOR_BGR2RGB)
eroded_image_rgb = cv2.cvtColor(eroded_image, cv2.COLOR_BGR2RGB)
blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

# Display the results
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(original_image_rgb)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Thresholded Color Image')
plt.imshow(thresholded_color_image_rgb)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Eroded Image')
plt.imshow(eroded_image_rgb)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Gaussian Blurred Image')
plt.imshow(sharpened_image)
plt.axis('off')

plt.show()
