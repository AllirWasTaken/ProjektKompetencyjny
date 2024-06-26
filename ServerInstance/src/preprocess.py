import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm 
from PIL import Image
import shutil
from multiprocessing import Pool, cpu_count

def make_square(image, size=512, padding_color=0):
    """
    Add white padding to an image to make it square and resize it to the given size.

    Parameters:
    image (numpy array): The input image.
    size (int): The desired size of the output image (default is 512).
    padding_color (int): The color of the padding (default is white, 255).

    Returns:
    numpy array: The resized square image with padding.
    """
    h, w = image.shape[:3]

    # Determine padding
    if h > w:
        pad_vert = 0
        pad_horz = (h - w) // 2
    else:
        pad_vert = (w - h) // 2
        pad_horz = 0

    # Add padding to make the image square
    padded_image = cv2.copyMakeBorder(image, pad_vert, pad_vert, pad_horz, pad_horz,
                                      cv2.BORDER_CONSTANT, value=[padding_color, padding_color, padding_color])

    # Resize the image to the desired size
    resized_image = cv2.resize(padded_image, (size, size))

    return resized_image

# Load the image
def preprocess_image_debug(path_src):
    original = cv2.imread(path_src, cv2.IMREAD_GRAYSCALE)
    image = original

    plt.figure(figsize=(15, 10))
    plt.subplot(3, 4, 1)
    plt.title("Oryginalny obraz")
    plt.imshow(image, cmap='gray')

    

    # Threshold to create a binary mask
    _, binary_mask = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)
    plt.subplot(3, 4, 2)
    plt.title("Wykrycie brzegowych białych obszarów")
    plt.imshow(binary_mask, cmap='gray')

    # Find contours of the white region
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw over the largest contour to remove it (fill it with black)
    cv2.drawContours(image, [largest_contour], -1, (0, 0, 0), thickness=cv2.FILLED)
    plt.subplot(3, 4, 3)
    plt.title("Wypełnie obszarów")
    plt.imshow(image, cmap='gray')

    # Apply Non-Local Means Denoising
    nlm_filtered = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    plt.subplot(3, 4, 4)
    plt.title("Rozmycie za pomocą N1FMD")
    plt.imshow(nlm_filtered, cmap='gray')

    # Calculate median value and create mask
    median_value = np.mean(nlm_filtered)
    _, mask = cv2.threshold(nlm_filtered, median_value, 255, cv2.THRESH_BINARY)
    plt.subplot(3, 4, 5)
    plt.title("Wydobycie maski z obrazu")
    plt.imshow(mask, cmap='gray')

    border_size=5
    height, width = image.shape
    cropped_image = mask[border_size:height-border_size, border_size:width-border_size]
    bordered_image = cv2.copyMakeBorder(
        cropped_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # Black color
    )

    # Find contours on the mask
    contours2, _ = cv2.findContours(bordered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the largest contour from the mask
    largest_contour2 = max(contours2, key=cv2.contourArea)

    black_image = original.copy()
    black_image[:, :] = 0

    # Draw the largest contour on the original image (fill it with white)
    cv2.drawContours(black_image, [largest_contour2], -1, (255, 255, 255), thickness=cv2.FILLED)


    # Cut out the region
    cut_out = cv2.bitwise_and(image, black_image)
    plt.subplot(3, 4, 6)
    plt.title("Wypełnienie dziur w masce")
    plt.imshow(black_image, cmap='gray')

    # Final image
    final_image = cut_out
    plt.subplot(3, 4, 7)
    plt.title("Nałożenie maski na oryginalny obraz")
    plt.imshow(final_image, cmap='gray')

    image=make_square(final_image)
    plt.subplot(3, 4, 8)
    plt.title("Square padding i skalowanie na 512x512")
    plt.imshow(image, cmap='gray')

    plt.tight_layout()
    plt.show()

    
def preprocess_image(path_src,path_dst):
    original = cv2.imread(path_src, cv2.IMREAD_GRAYSCALE)
    image = original

    # Threshold to create a binary mask
    _, binary_mask = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)
    
    # Find contours of the white region
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw over the largest contour to remove it (fill it with black)
    cv2.drawContours(image, [largest_contour], -1, (0, 0, 0), thickness=cv2.FILLED)
 
    # Apply Non-Local Means Denoising
    nlm_filtered = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
   
    # Calculate median value and create mask
    median_value = np.mean(nlm_filtered)
    _, mask = cv2.threshold(nlm_filtered, median_value, 255, cv2.THRESH_BINARY)

    border_size=5
    height, width = image.shape
    cropped_image = mask[border_size:height-border_size, border_size:width-border_size]
    bordered_image = cv2.copyMakeBorder(
        cropped_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # Black color
    )

    # Find contours on the mask
    contours2, _ = cv2.findContours(bordered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the largest contour from the mask
    largest_contour2 = max(contours2, key=cv2.contourArea)

    black_image = original.copy()
    black_image[:, :] = 0

    # Draw the largest contour on the original image (fill it with white)
    cv2.drawContours(black_image, [largest_contour2], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Cut out the region
    final_image = cv2.bitwise_and(image, black_image)

    # Final image
    image=make_square(final_image)
    cv2.imwrite(path_dst,image)


def process_image(args):
    image_file, input_folder, output_folder = args
    try:
        # Open image
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # Preprocess image
        preprocess_image(input_path, output_path)
        
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

def process_folder(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

    # Use all available CPUs
    num_processes = cpu_count()
    
    # Prepare arguments for the worker function
    args = [(image_file, input_folder, output_folder) for image_file in image_files]

    # Use multiprocessing Pool to process images
    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_image, args), total=len(image_files), desc="Processing "+input_folder))

def empty_processed_folder(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

def get_subfolder_paths(directory):
    """
    Returns a list of all folder paths one level below the given directory.

    Parameters:
    directory (str): The directory path to search for subfolders.

    Returns:
    list: A list of subfolder paths.
    """
    try:
        # Get the list of entries in the given directory
        entries = os.listdir(directory)
        # Filter out the subdirectories
        subfolders = [os.path.join(directory, entry) for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        return subfolders
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Example usage:
# directory_path = "/path/to/your/directory"
# subfolder_paths = get_subfolder_paths(directory_path)
# print(subfolder_paths)
def replace_word_in_path(original_path, old_word, new_word):
    """
    Replaces a specific word in the given path with a new word.

    Parameters:
    original_path (str): The original file path as a string.
    old_word (str): The word to be replaced.
    new_word (str): The word to replace with.

    Returns:
    str: The modified file path.
    """
    return original_path.replace(old_word, new_word)

def process_all_files():
    folders_train=get_subfolder_paths('../data/train')
    empty_processed_folder('../processed')
    folders_test=get_subfolder_paths('../data/test')
    folders_train_processed=[]
    folders_test_processed=[]
    for path in folders_train:
        folders_train_processed.append(path.replace('data','processed'))
    for path in folders_test:
        folders_test_processed.append(path.replace('data','processed'))

    for i in range(len(folders_test)):
        process_folder(folders_test[i],folders_test_processed[i])

    for i in range(len(folders_train)):
        process_folder(folders_train[i],folders_train_processed[i])
    
if __name__ == '__main__':
    process_all_files()