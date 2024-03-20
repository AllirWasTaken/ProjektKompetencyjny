import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm  # Import tqdm
import random

# Adjust the root directory of the dataset and the target directory for processed images
root_dir = '../data'
target_dir = '../processed'

# Define the transformation: resize to 256x256 and convert to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.ToPILImage()  # Add ToPILImage here for direct saving after transformation
])

# Function to process and save a percentage of the images in the new directory
def process_and_save_images(dataset_type, percentage):
    dataset_path = os.path.join(root_dir, dataset_type)
    target_path = os.path.join(target_dir, dataset_type)

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        target_class_path = os.path.join(target_path, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        image_names = os.listdir(class_path)
        num_files_to_process = int(len(image_names) * (percentage / 100))
        
        # Randomly select a subset of files based on the specified percentage
        selected_files = random.sample(image_names, num_files_to_process)

        # Wrap the loop with tqdm for a progress bar
        for img_name in tqdm(selected_files, desc=f'Processing {percentage}% of {dataset_type}/{class_name}'):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert('RGB')  # Ensure it's RGB
            
            # Apply transformation
            img_transformed = transform(img)
            
            save_path = os.path.join(target_class_path, img_name)
            img_transformed.save(save_path)  # Save directly

# Ask for the percentage of files to use
percentage = float(input("Enter the percentage of files you want to use (0-100): "))

# Validate the input percentage
if 0 <= percentage <= 100:
    process_and_save_images('train', percentage)
    process_and_save_images('test', percentage)
else:
    print("Invalid percentage. Please enter a value between 0 and 100.")