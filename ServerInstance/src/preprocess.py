import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm  # Import tqdm
import random
import shutil

# Adjust the root directory of the dataset and the target directory for processed images
root_dir = '../data'
target_dir = '../processed'

class MakeSquarePad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left
        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return transforms.Pad(padding, fill=self.fill, padding_mode=self.padding_mode)(img)


# Now, include the custom padding in your transformation pipeline
transform = transforms.Compose([
    MakeSquarePad(fill=255, padding_mode='constant'),  # Dynamically pad the image to make it square
    transforms.Resize((256, 256)),  # Then resize it to 256x256
    transforms.ToTensor(),
    transforms.ToPILImage()  # Optional: Convert back to PIL Image for further use or saving
])


def empty_processed_folder():
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)


def process_and_save_images(dataset_type, num_files=None):
    dataset_path = os.path.join(root_dir, dataset_type)
    target_path = os.path.join(target_dir, dataset_type)

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        target_class_path = os.path.join(target_path, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        image_names = os.listdir(class_path)
        
        if num_files is not None:  # For training, when num_files is specified
            selected_files = random.sample(image_names, min(num_files, len(image_names)))
        else:  # For testing, process all files
            selected_files = image_names

        for img_name in tqdm(selected_files, desc=f'Processing {dataset_type}/{class_name}'):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            
            # Apply transformation
            img_transformed = transform(img)
            
            save_path = os.path.join(target_class_path, img_name)
            img_transformed.save(save_path)

empty_processed_folder()

# Specify the number of files for each category in the training dataset
num_files_for_training = int(input("Enter the number of files you want to process for each category in training: "))

# Validate the input number
if num_files_for_training > 0:
    process_and_save_images('train', num_files_for_training)
    process_and_save_images('test')  # Process all files for testing
else:
    print("Invalid number. Please enter a value greater than 0.")
