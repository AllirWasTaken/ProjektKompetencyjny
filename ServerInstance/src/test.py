import os
os.environ["HSA_OVERRIDE_GFX_VERSION"]="10.3.0" 
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_definition import SimpleCNN  # Ensure this matches your model file and class name
from tqdm import tqdm
import shutil
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Directory where the model checkpoints are saved
checkpoint_path = './checkpoints/model_checkpoint.pth'

# Check for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define transformations for the test set
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3196], std=[0.2934]),
])

# Assuming the dataset directories are structured properly for the test set
test_dataset = datasets.ImageFolder(root='../processed/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Create a reverse mapping from index to class names
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

# Parameters
num_classes = 4  # Update this based on your specific dataset

# Load the trained model
model = SimpleCNN(num_classes=num_classes).to(device)

if os.path.isfile(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print(f"No checkpoint found at '{checkpoint_path}'")
    exit()

# Function to delete and recreate a directory
def reset_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# Path to save wrongly classified images
wrong_predictions_path = 'checkpoints/wrong_predictions'
reset_directory(wrong_predictions_path)  # Clear existing data

model.eval()  # Set model to evaluation mode

# Initialize lists to store labels and predictions
all_labels = []
all_predictions = []

# Test the model and collect labels and predictions
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing Model")):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        
        batch_start_idx = batch_idx * test_loader.batch_size

        # Process each image in the batch
        for i, (image, label, pred) in enumerate(zip(images, labels, predicted)):
            if label != pred:  # If the classification is wrong
                predicted_label_name = idx_to_class[pred.item()]
                true_label_name = idx_to_class[label.item()]

                pred_dir = os.path.join(wrong_predictions_path, predicted_label_name)
                os.makedirs(pred_dir, exist_ok=True)  # Create dir for predicted label name if not exists
                
                # Retrieve original filename
                img_idx = batch_start_idx + i
                _, original_filename = os.path.split(test_dataset.imgs[img_idx][0])
                
                image_name = f"{true_label_name}_{original_filename}"
                # Convert image tensor to PIL image
                pil_image = transforms.ToPILImage()(image.cpu())
                
                # Save the wrongly classified image
                image_path = os.path.join(pred_dir, image_name)
                pil_image.save(image_path)


# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('checkpoints/confusion_matrix.png')
plt.close()

# Calculate accuracy
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print(f'Accuracy of the model on the test images: {accuracy * 100:.2f}%')
