import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_definition import SimpleCNN  # Ensure this matches your model file and class name
import os

# Directory where the model checkpoints are saved
checkpoint_path = './checkpoints/model_checkpoint.pth'

# Define transformations for the test set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Assuming the dataset directories are structured properly for the test set
test_dataset = datasets.ImageFolder(root='../processed/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Parameters
num_classes = 4  # Update this based on your specific dataset

# Load the trained model
model = SimpleCNN(num_classes=num_classes)

if os.path.isfile(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("No checkpoint found at '{}'".format(checkpoint_path))
    exit()

model.eval()  # Set model to evaluation mode

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total}%')
