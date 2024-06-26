import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model_definition import SimpleCNN  

# Load the trained ResNet50 model
device=torch.device("cpu")
checkpoint = torch.load("serverFiles/model.pth", map_location=device)


model = SimpleCNN(num_classes=4).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define the image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0784], std=[0.1519]),
])

# Load and preprocess the image
img_path = 'test_images/processed/1.jpeg'  # Change this to the path of your image
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0)  # Create a mini-batch as expected by the model

# Hook to capture activations and gradients
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# Register hooks
target_layer = model.resnet50.layer4[2].conv3  # This targets the final convolutional layer of ResNet50
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Forward pass
output = model(input_tensor)
output_idx = output.argmax()
output[:, output_idx].backward()

# Process CAM
gradients_np = gradients.cpu().data.numpy()[0]
activations_np = activations.cpu().data.numpy()[0]
weights = np.mean(gradients_np, axis=(1, 2))
cam = np.zeros(activations_np.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * activations_np[i, :, :]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (512, 512))
cam -= np.min(cam)
cam /= np.max(cam)

# Invert the CAM values
cam = 1 - cam

# Visualize the heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
img = np.array(img)
heatmap = heatmap + np.float32(img) / 255
heatmap = heatmap / np.max(heatmap)

plt.imshow(heatmap)
plt.axis('off')
plt.show()
