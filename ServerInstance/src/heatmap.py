import os
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model_definition import SimpleCNN

# Load the pre-trained ResNet50 model
device = torch.device("cpu")
print(f'Using device: {device}')
# Parameters
num_classes = 4  # Update this based on your specific dataset

model_path="serverFiles/model.pth"

# Load the trained model
model = SimpleCNN(num_classes=num_classes).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


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


# Function to preprocess the input image
def preprocess_image(img_path):
    preprocess = transforms.Compose([
        MakeSquarePad(fill=255, padding_mode='constant'),  # Dynamically pad the image to make it square
        transforms.Resize((512, 512)),  # Then resize it to 512x512
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3196], std=[0.2934]),
    ])
    img = Image.open(img_path).convert("L")  # Convert to grayscale
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

# Function to generate Grad-CAM
def generate_gradcam(model, img_tensor, target_layer):
    def get_gradients_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    gradients = []
    activations = []
    target_layer.register_forward_hook(lambda module, input, output: activations.append(output))
    target_layer.register_backward_hook(get_gradients_hook)
    
    output = model(img_tensor)
    class_idx = torch.argmax(output).item()
    
    model.zero_grad()
    output[0, class_idx].backward()
    
    grads_val = gradients[0].cpu().data.numpy().squeeze()
    activations = activations[0].cpu().data.numpy().squeeze()
    
    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]
    
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = np.uint8(cam * 255)
    cam = np.uint8(Image.fromarray(cam).resize((512, 512), Image.Resampling.LANCZOS))
    return cam

# Load and preprocess the image
img_path = '6.jpeg'
img_tensor = preprocess_image(img_path)

# Choose the target layer (e.g., last convolutional layer before the fully connected layer)
target_layer = model.resnet50.layer4[2].conv3

# Generate the Grad-CAM
cam = generate_gradcam(model, img_tensor, target_layer)

# Display the image and Grad-CAM
img = Image.open(img_path).convert("L")
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray')
plt.imshow(cam, cmap='jet', alpha=0.5)  # Overlay Grad-CAM
plt.axis('off')
plt.show()