from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm  # Import tqdm for the progress bar

# Assuming your dataset of grayscale images is in 'path_to_grayscale_images'
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root='../processed', transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)  # Adjusted batch size for memory efficiency

mean = 0.
std = 0.
n_samples = 0
for images, _ in tqdm(loader, desc="Calculating Mean and Std"):  # Wrap loader with tqdm for the progress bar
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_samples += batch_samples

mean /= n_samples
std /= n_samples

print(f"Mean: {mean}")
print(f"Std: {std}")