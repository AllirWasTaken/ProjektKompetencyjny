import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_data_loaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Assuming 'data_dir' contains 'train' and 'test' subdirectories
    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
