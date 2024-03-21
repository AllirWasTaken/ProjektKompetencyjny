import os
import sys
import signal
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_definition import SimpleCNN  # Ensure this matches your model's import path

class Trainer:
    def __init__(self, model, train_loader, checkpoint_dir='./checkpoints'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)  # Move model to the appropriate device
        self.model = model
        self.train_loader = train_loader
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.main_process_id = os.getpid()
        self.configure()

    def configure(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        signal.signal(signal.SIGINT, self.signal_handler)
        self.start_epoch = 0
        self.batch_interval = 100  # Save checkpoint every 100 batches, adjust as needed

    def save_checkpoint(self, epoch, batch_idx=None):
        checkpoint_name = 'model_checkpoint' + '.pth'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        print(f"\nSaving checkpoint to {checkpoint_path}...")
        torch.save({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Resumed training from epoch {self.start_epoch + 1}, batch {checkpoint.get('batch_idx', 'start')}")
            return checkpoint.get('batch_idx')
        else:
            print("No checkpoint found, starting from scratch")
            return None

    def signal_handler(self, sig, frame):
        if os.getpid() == self.main_process_id:
            print('\nYou pressed Ctrl+C! Saving checkpoint before exiting.')
            self.save_checkpoint(self.current_epoch,self.batch_id_signal)
            sys.exit(0)

    def train(self, num_epochs, resume_from_checkpoint=True):
            last_batch_idx = None
            self.batch_id_signal = None
            if resume_from_checkpoint:
                last_batch_idx = self.load_checkpoint()  # Load checkpoint if resuming
            else:
                print("Starting a new training session without loading a checkpoint.")
            
            for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
                self.current_epoch = epoch
                self.model.train()
                running_loss = 0.0
                for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.start_epoch + num_epochs}'), start=1):
                    if last_batch_idx and batch_idx <= last_batch_idx:
                        continue  # Skip batches until we reach the last saved batch
                    # Move data to the appropriate device after the checkpoint batch check
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    
                    # Save intra-epoch checkpoint
                    self.batch_id_signal=batch_idx
                    if batch_idx % self.batch_interval == 0:
                        self.save_checkpoint(epoch, batch_idx)

                last_batch_idx = None  # Reset for the next epoch
                print(f'Epoch [{epoch+1}/{self.start_epoch + num_epochs}], Loss: {running_loss/len(self.train_loader)}')
                self.save_checkpoint(epoch)  # Save at the end of each epoch

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root='../processed/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_classes = 4
    model = SimpleCNN(num_classes=num_classes)

    trainer = Trainer(model, train_loader)

    user_choice = input("Press 'N' to start a new training session or any other key to resume: ").lower()
    resume_training = True
    if user_choice == 'n':
        resume_training = False  # Don't attempt to load a checkpoint for new training sessions
        if os.path.exists(trainer.checkpoint_path):
            os.remove(trainer.checkpoint_path)  # Optionally delete the old checkpoint
            print("Deleted the old checkpoint.")
        print("Starting a new training session.")
    else:
        print("Attempting to resume training from the last checkpoint.")

    epochs_input = input("Enter the number of epochs to train for: ")
    try:
        num_epochs = int(epochs_input)
    except ValueError:
        print("Invalid number of epochs. Using default 10 epochs.")
        num_epochs = 10

    trainer.train(num_epochs, resume_from_checkpoint=resume_training)

