import os
import shutil
import pickle

os.environ["HSA_OVERRIDE_GFX_VERSION"]="10.3.0" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_definition import SimpleCNN  # Ensure this matches your model's import path




def load_accuracy(path):
    try:
        with open(path, 'rb') as fi:
            return pickle.load(fi)
    except:
        return 0


class Trainer:

    best_accuracy_path = f"./checkpoints/best_accuracy.pk"
    best_accuracy = load_accuracy(best_accuracy_path) * 1.0
    best_checkpoint_path = f"./checkpoints/best_checkpoint_{best_accuracy:{1}.{2}}"

    def __init__(self, model, lr_in, checkpoint_dir='./checkpoints'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)  # Move model to the appropriate device
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.main_process_id = os.getpid()
        self.lr=lr_in
        self.configure()
        self.current_accuracy = 0


    def configure(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=pow(10,self.lr))
        self.start_epoch = 0

    def save_checkpoint(self, epoch):
        checkpoint_name = 'model_checkpoint' + '.pth'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        print(f"Saving checkpoint to {checkpoint_path}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch']+1
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Resumed training from epoch {self.start_epoch + 1}")
        else:
            print("No checkpoint found, starting from scratch")

    def save_accuracy(self):
        with open(Trainer.best_accuracy_path, 'wb') as fi:
            pickle.dump(self.current_accuracy, fi)

    def save_best_checkpoint(self):

        if Trainer.best_accuracy < self.current_accuracy:
            print("New best accuaracy!")
            if os.path.isfile(Trainer.best_checkpoint_path):
                os.remove(Trainer.best_checkpoint_path)

            Trainer.best_accuracy = self.current_accuracy * 1.0
            Trainer.best_checkpoint_path = f"./checkpoints/best_checkpoint_{self.current_accuracy:{1}.{2}}"
            shutil.copy(self.checkpoint_path, Trainer.best_checkpoint_path)
            self.save_accuracy()
        return Trainer.best_accuracy, Trainer.best_checkpoint_path


    def test_and_add_statistics(self, loss):

        stats_directory = 'checkpoints/stats'
        

        self.model.eval()  # Set the model to evaluation mode
        true_labels = []
        predictions = []

        
        
        print("Testting the model after current epoch...")

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing Model"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predictions.extend(predicted.cpu().numpy())
        
        # Calculate metrics
        self.current_accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        kappa = cohen_kappa_score(true_labels, predictions)
        
        if os.path.exists(stats_directory)==False:
            os.makedirs(stats_directory)
        # Append metrics to their respective files
        with open(os.path.join(stats_directory, 'accuracy.txt'), 'a') as f:
            f.write(f"{self.current_accuracy}\n")
        with open(os.path.join(stats_directory, 'f1.txt'), 'a') as f:
            f.write(f"{f1}\n")
        with open(os.path.join(stats_directory, 'kappa_cohens.txt'), 'a') as f:
            f.write(f"{kappa}\n")
        with open(os.path.join(stats_directory, 'loss.txt'), 'a') as f:
            f.write(f"{loss}\n")
        print("Saved the model detection statistics, resuming training...")





    def train(self, num_epochs, resume_from_checkpoint=True):
        if resume_from_checkpoint:
            self.load_checkpoint()  # Load checkpoint if resuming
        else:
            print("Starting a new training session without loading a checkpoint.")

        train_dataset = datasets.ImageFolder(root='../processed/train', transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        test_dataset = datasets.ImageFolder(root='../processed/test', transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            self.current_epoch = epoch
            self.model.train()
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.start_epoch + num_epochs}'), start=1):
                # Move data to the appropriate device after the checkpoint batch check
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f'Epoch [{epoch+1}/{self.start_epoch + num_epochs}], Loss: {running_loss/len(self.train_loader)}')
            self.save_checkpoint(epoch)  # Save at the end of each epoch
            self.test_and_add_statistics(running_loss/len(self.train_loader))
            print("Accuracy: ",self.current_accuracy)
            Trainer.best_accuracy, Trainer.best_checkpoint_path = self.save_best_checkpoint()
            print("\n")


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0784], std=[0.1519]),
    ])

    num_classes = 4
    model = SimpleCNN(num_classes=num_classes)

    learning_rate=input("input learning rate 10^")

    lr=-2
    try:
        lr=int(learning_rate)
    except:
        print("failed to parse")
        exit()

    if lr>0:
        print("wrong value")
        exit()

    trainer = Trainer(model,lr)

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

    print("Program has decided it finished training model. Program will now close")

