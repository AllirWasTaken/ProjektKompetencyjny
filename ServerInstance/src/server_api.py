print("Launching Server")
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"]="10.3.0" 
import socket
import struct
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_definition import SimpleCNN  # Ensure this matches your model file and class name
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from preprocess import preprocess_image

import torch.nn.functional as F
import random
import shutil
from torch.utils.data import Dataset

model_path='serverFiles/model.pth'
server_ip='localhost'
labelNames=['CNV','DME','DRUSEN','NORMAL']


class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpeg') or f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB
        if self.transform:
            image = self.transform(image)
        return image


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
# Define transformations for the test set

class UniqueRandom:
    def __init__(self):
        self.generated = set()

    def generate(self, range_start, range_end):
        if len(self.generated) == (range_end - range_start):
            raise ValueError("No more unique numbers available in the given range")
        while True:
            num = random.randint(range_start, range_end)
            if num not in self.generated:
                self.generated.add(num)
                return num
            
class NumberStack:
    def __init__(self):
        self.stack = []

    def push(self, number):
        self.stack.append(number)

    def is_present(self, number):
        return number in self.stack

    def delete(self, number):
        if number in self.stack:
            self.stack.reverse()  # Reverse to remove the first occurrence from the top
            self.stack.remove(number)
            self.stack.reverse()  # Reverse back to original order
            return True  # Number was found and removed
        return False  # Number not found

class StringSaver:
    def __init__(self, filepath):
        self.filepath = filepath
        # Ensure the file exists on initialization
        open(self.filepath, 'a').close()

    def add_user(self, string):
        with open(self.filepath, 'a') as file:
            file.write(string + '\n')
            return f"String added: {string}"
        

    def check_user(self, string):
        with open(self.filepath, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
        return string in lines
    

class ServerApi:
    def Server(self):
        print("launching server api")
        # Set up the socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = server_ip  # Server's hostname or IP address
        port = 8211  # Port to listen on
        server_socket.bind((host, port))
        server_socket.listen(5)
        print("Listening for incoming connections...")

        # Accept a connection
        while(True):
            self.client_socket, addr = server_socket.accept()
            print("Got a connection from", addr)
            type =self.client_socket.recv(1024).decode('utf-8')
            type=int(type)
            if(type==0):
                print("login")
                self.GenerateAuthKey()
            if(type==1):
                print('shutdown')
                if(self.shutdown()):
                    print('shuting down')
                    break
            if(type==2):
                print('label names')
                self.LabelNames()
            if(type==3):
                print('folder')
                self.CreateUserFolder()
            if(type==4):
                print('create user')
                self.CreateUser()
            if(type==5):
                print('log out')
                self.LogOut()
            if(type==6):
                print('add image')
                self.AddImage()
            if(type==7):
                print('analyze')
                self.Analyze()
        server_socket.close()
                
    def list_directories(self,path):
        # List all entries in the directory and filter them to include only directories
        folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        return folder_names

    def LogOut(self):
        key=self.Autheticate()
        if(key==False):
            return
        self.authStack.delete(key)
        self.client_socket.close()

    def get_dataloader(self,folder_path, batch_size=4):
        transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0784], std=[0.1519]),
        ])
        dataset = ImageDataset(folder_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader
    
    def count_files_in_directory(self,directory_path):
    # Get the list of all files and directories in the specified directory
        files_and_dirs = os.listdir(directory_path)
        
        # Filter out directories, keeping only files
        files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory_path, f))]
        
        # Return the count of files
        return len(files)
        
    def Analyze(self):
        if(self.Autheticate()==False):
            return
        self.client_socket.send(str('ok').encode('utf-8'))

        folder=int(self.client_socket.recv(1024).decode('utf-8'))
        if(self.folderStack.is_present(folder)==False):
            self.client_socket.close()
            return False
        
        folder_path='serverFiles/images/'+str(folder)
        
        ret=[]
        dataloader=self.get_dataloader(folder_path,4)

        with torch.no_grad():
            for batch in dataloader:
                batch=batch.to(self.device)
                outputs = self.model(batch)# Adjust this depending on your model's output
                # Convert each item in the batch
                for output in outputs:
                    output_2d = output.squeeze().cpu()
                    numpyFloat=F.softmax(output_2d,dim=0).numpy()
                    ret.append(numpyFloat.tolist())


        raw=''
        for item in ret:
            for number in item:
                raw+=str(number)+'_'
            raw+='|'

        self.client_socket.send(str(raw).encode('utf-8'))
        self.client_socket.close()
        shutil.rmtree(folder_path)
        self.folderStack.delete(folder)

        


    def AddImage(self):
        if(self.Autheticate()==False):
            return
        self.client_socket.send(str('ok').encode('utf-8'))

        folder=int(self.client_socket.recv(1024).decode('utf-8'))
        if(self.folderStack.is_present(folder)==False):
            self.client_socket.close()
            return False
        
        self.client_socket.send(str('unpause').encode('utf-8'))
        
        self.size = int.from_bytes(self.client_socket.recv(4),'little')
        path='serverFiles/images/'+str(folder)+'/ob.jpeg'
        self.SaveImage(path)
        self.client_socket.close()

    def LabelNames(self):
        if(self.Autheticate()==False):
            return
        name=''
        for label in labelNames:
            name+=label+'|'
        self.client_socket.send(str(name).encode('utf-8'))
        self.client_socket.close()
    
    def CreateUser(self):
        key=self.Autheticate()
        if(key==False):
            return
        if(key!=self.admin):
            self.client_socket.close()
            return
        self.client_socket.send(str('unpause').encode('utf-8'))
        data=self.client_socket.recv(1024).decode('utf-8')

        if(self.users.check_user(data)):
            self.client_socket.send(str('not ok').encode('utf-8'))
            self.client_socket.close()
            return
        
        self.users.add_user(data)
        self.client_socket.send(str('ok').encode('utf-8'))
        self.client_socket.close()

           
    def GenerateAuthKey(self):
        self.client_socket.send(str('unpause').encode('utf-8'))
        data = self.client_socket.recv(1024).decode('utf-8')  # Receive data from the client
        if(data=='admin|papuga12'):    
            key = self.authGen.generate(0,1000000000)
            self.authStack.push(key)
            self.admin=key
            self.client_socket.sendall(str(key).encode('utf-8'))
        else:
            if(self.users.check_user(data)):
                key = self.authGen.generate(0,1000000000)
                self.authStack.push(key)
                self.client_socket.sendall(str(key).encode('utf-8'))
            else:
                self.client_socket.sendall(str(-1).encode('utf-8'))

        self.client_socket.close()


    def Autheticate(self):
        self.client_socket.send(str('unpause').encode('utf-8'))
        data = self.client_socket.recv(1024).decode('utf-8')
        if(self.authStack.is_present(int(data))):
            return int(data)
        self.client_socket.close()
        return False
    
    def CreateUserFolder(self):
        if(self.Autheticate()==False):
            return
        newfolder=self.authGen.generate(0,1000000000)
        self.folderStack.push(newfolder)
        os.mkdir('serverFiles/images/'+str(newfolder))

        self.client_socket.send(str(newfolder).encode('utf-8'))
        self.client_socket.close()
        

    def shutdown(self):
        key=self.Autheticate()
        if(key==False):
            return False
        if(key!=self.admin):
            self.client_socket.close()
            return False
        return True

    def SaveImage(self,path):
        def get_unique_filename(base_path):
            directory, filename = os.path.split(base_path)
            base, extension = os.path.splitext(filename)
            counter = 1
            new_path = os.path.join(directory, f"{base}_{counter}{extension}")
            while os.path.exists(new_path):
                new_path = os.path.join(directory, f"{base}_{counter}{extension}")
                counter += 1
            return new_path
        # Receive the data in small chunks and write it into a file
        toRecieve=self.size
        data=self.client_socket.recv(toRecieve)
        while True:
            print("recived ",len(data)," bytes")
            toRecieve=self.size-len(data)
            if(toRecieve==0):
                break
            data += self.client_socket.recv(toRecieve)
            
        temp_path=path.replace('ob','temp')
        with open(temp_path, 'wb') as f:
            f.write(data)

        unique=get_unique_filename(path)
        preprocess_image(temp_path,unique)
        os.remove(temp_path)




    def Launch(self):
        print("Loading model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        # Parameters
        num_classes = 4  # Update this based on your specific dataset

        # Load the trained model
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.authGen=UniqueRandom()
        self.authStack=NumberStack()
        self.folderStack=NumberStack()
        self.users=StringSaver('serverFiles/users')
        self.model.eval()
        self.admin=-1

        self.Server()





    #def SendResult(self):




if __name__ == '__main__':
    server=ServerApi()
    server.Launch()
