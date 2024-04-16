print("Launching Server")
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
import os
import torch.nn.functional as F

model_path='serverFiles/model.pth'
image_path='serverFiles/images/received_image.jpeg'



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
transform = transforms.Compose([
    MakeSquarePad(fill=255, padding_mode='constant'),  # Dynamically pad the image to make it square
    transforms.Resize((512, 512)),  # Then resize it to 512x512
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3196], std=[0.2934]),
])



class ServerApi:
    def Server(self):
        print("launching server api")
        # Set up the socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = '192.168.0.129'  # Server's hostname or IP address
        port = 8211  # Port to listen on
        server_socket.bind((host, port))
        server_socket.listen(5)
        print("Listening for incoming connections...")

        # Accept a connection
        while(True):
            self.client_socket, addr = server_socket.accept()
            print("Got a connection from", addr)
            self.size = int.from_bytes(self.client_socket.recv(4),'little')
            self.SaveImage()
            print("Received the image successfully.")
            floats =self.MakePrediction()
            print("Made a prediction, deleting image")
            os.remove(image_path)
            data = struct.pack(f'{len(floats)}f', *floats)
            print("Sending result to client")
            self.client_socket.send(data)
            # Close the connection
            print("Closing client connection")
            self.client_socket.close()
            
        

    def SaveImage(self):
        # Receive the data in small chunks and write it into a file
        toRecieve=self.size
        data=self.client_socket.recv(toRecieve)
        while True:
            print("recived ",len(data)," bytes")
            toRecieve=self.size-len(data)
            if(toRecieve==0):
                break
            data += self.client_socket.recv(toRecieve)
            
            
        with open(image_path, 'wb') as f:
            f.write(data)


    def MakePrediction(self):
        image = Image.open(image_path)
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():  # Temporarily set all the requires_grad flag to false
            raw_output = self.model(input_tensor)


        numpyFloat=F.softmax(raw_output,dim=1).numpy()
        list=numpyFloat.tolist()
        result=list[0]

        return result







    def Launch(self):
        print("Loading model")
        self.device = torch.device("cpu")
        print(f'Using device: {self.device}')
        # Parameters
        num_classes = 4  # Update this based on your specific dataset

        # Load the trained model
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()

        self.Server()





    #def SendResult(self):




if __name__ == '__main__':
    server=ServerApi()
    server.Launch()