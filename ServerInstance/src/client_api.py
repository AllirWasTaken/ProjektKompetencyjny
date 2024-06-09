import socket
import os
import struct

host='none'

def client_prediction(path_to_image):
    # Set up the socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    port = 8211  # Server's port
    
    # Connect to the server
    client_socket.connect((host, port))

    client_socket.send(os.stat(path_to_image).st_size.to_bytes(4,'little'))
    

    # Open the JPEG file and send its contents
    with open(path_to_image, 'rb') as f:
        data = f.read()
        client_socket.send(data)


    result = client_socket.recv(32)
    floats = struct.unpack('f' * 4, result)
    
    

    # Close the connection
    client_socket.close()
    return floats

def configure(host_adress):
    global host
    host=host_adress
    return True


def send_auth(auth_key,socket):
    socket.recv(1024)
    socket.send(str(auth_key).encode('utf-8'))

def get_label_names(auth_key):
    global host
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    port = 8211  # Server's port
    
    # Connect to the server
    client_socket.connect((host, port))
    client_socket.send(str(2).encode('utf-8'))
    send_auth(auth_key,client_socket)
    raw=client_socket.recv(1024).decode('utf-8')

    labels=raw.split('|')
    labels.pop()

    return labels

def get_auth_key(login,password):
    global host
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    port = 8211  # Server's port
    
    # Connect to the server
    client_socket.connect((host, port))
    client_socket.send(str(0).encode('utf-8'))
    client_socket.recv(1024).decode('utf-8')
    data=login+'|'+password
    client_socket.send(str(data).encode('utf-8'))
    key =int(client_socket.recv(1024).decode('utf-8'))

    if(key==-1):
        print("Failed to login")
        return -1

    return key

def create_prediction_folder(auth_key):
    global host
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    port = 8211  # Server's port
    
    # Connect to the server
    client_socket.connect((host, port))
    client_socket.send(str(3).encode('utf-8'))
    send_auth(auth_key,client_socket)
    folder_key=int(client_socket.recv(1024).decode('utf-8'))
    
    return folder_key

def shutdown_server(auth_key):
    global host
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    port = 8211  # Server's port
    
    # Connect to the server
    client_socket.connect((host, port))
    client_socket.send(str(1).encode('utf-8'))
    send_auth(auth_key,client_socket)
    client_socket.recv(1024)

    return

def log_out(auth_key):
    global host
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    port = 8211  # Server's port
    
    # Connect to the server
    client_socket.connect((host, port))
    client_socket.send(str(5).encode('utf-8'))
    send_auth(auth_key,client_socket)
    return

def create_user(auth_key,login,password):
    global host
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    port = 8211  # Server's port
    
    # Connect to the server
    client_socket.connect((host, port))
    client_socket.send(str(4).encode('utf-8'))
    send_auth(auth_key,client_socket)
    client_socket.recv(1024)

    data=login+'|'+password

    client_socket.send(str(data).encode('utf-8'))
    client_socket.recv(1024)


    return



def add_image_to_prediction(auth_key,folder_key,path_to_image):
    global host
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    port = 8211  # Server's port
    
    # Connect to the server
    client_socket.connect((host, port))
    client_socket.send(str(6).encode('utf-8'))
    send_auth(auth_key,client_socket)
    client_socket.recv(1024)
    client_socket.send(str(folder_key).encode('utf-8'))
    client_socket.recv(1024)

    client_socket.send(os.stat(path_to_image).st_size.to_bytes(4,'little'))
    

    # Open the JPEG file and send its contents
    with open(path_to_image, 'rb') as f:
        data = f.read()
        client_socket.send(data)

    client_socket.recv(1024)


def make_mass_prediction(auth_key,folder_key):
    global host
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    port = 8211  # Server's port
    
    # Connect to the server
    client_socket.connect((host, port))
    client_socket.send(str(7).encode('utf-8'))
    send_auth(auth_key,client_socket)

    client_socket.recv(1024)
    client_socket.send(str(folder_key).encode('utf-8'))

    raw=client_socket.recv(1024).decode('utf-8')
    raw_images=raw.split('|')
    raw_images.pop()

    result=[]
    

    for image in raw_images:
        line=[]
        temp=image.split('_')
        temp.pop()
        for element in temp:
            line.append(float(element))
        result.append(line)
        
    return result
    
