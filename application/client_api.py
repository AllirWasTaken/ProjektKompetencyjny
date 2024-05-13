import socket
import os
import struct

def client_prediction(path_to_image):
    # Set up the socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    port = 8211  # Server's port
    host ='localhost'
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

