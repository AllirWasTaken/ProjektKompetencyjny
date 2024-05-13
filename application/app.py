# Import necessary modules
import customtkinter
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import socket
import os
import struct
from client_api import client_prediction

# Set appearance mode and color theme
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

# Global variable to store the currently displayed image
current_image = None


# Placeholder function for image analysis
def analyze_image(image_path):
    try:
        # Placeholder for actual image analysis logic
        messagebox.showinfo("Analysis Result", "Eye disease detected: Glaucoma")
    except Exception as e:
        messagebox.showerror("Analysis Error", f"An error occurred during analysis: {str(e)}")


# Function to open an image and display it on the canvas
def open_image():
    global current_image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        try:
            # Open the image file
            image = Image.open(file_path)

            # Store the current image
            current_image = image

            # Calculate the dimensions to resize the image while maintaining aspect ratio
            update_image_size()

        except Exception as e:
            messagebox.showerror("Image Error", f"An error occurred while opening the image: {str(e)}")


# Function to analyze the currently displayed image
# Function to analyze the currently displayed image
# Function to analyze the currently displayed image
def float_array_to_string(float_array):
    # Convert each float in the array to a string and join them with commas
    return ', '.join(map(str, float_array))

def analyze_current_image():
    global current_image
    if current_image:
        # Convert RGBA image to RGB mode if it's RGBA
        if current_image.mode == 'RGBA':
            current_image = current_image.convert('RGB')

        # Save the image temporarily
        temp_image_path = "temp_image.jpeg"
        current_image.save(temp_image_path)
        try:
            # Call the client_prediction function to analyze the image
            result = client_prediction(temp_image_path)

            # Process the result (for example, display it)
            # Here you may need to handle binary data according to your server's response
            # For example, you may need to convert binary data to a string or parse it differently
            # For demonstration purposes, let's assume the result is a string
            result_str = float_array_to_string(result)
            

            messagebox.showinfo("Analysis Result", f"Eye disease detected: {result_str}")
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis: {str(e)}")
        finally:
            # Remove the temporary image file
            os.remove(temp_image_path)
    else:
        messagebox.showinfo("Information", "No image is currently displayed.")


# Function to view analysis history
def view_analysis_history():
    # Placeholder function for viewing analysis history
    messagebox.showinfo("Analysis History", "No analysis history available yet.")


# Function to update the size of the displayed image on the canvas
def update_image_size():
    global current_image
    try:
        if current_image:
            # Retrieve the current image
            image = current_image

            # Calculate the dimensions to resize the image while maintaining aspect ratio
            container_width = canvas.winfo_width()
            container_height = canvas.winfo_height()
            image_width, image_height = image.size
            aspect_ratio = min(container_width / image_width, container_height / image_height)
            new_width = int(image_width * aspect_ratio)
            new_height = int(image_height * aspect_ratio)

            # Resize the image with LANCZOS filter (anti-aliasing)
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)

            # Convert the resized image for Tkinter
            tk_image = ImageTk.PhotoImage(resized_image)

            # Clear the canvas before drawing the resized image
            canvas.delete("all")

            # Display the resized image on the canvas
            canvas.create_image(container_width / 2, container_height / 2, anchor="center", image=tk_image)

            # Configure the canvas to expand and fill its parent frame
            canvas.config(scrollregion=canvas.bbox("all"))

            # Store the reference to the image to prevent it from being garbage collected
            canvas.image = tk_image

    except Exception as e:
        messagebox.showerror("Image Error", f"An error occurred while resizing the image: {str(e)}")


# Function to connect to the server and perform prediction


# Function to check if the server is up
def is_server_up(host, port):
    try:
        # Set up a temporary socket to check if connection is successful
        with socket.create_connection((host, port), timeout=2) as temp_socket:
            return True
    except (socket.timeout, ConnectionRefusedError):
        return False

# Function to connect to the server and perform prediction




# Create the main window
root = customtkinter.CTk()
root.geometry("500x500")  # Increased height for better image display
root.title("Eye Disease Analyzer")

# Create the main frame
frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

# Create a label for instructions
label = customtkinter.CTkLabel(master=frame, text="Click the buttons to interact with the application:")
label.pack(pady=12, padx=10)

# Create a canvas for displaying images
canvas = customtkinter.CTkCanvas(master=frame)
canvas.pack(pady=12, padx=10, fill="both", expand=True)

# Create toolbar for buttons
toolbar = customtkinter.CTkFrame(master=frame)

# Create a button to analyze the current image
analyze_button = customtkinter.CTkButton(master=toolbar, text="Analyze Image", command=analyze_current_image)
analyze_button.pack(side="left", padx=5)

# Create a button to select an image
select_button = customtkinter.CTkButton(master=toolbar, text="Select Image", command=open_image)
select_button.pack(side="left", padx=5)

# Create a button to view analysis history
history_button = customtkinter.CTkButton(master=toolbar, text="View Analysis History", command=view_analysis_history)
history_button.pack(side="left", padx=5)

# Pack the toolbar below the canvas
toolbar.pack(side="bottom", fill="x", pady=5)

# Bind a function to the window's <Configure> event to handle window resizing
root.bind("<Configure>", lambda event: update_image_size())

# Start the Tkinter event loop
root.mainloop()
