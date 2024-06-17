import customtkinter
from tkinter import filedialog, messagebox, Toplevel, Scrollbar, Text, RIGHT, Y, END, Frame, LEFT, BOTH
from tkinter.ttk import Treeview, Progressbar
from PIL import Image, ImageTk
import os
import json
import datetime
import tkinter as tk
import client_api as api
import threading

# Set appearance mode and color theme
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

# Automatically configure the server address when the application starts
server_address = "localhost"
api.configure(server_address)

# Global variables
current_images = []
original_images = []  # To store the original images
current_user = None
auth_key = None  # Authentication key for the server
disease_labels = []  # To store the disease labels
history_file = 'history.json'
analysis_result = ""  # To store the analysis result

# Function to handle login
def handle_login(username, password):
    global current_user, auth_key, disease_labels
    auth_key = api.get_auth_key(username, password)
    if auth_key != -1:
        current_user = username
        messagebox.showinfo("Login Successful", f"Logged in as: {current_user}")
        # Fetch disease labels
        disease_labels = api.get_label_names(auth_key)
        if not disease_labels:
            disease_labels = ["Choroba 1", "Choroba 2", "Choroba 3", "Zdrowe Oko"]  # Default fallback
        # Enable the buttons related to image analysis, opening images, viewing history, and submitting comments
        analyze_button.configure(state="normal")
        open_image_button.configure(state="normal")
        view_history_button.configure(state="normal")
        submit_comment_button.configure(state="normal")
        logout_button.configure(state="normal")
    else:
        messagebox.showerror("Login Failed", "Failed to login. Check your credentials or server configuration.")

# Function to handle logout
def handle_logout():
    global current_user, auth_key
    if auth_key:
        api.log_out(auth_key)
        current_user = None
        auth_key = None
        messagebox.showinfo("Logout Successful", "You have been logged out.")
        # Disable buttons after logout
        analyze_button.configure(state="disabled")
        open_image_button.configure(state="disabled")
        view_history_button.configure(state="disabled")
        submit_comment_button.configure(state="disabled")
        logout_button.configure(state="disabled")

# Function to open the login window
def open_login_window():
    login_window = Toplevel(root)
    login_window.title("Login")
    login_window.geometry("300x200")

    username_label = customtkinter.CTkLabel(login_window, text="Username:")
    username_label.pack(pady=5)
    username_entry = customtkinter.CTkEntry(login_window)
    username_entry.pack(pady=5)

    password_label = customtkinter.CTkLabel(login_window, text="Password:")
    password_label.pack(pady=5)
    password_entry = customtkinter.CTkEntry(login_window, show='*')
    password_entry.pack(pady=5)

    login_button = customtkinter.CTkButton(login_window, text="Login", command=lambda: handle_login(username_entry.get(), password_entry.get()))
    login_button.pack(pady=10)

# Function to open an image and display it on the canvas
def open_image():
    global current_images, original_images
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        try:
            image = Image.open(file_path)
            original_images = [image.copy()]
            current_images = [image]
            update_image_size()
        except Exception as e:
            messagebox.showerror("Image Error", f"An error occurred while opening the image: {str(e)}")

# Function to open up to 4 images and display them on the canvas
def open_images():
    global current_images, original_images
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")], title="Select up to 4 images", multiple=True)
    if file_paths:
        try:
            images = [Image.open(file_path) for file_path in file_paths[:4]]
            original_images = [image.copy() for image in images]
            current_images = images
            update_image_size()
        except Exception as e:
            messagebox.showerror("Image Error", f"An error occurred while opening the images: {str(e)}")

# Function to translate the raw analysis result
def translate_result(result):
    global disease_labels
    translated_results = []

    for idx, probabilities in enumerate(result):
        if len(probabilities) != len(disease_labels):
            continue  # Skip this result if it doesn't match the number of labels
        max_prob_index = probabilities.index(max(probabilities))
        max_prob_disease = disease_labels[max_prob_index]
        max_prob_percentage = max(probabilities) * 100
        translated_results.append(f"{idx + 1}. {max_prob_disease}: {max_prob_percentage:.2f}%")

    return "\n".join(translated_results)

# Function to run analysis in a separate thread
def run_analysis():
    global current_images, current_user, analysis_result, auth_key
    results = []
    for image in current_images:
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        try:
            folder = api.create_prediction_folder(auth_key)
            api.add_image_to_prediction(auth_key, folder, temp_image_path)
            prediction = api.make_mass_prediction(auth_key, folder)

            results.append(prediction[0])
            log_analysis(temp_image_path, str(prediction[0]), current_user)
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis: {str(e)}")
        finally:
            os.remove(temp_image_path)

    translated_results = translate_result(results)
    analysis_result = f"Analysis Results:\n{translated_results}"
    result_display.configure(state='normal')
    result_display.delete('1.0', END)
    result_display.insert('1.0', analysis_result)
    result_display.configure(state='disabled')
    comment_entry.configure(state='normal')
    progress_bar.stop()
    progress_bar.pack_forget()  # Hide the progress bar after analysis

# Function to analyze the currently displayed image
def analyze_current_image():
    global current_images, current_user, auth_key, progress_bar
    if current_images:
        if current_user is None:
            messagebox.showinfo("Login Required", "Please log in before analyzing images.")
            return

        if auth_key is None:
            messagebox.showinfo("Configuration Required", "Please configure the server and log in.")
            return

        progress_bar.pack(pady=10)  # Show the progress bar
        progress_bar.start()
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.start()
    else:
        messagebox.showinfo("Information", "No image is currently displayed.")

# Function to view analysis history
def view_analysis_history():
    try:
        with open(history_file, 'r') as file:
            history_data = json.load(file)
            history = history_data.get('history', [])

            history_window = Toplevel(root)
            history_window.title("Analysis History")
            history_window.geometry("800x400")

            tree_frame = Frame(history_window)
            tree_frame.pack(fill='both', expand=True)

            columns = ("Date", "User", "Result", "Comment")
            tree = Treeview(tree_frame, columns=columns, show='headings')

            tree.heading("Date", text="Date")
            tree.heading("User", text="User")
            tree.heading("Result", text="Result")
            tree.heading("Comment", text="Comment")

            tree.column("Date", width=150)
            tree.column("User", width=100)
            tree.column("Result", width=200)
            tree.column("Comment", width=300)

            scrollbar = Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=RIGHT, fill=Y)

            for entry in history:
                date = entry['timestamp']
                user = entry['user']
                try:
                    result = translate_result([eval(entry['disease'])])
                except IndexError:
                    result = "N/A"  # Handle cases where result is not in expected format
                comment = entry.get('comment', '')  # Handle missing comments
                tree.insert("", "end", values=(date, user, result, comment))

            tree.pack(fill='both', expand=True)

    except FileNotFoundError:
        messagebox.showinfo("Analysis History", "No analysis history available yet.")
    except Exception as e:
        messagebox.showerror("History Error", f"An error occurred while loading the history: {str(e)}")

# Function to log the analysis to a JSON file
def log_analysis(image_path, disease, user):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis_entry = {
        "image": image_path,
        "disease": disease,
        "timestamp": timestamp,
        "user": user,
        "comment": comment_entry.get("1.0", "end-1c").strip()  # Capture the current comment
    }

    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as file:
                history_data = json.load(file)
        else:
            history_data = {"history": []}

        history_data["history"].append(analysis_entry)

        with open(history_file, 'w') as file:
            json.dump(history_data, file, indent=4)

    except Exception as e:
        messagebox.showerror("Logging Error", f"An error occurred while logging the analysis: {str(e)}")

# Function to display an image in a new window
def show_image_in_new_window(original_image):
    new_window = Toplevel(root)
    new_window.title("Image Viewer")

    # Set the new window size to 60% of the main window size
    main_window_width = root.winfo_width()
    main_window_height = root.winfo_height()
    new_window.geometry(f"{int(main_window_width * 0.6)}x{int(main_window_height * 0.6)}")

    display_frame = Frame(new_window, bg="black")
    display_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(display_frame, bg="black")
    canvas.pack(fill="both", expand=True)

    def resize_image(event):
        display_width = display_frame.winfo_width()
        display_height = display_frame.winfo_height()
        aspect_ratio = min(display_width / original_image.width, display_height / original_image.height)
        new_width = int(original_image.width * aspect_ratio)
        new_height = int(original_image.height * aspect_ratio)

        if new_width <= 0 or new_height <= 0:
            return  # Avoid resizing to zero or negative dimensions

        resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_image)

        canvas.delete("all")
        canvas.create_image(display_width / 2, display_height / 2, anchor="center", image=tk_image)

        if not hasattr(canvas, 'images'):
            canvas.images = []
        canvas.images.append(tk_image)

    new_window.bind("<Configure>", resize_image)

# Function to update the size of the displayed images on the canvas
def update_image_size():
    global current_images
    try:
        if current_images:
            images = current_images
            canvas.delete("all")

            container_width = image_frame.winfo_width()
            container_height = image_frame.winfo_height()
            grid_size = 2

            padding = 20
            cell_width = (container_width - padding * (grid_size + 1)) / grid_size
            cell_height = (container_height - padding * (grid_size + 1)) / grid_size

            total_grid_width = cell_width * grid_size + padding * (grid_size + 1)
            total_grid_height = cell_height * grid_size + padding * (grid_size + 1)

            start_x = (container_width - total_grid_width) / 2
            start_y = (container_height - total_grid_height) / 2

            for idx, image in enumerate(images):
                image_width, image_height = image.size
                aspect_ratio = min((cell_width - padding * 2) / image_width, (cell_height - padding * 2) / image_height)
                new_width = int(image_width * aspect_ratio)
                new_height = int(image_height * aspect_ratio)
                resized_image = image.resize((new_width, new_height), Image.LANCZOS)
                tk_image = ImageTk.PhotoImage(resized_image)

                row, col = divmod(idx, grid_size)
                x = start_x + padding + col * (cell_width + padding)
                y = start_y + padding + row * (cell_height + padding)
                x_centered = x + (cell_width - new_width) / 2
                y_centered = y + (cell_height - new_height) / 2

                img_id = canvas.create_image(x_centered, y_centered, anchor="nw", image=tk_image)
                canvas.tag_bind(img_id, "<Button-1>", lambda event, img=original_images[idx]: show_image_in_new_window(img))

                border_padding = 5
                canvas.create_rectangle(
                    x_centered - border_padding, y_centered - border_padding,
                    x_centered + new_width + border_padding, y_centered + new_height + border_padding,
                    outline="red", width=2
                )

                # Draw the image index number in the top-left corner
                number_x = x_centered + padding
                number_y = y_centered + padding
                canvas.create_text(number_x, number_y, text=f"{idx + 1}", fill="red", font=("Arial", 20))

                if not hasattr(canvas, 'images'):
                    canvas.images = []
                canvas.images.append(tk_image)

            for i in range(1, grid_size):
                canvas.create_line(start_x + i * (cell_width + padding), start_y,
                                   start_x + i * (cell_width + padding), start_y + total_grid_height,
                                   fill="red", width=2)
                canvas.create_line(start_x, start_y + i * (cell_height + padding),
                                   start_x + total_grid_width, start_y + i * (cell_height + padding),
                                   fill="red", width=2)

            canvas.config(scrollregion=canvas.bbox("all"))

    except Exception as e:
        messagebox.showerror("Image Error", f"An error occurred while displaying the images: {str(e)}")

# Function to handle comment submission
def submit_comment():
    global comment_entry, current_user
    comment = comment_entry.get("1.0", "end-1c").strip()
    if comment:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        comment_entry.configure(state='disabled')

        messagebox.showinfo("Comment Submitted", f"Comment by {current_user} at {timestamp}:\n\n{comment}")
        comment_entry.delete("1.0", "end")

    else:
        messagebox.showwarning("Empty Comment", "Comment cannot be empty.")

# Main window setup
def show_main_window():
    global canvas, comment_entry, result_display, analyze_button, open_image_button, view_history_button, submit_comment_button, image_frame, logout_button, progress_bar
    global root

    root = customtkinter.CTk()
    root.title("Eye Disease Detection")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    initial_width = int(screen_width * 0.8)
    initial_height = int(screen_height * 0.8)

    # Calculate the position to center the window
    position_right = int(screen_width / 2 - initial_width / 2)
    position_down = int(screen_height / 2 - initial_height / 2)
    root.geometry(f"{initial_width}x{initial_height}+{position_right}+{position_down}")

    # Adjust layout for image display (60%) and buttons (40%)
    image_frame = Frame(root, bg="black", width=int(initial_width * 0.6), height=initial_height)
    image_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    image_frame.pack_propagate(False)

    canvas = tk.Canvas(image_frame, bg="black")
    canvas.pack(fill="both", expand=True)

    button_frame = customtkinter.CTkFrame(root, width=int(initial_width * 0.4))
    button_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    button_frame_inner = customtkinter.CTkFrame(button_frame)
    button_frame_inner.pack(pady=10)

    # Arrange buttons horizontally and wrap to new line if necessary
    analyze_button = customtkinter.CTkButton(button_frame_inner, text="Analyze Image/Images", command=analyze_current_image, state="disabled")
    analyze_button.grid(row=0, column=0, padx=5, pady=5)

    open_image_button = customtkinter.CTkButton(button_frame_inner, text="Open Image/Images", command=open_images, state="disabled")
    open_image_button.grid(row=0, column=1, padx=5, pady=5)

    view_history_button = customtkinter.CTkButton(button_frame_inner, text="View History", command=view_analysis_history, state="normal")
    view_history_button.grid(row=0, column=2, padx=5, pady=5)

    login_button = customtkinter.CTkButton(button_frame_inner, text="Login", command=open_login_window)
    login_button.grid(row=1, column=0, padx=5, pady=5)

    logout_button = customtkinter.CTkButton(button_frame_inner, text="Logout", command=handle_logout, state="disabled")
    logout_button.grid(row=1, column=1, padx=5, pady=5)

    result_label = customtkinter.CTkLabel(button_frame, text="Analysis Result:")
    result_label.pack(pady=10)

    result_display = tk.Text(button_frame, height=10, width=60, state='disabled', wrap='word')
    result_display.pack(pady=10)

    comment_label = customtkinter.CTkLabel(button_frame, text="Comment:")
    comment_label.pack(pady=10)

    comment_frame = Frame(button_frame)
    comment_frame.pack(pady=10)

    comment_entry = tk.Text(comment_frame, height=15, width=80, wrap='word', state='disabled')
    comment_scrollbar = Scrollbar(comment_frame, command=comment_entry.yview)
    comment_scrollbar.pack(side=RIGHT, fill=Y)
    comment_entry.config(yscrollcommand=comment_scrollbar.set)
    comment_entry.pack(side=LEFT, fill=BOTH, expand=True)

    submit_comment_button = customtkinter.CTkButton(button_frame, text="Submit Comment", command=submit_comment, state="disabled")
    submit_comment_button.pack(pady=10)

    # Create a progress bar but keep it hidden initially
    progress_bar = Progressbar(button_frame, orient=tk.HORIZONTAL, length=300, mode='indeterminate')

    root.bind("<Configure>", lambda event: update_image_size())

    root.mainloop()

if __name__ == "__main__":
    show_main_window()