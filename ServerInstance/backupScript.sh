#!/bin/bash

# Define source and destination paths
src_dir="src/checkpoints"
backup_dir="model_backups"

# Function to backup a model
backup_model() {
    echo "Backing up the model..."
    # Get the provided name for the backup
    backup_name="$1"
    # Check if the provided name is empty
    if [ -z "$backup_name" ]; then
        echo "Error: Please provide a name for the backup."
        return 1
    fi
    # Define source file path
    src_file="$src_dir/model_checkpoint.pth"
    # Define destination directory
    dest_dir="$backup_dir/$backup_name"
    # Create destination directory if it doesn't exist
    mkdir -p "$dest_dir"
    # Copy the file to the destination directory
    cp "$src_file" "$dest_dir/"
    # Check if the copy was successful
    if [ $? -eq 0 ]; then
        echo "File copied successfully to $dest_dir."
    else
        echo "Error: Failed to copy file to $dest_dir."
        return 1
    fi
}

# Function to load a model
load_model() {
    echo "Loading a model..."
    # Get the provided name for the backup
    backup_name="$1"
    # Check if the provided name is empty
    if [ -z "$backup_name" ]; then
        echo "Error: Please provide a name for the backup."
        return 1
    fi
    # Define source directory path
    src_dir="$backup_dir/$backup_name"
    # Check if the backup directory exists
    if [ ! -d "$src_dir" ]; then
        echo "Error: Backup '$src_dir' not found."
        return 1
    fi
    # Define destination directory
    dest_dir="src/checkpoints/model_checkpoint.pth"
    # Move the file to the source directory with the name 'model_checkpoint.pth'
    cp "$src_dir/model_checkpoint.pth" "$dest_dir"
    # Check if the move was successful
    if [ $? -eq 0 ]; then
        echo "File moved successfully to $dest_dir."
    else
        echo "Error: Failed to move file to $dest_dir."
        return 1
    fi
}

# Ask the user whether to backup or load a model
echo "Do you want to backup or load a model?"
select option in "Backup" "Load"; do
    case "$option" in
        "Backup")
            read -p "Enter a name for the backup: " backup_name
            backup_model "$backup_name"
            break
            ;;
        "Load")
            read -p "Enter the name of the backup to load: " backup_name
            load_model "$backup_name"
            break
            ;;
        *)
            echo "Invalid option. Please select 1 or 2."
            ;;
    esac
done

