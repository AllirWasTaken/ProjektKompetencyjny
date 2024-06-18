#!/bin/bash

# Function to backup a model
backup_model() {
    src_dir="src/checkpoints"
    backup_dir="model_backups"
    echo "Backing up the model..."
    # Get the provided name for the backup
    backup_name="$1"
    # Check if the provided name is empty
    if [ -z "$backup_name" ]; then
        echo "Error: Please provide a name for the backup."
        return 1
    fi
    # Define destination directory
    dest_dir="$backup_dir/$backup_name"
    # Create destination directory if it doesn't exist
    mkdir -p "$dest_dir"
    # Copy the file to the destination directory
    cp "$src_dir/." "$dest_dir" -r
    # Check if the copy was successful
    if [ $? -eq 0 ]; then
        echo "Folder copied successfully to $dest_dir."
    else
        echo "Error: Failed to copy folder to $dest_dir."
        return 1
    fi
}

# Function to load a model
load_model() {
    backup_dir="model_backups"
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
    dest_dir="src/checkpoints"
    # Move the file to the source directory
    cp "$src_dir/." "$dest_dir" -r
    # Check if the move was successful
    if [ $? -eq 0 ]; then
        echo "Folder moved successfully to $dest_dir."
    else
        echo "Error: Failed to move folder to $dest_dir."
        return 1
    fi
}

# Function to remove current model data
remove_model_data() {
    echo "Are you sure you want to delete current model data? (yes/no)"
    read confirmation
    if [ "$confirmation" = "yes" ]; then
        rm -r src/checkpoints/*
        echo "Current model data deleted."
    else
        echo "Deletion canceled."
    fi
}

# Ask the user whether to backup, load a model, or remove current model data
echo "Do you want to backup, load a model, or remove current model data?"
select option in "Backup" "Load" "Remove"; do
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
        "Remove")
            remove_model_data
            break
            ;;
        *)
            echo "Invalid option. Please select 1, 2, or 3."
            ;;
    esac
done

