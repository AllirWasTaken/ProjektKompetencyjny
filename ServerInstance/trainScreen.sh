#!/bin/bash

# Specify the script to execute within the screen session
SCRIPT="./AutoScreenScript.sh"
SESSION_NAME="training_console"

# Check if the screen session exists with the specified name
if screen -ls | grep -q "$SESSION_NAME"; then
    # If the session exists, reattach to it
    screen -r "$SESSION_NAME"
else
    # If the session does not exist, start a new screen session and execute the script
    screen -S "$SESSION_NAME" bash -c "$SCRIPT"
fi
