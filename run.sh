#!/bin/bash

# activate virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment is not activated. Activating..."
    # Activate the virtual environment
    source "./.venv/bin/activate"
else
    echo "Virtual environment is already activated."
fi

# Export the LD_LIBRARY_PATH based on the output of the Python command
export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')

# # Function to handle keyboard interrupt (Ctrl+C)
# trap "echo 'Keyboard interrupt. Exiting...'; exit" SIGINT

while true
do
    # Run the Python script
    python3 start.py

    # Check the exit code of the Python script
    EXIT_CODE=$?

    # If the script exited with 0, exit the loop
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Exiting"
        break
    else
        # Otherwise, print a message and restart the script
        echo "Unexpected stopped with exit code $EXIT_CODE. Restarting..."
    fi
done
