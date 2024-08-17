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

# Run the Python script
./.venv/bin/python3 start.py
