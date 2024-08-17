# Check if the virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Output "Virtual environment is not activated. Activating..."
    # Activate the virtual environment
    .\venv\Scripts\Activate.ps1
} else {
    Write-Output "Virtual environment is already activated."
}

# Export the LD_LIBRARY_PATH based on the output of the Python command
$ld_library_path = python -c "import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ';' + os.path.dirname(nvidia.cudnn.lib.__file__))"
[System.Environment]::SetEnvironmentVariable('LD_LIBRARY_PATH', $ld_library_path, [System.EnvironmentVariableTarget]::Process)

# Run the Python script
python start.py