#!/bin/bash

# Check for Python 3.9 installation
if command -v python3.9 &>/dev/null; then
    PYTHON_BIN=python3.9
elif command -v python3 &>/dev/null && python3 --version | grep -q "3.9"; then
    PYTHON_BIN=python3
else
    echo "Python 3.9 not found. Installing via Homebrew..."
    brew install python@3.9
    PYTHON_BIN=python3.9
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "Creating a virtual environment..."
    $PYTHON_BIN -m venv env
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! Use 'source env/bin/activate' to activate the virtual environment."


