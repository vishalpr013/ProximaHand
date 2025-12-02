#!/bin/bash

echo "========================================"
echo "  ProximaHand - Hand Proximity Demo"
echo "========================================"
echo ""

# Check if Python is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

python3 --version
echo ""

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Starting ProximaHand..."
echo ""

# Run the application
python3 main.py
