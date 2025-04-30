#!/bin/bash

# Install Python 3.12.8
# echo "Installing Python 3.12.8..."
# You might need to adjust this command based on your system's package manager
# For Debian/Ubuntu:
# sudo apt-get update
# sudo apt-get install python3.12 python3.12-pip 
# echo "Please ensure Python 3.12.8 and pip are installed correctly."
echo "Assumes Python and Pip are allready pre-installed on the system"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the specified Python packages
echo "Installing required Python packages..."
pip install numpy==2.0.2
pip install pandas==2.2.3
pip install pillow==11.1.0
pip install matplotlib==3.10.0
pip install torch==2.5.1 torchvision==0.20.1
pip install datasets==3.2.0
pip install scikit-learn==1.6.0
pip install tensorflow==2.18.0
pip install seaborn==0.13.2
pip install GitPython==3.1.44
pip install optuna==4.1.0
pip install optuna-dashboard==0.17.0

echo "All required packages have been installed."