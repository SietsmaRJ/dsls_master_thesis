#!/bin/bash
echo "Making venv direction"
mkdir venv
cd venv
echo "Creating virtual environment"
python3 -m venv ./
cd ..
echo "Installing required packages"
source "./venv/bin/activate"
pip install -r requirements.txt
echo "All done! Do not forget to activate the venv using:"
echo "source './venv/bin/activate' (you can use ctrl+shift+c to copy the command)"
