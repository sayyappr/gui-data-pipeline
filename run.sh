#!/bin/bash

# Step 1: Create a virtual environment (if not already created)
# Check if the virtual environment exists, and create one if not
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Step 2: Activate the virtual environment
source venv/bin/activate

# Step 3: Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 4: Run the download and unzip script
echo "Downloading and unzipping the dataset..."
python download_and_unzip.py

# Step 5: Parse the OS-Atlas dataset
echo "Parsing OS-Atlas data..."
python parse_osatlas.py

# Step 6: Parse the AGUVIS data
echo "Parsing AGUVIS data (OmniAct)..."
python parse_aguvis1_omniact.py

# Step 7: Run AGUVIS parsing script
echo "Running AGUVIS parsing..."
python aguvis.py

# Deactivate the virtual environment after running all scripts
deactivate

echo "Pipeline executed successfully."
