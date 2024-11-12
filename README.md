# GNN-Verification

**STOR 566 Project**

This project utilizes Graph Neural Networks (GNNs) for the verification of SystemVerilog (SV) files. The goal is to parse `.sv` files, convert their Abstract Syntax Trees (ASTs) into graph structures, and prepare them as data objects suitable for GNN training.

---

## Prerequisites

Before starting, ensure you have the following installed on your system:

- **Git**: For cloning the repository.
- **Homebrew** (macOS users): For installing Python 3.9 if it's not already installed.
- **Python 3.9**: Required for compatibility with PyTorch.

---

## Setup Instructions

Follow these steps to set up the project environment and prepare for processing `.sv` files.

### 1. Clone the Repository

2. Place .sv Files in the sv_files Folder
   Create a folder named sv_files in the root directory of the project if it doesn't exist:

Place all your .sv files inside the sv_files folder. The script will process all .sv files within this folder and its subdirectories.

3. Make the setup.sh Script Executable
   Ensure the setup.sh script has the necessary permissions by running this in the terminal:

chmod +x setup.sh

4. Run the Setup Script

Execute the setup.sh script to set up the Python environment and install all required dependencies:

./setup.sh

This script will:

Check for Python 3.9 and install it via Homebrew if it's not found.
Create a virtual environment named env in the project directory.
Upgrade pip to the latest version.
Install all dependencies listed in requirements.txt.

5. Activate the Virtual Environment
   After the setup script completes, activate the virtual environment:

source env/bin/activate

Note: You need to activate the virtual environment every time you start a new terminal session to work on this project.

Processing .sv Files
Now that the environment is set up and your .sv files are in place, you can process them to generate the data objects required for GNN training.

1. Run process_sv_files.py
   Execute the following command:

python process_sv_files.py

This script will:

Traverse the sv_files folder and its subdirectories to find all .sv files.
Use ast_to_pyg.py to parse each .sv file and convert its AST into a PyTorch Geometric Data object.
Collect all Data objects into a list called data_objects.
Save data_objects to a file named data_objects.pt in the project root directory.

2. Confirm Successful Processing
   After running the script, you should see output similar to:

Processing /path/to/GNN-Verification/sv_files/your_file.sv
Total .sv files processed: X
Saved data_objects to data_objects.pt

Where X is the number of .sv files processed.

Using data_objects.pt in Training
The data_objects.pt file contains the serialized list of Data objects, each representing an .sv file's AST. You can now use this file in other scripts.

deactivate the virtual environment when you're done working:

deactivate
