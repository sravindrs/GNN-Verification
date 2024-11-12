# process_sv_files.py
"""
This script iterates through all `.sv` files in the `sv_files` folder located within the 
specified repository directory, converts each file's AST to a PyTorch Geometric Data object 
using the `sv_to_data` function from `ast_to_pyg.py`, and stores all Data objects in a list.

Expectations:
- `.sv` files are expected to be located within the `sv_files` folder in the specified 
  repository directory (and any of its subdirectories).
- `ast_to_pyg.py` should be in the same directory as this script or accessible within the Python path.

Main Output:
- This script creates a file named `data_objects.pt` in the project directory, which contains 
  the list of `Data` objects, one for each `.sv` file processed.
- Each `Data` object in `data_objects.pt` represents the AST structure of a `.sv` file, 
  making it ready for use in GNN training or further processing.

Usage:
1. Set the `repo_dir` variable to the path of the cloned repository containing the `sv_files` folder.
2. Ensure that all `.sv` files are located inside `sv_files`.
3. Run this script to generate and store the `data_objects.pt` file.
4. (Optional) You can use `data_objects.pt` with a PyTorch Geometric DataLoader for batch processing 
   in your training scripts.
"""

import os
import torch
from ast_to_pyg import sv_to_data

# Specify the path to the cloned repository containing the `sv_files` folder
repo_dir = '/Local/Path/To/Directory'  # Set this path to the repository directory

data_objects = []
sv_files_dir = os.path.join(repo_dir, 'sv_files')

for root, dirs, files in os.walk(sv_files_dir):
    for file in files:
        if file.endswith('.sv'):
            file_path = os.path.join(root, file)
            print(f"Processing {file_path}")
            data = sv_to_data(file_path)
            data_objects.append(data)

print(f"Total .sv files processed: {len(data_objects)}")

torch.save(data_objects, 'data_objects.pt')
print("Saved data_objects to data_objects.pt")
