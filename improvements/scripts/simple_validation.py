#!/usr/bin/env python3
import numpy as np
import json
import os

# Configuration
data_path = 'data/sudoku-extreme-1k-aug-1000-new'
train_path = os.path.join(data_path, 'train')
output_file = 'validation_results.txt'

with open(output_file, 'w') as f:
    # Check if data exists
    f.write(f"Checking data in {data_path}...\n")
    if os.path.exists(data_path):
        f.write("✓ Data directory exists\n")
    else:
        f.write("✗ Data directory does not exist\n")
        exit(1)
    
    # Load identifiers
    identifiers_path = os.path.join(data_path, 'identifiers.json')
    if os.path.exists(identifiers_path):
        with open(identifiers_path, 'r') as ident_file:
            identifiers = json.load(ident_file)
            f.write(f"✓ Identifiers loaded: {identifiers}\n")
    else:
        f.write("✗ identifiers.json does not exist\n")
    
    # Load dataset metadata
    dataset_json_path = os.path.join(train_path, 'dataset.json')
    if os.path.exists(dataset_json_path):
        with open(dataset_json_path, 'r') as meta_file:
            metadata = json.load(meta_file)
            f.write("✓ Dataset metadata loaded\n")
            f.write(json.dumps(metadata, indent=2) + "\n")
    else:
        f.write("✗ dataset.json does not exist\n")
    
    # Load data files
    files_to_check = [
        'all__inputs.npy',
        'all__labels.npy',
        'all__puzzle_indices.npy',
        'all__puzzle_identifiers.npy',
        'all__group_indices.npy'
    ]
    
    for file_name in files_to_check:
        file_path = os.path.join(train_path, file_name)
        if os.path.exists(file_path):
            data = np.load(file_path)
            f.write(f"✓ {file_name} loaded, shape: {data.shape}\n")
            
            # For inputs, check for empty cells (zeros)
            if file_name == 'all__inputs.npy':
                empty_count = np.sum(data == 0)
                f.write(f"  Empty cells (zeros): {empty_count}\n")
                
                # Print first example
                first_grid = data[0].reshape(9, 9)
                f.write("\nFirst input grid:\n")
                for row in first_grid:
                    f.write(" ".join([str(int(x)) if x != 0 else "*" for x in row]) + "\n")
        else:
            f.write(f"✗ {file_name} does not exist\n")
    
    f.write("\nValidation complete!\n")

print(f"Validation complete. Results written to {output_file}")
