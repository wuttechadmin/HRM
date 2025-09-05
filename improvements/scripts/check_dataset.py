#!/usr/bin/env python3
import numpy as np
import json
import os

def check_dataset(data_path):
    train_path = os.path.join(data_path, 'train')
    
    # Load the training data
    inputs = np.load(os.path.join(train_path, 'all__inputs.npy'))
    labels = np.load(os.path.join(train_path, 'all__labels.npy'))
    
    # Load dataset metadata
    with open(os.path.join(train_path, 'dataset.json'), 'r') as f:
        dataset_info = json.load(f)
    
    # Check identifiers
    with open(os.path.join(data_path, 'identifiers.json'), 'r') as f:
        identifiers = json.load(f)
    
    # Write results to file
    with open('dataset_check_results.txt', 'w') as f:
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Input shape: {inputs.shape}\n")
        f.write(f"Label shape: {labels.shape}\n")
        f.write(f"Identifiers: {identifiers}\n\n")
        
        f.write("Dataset Info:\n")
        f.write(json.dumps(dataset_info, indent=2) + "\n\n")
        
        # Check for empty cells (zeros)
        empty_cells_count = np.sum(inputs == 0)
        f.write(f"Number of empty cells (zeros): {empty_cells_count}\n")
        
        # Sample first grid
        f.write("\nSample input grid:\n")
        input_grid = inputs[0].reshape(9, 9)
        for row in input_grid:
            f.write(" ".join([str(int(x)) if x != 0 else "*" for x in row]) + "\n")
        
        f.write("\nSample label grid:\n")
        label_grid = labels[0].reshape(9, 9)
        for row in label_grid:
            f.write(" ".join([str(int(x)) for x in row]) + "\n")

if __name__ == "__main__":
    check_dataset('data/sudoku-extreme-1k-aug-1000-new')
    print("Check complete. Results written to dataset_check_results.txt")
