#!/usr/bin/env python3
import numpy as np
import json
import os

data_path = '/Users/robertburkhall/Development/HRM/data/sudoku-extreme-1k-aug-1000'
train_path = os.path.join(data_path, 'train')

# Load the training data
inputs = np.load(os.path.join(train_path, 'all__inputs.npy'))
labels = np.load(os.path.join(train_path, 'all__labels.npy'))
puzzle_indices = np.load(os.path.join(train_path, 'all__puzzle_indices.npy'))
puzzle_identifiers = np.load(os.path.join(train_path, 'all__puzzle_identifiers.npy'))
group_indices = np.load(os.path.join(train_path, 'all__group_indices.npy'))

# Load dataset metadata
with open(os.path.join(train_path, 'dataset.json'), 'r') as f:
    dataset_info = json.load(f)

# Print dataset info
print("Dataset Info:")
print(json.dumps(dataset_info, indent=2))

print("\nData Shapes:")
print(f"Inputs shape: {inputs.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Puzzle indices shape: {puzzle_indices.shape}")
print(f"Puzzle identifiers shape: {puzzle_identifiers.shape}")
print(f"Group indices shape: {group_indices.shape}")

# Print first 20 examples
print("\nFirst 20 examples:")
for i in range(min(20, len(inputs))):
    print(f"\nExample {i+1}:")
    print(f"Puzzle Index: {puzzle_indices[i]}")
    print(f"Group Index: {group_indices[i]}")
    
    # Format Sudoku grid for input
    input_grid = inputs[i].reshape(9, 9)
    print("Input Grid:")
    for row in input_grid:
        # Replace "1" with "*" to represent empty cells
        print(" ".join([str(int(x)) if int(x) != 1 else "*" for x in row]))
    
    # Format Sudoku grid for label
    label_grid = labels[i].reshape(9, 9)
    print("Label Grid:")
    for row in label_grid:
        print(" ".join([str(int(x)) for x in row]))
