#!/usr/bin/env python3
import numpy as np
import json
import os
import argparse

def validate_data(data_path, num_examples=20):
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

    # Print statistics about empty cells
    empty_cells = np.sum(inputs == 0, axis=1)
    print(f"\nEmpty cells statistics:")
    print(f"Average empty cells per puzzle: {np.mean(empty_cells):.2f}")
    print(f"Min empty cells: {np.min(empty_cells)}")
    print(f"Max empty cells: {np.max(empty_cells)}")

    # Print examples
    print(f"\nFirst {num_examples} examples:")
    for i in range(min(num_examples, len(inputs))):
        print(f"\nExample {i+1}:")
        print(f"Puzzle Index: {puzzle_indices[i]}")
        print(f"Group Index: {group_indices[i] if i < len(group_indices) else 'N/A'}")
        
        # Format Sudoku grid for input
        input_grid = inputs[i].reshape(9, 9)
        print("Input Grid:")
        for row in input_grid:
            # Replace 0 with "*" to represent empty cells
            print(" ".join([str(int(x)) if x != 0 else "*" for x in row]))
        
        # Format Sudoku grid for label
        label_grid = labels[i].reshape(9, 9)
        print("Label Grid:")
        for row in label_grid:
            print(" ".join([str(int(x)) for x in row]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Sudoku dataset")
    parser.add_argument("--data_path", type=str, default="/Users/robertburkhall/Development/HRM/data/sudoku-extreme-1k-aug-1000", 
                        help="Path to the dataset directory")
    parser.add_argument("--num_examples", type=int, default=20,
                        help="Number of examples to display")
    args = parser.parse_args()
    
    validate_data(args.data_path, args.num_examples)
