#!/usr/bin/env python
# quick_dataset_check.py - Simple script to check dataset consistency

import numpy as np
import os
from pathlib import Path
import sys
import random

def is_valid_sudoku(grid_flat):
    """Check if a flattened 9x9 grid is a valid Sudoku (no duplicates in rows/cols/boxes)"""
    grid = grid_flat.reshape(9, 9)
    
    # Check rows
    for i in range(9):
        row = grid[i, :]
        row_no_zeros = row[row != 0]
        if len(row_no_zeros) != len(set(row_no_zeros)):
            return False
            
    # Check columns
    for i in range(9):
        col = grid[:, i]
        col_no_zeros = col[col != 0]
        if len(col_no_zeros) != len(set(col_no_zeros)):
            return False
            
    # Check 3x3 boxes
    for box_row in range(3):
        for box_col in range(3):
            box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
            box_no_zeros = box[box != 0]
            if len(box_no_zeros) != len(set(box_no_zeros)):
                return False
                
    return True

def print_puzzle(puzzle):
    """Print a Sudoku puzzle with grid lines"""
    puzzle_grid = puzzle.reshape(9, 9)
    print("-" * 25)
    for i in range(9):
        row = puzzle_grid[i]
        row_str = ""
        for j, val in enumerate(row):
            if j % 3 == 0:
                row_str += "| "
            row_str += f"{int(val) if val > 0 else '.'} "
        row_str += "|"
        print(row_str)
        if i % 3 == 2:
            print("-" * 25)

def main():
    # Set data directory
    data_dir = Path("data/sudoku-extreme-1k-aug-1000")
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
        
    # Check each split
    for split in ['train', 'test']:
        split_dir = data_dir / split
        
        if not split_dir.exists():
            print(f"❌ {split} directory not found: {split_dir}")
            continue
        
        inputs_file = split_dir / "all__inputs.npy"
        labels_file = split_dir / "all__labels.npy"
        
        if not inputs_file.exists() or not labels_file.exists():
            print(f"❌ Required files missing in {split}")
            continue
            
        try:
            # Load arrays
            inputs = np.load(inputs_file)
            labels = np.load(labels_file)
            
            print(f"\n📊 {split.upper()} Split:")
            print(f"  - Inputs: {inputs.shape}, dtype={inputs.dtype}")
            print(f"  - Labels: {labels.shape}, dtype={labels.dtype}")
            
            # Check random samples
            num_samples = min(20, len(inputs))
            sample_indices = random.sample(range(len(inputs)), num_samples)
            
            valid_count = 0
            invalid_count = 0
            issues = []
            
            for idx in sample_indices:
                input_sample = inputs[idx]
                label_sample = labels[idx]
                
                # Check clue consistency
                non_zero_mask = input_sample > 0
                clues_match = np.all(input_sample[non_zero_mask] == label_sample[non_zero_mask])
                
                # Check solution validity
                solution_valid = is_valid_sudoku(label_sample)
                
                if clues_match and solution_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    
                    issue = f"Sample {idx}: "
                    if not clues_match:
                        mismatches = np.sum(input_sample[non_zero_mask] != label_sample[non_zero_mask])
                        issue += f"Clue mismatch ({mismatches} positions)"
                    if not solution_valid:
                        issue += " Invalid solution"
                    
                    issues.append((idx, input_sample, label_sample, issue))
            
            print(f"  - Samples checked: {num_samples}")
            print(f"  - Valid samples: {valid_count} ({valid_count/num_samples*100:.1f}%)")
            print(f"  - Invalid samples: {invalid_count} ({invalid_count/num_samples*100:.1f}%)")
            
            # Show details for first invalid sample
            if issues:
                idx, input_sample, label_sample, issue = issues[0]
                print(f"\n⚠️ Issue details for sample {idx}:")
                print(f"  {issue}")
                
                print("\nInput puzzle:")
                print_puzzle(input_sample)
                
                print("\nSolution:")
                print_puzzle(label_sample)
                
                # Show first few mismatches
                non_zero_mask = input_sample > 0
                if not np.all(input_sample[non_zero_mask] == label_sample[non_zero_mask]):
                    print("\nMismatched positions:")
                    mismatch_indices = np.where((input_sample > 0) & (input_sample != label_sample))[0]
                    for i, pos in enumerate(mismatch_indices[:5]):
                        row, col = pos // 9, pos % 9
                        print(f"  Position ({row+1},{col+1}): Input={input_sample[pos]}, Solution={label_sample[pos]}")
        
        except Exception as e:
            print(f"❌ Error inspecting {split} files: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
