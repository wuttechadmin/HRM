#!/usr/bin/env python3
"""
Sudoku Solution Validator

This script validates Sudoku solutions by checking that:
1. Each solution is a valid Sudoku grid (no repeated digits in any row, column, or 3x3 box)
2. Non-empty cells in the input puzzle match the corresponding cells in the solution
3. Solutions solve the puzzles

Usage:
    python validate_sudoku.py [--file FILEPATH] [--num-samples NUM_SAMPLES]

Example:
    python validate_sudoku.py --file data/sudoku-extreme-1k-aug-1000/train/all__inputs.npy --num-samples 50
"""

import os
import sys
import numpy as np
import argparse
import random
from pathlib import Path
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Validate Sudoku puzzles and solutions")
    parser.add_argument('--file', type=str, default=None,
                       help='Path to the input .npy file (solution file will be inferred)')
    parser.add_argument('--train', action='store_true',
                       help='Check train dataset')
    parser.add_argument('--test', action='store_true',
                       help='Check test dataset')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to check (use -1 for all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--data-dir', type=str, 
                       default='/Users/robertburkhall/Development/HRM/data/sudoku-extreme-1k-aug-1000',
                       help='Root directory for dataset')
    return parser.parse_args()

def check_sudoku_validity(solution_grid):
    """
    Check if a completed Sudoku grid is valid.
    
    Args:
        solution_grid: 9x9 NumPy array with Sudoku solution
        
    Returns:
        Boolean indicating validity and a list of issues if any
    """
    # Reshape to ensure 9x9
    solution_grid = solution_grid.reshape(9, 9)
    issues = []
    
    # 1. Check if all values are 1-9 (no zeros)
    if np.any(solution_grid == 0):
        zero_count = np.sum(solution_grid == 0)
        issues.append(f"Solution contains {zero_count} zero(s)/empty cells")
    
    # 2. Check rows
    for i in range(9):
        row = solution_grid[i, :]
        row_values = [x for x in row if x != 0]
        if len(row_values) != len(set(row_values)):
            issues.append(f"Row {i+1} has duplicate values: {row}")
    
    # 3. Check columns
    for i in range(9):
        col = solution_grid[:, i]
        col_values = [x for x in col if x != 0]
        if len(col_values) != len(set(col_values)):
            issues.append(f"Column {i+1} has duplicate values: {col}")
    
    # 4. Check 3x3 boxes
    for box_row in range(3):
        for box_col in range(3):
            box = solution_grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
            box_values = [x for x in box if x != 0]
            if len(box_values) != len(set(box_values)):
                issues.append(f"Box at ({box_row+1},{box_col+1}) has duplicate values: {box}")
    
    return len(issues) == 0, issues

def check_input_matches_solution(input_grid, solution_grid):
    """
    Check if non-empty cells in the input match the solution.
    
    Args:
        input_grid: 9x9 NumPy array with input puzzle
        solution_grid: 9x9 NumPy array with solution
        
    Returns:
        Boolean indicating validity and a list of mismatches if any
    """
    # Reshape to ensure 9x9
    input_grid = input_grid.reshape(9, 9)
    solution_grid = solution_grid.reshape(9, 9)
    
    mismatches = []
    for i in range(9):
        for j in range(9):
            if input_grid[i, j] != 0 and input_grid[i, j] != solution_grid[i, j]:
                mismatches.append((i, j, input_grid[i, j], solution_grid[i, j]))
    
    return len(mismatches) == 0, mismatches

def print_sudoku(grid, title=None):
    """Pretty print sudoku grid"""
    if title:
        print(f"\n{title}")
    
    # Reshape if needed
    grid = grid.reshape(9, 9)
    
    for i in range(9):
        if i % 3 == 0 and i > 0:
            print("------+-------+------")
        row = ""
        for j in range(9):
            if j % 3 == 0 and j > 0:
                row += "| "
            val = grid[i, j]
            row += f"{val if val != 0 else '.'} "
        print(row)

def validate_sample(input_grid, solution_grid, index=None):
    """
    Validate a single Sudoku puzzle and its solution.
    
    Args:
        input_grid: NumPy array with input puzzle
        solution_grid: NumPy array with solution
        index: Optional index for reporting
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "index": index,
        "valid_solution": False,
        "input_matches_solution": False,
        "solution_issues": [],
        "mismatch_issues": []
    }
    
    # Check if solution is valid
    result["valid_solution"], result["solution_issues"] = check_sudoku_validity(solution_grid)
    
    # Check if input matches solution
    result["input_matches_solution"], result["mismatch_issues"] = check_input_matches_solution(input_grid, solution_grid)
    
    return result

def load_and_validate_samples(input_file, label_file, num_samples=-1):
    """
    Load and validate Sudoku puzzles and solutions.
    
    Args:
        input_file: Path to .npy file with input puzzles
        label_file: Path to .npy file with solutions
        num_samples: Number of samples to validate (-1 for all)
        
    Returns:
        Dictionary with validation results
    """
    print(f"\nValidating Sudoku puzzles from:")
    print(f"  Inputs: {input_file}")
    print(f"  Labels: {label_file}")
    
    try:
        # Get basic info about the files
        input_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        label_size_mb = os.path.getsize(label_file) / (1024 * 1024)
        print(f"Input file size: {input_size_mb:.2f} MB")
        print(f"Label file size: {label_size_mb:.2f} MB")
        
        # Memory-mapped loading to handle large files
        inputs = np.load(input_file, mmap_mode='r')
        labels = np.load(label_file, mmap_mode='r')
        
        total_samples = len(inputs)
        print(f"Total samples in dataset: {total_samples}")
        
        # Determine number of samples to validate
        if num_samples <= 0 or num_samples > total_samples:
            num_samples = total_samples
            indices = list(range(total_samples))
            print(f"Validating all {num_samples} samples")
        else:
            indices = np.random.choice(total_samples, num_samples, replace=False)
            print(f"Validating random {num_samples} samples")
        
        # Results tracker
        results = {
            "total_checked": num_samples,
            "valid_solution_count": 0,
            "input_matches_solution_count": 0,
            "issue_count": 0,
            "issues": []
        }
        
        # Process each sample
        start_time = time.time()
        for count, idx in enumerate(indices):
            if count > 0 and count % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {count}/{num_samples} samples ({elapsed:.1f}s)...")
            
            input_grid = inputs[idx]
            solution_grid = labels[idx]
            
            validation = validate_sample(input_grid, solution_grid, idx)
            
            # Update counters
            if validation["valid_solution"]:
                results["valid_solution_count"] += 1
            
            if validation["input_matches_solution"]:
                results["input_matches_solution_count"] += 1
            
            # Track issues
            if not validation["valid_solution"] or not validation["input_matches_solution"]:
                results["issue_count"] += 1
                results["issues"].append(validation)
        
        # Calculate percentages
        results["valid_solution_pct"] = (results["valid_solution_count"] / results["total_checked"]) * 100
        results["input_matches_solution_pct"] = (results["input_matches_solution_count"] / results["total_checked"]) * 100
        results["issue_pct"] = (results["issue_count"] / results["total_checked"]) * 100
        
        return results
    
    except Exception as e:
        print(f"Error loading or validating data: {e}")
        return None

def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print("\n🔍 Sudoku Solution Validator")
    print("=" * 60)
    
    results = []
    
    # Process specific file if provided
    if args.file:
        # Infer the label file from the input file
        input_file = args.file
        if 'inputs' in os.path.basename(args.file):
            label_file = args.file.replace('inputs', 'labels')
        else:
            # If we can't infer, assume the label file is in the same directory
            print("Could not infer label file path, please specify both files.")
            return
        
        results.append(load_and_validate_samples(input_file, label_file, args.num_samples))
    
    # Otherwise check train and/or test datasets
    else:
        # Check train dataset if requested
        if args.train:
            train_input = os.path.join(args.data_dir, 'train', 'all__inputs.npy')
            train_label = os.path.join(args.data_dir, 'train', 'all__labels.npy')
            if os.path.exists(train_input) and os.path.exists(train_label):
                print("\nChecking TRAIN dataset:")
                results.append(load_and_validate_samples(train_input, train_label, args.num_samples))
            else:
                print("❌ Train dataset files not found")
        
        # Check test dataset if requested
        if args.test:
            test_input = os.path.join(args.data_dir, 'test', 'all__inputs.npy')
            test_label = os.path.join(args.data_dir, 'test', 'all__labels.npy')
            if os.path.exists(test_input) and os.path.exists(test_label):
                print("\nChecking TEST dataset:")
                results.append(load_and_validate_samples(test_input, test_label, args.num_samples))
            else:
                print("❌ Test dataset files not found")
        
        # If neither train nor test specified, check both
        if not args.train and not args.test:
            train_input = os.path.join(args.data_dir, 'train', 'all__inputs.npy')
            train_label = os.path.join(args.data_dir, 'train', 'all__labels.npy')
            test_input = os.path.join(args.data_dir, 'test', 'all__inputs.npy')
            test_label = os.path.join(args.data_dir, 'test', 'all__labels.npy')
            
            if os.path.exists(train_input) and os.path.exists(train_label):
                print("\nChecking TRAIN dataset:")
                results.append(load_and_validate_samples(train_input, train_label, args.num_samples))
            else:
                print("❌ Train dataset files not found")
            
            if os.path.exists(test_input) and os.path.exists(test_label):
                print("\nChecking TEST dataset:")
                results.append(load_and_validate_samples(test_input, test_label, args.num_samples))
            else:
                print("❌ Test dataset files not found")
    
    # Print summary
    print("\n📊 Validation Summary:")
    print("=" * 60)
    
    all_valid = True
    
    for i, result in enumerate(results):
        if result:
            print(f"\nResult {i+1}:")
            print(f"  Samples checked: {result['total_checked']}")
            print(f"  Valid solutions: {result['valid_solution_count']} ({result['valid_solution_pct']:.1f}%)")
            print(f"  Input-solution matches: {result['input_matches_solution_count']} ({result['input_matches_solution_pct']:.1f}%)")
            print(f"  Issues found: {result['issue_count']} ({result['issue_pct']:.1f}%)")
            
            if result['issue_count'] > 0:
                all_valid = False
                
                # Show up to 3 issues
                print("\n  Issue details (up to 3):")
                for issue_idx, issue in enumerate(result['issues'][:3]):
                    print(f"\n  Issue {issue_idx+1} at sample index {issue['index']}:")
                    if not issue['valid_solution']:
                        print(f"    ❌ Invalid solution: {issue['solution_issues']}")
                    if not issue['input_matches_solution']:
                        print(f"    ❌ Input-solution mismatch: {issue['mismatch_issues']}")
                        
                    # Print the puzzle and solution for visual inspection
                    input_idx = issue['index']
                    inputs = np.load(args.file or 
                                  (os.path.join(args.data_dir, 'train' if i==0 else 'test', 'all__inputs.npy')), 
                                  mmap_mode='r')
                    labels = np.load((args.file or 
                                   (os.path.join(args.data_dir, 'train' if i==0 else 'test', 'all__inputs.npy'))).replace('inputs', 'labels'), 
                                  mmap_mode='r')
                    
                    print_sudoku(inputs[input_idx], "Input Puzzle")
                    print_sudoku(labels[input_idx], "Solution")
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED: No issues found in the checked samples!")
        print("   The dataset appears to be valid and ready to use.")
    else:
        print("❌ VALIDATION FAILED: Issues were found in the dataset.")
        print("   Please review the issues above and consider repairing the dataset.")

if __name__ == "__main__":
    main()
