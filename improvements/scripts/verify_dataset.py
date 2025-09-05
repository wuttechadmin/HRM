#!/usr/bin/env python3
"""
Sudoku Dataset Verification Script

This script performs comprehensive verification of Sudoku datasets to ensure:
1. Input-Solution Correspondence: Non-empty cells in inputs match solutions
2. Solution Validity: Solutions are valid Sudoku grids (no repeats in rows, columns, or boxes)
3. Dataset Integrity: Input puzzles and solutions are properly aligned

Usage:
    python verify_dataset.py [--data-path PATH] [--samples NUM] [--seed SEED]

Example:
    python verify_dataset.py --data-path ./data/sudoku-extreme-1k-aug-1000 --samples 100 --seed 42
"""

import os
import sys
import numpy as np
import argparse
import random
from pathlib import Path
from tqdm import tqdm

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Verify Sudoku dataset integrity")
    parser.add_argument('--data-path', type=str, 
                        default='/Users/robertburkhall/Development/HRM/data/sudoku-extreme-1k-aug-1000',
                        help='Path to the dataset directory')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to verify (per split)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--chunks', type=int, default=5,
                        help='Number of chunks for large-scale verification')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Size of each chunk for large-scale verification')
    return parser.parse_args()

def load_sample_from_dataset(input_file, label_file, num_samples=100):
    """Load samples from the dataset files with memory efficiency"""
    try:
        # Check file sizes
        input_size = os.path.getsize(input_file) / (1024**2)  # Size in MB
        label_size = os.path.getsize(label_file) / (1024**2)  # Size in MB
        print(f"Input file size: {input_size:.2f} MB")
        print(f"Label file size: {label_size:.2f} MB")
        
        # Memory-efficient loading for large files
        input_array_info = np.load(input_file, mmap_mode='r')
        total_samples = len(input_array_info)
        print(f"Total samples in dataset: {total_samples}")
        
        # Generate random indices
        indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
        print(f"Sampling {len(indices)} random indices")
        
        # Load only the selected samples
        inputs = np.array([input_array_info[i] for i in indices])
        labels = np.array([np.load(label_file, mmap_mode='r')[i] for i in indices])
        
        return inputs, labels
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def verify_puzzle_solution_correspondence(inputs, solutions):
    """Verify input puzzles correspond to their solutions"""
    results = {
        "total": len(inputs),
        "non_empty_matches": 0,
        "valid_solutions": 0,
        "mismatches": [],
        "invalid_solutions": []
    }
    
    for i in range(len(inputs)):
        input_puzzle = inputs[i].reshape(9, 9)
        solution = solutions[i].reshape(9, 9)
        
        # Check 1: Do non-empty cells in input match solution?
        mask = input_puzzle != 0
        non_empty_count = mask.sum()
        
        if non_empty_count == 0:
            # Skip puzzles with no clues (shouldn't happen in real datasets)
            results["total"] -= 1
            continue
            
        matches = (input_puzzle[mask] == solution[mask]).all()
        if matches:
            results["non_empty_matches"] += 1
        else:
            mismatch_positions = []
            for r in range(9):
                for c in range(9):
                    if input_puzzle[r, c] != 0 and input_puzzle[r, c] != solution[r, c]:
                        mismatch_positions.append((r, c, input_puzzle[r, c], solution[r, c]))
            results["mismatches"].append((i, mismatch_positions))
        
        # Check 2: Is the solution a valid Sudoku solution?
        valid_solution = True
        
        # Check rows
        for r in range(9):
            if len(set(solution[r, :])) != 9 or 0 in solution[r, :]:
                valid_solution = False
                break
        
        # Check columns
        if valid_solution:
            for c in range(9):
                if len(set(solution[:, c])) != 9 or 0 in solution[:, c]:
                    valid_solution = False
                    break
        
        # Check 3x3 boxes
        if valid_solution:
            for box_r in range(3):
                for box_c in range(3):
                    box = solution[box_r*3:(box_r+1)*3, box_c*3:(box_c+1)*3].flatten()
                    if len(set(box)) != 9 or 0 in box:
                        valid_solution = False
                        break
                if not valid_solution:
                    break
        
        if valid_solution:
            results["valid_solutions"] += 1
        else:
            results["invalid_solutions"].append(i)
    
    # Calculate percentages
    if results["total"] > 0:
        results["pct_non_empty_match"] = results["non_empty_matches"] / results["total"] * 100
        results["pct_valid_solutions"] = results["valid_solutions"] / results["total"] * 100
    else:
        results["pct_non_empty_match"] = 0
        results["pct_valid_solutions"] = 0
        
    return results

def verify_dataset_in_chunks(input_file, label_file, chunk_size=1000, max_chunks=10):
    """Verify a large dataset by processing it in chunks"""
    try:
        # Memory-mapped arrays
        inputs_mmap = np.load(input_file, mmap_mode='r')
        labels_mmap = np.load(label_file, mmap_mode='r')
        
        total_samples = len(inputs_mmap)
        chunks_to_check = min(max_chunks, (total_samples + chunk_size - 1) // chunk_size)
        
        results = {
            "total": 0,
            "non_empty_matches": 0,
            "valid_solutions": 0,
            "mismatches": [],
            "invalid_solutions": []
        }
        
        print(f"\nVerifying {chunks_to_check} chunks of {chunk_size} samples each")
        print(f"Total dataset size: {total_samples} samples")
        
        for chunk_idx in tqdm(range(chunks_to_check), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            
            # Load current chunk
            inputs_chunk = np.array([inputs_mmap[i] for i in range(start_idx, end_idx)])
            labels_chunk = np.array([labels_mmap[i] for i in range(start_idx, end_idx)])
            
            # Verify this chunk
            chunk_results = verify_puzzle_solution_correspondence(inputs_chunk, labels_chunk)
            
            # Aggregate results
            results["total"] += chunk_results["total"]
            results["non_empty_matches"] += chunk_results["non_empty_matches"]
            results["valid_solutions"] += chunk_results["valid_solutions"]
            
            # Adjust indices for mismatches and invalid solutions
            for idx, mismatches in chunk_results["mismatches"]:
                results["mismatches"].append((idx + start_idx, mismatches))
                
            for idx in chunk_results["invalid_solutions"]:
                results["invalid_solutions"].append(idx + start_idx)
        
        # Calculate percentages
        if results["total"] > 0:
            results["pct_non_empty_match"] = results["non_empty_matches"] / results["total"] * 100
            results["pct_valid_solutions"] = results["valid_solutions"] / results["total"] * 100
        else:
            results["pct_non_empty_match"] = 0
            results["pct_valid_solutions"] = 0
            
        return results
    except Exception as e:
        print(f"Error during large-scale verification: {e}")
        return None

def print_sudoku_grid(grid, title="Puzzle"):
    """Print a Sudoku grid in a readable format"""
    print(f"\n{title}:")
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

def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print("\n🔍 Sudoku Dataset Verification Script")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Samples per split: {args.samples}")
    print(f"Random seed: {args.seed}")
    
    # Files to check
    train_input_file = os.path.join(args.data_path, 'train', 'all__inputs.npy')
    train_label_file = os.path.join(args.data_path, 'train', 'all__labels.npy')
    test_input_file = os.path.join(args.data_path, 'test', 'all__inputs.npy')
    test_label_file = os.path.join(args.data_path, 'test', 'all__labels.npy')
    
    # Verify files exist
    files_exist = True
    for file_path in [train_input_file, train_label_file, test_input_file, test_label_file]:
        exists = os.path.exists(file_path)
        print(f"{os.path.basename(file_path)} exists: {exists}")
        if not exists:
            files_exist = False
    
    if not files_exist:
        print("❌ Some required files are missing. Please check the dataset path.")
        return
        
    # Initialize results variables
    train_results = None
    test_results = None
    large_train_results = None
    
    # Load and verify train dataset
    print("\n🔍 Verifying TRAIN Dataset:")
    print("=" * 50)
    train_inputs, train_labels = load_sample_from_dataset(train_input_file, train_label_file, args.samples)
    
    if train_inputs is not None and train_labels is not None:
        train_results = verify_puzzle_solution_correspondence(train_inputs, train_labels)
        
        print(f"Total puzzles checked: {train_results['total']}")
        print(f"Non-empty cells match: {train_results['non_empty_matches']} ({train_results['pct_non_empty_match']:.1f}%)")
        print(f"Valid Sudoku solutions: {train_results['valid_solutions']} ({train_results['pct_valid_solutions']:.1f}%)")
        
        if train_results['mismatches']:
            print(f"\n⚠️ Found {len(train_results['mismatches'])} puzzles with input-solution mismatches:")
            for idx, mismatches in train_results['mismatches'][:3]:  # Show first 3 examples
                print(f"  Puzzle {idx}: {mismatches}")
                
                # Print the first mismatch example
                if idx == train_results['mismatches'][0][0]:
                    print_sudoku_grid(train_inputs[idx].reshape(9, 9), "Input Puzzle")
                    print_sudoku_grid(train_labels[idx].reshape(9, 9), "Solution")
        else:
            print("\n✅ No input-solution mismatches found!")
            
        if train_results['invalid_solutions']:
            print(f"\n⚠️ Found {len(train_results['invalid_solutions'])} puzzles with invalid solutions:")
            for idx in train_results['invalid_solutions'][:3]:  # Show first 3 examples
                print(f"  Puzzle {idx}")
                
                # Print the first invalid solution
                if idx == train_results['invalid_solutions'][0]:
                    print_sudoku_grid(train_inputs[idx].reshape(9, 9), "Input Puzzle")
                    print_sudoku_grid(train_labels[idx].reshape(9, 9), "Invalid Solution")
        else:
            print("\n✅ All solutions are valid Sudoku solutions!")
    else:
        print("❌ Could not verify train dataset due to loading errors")
    
    # Load and verify test dataset
    print("\n🔍 Verifying TEST Dataset:")
    print("=" * 50)
    test_inputs, test_labels = load_sample_from_dataset(test_input_file, test_label_file, args.samples)
    
    if test_inputs is not None and test_labels is not None:
        test_results = verify_puzzle_solution_correspondence(test_inputs, test_labels)
        
        print(f"Total puzzles checked: {test_results['total']}")
        print(f"Non-empty cells match: {test_results['non_empty_matches']} ({test_results['pct_non_empty_match']:.1f}%)")
        print(f"Valid Sudoku solutions: {test_results['valid_solutions']} ({test_results['pct_valid_solutions']:.1f}%)")
        
        if test_results['mismatches']:
            print(f"\n⚠️ Found {len(test_results['mismatches'])} puzzles with input-solution mismatches:")
            for idx, mismatches in test_results['mismatches'][:3]:  # Show first 3 examples
                print(f"  Puzzle {idx}: {mismatches}")
        else:
            print("\n✅ No input-solution mismatches found!")
            
        if test_results['invalid_solutions']:
            print(f"\n⚠️ Found {len(test_results['invalid_solutions'])} puzzles with invalid solutions:")
            for idx in test_results['invalid_solutions'][:3]:  # Show first 3 examples
                print(f"  Puzzle {idx}")
        else:
            print("\n✅ All solutions are valid Sudoku solutions!")
    else:
        print("❌ Could not verify test dataset due to loading errors")
    
    # Large-scale verification
    print("\n🔍 Large-Scale Verification:")
    print("=" * 60)
    print("Performing large-scale verification on training dataset...")
    large_train_results = verify_dataset_in_chunks(train_input_file, train_label_file, 
                                                 chunk_size=args.chunk_size, max_chunks=args.chunks)
    
    if large_train_results:
        print(f"Verified {large_train_results['total']} puzzles in total")
        print(f"Non-empty cells match: {large_train_results['non_empty_matches']} ({large_train_results['pct_non_empty_match']:.1f}%)")
        print(f"Valid Sudoku solutions: {large_train_results['valid_solutions']} ({large_train_results['pct_valid_solutions']:.1f}%)")
        
        if large_train_results['mismatches']:
            print(f"\n⚠️ Found {len(large_train_results['mismatches'])} puzzles with input-solution mismatches")
        else:
            print("\n✅ No input-solution mismatches found in large-scale verification!")
            
        if large_train_results['invalid_solutions']:
            print(f"\n⚠️ Found {len(large_train_results['invalid_solutions'])} puzzles with invalid solutions")
        else:
            print("\n✅ All solutions are valid Sudoku solutions in large-scale verification!")
    else:
        print("❌ Could not perform large-scale verification")
    
    # Final summary
    print("\n📊 Final Dataset Verification Summary:")
    print("=" * 60)
    
    all_checks_passed = True
    
    if train_results is not None:
        print(f"\nTRAIN Dataset:")
        print(f"- Checked {train_results['total']} samples")
        print(f"- Input-Solution Correspondence: {train_results['pct_non_empty_match']:.1f}% match")
        print(f"- Solution Validity: {train_results['pct_valid_solutions']:.1f}% valid")
        
        if len(train_results['mismatches']) > 0 or len(train_results['invalid_solutions']) > 0:
            all_checks_passed = False
    
    if test_results is not None:
        print(f"\nTEST Dataset:")
        print(f"- Checked {test_results['total']} samples")
        print(f"- Input-Solution Correspondence: {test_results['pct_non_empty_match']:.1f}% match")
        print(f"- Solution Validity: {test_results['pct_valid_solutions']:.1f}% valid")
        
        if len(test_results['mismatches']) > 0 or len(test_results['invalid_solutions']) > 0:
            all_checks_passed = False
    
    if large_train_results is not None:
        print(f"\nLarge-Scale Verification:")
        print(f"- Checked {large_train_results['total']} samples")
        print(f"- Input-Solution Correspondence: {large_train_results['pct_non_empty_match']:.1f}% match")
        print(f"- Solution Validity: {large_train_results['pct_valid_solutions']:.1f}% valid")
        
        if len(large_train_results['mismatches']) > 0 or len(large_train_results['invalid_solutions']) > 0:
            all_checks_passed = False
    
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ VERIFICATION PASSED: The dataset appears to be correct and ready for use!")
        print("   - All non-empty cells in input puzzles match their corresponding solutions")
        print("   - All solutions are valid Sudoku grids")
        print("   - No misalignment detected between inputs and solutions")
        print("\nRecommendation: This dataset should work well with the HRM Colab notebook.")
    else:
        print("❌ VERIFICATION FAILED: Issues were detected in the dataset")
        print("\nRecommendations:")
        print("1. Run a repair script to fix any mismatches")
        print("2. Regenerate any invalid Sudoku solutions")
        print("3. Ensure proper alignment between inputs and solutions")
        print("4. Consider using a smaller subset of verified-correct puzzles")
        print("\nConsult the mismatch and invalid solution details above for specific issues.")

if __name__ == "__main__":
    main()
