import torch
import numpy as np
from pathlib import Path

class SimpleDataLoader:
    def __init__(self, data_dir, split='train', max_samples=100):
        self.data_path = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        
        # Load data
        inputs_file = self.data_path / split / "all__inputs.npy"
        labels_file = self.data_path / split / "all__labels.npy"
        
        self.inputs = np.load(inputs_file)
        self.labels = np.load(labels_file)
        
        print(f"Loaded {split} data: {self.inputs.shape}, {self.labels.shape}")
        
        # Limit samples if needed
        if max_samples and max_samples < len(self.inputs):
            self.inputs = self.inputs[:max_samples]
            self.labels = self.labels[:max_samples]
            print(f"Using {max_samples} samples")

def is_valid_sudoku(grid):
    """Check if a Sudoku grid is valid (all rows, columns, and boxes have unique values)"""
    grid_reshaped = grid.reshape(9, 9)
    
    # Check rows
    for i in range(9):
        row = grid_reshaped[i]
        row_no_zeros = row[row != 0]
        if len(row_no_zeros) != len(set(row_no_zeros)):
            return False
    
    # Check columns
    for i in range(9):
        col = grid_reshaped[:, i]
        col_no_zeros = col[col != 0]
        if len(col_no_zeros) != len(set(col_no_zeros)):
            return False
    
    # Check boxes
    for box_row in range(3):
        for box_col in range(3):
            box = grid_reshaped[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
            box_no_zeros = box[box != 0]
            if len(box_no_zeros) != len(set(box_no_zeros)):
                return False
    
    return True

def check_dataset_consistency(data_path):
    """Check if the dataset has consistent inputs and targets"""
    data_loader = SimpleDataLoader(data_path, 'train', max_samples=1000)
    
    # Check if clue cells match solution values
    clue_matches = 0
    clue_mismatches = 0
    valid_solutions = 0
    invalid_solutions = 0
    clue_counts = []
    
    for i in range(len(data_loader.inputs)):
        input_puzzle = data_loader.inputs[i]
        solution = data_loader.labels[i]
        
        # Count clues
        clue_count = np.count_nonzero(input_puzzle)
        clue_counts.append(clue_count)
        
        # Check clue consistency
        clue_mask = input_puzzle > 0
        if np.all(input_puzzle[clue_mask] == solution[clue_mask]):
            clue_matches += 1
        else:
            clue_mismatches += 1
        
        # Check solution validity
        if is_valid_sudoku(solution):
            valid_solutions += 1
        else:
            invalid_solutions += 1
        
        # Print details for a few samples
        if i < 5:
            print(f"\nSample {i}:")
            print(f"Clues: {clue_count}/81")
            print(f"Clues match solution: {'✅' if np.all(input_puzzle[clue_mask] == solution[clue_mask]) else '❌'}")
            print(f"Valid solution: {'✅' if is_valid_sudoku(solution) else '❌'}")
    
    print(f"\nSummary of {len(data_loader.inputs)} samples:")
    print(f"Clue counts: min={min(clue_counts)}, avg={sum(clue_counts)/len(clue_counts):.1f}, max={max(clue_counts)}")
    print(f"Clue consistency: {clue_matches}/{clue_matches + clue_mismatches} consistent ({clue_matches/(clue_matches + clue_mismatches):.2%})")
    print(f"Solution validity: {valid_solutions}/{valid_solutions + invalid_solutions} valid ({valid_solutions/(valid_solutions + invalid_solutions):.2%})")

if __name__ == "__main__":
    # Update path to your dataset
    DATA_DIR = Path("/Users/robertburkhall/Development/HRM/data/sudoku-extreme-1k-aug-1000")
    print(f"Checking dataset at {DATA_DIR}")
    check_dataset_consistency(DATA_DIR)
