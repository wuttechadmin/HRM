# HRM Dataset Comparison for Colab

This script helps you quickly compare the original and repaired datasets to determine if the repair process corrected the issues properly.

```python
import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Update these paths to match your environment
ORIGINAL_DATA_PATH = "/content/original_dataset"  # Path to the uploaded original dataset
REPAIRED_DATA_PATH = "/content/repaired_dataset"  # Path to the uploaded repaired dataset

class HRMSudokuDataset(Dataset):
    """Smart dataset loader for HRM Sudoku data format"""
    
    def __init__(self, data_path, split='train', max_samples=10):
        # Same implementation as in test_colab_compatibility.ipynb
        self.data_path = Path(data_path)
        self.split = split
        self.samples = []
        self.vocab_size = 11  # HRM uses 0-10

        print(f"\n🔍 Loading HRM dataset from: {self.data_path / split}")

        split_dir = self.data_path / split
        if not split_dir.exists():
            print(f"❌ Directory {split_dir} not found")
            return

        # Load metadata
        metadata_file = split_dir / "dataset.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"📊 Metadata: vocab_size={metadata.get('vocab_size', 11)}")
                self.vocab_size = metadata.get('vocab_size', 11)
            except Exception as e:
                print(f"⚠️ Could not load metadata: {e}")
        
        # Try to load directly
        try:
            inputs = np.load(os.path.join(split_dir, 'all__inputs.npy'))
            labels = np.load(os.path.join(split_dir, 'all__labels.npy'))
            
            print(f"📊 Found {len(inputs)} samples")
            
            # Load a subset of samples
            for i in range(min(len(inputs), max_samples)):
                self.samples.append({
                    'input_ids': torch.tensor(inputs[i], dtype=torch.long),
                    'target': torch.tensor(labels[i], dtype=torch.long)
                })
            
            print(f"✅ Loaded {len(self.samples)} samples")
        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def print_sudoku(grid, title):
    """Pretty print sudoku grid"""
    print(f"\n{title}:")
    grid = grid.reshape(9, 9)
    for i in range(9):
        if i % 3 == 0 and i > 0:
            print("------+-------+------")
        row = ""
        for j in range(9):
            if j % 3 == 0 and j > 0:
                row += "| "
            val = grid[i, j].item() if hasattr(grid[i, j], 'item') else grid[i, j]
            row += f"{val if val != 0 else '.'} "
        print(row)

def compare_datasets():
    """Compare original and repaired datasets"""
    print("=" * 60)
    print("🔍 DATASET COMPARISON")
    print("=" * 60)
    
    # Load both datasets
    original_dataset = HRMSudokuDataset(ORIGINAL_DATA_PATH, 'train')
    repaired_dataset = HRMSudokuDataset(REPAIRED_DATA_PATH, 'train')
    
    print(f"\nOriginal dataset: {len(original_dataset)} samples")
    print(f"Repaired dataset: {len(repaired_dataset)} samples")
    
    # Check if we have samples to compare
    if len(original_dataset) == 0 or len(repaired_dataset) == 0:
        print("❌ Not enough samples to compare")
        return
    
    # Compare a few samples
    samples_to_compare = min(3, len(original_dataset), len(repaired_dataset))
    
    for i in range(samples_to_compare):
        original = original_dataset[i]
        repaired = repaired_dataset[i]
        
        print(f"\n{'='*60}")
        print(f"Sample {i+1}")
        
        # Print original puzzle
        print("\n📊 ORIGINAL DATASET")
        print_sudoku(original['input_ids'], "Input Puzzle")
        print_sudoku(original['target'], "Solution")
        
        # Check mismatches in original
        orig_input = original['input_ids']
        orig_target = original['target']
        mask = orig_input != 0
        matching_cells = (orig_input[mask] == orig_target[mask]).sum().item()
        total_non_empty = mask.sum().item()
        
        print(f"\nMatching non-empty cells: {matching_cells}/{total_non_empty} ({matching_cells/total_non_empty*100:.1f}%)")
        
        # Show mismatch details for original
        if matching_cells < total_non_empty:
            print("\nMismatches in original:")
            orig_input_2d = orig_input.reshape(9, 9)
            orig_target_2d = orig_target.reshape(9, 9)
            
            for r in range(9):
                for c in range(9):
                    if orig_input_2d[r, c] != 0 and orig_input_2d[r, c] != orig_target_2d[r, c]:
                        print(f"  Position [{r},{c}]: Input={orig_input_2d[r, c].item()}, Solution={orig_target_2d[r, c].item()}")
        
        # Print repaired puzzle
        print("\n📊 REPAIRED DATASET")
        print_sudoku(repaired['input_ids'], "Input Puzzle")
        print_sudoku(repaired['target'], "Solution")
        
        # Check mismatches in repaired
        rep_input = repaired['input_ids']
        rep_target = repaired['target']
        mask = rep_input != 0
        matching_cells = (rep_input[mask] == rep_target[mask]).sum().item()
        total_non_empty = mask.sum().item()
        
        print(f"\nMatching non-empty cells: {matching_cells}/{total_non_empty} ({matching_cells/total_non_empty*100:.1f}%)")
        
        # Show mismatch details for repaired
        if matching_cells < total_non_empty:
            print("\nMismatches in repaired:")
            rep_input_2d = rep_input.reshape(9, 9)
            rep_target_2d = rep_target.reshape(9, 9)
            
            for r in range(9):
                for c in range(9):
                    if rep_input_2d[r, c] != 0 and rep_input_2d[r, c] != rep_target_2d[r, c]:
                        print(f"  Position [{r},{c}]: Input={rep_input_2d[r, c].item()}, Solution={rep_target_2d[r, c].item()}")
    
    print("\n" + "=" * 60)
    print("Comparison complete!")

# Run the comparison
compare_datasets()
```

To use this script in Colab:

1. Upload both your original and repaired datasets to your Colab instance
2. Update the `ORIGINAL_DATA_PATH` and `REPAIRED_DATA_PATH` variables to point to your uploaded datasets
3. Run the script to see a comparison of the datasets

The script will show you:
- If both datasets loaded successfully
- A side-by-side comparison of a few samples from each dataset
- Any mismatches between inputs and solutions in both datasets
- Verification that the repaired dataset fixed the mismatches
