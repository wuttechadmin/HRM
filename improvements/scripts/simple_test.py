import torch
import numpy as np
from pathlib import Path

# Basic environment check
print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

# Data paths
DATA_DIR = Path('./data/sudoku-extreme-1k-aug-1000')
print("Dataset exists:", DATA_DIR.exists())

# Check dataset files
test_dir = DATA_DIR / "test"
train_dir = DATA_DIR / "train"

if test_dir.exists() and train_dir.exists():
    inputs_file = test_dir / "all__inputs.npy"
    labels_file = test_dir / "all__labels.npy"
    
    if inputs_file.exists() and labels_file.exists():
        inputs = np.load(inputs_file)
        labels = np.load(labels_file)
        print(f"Test inputs shape: {inputs.shape}")
        print(f"Test labels shape: {labels.shape}")
        
        # Check a sample
        if len(inputs) > 0:
            input_sample = inputs[0]
            label_sample = labels[0]
            clue_count = (input_sample > 0).sum()
            print(f"Sample clue count: {clue_count}")
            
            # Check clue consistency
            non_zero_mask = input_sample > 0
            clues_match = np.all(input_sample[non_zero_mask] == label_sample[non_zero_mask])
            print(f"Clues match solution: {clues_match}")
    else:
        print("Required dataset files not found")
else:
    print("Dataset directories not found")
