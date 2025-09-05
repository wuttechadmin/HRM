#!/usr/bin/env python3
"""
Create a modified version of pretrain.py with MPS support for macOS
"""
import torch
import os
import shutil
from pathlib import Path

# Copy pretrain.py.bak to pretrain_macos.py
shutil.copy('pretrain.py.bak', 'pretrain_macos.py')

# Read the file
with open('pretrain_macos.py', 'r') as f:
    lines = f.readlines()

# Define device selection function
device_selection = """
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Get the appropriate device
DEVICE = get_device()
"""

# Insert device selection after imports
import_end_idx = 0
for i, line in enumerate(lines):
    if 'from omegaconf import DictConfig' in line:
        import_end_idx = i + 1
        break

if import_end_idx > 0:
    lines.insert(import_end_idx, device_selection)

# Replace "cuda" device references with DEVICE
for i, line in enumerate(lines):
    if 'device="cuda"' in line:
        lines[i] = line.replace('device="cuda"', 'device=DEVICE')

# Fix the distributed initialization
for i, line in enumerate(lines):
    if 'dist.init_process_group(backend="nccl")' in line:
        lines[i] = line.replace('backend="nccl"', 'backend="gloo" if DEVICE == "mps" else "nccl"')
    
    # Fix the CUDA device setting line
    if 'torch.cuda.set_device(' in line:
        indentation = line.split('torch')[0]
        lines[i] = f"{indentation}if DEVICE == 'cuda':\n{indentation}    {line.strip()}\n"

# Add MPS device move after model creation
for i, line in enumerate(lines):
    if 'model, optimizers, optimizer_lrs = create_model(' in line:
        indentation = line.split('model')[0]
        lines.insert(i + 1, f"\n{indentation}# Move model to the appropriate device\n{indentation}model = model.to(DEVICE)\n")
        break

# Write the modified file
with open('pretrain_macos.py', 'w') as f:
    f.writelines(lines)

print("✅ Successfully created pretrain_macos.py with MPS support")
print("You can now run training on macOS with:")
print("  python pretrain_macos.py data_path=data/sudoku-extreme-1k-aug-1000-new epochs=1 eval_interval=1 global_batch_size=16")
