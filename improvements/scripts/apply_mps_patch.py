#!/usr/bin/env python3
"""
This patch modifies the pretrain.py file to add MPS (Metal Performance Shaders) support
for Apple Silicon Macs. Apply this patch using:

python apply_mps_patch.py

This will backup the original file and create a new version with MPS support.
"""

import os
import shutil
import re
from pathlib import Path

def patch_pretrain():
    # Define the file path
    pretrain_path = Path("pretrain.py")
    backup_path = Path("pretrain.py.bak")
    
    # Check if file exists
    if not pretrain_path.exists():
        print(f"Error: {pretrain_path} not found!")
        return False
    
    # Create backup
    print(f"Creating backup of {pretrain_path} as {backup_path}")
    shutil.copy(pretrain_path, backup_path)
    
    # Read the file
    with open(pretrain_path, 'r') as f:
        content = f.read()
    
    # Apply patches
    
    # Patch 1: Add MPS device selection
    device_selection_code = """
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
    
    # Add device selection function after imports
    import_pattern = r'(from omegaconf import DictConfig.*?\n)'
    content = re.sub(import_pattern, r'\1\n' + device_selection_code, content, flags=re.DOTALL)
    
    # Patch 2: Replace "cuda" references with DEVICE
    content = content.replace('device="cuda"', 'device=DEVICE')
    content = content.replace("backend=\"nccl\"", "backend=\"gloo\" if DEVICE == \"mps\" else \"nccl\"")
    
    # Find and fix the CUDA device setting with proper indentation
    cuda_pattern = r'(dist\.init_process_group\(backend="[^"]+"\)\n\n\s+RANK = dist\.get_rank\(\)\n\s+WORLD_SIZE = dist\.get_world_size\(\)\n\n\s+)torch\.cuda\.set_device\(int\(os\.environ\["LOCAL_RANK"\]\)\)'
    cuda_replacement = r'\1if DEVICE == "cuda":\n            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))'
    content = re.sub(cuda_pattern, cuda_replacement, content, flags=re.DOTALL)
    
    # Patch 3: Add MPS device initialization for model
    model_init_pattern = r'(model, optimizers, optimizer_lrs = create_model\(config, train_metadata, world_size=world_size\))'
    model_device_code = r'\1\n\n    # Move model to the appropriate device\n    model = model.to(DEVICE)'
    content = re.sub(model_init_pattern, model_device_code, content)
    
    # Patch 4: Add warning when using MPS with distributed setup
    launch_pattern = r'(def launch\(hydra_config: DictConfig\):.*?WORLD_SIZE = 1\n)'
    mps_warning_code = r'\1\n    # Check for MPS with distributed setup\n    if torch.backends.mps.is_available() and "LOCAL_RANK" in os.environ:\n        print("Warning: Distributed training with MPS is not fully supported. Falling back to single GPU.")\n        os.environ.pop("LOCAL_RANK", None)\n'
    content = re.sub(launch_pattern, mps_warning_code, content, flags=re.DOTALL)
    
    # Write the modified content back
    with open(pretrain_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Successfully patched {pretrain_path} with MPS support")
    print("You can now run training on macOS with:")
    print("  python run_macos.py")
    return True

if __name__ == "__main__":
    patch_pretrain()
