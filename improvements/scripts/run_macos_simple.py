#!/usr/bin/env python3
"""
Simplified MacOS runner for HRM that fixes the indentation issues
"""

import os
import sys
import argparse
import torch
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run HRM on macOS with Apple Silicon")
    parser.add_argument("--data_path", type=str, default="data/sudoku-extreme-1k-aug-1000-new",
                        help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=1000, 
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="Evaluation interval in epochs")
    args = parser.parse_args()

    # Verify dataset exists
    if not Path(args.data_path).exists():
        print(f"Error: Dataset path '{args.data_path}' does not exist.")
        print("You may need to generate the dataset first with:")
        print("  python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000-new --subsample-size 1000 --num-aug 1000")
        sys.exit(1)

    # Verify MPS availability
    if not torch.backends.mps.is_available():
        print("Error: MPS is not available on this device. Make sure you're using macOS 12.3+ with an Apple Silicon chip.")
        sys.exit(1)
    
    print(f"✓ MPS is available, using device: {torch.device('mps')}")
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Create command with appropriate parameters for macOS
    cmd = [
        sys.executable,  # Use the current Python interpreter
        "-c",
        """
import os
import sys
import torch
import subprocess

# Set environment variables
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Build the command
cmd = [
    sys.executable,
    "pretrain.py.bak",  # Use the original script
    f"data_path={sys.argv[1]}",
    f"epochs={sys.argv[2]}",
    f"eval_interval={sys.argv[3]}",
    f"global_batch_size={sys.argv[4]}",
    "lr=7e-5",
    "puzzle_emb_lr=7e-5",
    "weight_decay=1.0",
    "puzzle_emb_weight_decay=1.0"
]

# Print command
print("\\nRunning command:", " ".join(cmd))

# Run the command with MPS environment variables
env = os.environ.copy()
env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

result = subprocess.run(cmd, env=env)
sys.exit(result.returncode)
        """,
        args.data_path,
        str(args.epochs),
        str(args.eval_interval),
        str(args.batch_size)
    ]
    
    # Set environment variables for MPS
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable fallback for ops not supported in MPS
    
    print(f"\nRunning HRM with the following configuration:")
    print(f"  - Dataset: {args.data_path}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Evaluation interval: {args.eval_interval}")
    
    # Execute the command
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
