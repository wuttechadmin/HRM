#!/usr/bin/env python3
import os
import torch
import argparse
import sys
from pathlib import Path

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

    # Verify MPS availability
    if not torch.backends.mps.is_available():
        print("Error: MPS is not available on this device. Make sure you're using macOS 12.3+ with an Apple Silicon chip.")
        sys.exit(1)
    
    print(f"✓ MPS is available, using device: {torch.device('mps')}")
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Create command with appropriate parameters for macOS
    python_path = sys.executable
    cmd = [
        python_path, "pretrain.py",
        f"data_path={args.data_path}",
        f"epochs={args.epochs}",
        f"eval_interval={args.eval_interval}",
        f"global_batch_size={args.batch_size}",
        "lr=7e-5",
        "puzzle_emb_lr=7e-5",
        "weight_decay=1.0",
        "puzzle_emb_weight_decay=1.0"
    ]
    
    # Set environment variables for MPS
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable fallback for ops not supported in MPS
    
    print(f"\nRunning HRM with the following configuration:")
    print(f"  - Dataset: {args.data_path}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Evaluation interval: {args.eval_interval}")
    print("\nCommand:", " ".join(cmd))
    
    # Execute the command
    os.execvpe(cmd[0], cmd, env)

if __name__ == "__main__":
    main()
