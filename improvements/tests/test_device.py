#!/usr/bin/env python3
import torch
import os
import time

def test_device():
    print("Testing PyTorch device availability:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create tensors on the device
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Perform a matrix multiplication
    print("\nPerforming a 1000x1000 matrix multiplication...")
    
    # Time the operation
    start = time.time()
    z = torch.matmul(x, y)
    end = time.time()
    
    elapsed_time = (end - start) * 1000  # Convert to milliseconds
    print(f"Computation time: {elapsed_time:.2f} ms")
    
    print("Matrix multiplication completed successfully!")
    print(f"Output tensor shape: {z.shape}, device: {z.device}")
    
    return True

if __name__ == "__main__":
    success = test_device()
    print(f"\nTest {'successful' if success else 'failed'}")
