#!/usr/bin/env python3
"""
Minimal training script for HRM on macOS with MPS
"""
import torch
import numpy as np
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Validate HRM dataset on macOS")
    parser.add_argument("--data_path", type=str, default="data/sudoku-extreme-1k-aug-1000-new",
                        help="Path to the dataset directory")
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Number of examples to load and process")
    args = parser.parse_args()
    
    # Check if MPS is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load the training data
    train_path = os.path.join(args.data_path, 'train')
    
    print(f"Loading data from {train_path}...")
    
    # Load first few examples
    inputs = np.load(os.path.join(train_path, 'all__inputs.npy'))[:args.num_examples]
    labels = np.load(os.path.join(train_path, 'all__labels.npy'))[:args.num_examples]
    
    # Convert to PyTorch tensors and move to device
    input_tensors = torch.tensor(inputs, dtype=torch.float32).to(device)
    label_tensors = torch.tensor(labels, dtype=torch.float32).to(device)
    
    print(f"Successfully loaded {len(inputs)} examples")
    print(f"Input shape: {input_tensors.shape}, device: {input_tensors.device}")
    print(f"Label shape: {label_tensors.shape}, device: {label_tensors.device}")
    
    # Create a simple model
    print("\nCreating a simple model...")
    model = torch.nn.Sequential(
        torch.nn.Linear(81, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 81)
    ).to(device)
    
    print(f"Model created on device: {next(model.parameters()).device}")
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simple training loop
    print("\nRunning a mini training loop...")
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(input_tensors)
        loss = criterion(outputs, label_tensors)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/3, Loss: {loss.item():.6f}")
    
    print("\nTraining completed successfully!")
    
    # Print sample predictions
    with torch.no_grad():
        sample_output = model(input_tensors[0:1]).cpu().numpy().reshape(9, 9)
        sample_input = inputs[0].reshape(9, 9)
        
        print("\nSample input grid:")
        for row in sample_input:
            print(" ".join([str(int(x)) if x != 0 else "*" for x in row]))
        
        print("\nSample prediction (raw values):")
        for row in sample_output:
            print(" ".join([f"{x:.1f}" for x in row]))

if __name__ == "__main__":
    main()
