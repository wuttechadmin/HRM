import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# Check environment
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

# Set device
device_name = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)
print("Using device:", device_name)

# Set paths
ROOT_DIR = Path(os.getcwd())
DATA_DIR = ROOT_DIR / "data" / "sudoku-extreme-1k-aug-1000"
print("Data directory exists:", DATA_DIR.exists())
print("Test directory exists:", (DATA_DIR / "test").exists())
print("Train directory exists:", (DATA_DIR / "train").exists())

# Check for necessary files
test_files = list((DATA_DIR / "test").glob("*.npy"))
train_files = list((DATA_DIR / "train").glob("*.npy"))
print(f"Found {len(test_files)} test files and {len(train_files)} train files")

# Test HRMSudokuDataset class import (simplified version for testing)
class SimpleSudokuDataset(Dataset):
    def __init__(self, data_dir, split="train", max_samples=None):
        self.split = split
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        self.samples = []
        
        # Check if directory exists
        if not self.split_dir.exists():
            print(f"Error: Split directory {self.split_dir} does not exist")
            return
            
        # Load inputs and labels
        inputs_file = self.split_dir / "all__inputs.npy"
        labels_file = self.split_dir / "all__labels.npy"
        
        if not inputs_file.exists() or not labels_file.exists():
            print(f"Error: Required files missing in {split}")
            return
            
        # Load arrays
        inputs = np.load(inputs_file)
        labels = np.load(labels_file)
        
        print(f"{split.upper()} Split:")
        print(f"  - Inputs: {inputs.shape}, dtype={inputs.dtype}")
        print(f"  - Labels: {labels.shape}, dtype={labels.dtype}")
        
        # Create sample list (limit to max_samples if specified)
        max_idx = len(inputs) if max_samples is None else min(len(inputs), max_samples)
        
        for i in range(max_idx):
            self.samples.append({
                'input_ids': torch.tensor(inputs[i], dtype=torch.long),
                'target': torch.tensor(labels[i], dtype=torch.long)
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# Test SudokuTransformer class (simplified version for testing)
class SimpleSudokuTransformer(nn.Module):
    def __init__(self, vocab_size=10, hidden_size=128, num_layers=4, num_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(81, hidden_size)  # 9x9 Sudoku
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Position indices
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos_ids)
        
        # Combined embeddings
        x = token_emb + pos_emb
        
        # Transformer
        x = self.transformer(x)
        
        # Output projection
        logits = self.head(x)
        
        return logits

# Try to create a dataset and model
try:
    print("\nTesting dataset creation...")
    test_dataset = SimpleSudokuDataset(DATA_DIR, split="test", max_samples=10)
    print(f"Created test dataset with {len(test_dataset)} samples")
    
    # Check a sample
    if len(test_dataset) > 0:
        sample = test_dataset[0]
        print(f"Sample input shape: {sample['input_ids'].shape}")
        print(f"Sample target shape: {sample['target'].shape}")
        print(f"Sample clue count: {(sample['input_ids'] > 0).sum().item()}")
    
    print("\nTesting model creation...")
    config = {
        'hidden_size': 128,
        'num_layers': 4,
        'num_heads': 4,
    }
    model = SimpleSudokuTransformer(
        vocab_size=10,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads']
    ).to(device)
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("\nTesting forward pass...")
    if len(test_dataset) > 0:
        # Get a batch
        input_ids = test_dataset[0]['input_ids'].unsqueeze(0).to(device)
        print(f"Input batch shape: {input_ids.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(input_ids)
            print(f"Output logits shape: {logits.shape}")
            
            # Get predictions
            predictions = logits.argmax(dim=-1)
            print(f"Predictions shape: {predictions.shape}")
    
    print("\nBasic functionality test completed successfully!")
except Exception as e:
    print(f"Error during test: {e}")
