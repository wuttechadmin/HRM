import torch
import numpy as np
from pathlib import Path
import json

# Define SudokuTransformer class
class SudokuTransformer(torch.nn.Module):
    def __init__(self, vocab_size=10, hidden_size=128, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = torch.nn.Parameter(torch.zeros(1, 81, hidden_size))
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads,
            dim_feedforward=4*hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = torch.nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        # Embedding and positional encoding
        x = self.embedding(x)
        x = x + self.pos_encoding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Project back to vocabulary space
        return self.output_proj(x)

def analyze_model_weights(model):
    """Print statistics about model weights"""
    total_params = 0
    weight_norms = []
    gradient_norms = []
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        weight_norms.append((name, param.norm().item()))
        if param.grad is not None:
            gradient_norms.append((name, param.grad.norm().item()))
    
    print(f"Total parameters: {total_params:,}")
    print("\nWeight norms (top 5 largest):")
    for name, norm in sorted(weight_norms, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {norm:.4f}")
    
    if gradient_norms:
        print("\nGradient norms (top 5 largest):")
        for name, norm in sorted(gradient_norms, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {name}: {norm:.4f}")
    else:
        print("\nNo gradients found")

def main():
    # Create a simple model with default parameters
    model = SudokuTransformer()
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Create a small batch for testing
    batch_size = 8
    seq_len = 81
    input_ids = torch.randint(0, 10, (batch_size, seq_len), device=device)
    
    # Forward pass
    print("Running forward pass...")
    output = model(input_ids)
    print(f"Output shape: {output.shape}")
    
    # Loss calculation
    target = torch.randint(0, 10, (batch_size, seq_len), device=device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    print("Running backward pass...")
    loss.backward()
    
    # Analyze model parameters
    analyze_model_weights(model)

if __name__ == "__main__":
    main()
