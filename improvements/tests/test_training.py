import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

class SudokuTransformer(nn.Module):
    def __init__(self, vocab_size=10, hidden_size=64, num_layers=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 81, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads,
            dim_feedforward=2*hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        return self.output_proj(x)

def load_data(data_path, split='train', max_samples=100, batch_size=32):
    """Load dataset and create DataLoader"""
    inputs_file = data_path / split / "all__inputs.npy"
    labels_file = data_path / split / "all__labels.npy"
    
    inputs = np.load(inputs_file)
    labels = np.load(labels_file)
    
    print(f"Loading {split} data: {inputs.shape}")
    
    # Limit samples
    if max_samples and max_samples < len(inputs):
        indices = np.random.choice(len(inputs), max_samples, replace=False)
        inputs = inputs[indices]
        labels = labels[indices]
    
    # Convert to tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == 'train')
    )
    
    return dataloader

def is_valid_sudoku(grid):
    """Check if a Sudoku grid is valid"""
    grid_reshaped = grid.reshape(9, 9)
    
    # Check rows
    for i in range(9):
        row = grid_reshaped[i]
        row_no_zeros = row[row != 0]
        if len(row_no_zeros) != len(set(row_no_zeros)):
            return False
    
    # Check columns
    for i in range(9):
        col = grid_reshaped[:, i]
        col_no_zeros = col[col != 0]
        if len(col_no_zeros) != len(set(col_no_zeros)):
            return False
    
    # Check boxes
    for box_row in range(3):
        for box_col in range(3):
            box = grid_reshaped[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
            box_no_zeros = box[box != 0]
            if len(box_no_zeros) != len(set(box_no_zeros)):
                return False
    
    return True

def evaluate_model(model, dataloader, device):
    """Evaluate model on the dataloader"""
    model.eval()
    val_correct = 0
    val_total = 0
    exact_matches = 0
    valid_solutions = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(inputs)
            predictions = logits.argmax(dim=-1)
            
            # Calculate cell-level accuracy (for non-clue positions)
            non_clue_mask = inputs == 0
            val_correct += ((predictions == targets) & non_clue_mask).sum().item()
            val_total += non_clue_mask.sum().item()
            
            # Check each sample for exact match and valid solution
            for i in range(inputs.size(0)):
                input_grid = inputs[i].cpu().numpy()
                target_grid = targets[i].cpu().numpy()
                pred_grid = predictions[i].cpu().numpy()
                
                # Ensure clues are preserved in prediction
                non_zero_mask = input_grid > 0
                pred_grid[non_zero_mask] = input_grid[non_zero_mask]
                
                # Check for exact match
                if np.array_equal(pred_grid, target_grid):
                    exact_matches += 1
                
                # Check for valid solution
                if is_valid_sudoku(pred_grid):
                    valid_solutions += 1
    
    # Calculate metrics
    cell_accuracy = val_correct / val_total if val_total > 0 else 0
    exact_match_rate = exact_matches / len(dataloader.dataset)
    valid_solution_rate = valid_solutions / len(dataloader.dataset)
    
    return {
        'cell_accuracy': cell_accuracy,
        'exact_match_rate': exact_match_rate,
        'valid_solution_rate': valid_solution_rate
    }

def train_model(data_path, config=None):
    """Train a model with the given configuration"""
    if config is None:
        config = {
            'hidden_size': 64,
            'num_layers': 3,
            'num_heads': 4,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'weight_decay': 0.01,
            'epochs': 20,
            'max_train_samples': 1000,
            'max_val_samples': 200
        }
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Load data
    train_loader = load_data(
        data_path, 'train', 
        max_samples=config['max_train_samples'], 
        batch_size=config['batch_size']
    )
    val_loader = load_data(
        data_path, 'test', 
        max_samples=config['max_val_samples'], 
        batch_size=config['batch_size']
    )
    
    # Create model
    model = SudokuTransformer(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads']
    ).to(device)
    
    # Setup optimization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training history
    history = {
        'train_loss': [],
        'cell_accuracy': [],
        'exact_match_rate': [],
        'valid_solution_rate': []
    }
    
    # Train loop
    start_time = time.time()
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        # Validate
        metrics = evaluate_model(model, val_loader, device)
        
        # Update history
        avg_epoch_loss = epoch_loss / batch_count
        history['train_loss'].append(avg_epoch_loss)
        history['cell_accuracy'].append(metrics['cell_accuracy'])
        history['exact_match_rate'].append(metrics['exact_match_rate'])
        history['valid_solution_rate'].append(metrics['valid_solution_rate'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_epoch_loss:.4f}, Accuracy: {metrics['cell_accuracy']:.2%}")
        print(f"  Exact Matches: {metrics['exact_match_rate']:.2%}, Valid Solutions: {metrics['valid_solution_rate']:.2%}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['cell_accuracy'])
    plt.title('Cell Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['valid_solution_rate'], label='Valid')
    plt.plot(history['exact_match_rate'], label='Exact')
    plt.title('Solution Quality')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to 'training_history.png'")
    
    return model, history

if __name__ == "__main__":
    DATA_DIR = Path("/Users/robertburkhall/Development/HRM/data/sudoku-extreme-1k-aug-1000")
    
    # Use a smaller model with faster training time
    config = {
        'hidden_size': 32,       # Small hidden size for quicker training
        'num_layers': 2,         # Reduced number of layers
        'num_heads': 4,          # Multiple attention heads
        'batch_size': 32,        # Moderate batch size
        'learning_rate': 1e-3,   # Slightly higher learning rate
        'weight_decay': 0.01,    # Regularization
        'epochs': 20,            # More epochs to see learning trend
        'max_train_samples': 500, # Reduced samples for quicker training
        'max_val_samples': 100   # Reduced validation set
    }
    
    model, history = train_model(DATA_DIR, config)
