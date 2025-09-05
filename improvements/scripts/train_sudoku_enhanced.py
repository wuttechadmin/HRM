# HRM Sudoku Training Script with Enhanced Configuration
# This script implements the key improvements from the notebook in a simpler format
# that can be run directly in the terminal.

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import math
import random
import matplotlib.pyplot as plt

print("🚀 HRM Sudoku Enhanced Training Script")
print("=" * 60)

# Set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS (Metal Performance Shaders) for Apple Silicon")
else:
    device = torch.device("cpu")
    print(f"Using CPU")

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # MPS doesn't support manual_seed yet, but we set the others for future compatibility
    pass

# Project paths
ROOT_DIR = Path(os.getcwd())
DATA_DIR = ROOT_DIR / "data" / "sudoku-extreme-1k-aug-1000"
CONFIG_DIR = ROOT_DIR / "config"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"📁 ROOT_DIR: {ROOT_DIR}")
print(f"📁 DATA_DIR: {DATA_DIR}")
print(f"📁 OUTPUT_DIR: {OUTPUT_DIR}")

# Dataset class
class HRMSudokuDataset(Dataset):
    """Smart dataset loader for HRM Sudoku data format"""

    def __init__(self, data_path, split='train', max_samples=100):
        self.data_path = Path(data_path)
        self.split = split
        self.samples = []
        self.vocab_size = 10  # Using 0-9 for Sudoku
        
        print(f"\n🔍 Loading HRM dataset from: {self.data_path / split}")
        
        split_dir = self.data_path / split
        if not split_dir.exists():
            print(f"❌ Directory {split_dir} not found, creating synthetic data")
            self.samples = self._create_synthetic_samples(max_samples)
            return
            
        # Try to directly load the numpy files we expect
        inputs_file = split_dir / "all__inputs.npy"
        labels_file = split_dir / "all__labels.npy"
        
        if inputs_file.exists() and labels_file.exists():
            print(f"✅ Found standard HRM format files")
            try:
                inputs = np.load(inputs_file)
                labels = np.load(labels_file)
                
                print(f"📊 Loaded arrays - inputs: {inputs.shape}, labels: {labels.shape}")
                
                if len(inputs) == len(labels):
                    # Verify and add samples with validation
                    valid_count = 0
                    for i in range(min(len(inputs), max_samples)):
                        if self._add_validated_sample(inputs[i], labels[i]):
                            valid_count += 1
                    
                    print(f"✅ Added {valid_count} validated samples from standard files")
                    return
            except Exception as e:
                print(f"⚠️ Error loading standard files: {e}")
                
        # Fallback to synthetic data if nothing loaded
        if len(self.samples) == 0:
            print("⚠️ No real data loaded, creating synthetic puzzles...")
            self.samples = self._create_synthetic_samples(max_samples)
            
        print(f"✅ Final dataset: {len(self.samples)} {split} samples")
    
    def _is_valid_sudoku(self, grid):
        """Check if 9x9 grid is valid Sudoku solution"""
        # Check rows
        for i in range(9):
            row = grid[i, :]
            row_no_zeros = row[row != 0]
            if len(row_no_zeros) != len(set(row_no_zeros)):
                return False
                
        # Check columns
        for i in range(9):
            col = grid[:, i]
            col_no_zeros = col[col != 0]
            if len(col_no_zeros) != len(set(col_no_zeros)):
                return False
                
        # Check 3x3 boxes
        for box_row in range(3):
            for box_col in range(3):
                box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
                box_no_zeros = box[box != 0]
                if len(box_no_zeros) != len(set(box_no_zeros)):
                    return False
                    
        return True
    
    def _add_validated_sample(self, input_data, target_data):
        """Add a sample with validation to ensure input/solution consistency"""
        try:
            input_array = np.array(input_data, dtype=np.int64)
            target_array = np.array(target_data, dtype=np.int64)

            # Cap values at 9 for Sudoku
            input_array = np.clip(input_array, 0, 9)
            target_array = np.clip(target_array, 0, 9)

            if not (len(input_array) == 81 and len(target_array) == 81):
                return False

            if not (np.all(input_array >= 0) and np.all(input_array < self.vocab_size) and
                   np.all(target_array >= 0) and np.all(target_array < self.vocab_size)):
                return False

            # CRITICAL: Ensure all non-zero input values match the target values
            # This is essential for valid Sudoku puzzles
            non_zero_mask = input_array > 0
            if not np.all(input_array[non_zero_mask] == target_array[non_zero_mask]):
                return False
                
            # Validate solution is a proper Sudoku grid
            if not self._is_valid_sudoku(target_array.reshape(9, 9)):
                return False

            self.samples.append({
                'input_ids': torch.tensor(input_array, dtype=torch.long),
                'target': torch.tensor(target_array, dtype=torch.long)
            })
            return True
        except:
            pass
        return False
    
    def _create_synthetic_samples(self, num_samples):
        """Create synthetic Sudoku samples"""
        samples = []

        # High-quality Sudoku puzzle for demo
        base_puzzle = {
            'input': [5,3,0,0,7,0,0,0,0,6,0,0,1,9,5,0,0,0,0,9,8,0,0,0,0,6,0,8,0,0,0,6,0,0,0,3,4,0,0,8,0,3,0,0,1,7,0,0,0,2,0,0,0,6,0,6,0,0,0,0,2,8,0,0,0,0,4,1,9,0,0,5,0,0,0,0,8,0,0,7,9],
            'target': [5,3,4,6,7,8,9,1,2,6,7,2,1,9,5,3,4,8,1,9,8,3,4,2,5,6,7,8,5,9,7,6,1,4,2,3,4,2,6,8,5,3,7,9,1,7,1,3,9,2,4,8,5,6,9,6,1,5,3,7,2,8,4,2,8,7,4,1,9,6,3,5,3,4,5,2,8,6,1,7,9]
        }

        for i in range(num_samples):
            input_data = base_puzzle['input'].copy()
            target_data = base_puzzle['target'].copy()

            # Add variation by removing more clues
            if i > 0:
                non_zero_indices = [idx for idx, val in enumerate(input_data) if val != 0]
                if non_zero_indices:
                    remove_count = min(3 + i % 8, len(non_zero_indices) // 2)
                    indices_to_zero = np.random.choice(non_zero_indices, size=remove_count, replace=False)
                    for idx in indices_to_zero:
                        input_data[idx] = 0

            samples.append({
                'input_ids': torch.tensor(input_data, dtype=torch.long),
                'target': torch.tensor(target_data, dtype=torch.long)
            })

        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Enhanced model with Grid-Aware Positional Encoding
class EnhancedSudokuTransformer(nn.Module):
    """Enhanced Transformer model for Sudoku solving with grid-aware positional encoding"""

    def __init__(self, vocab_size=10, hidden_size=256, num_layers=4, num_heads=8, 
                 dropout=0.1, attention_dropout=0.1, expansion_factor=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Separate embeddings for row, column, and box positions to better represent Sudoku structure
        # Make sure the total size matches hidden_size (must be divisible by 3)
        pos_size = hidden_size // 3
        self.row_embedding = nn.Embedding(9, pos_size)
        self.col_embedding = nn.Embedding(9, pos_size)
        self.box_embedding = nn.Embedding(9, pos_size)
        
        # Projection layer is not needed if the total size is exactly hidden_size
        self.pos_projection = nn.Linear(pos_size * 3, hidden_size)

        # Transformer layers with norm_first for better training dynamics
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * expansion_factor,  # Increased expansion factor
            dropout=dropout,
            activation='gelu',  # GELU activation for better training
            batch_first=True,
            norm_first=True  # Apply layer norm before attention (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Calculate Sudoku grid positions
        positions = torch.arange(81, device=device)
        rows = positions // 9
        cols = positions % 9
        boxes = (rows // 3) * 3 + (cols // 3)  # Box index (0-8)
        
        # Expand for batch dimension
        rows = rows.unsqueeze(0).expand(batch_size, -1)
        cols = cols.unsqueeze(0).expand(batch_size, -1)
        boxes = boxes.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        row_emb = self.row_embedding(rows)
        col_emb = self.col_embedding(cols)
        box_emb = self.box_embedding(boxes)
        
        # Concatenate position embeddings
        pos_emb = torch.cat([row_emb, col_emb, box_emb], dim=-1)
        pos_emb = self.pos_projection(pos_emb)
        
        # Combine with token embeddings
        x = self.token_embedding(input_ids) + pos_emb
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Output
        x = self.ln_f(x)
        return self.head(x)

# Utility functions
def is_valid_sudoku(grid_flat):
    """Check if a flattened 9x9 grid is a valid Sudoku"""
    if isinstance(grid_flat, torch.Tensor):
        grid_flat = grid_flat.cpu().numpy()
        
    grid = grid_flat.reshape(9, 9)
    
    # Check rows
    for i in range(9):
        row = grid[i, :]
        row_no_zeros = row[row != 0]
        if len(row_no_zeros) != len(set(row_no_zeros)):
            return False
            
    # Check columns
    for i in range(9):
        col = grid[:, i]
        col_no_zeros = col[col != 0]
        if len(col_no_zeros) != len(set(col_no_zeros)):
            return False
            
    # Check 3x3 boxes
    for box_row in range(3):
        for box_col in range(3):
            box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
            box_no_zeros = box[box != 0]
            if len(box_no_zeros) != len(set(box_no_zeros)):
                return False
                
    return True

def print_sudoku(grid, title="Sudoku Puzzle"):
    """Pretty print a Sudoku grid"""
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    
    grid = grid.reshape(9, 9)
    print(f"\n{title}:")
    for i in range(9):
        if i % 3 == 0 and i > 0:
            print("------+-------+------")
        row = ""
        for j in range(9):
            if j % 3 == 0 and j > 0:
                row += "| "
            val = grid[i, j].item() if hasattr(grid[i, j], 'item') else grid[i, j]
            # Make sure we display valid Sudoku values (0-9)
            if val > 9:
                val = 9  # Cap at 9 for display
            row += f"{val if val != 0 else '.'} "
        print(row)

def solve_sudoku(board):
    """Traditional Sudoku solver using backtracking"""
    # Find empty cell
    def find_empty(bo):
        for i in range(9):
            for j in range(9):
                if bo[i][j] == 0:
                    return (i, j)
        return None
    
    # Check if number is valid in position
    def valid(bo, num, pos):
        # Check row
        for j in range(9):
            if bo[pos[0]][j] == num and pos[1] != j:
                return False
        
        # Check column
        for i in range(9):
            if bo[i][pos[1]] == num and pos[0] != i:
                return False
        
        # Check 3x3 box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        
        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x*3, box_x*3 + 3):
                if bo[i][j] == num and (i, j) != pos:
                    return False
                
        return True
    
    # Main solver function
    def solve(bo):
        find = find_empty(bo)
        if not find:
            return True
        else:
            row, col = find
        
        for i in range(1, 10):
            if valid(bo, i, (row, col)):
                bo[row][col] = i
                
                if solve(bo):
                    return True
                
                bo[row][col] = 0
        
        return False
    
    # Work on a copy of the board
    board_copy = board.copy()
    if solve(board_copy):
        return board_copy
    return None

# Uniqueness constraint loss for Sudoku
def uniqueness_loss(logits, temperature=1.0):
    """Calculate the loss for ensuring uniqueness in rows, columns, and boxes"""
    batch_size = logits.shape[0]
    
    # Reshape predictions to 2D grid
    logits = logits.view(batch_size, 9, 9, -1)
    
    # Get probabilities
    probs = F.softmax(logits / temperature, dim=-1)
    
    # Only consider probabilities for digits 1-9 (not 0)
    probs = probs[:, :, :, 1:10]  # Shape [batch_size, 9, 9, 9]
    
    # Row uniqueness (penalize having the same digit in the same row)
    row_sum = probs.sum(dim=2)  # Sum over columns, shape [batch_size, 9, 9]
    row_loss = ((row_sum - 1.0) ** 2).mean()
    
    # Column uniqueness (penalize having the same digit in the same column)
    col_sum = probs.sum(dim=1)  # Sum over rows, shape [batch_size, 9, 9]
    col_loss = ((col_sum - 1.0) ** 2).mean()
    
    # Box uniqueness (penalize having the same digit in the same 3x3 box)
    # Reshape to get boxes
    boxes = probs.reshape(batch_size, 3, 3, 3, 3, 9)
    boxes = boxes.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, 9, 9, 9)
    box_sum = boxes.sum(dim=2)  # Sum within each 3x3 box
    box_loss = ((box_sum - 1.0) ** 2).mean()
    
    # Combine all constraints
    return row_loss + col_loss + box_loss

# Function to compute metrics
def compute_metrics(model, dataloader, device):
    model.eval()
    val_correct = 0
    val_total = 0
    valid_solutions = 0
    exact_matches = 0
    total_samples = 0
    empty_correct = 0
    empty_total = 0
    clue_preserved_count = 0
    clue_total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)
            
            logits = model(input_ids)
            # Ensure we only consider valid Sudoku digits (0-9)
            logits = logits[:, :, :10]
            predictions = logits.argmax(dim=-1)
            
            # Calculate cell-level accuracy (for all cells)
            val_correct += (predictions == targets).sum().item()
            val_total += targets.numel()
            
            # Calculate cell-level accuracy (only for non-clue positions)
            non_clue_mask = input_ids == 0
            empty_correct += ((predictions == targets) & non_clue_mask).sum().item()
            empty_total += non_clue_mask.sum().item()
            
            # Calculate clue preservation (should be 100%)
            clue_mask = input_ids > 0
            clue_preserved_count += ((predictions == input_ids) & clue_mask).sum().item()
            clue_total += clue_mask.sum().item()
            
            # Calculate puzzle-level metrics (exact match and valid solutions)
            for i in range(input_ids.size(0)):
                total_samples += 1
                
                # Get individual sample
                sample_input = input_ids[i].cpu().numpy()
                sample_target = targets[i].cpu().numpy()
                sample_pred = predictions[i].cpu().numpy()
                
                # Ensure clues are preserved
                non_zero_mask = sample_input != 0
                sample_pred[non_zero_mask] = sample_input[non_zero_mask]
                
                # Check for exact match
                if np.array_equal(sample_pred, sample_target):
                    exact_matches += 1
                
                # Check if solution is valid
                if is_valid_sudoku(sample_pred):
                    valid_solutions += 1
    
    # Calculate metrics
    cell_accuracy = val_correct / val_total if val_total > 0 else 0
    empty_cell_accuracy = empty_correct / empty_total if empty_total > 0 else 0
    clue_preservation = clue_preserved_count / clue_total if clue_total > 0 else 0
    exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0
    valid_solution_rate = valid_solutions / total_samples if total_samples > 0 else 0
    
    model.train()
    return {
        'cell_accuracy': cell_accuracy,
        'empty_cell_accuracy': empty_cell_accuracy,
        'clue_preservation': clue_preservation,
        'exact_match_rate': exact_match_rate,
        'valid_solution_rate': valid_solution_rate
    }

# Function to save training metrics and create visualizations
def save_training_metrics(history, filename='training_metrics.json'):
    """Save training metrics to a JSON file for later analysis"""
    import json
    
    # Convert tensor values to Python floats
    serializable_history = {}
    for key, values in history.items():
        serializable_history[key] = [float(val) for val in values]
    
    # Save to file
    out_path = OUTPUT_DIR / filename
    with open(out_path, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    
    print(f"✅ Training metrics saved to {out_path}")
    
    # Try to create visualizations if matplotlib is available
    try:
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot training loss
        axs[0, 0].plot(serializable_history['train_loss'])
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        
        # Plot cell accuracy
        axs[0, 1].plot(serializable_history['val_acc'])
        axs[0, 1].set_title('Cell Accuracy')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Accuracy')
        
        # Plot exact match rate
        axs[1, 0].plot(serializable_history['exact_match'])
        axs[1, 0].set_title('Exact Match Rate')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Rate')
        
        # Plot valid solution rate
        axs[1, 1].plot(serializable_history['valid_solution'])
        axs[1, 1].set_title('Valid Solution Rate')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Rate')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'training_metrics.png')
        print(f"✅ Training visualizations saved to {OUTPUT_DIR / 'training_metrics.png'}")
    except Exception as e:
        print(f"⚠️ Error creating visualizations: {e}")

# Main function for training
def train_enhanced_model():
    print("\n🔄 Training a new Sudoku model with improved configuration...")
    
    # Enhanced configuration with improved learning parameters
    config = {
        'data_path': DATA_DIR,
        'epochs': 100,                # More epochs for better convergence
        'batch_size': 64,             # Balanced batch size
        'learning_rate': 1e-4,        # Good starting learning rate
        'weight_decay': 0.01,         # Regularization
        'hidden_size': 256,           # Model size
        'num_layers': 6,              # More layers for deeper reasoning
        'num_heads': 8,               # Multiple attention heads
        'expansion_factor': 4,        # Feedforward expansion factor
        'max_train_samples': 2000,    # Use more training samples
        'max_val_samples': 400,       # More validation samples
        'min_epochs': 15,             # Train for at least this many epochs
        'early_stopping_patience': 15, # Patient early stopping
        'early_stopping_threshold': 0.0005, # Sensitive improvement detection
        'early_stopping_metric': 'valid_solution_rate', # Use valid solutions as metric
        'gradient_clip': 1.0,         # Gradient clipping for stability
        'lr_scheduler': 'cosine',     # Cosine learning rate scheduler
        'warmup_steps': 200,          # Learning rate warmup
        'min_lr_factor': 0.05,        # Minimum learning rate
        'dropout': 0.1,               # Dropout rate
        'attention_dropout': 0.1,     # Attention dropout
        'label_smoothing': 0.05,      # Label smoothing
        'enable_constraint_losses': True, # Use Sudoku-specific losses
        'constraint_loss_weight': 0.3, # Weight for constraint losses
        'save_path': OUTPUT_DIR / 'sudoku_model_best.pt', # Save path
    }
    
    print("\n📋 Enhanced Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create datasets
    train_dataset = HRMSudokuDataset(config['data_path'], 'train', config['max_train_samples'])
    val_dataset = HRMSudokuDataset(config['data_path'], 'test', config['max_val_samples'])
    
    if len(train_dataset) == 0:
        print("❌ No training data available")
        return None
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Create the model
    vocab_size = 10  # Standard for Sudoku (0-9)
    model = EnhancedSudokuTransformer(
        vocab_size=vocab_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        expansion_factor=config['expansion_factor']
    ).to(device)
    
    print(f"📊 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"📊 Training on {len(train_dataset)} samples")
    
    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)  # Higher beta2 for more stable updates
    )
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,  # Ignore padding in loss calculation
        label_smoothing=config['label_smoothing']
    )
    
    # Cosine learning rate scheduler
    def cosine_schedule_with_warmup(step, total_steps, warmup_steps, base_lr, min_lr_factor=0.1):
        min_lr = base_lr * min_lr_factor
        if step < warmup_steps:
            return base_lr * (step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    
    # Training loop
    start_time = time.time()
    model.train()
    best_val_acc = 0
    best_exact_match = 0
    best_valid_solution = 0
    patience_counter = 0
    training_history = {
        'train_loss': [], 
        'val_acc': [], 
        'empty_cell_acc': [],
        'clue_preservation': [],
        'exact_match': [], 
        'valid_solution': []
    }
    
    # Training progress bar
    for epoch in range(config['epochs']):
        total_loss = 0
        num_batches = 0
        
        # Training phase
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)
            
            # Get current learning rate from scheduler
            current_lr = config['learning_rate']  # Default value
            if config['lr_scheduler'] == 'cosine':
                current_step = epoch * len(train_loader) + num_batches
                total_steps = config['epochs'] * len(train_loader)
                current_lr = cosine_schedule_with_warmup(
                    current_step, 
                    total_steps, 
                    config['warmup_steps'], 
                    config['learning_rate'],
                    min_lr_factor=config['min_lr_factor']
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            optimizer.zero_grad()
            logits = model(input_ids)
            
            # Ensure we only consider valid Sudoku digits (0-9)
            logits = logits[:, :, :10]
            
            # Standard cross-entropy loss
            ce_loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Add Sudoku-specific constraint losses if enabled
            constraint_loss = 0
            constraint_weight = 0
            if config.get('enable_constraint_losses', False):
                constraint_weight = config.get('constraint_loss_weight', 0.3)
                # Gradually increase the constraint loss weight over time
                epoch_fraction = epoch / config['epochs']
                constraint_weight = constraint_weight * min(1.0, epoch_fraction * 2.0)
                constraint_loss = uniqueness_loss(logits)
                
            # Combined loss
            loss = ce_loss + constraint_weight * constraint_loss
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})
        
        avg_loss = total_loss / num_batches
        training_history['train_loss'].append(avg_loss)
        
        # Compute all validation metrics
        metrics = compute_metrics(model, val_loader, device)
        val_acc = metrics['cell_accuracy']
        empty_cell_acc = metrics['empty_cell_accuracy']
        clue_preservation = metrics['clue_preservation']
        exact_match_rate = metrics['exact_match_rate']
        valid_solution_rate = metrics['valid_solution_rate']
        
        # Update history
        training_history['val_acc'].append(val_acc)
        training_history['empty_cell_acc'].append(empty_cell_acc)
        training_history['clue_preservation'].append(clue_preservation)
        training_history['exact_match'].append(exact_match_rate)
        training_history['valid_solution'].append(valid_solution_rate)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, "
              f"Exact Matches={exact_match_rate:.4f}, Valid Solutions={valid_solution_rate:.4f}")
        
        # Determine the primary metric for early stopping based on the selected metric
        if config['early_stopping_metric'] == 'valid_solution_rate':
            primary_metric = valid_solution_rate
            best_metric_value = best_valid_solution
            best_metric_name = "valid solution rate"
        elif config['early_stopping_metric'] == 'exact_match_rate':
            primary_metric = exact_match_rate
            best_metric_value = best_exact_match
            best_metric_name = "exact match rate"
        else:  # Default to cell accuracy
            primary_metric = val_acc
            best_metric_value = best_val_acc
            best_metric_name = "cell accuracy"
        
        # Check if we've improved
        if primary_metric > best_metric_value + config['early_stopping_threshold']:
            if config['early_stopping_metric'] == 'valid_solution_rate':
                best_valid_solution = primary_metric
            elif config['early_stopping_metric'] == 'exact_match_rate':
                best_exact_match = primary_metric
            else:
                best_val_acc = primary_metric
                
            best_metric_value = primary_metric
            patience_counter = 0
            print(f"✅ New best {best_metric_name}: {best_metric_value:.4f}")
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'exact_match_rate': exact_match_rate,
                'valid_solution_rate': valid_solution_rate,
            }, config['save_path'])
        else:
            patience_counter += 1
            print(f"⏳ No improvement in {patience_counter}/{config['early_stopping_patience']} epochs")
            
            if patience_counter >= config['early_stopping_patience'] and epoch >= config.get('min_epochs', 0):
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        # Additional early stopping if we reach very high accuracy
        if exact_match_rate > 0.95:
            print(f"✅ Reached excellent exact match rate ({exact_match_rate:.4f}). Early stopping.")
            break
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    print(f"✅ Best validation accuracy: {best_val_acc:.4f}")
    print(f"✅ Best exact match rate: {best_exact_match:.4f}")
    print(f"✅ Best valid solution rate: {best_valid_solution:.4f}")
    
    # Save training metrics and visualizations
    save_training_metrics(training_history)
    
    # Test with user-provided example
    print("\n\n" + "="*50)
    print("🧩 TESTING WITH USER-PROVIDED EXAMPLE")
    print("="*50)
    
    # User example from the conversation
    user_input = np.zeros((9, 9), dtype=np.int64)
    user_input[0, 1] = 6; user_input[0, 2] = 4; user_input[0, 5] = 7; user_input[0, 8] = 1
    user_input[1, 0] = 7; user_input[1, 3] = 1
    user_input[2, 1] = 9; user_input[2, 4] = 4; user_input[2, 6] = 8; user_input[2, 7] = 7
    user_input[3, 3] = 3
    user_input[4, 2] = 2; user_input[4, 5] = 9; user_input[4, 7] = 5
    user_input[5, 0] = 9; user_input[5, 4] = 1; user_input[5, 8] = 8
    user_input[6, 0] = 4; user_input[6, 4] = 6; user_input[6, 8] = 9
    user_input[7, 1] = 3; user_input[7, 6] = 2
    user_input[8, 5] = 5; user_input[8, 7] = 4
    
    user_solution = np.array([
        [5, 6, 4, 8, 2, 7, 3, 9, 1],
        [7, 2, 8, 1, 9, 3, 5, 6, 4],
        [3, 9, 1, 5, 4, 6, 8, 7, 2],
        [8, 4, 6, 3, 5, 2, 9, 1, 7],
        [1, 7, 2, 6, 8, 9, 4, 5, 3],
        [9, 5, 3, 7, 1, 4, 6, 2, 8],
        [4, 1, 5, 2, 6, 8, 7, 3, 9],
        [6, 3, 9, 4, 7, 1, 2, 8, 5],
        [2, 8, 7, 9, 3, 5, 1, 4, 6]
    ], dtype=np.int64)
    
    # Flatten for processing
    user_input_flat = user_input.flatten()
    user_solution_flat = user_solution.flatten()
    
    # Validate the user example
    non_zero_mask = user_input_flat > 0
    clues_match = np.all(user_input_flat[non_zero_mask] == user_solution_flat[non_zero_mask])
    solution_valid = is_valid_sudoku(user_solution_flat)
    
    print(f"Validating user example:")
    print(f"- Clues match solution: {'✅' if clues_match else '❌'}")
    print(f"- Solution is valid: {'✅' if solution_valid else '❌'}")
    
    # Only proceed if the example is valid
    if clues_match and solution_valid:
        # Convert to tensors
        input_tensor = torch.tensor(user_input_flat, dtype=torch.long).to(device)
        target_tensor = torch.tensor(user_solution_flat, dtype=torch.long).to(device)
        
        print_sudoku(user_input_flat, "Input Puzzle (User Example)")
        
        # Solve with traditional algorithm for comparison
        traditional_solution = solve_sudoku(user_input.copy())
        if traditional_solution is not None:
            traditional_solution_flat = np.array(traditional_solution).flatten()
            print_sudoku(traditional_solution_flat, "Traditional Solver Solution")
            
            # Check if traditional solution matches expected solution
            trad_matches_expected = np.array_equal(traditional_solution_flat, user_solution_flat)
            print(f"Traditional solution matches expected: {'✅' if trad_matches_expected else '❌'}")
        else:
            print("❌ Traditional solver could not find a solution")
        
        print_sudoku(user_solution_flat, "Expected Solution (User Example)")
        
        # Run model inference
        model.eval()
        with torch.no_grad():
            output_logits = model(input_tensor.unsqueeze(0))
            # Only consider logits for digits 0-9 (ignore any higher values)
            output_logits = output_logits[:, :, :10]
            output_tokens = torch.argmax(output_logits, dim=-1).squeeze(0)
            
            # Ensure input clues are preserved
            non_zero_mask_tensor = input_tensor > 0
            output_tokens[non_zero_mask_tensor] = input_tensor[non_zero_mask_tensor]
        
        print_sudoku(output_tokens.cpu().numpy(), "Model Output")
        
        # Calculate accuracy metrics
        non_zero_mask_tensor = input_tensor > 0
        zero_mask_tensor = input_tensor == 0
        
        # Check if model preserves input clues
        input_preserved = (output_tokens[non_zero_mask_tensor] == input_tensor[non_zero_mask_tensor]).float().mean()
        # Check accuracy on filled cells
        filled_accuracy = (output_tokens[zero_mask_tensor] == target_tensor[zero_mask_tensor]).float().mean()
        # Check overall accuracy
        overall_accuracy = (output_tokens == target_tensor).float().mean()
        
        print(f"\n📊 Metrics for User Example:")
        print(f"- Input clues preserved: {input_preserved.item()*100:.2f}%")
        print(f"- Empty cell accuracy: {filled_accuracy.item()*100:.2f}%")
        print(f"- Overall accuracy: {overall_accuracy.item()*100:.2f}%")
        
        # Validate model output as valid Sudoku
        model_output = output_tokens.cpu().numpy()
        model_output_valid = is_valid_sudoku(model_output)
        print(f"- Model output is valid Sudoku: {'✅' if model_output_valid else '❌'}")
        
        # Show a detailed comparison of cells
        print("\n📊 Cell-by-cell comparison:")
        zero_positions = np.where(user_input_flat == 0)[0]
        correct_count = 0
        wrong_count = 0
        mismatch_count = 0
        print("\nDetailed comparison of model vs expected solution (first 10 mismatches if any):")
        for pos in zero_positions:
            row, col = pos // 9, pos % 9
            model_pred = output_tokens[pos].item()
            correct = user_solution_flat[pos]
            if model_pred == correct:
                correct_count += 1
            else:
                wrong_count += 1
                if mismatch_count < 10:
                    print(f"  Position ({row+1},{col+1}): Model={model_pred}, Correct={correct} ❌")
                    mismatch_count += 1
        
        print(f"\nSummary: {correct_count} correct cells, {wrong_count} incorrect cells out of {len(zero_positions)} empty cells")
        
        # Compare model with traditional solver
        if traditional_solution is not None:
            traditional_solution_flat = np.array(traditional_solution).flatten()
            model_matches_trad = np.sum(model_output == traditional_solution_flat) / 81
            print(f"\nModel agreement with traditional solver: {model_matches_trad*100:.2f}%")
    else:
        print("❌ User example is invalid. Please verify the input and solution.")
    
    return model, train_dataset, val_dataset

if __name__ == "__main__":
    train_enhanced_model()
