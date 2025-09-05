import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Simple dataset class
class SudokuDataset(torch.utils.data.Dataset):
    def __init__(self, inputs_path, labels_path, max_samples=None):
        self.inputs = np.load(inputs_path)
        self.labels = np.load(labels_path)
        
        if max_samples is not None:
            self.inputs = self.inputs[:max_samples]
            self.labels = self.labels[:max_samples]
            
        print(f"Dataset loaded: {len(self.inputs)} samples")
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.inputs[idx], dtype=torch.long),
            'target': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Define a simple Transformer model
class SudokuTransformer(nn.Module):
    def __init__(self, vocab_size=10, hidden_size=128, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Embedding(81, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.embedding(x) + self.pos_encoding(pos)
        x = self.transformer(x)
        return self.output(x)

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10, validate_every=50):
    model.train()
    history = {'train_loss': [], 'val_acc': []}
    global_step = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            global_step += 1
            
            # Get data
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            avg_loss = epoch_loss / batch_count
            
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            history['train_loss'].append(loss.item())
            
            # Validation
            if global_step % validate_every == 0:
                val_acc = validate_model(model, val_loader)
                history['val_acc'].append(val_acc)
                print(f"Step {global_step}: Validation accuracy = {val_acc:.4f}")
                model.train()
        
        # End of epoch validation
        val_acc = validate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return history

# Validation function
def validate_model(model, val_loader, max_batches=10):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            predictions = logits.argmax(dim=-1)
            
            # Calculate accuracy (only on non-zero targets)
            mask = targets != 0
            correct += (predictions[mask] == targets[mask]).sum().item()
            total += mask.sum().item()
    
    return correct / total if total > 0 else 0

# Run a test training session
def main():
    # Paths
    data_path = Path('data/sudoku-extreme-1k-aug-1000')
    train_inputs = data_path / 'train' / 'all__inputs.npy'
    train_labels = data_path / 'train' / 'all__labels.npy'
    test_inputs = data_path / 'test' / 'all__inputs.npy'
    test_labels = data_path / 'test' / 'all__labels.npy'
    
    # Parameters
    batch_size = 64
    max_samples = 1000
    hidden_size = 128
    num_layers = 4
    num_heads = 4
    learning_rate = 1e-4
    epochs = 10
    
    # Create datasets
    train_dataset = SudokuDataset(train_inputs, train_labels, max_samples)
    val_dataset = SudokuDataset(test_inputs, test_labels, max_samples // 5)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = SudokuTransformer(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads
    ).to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Train model
    print(f"Starting training with {len(train_dataset)} samples")
    start_time = time.time()
    history = train_model(model, train_loader, val_loader, optimizer, criterion, epochs)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Validation Steps')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("Results saved to training_results.png")

if __name__ == '__main__':
    main()
