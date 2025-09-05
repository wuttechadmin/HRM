# Memory Optimization Strategies for MPS Training

This document provides recommendations for optimizing memory usage and training performance for PyTorch models running on Apple's MPS (Metal Performance Shaders) backend.

## Memory Boundary Alignment

Apple's Metal framework performs best when data is properly aligned to memory boundaries:

```python
# Optimal batch size values (powers of 2)
batch_sizes = [32, 64, 128, 256]  # Choose based on model size and memory constraints

# Optimal model dimensions (also powers of 2)
hidden_dims = [512, 1024, 2048]
attention_heads = [8, 16]  # Should divide hidden_dim evenly
```

## Mixed Precision Training

Implementing automatic mixed precision on MPS:

```python
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

# Create scaler for gradient scaling
scaler = GradScaler()

# In training loop
with autocast(device_type='mps', dtype=torch.float16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Scale gradients and optimize
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Gradient Accumulation

Reduce memory footprint by accumulating gradients over multiple smaller batches:

```python
# Configuration
accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps
optimizer.zero_grad()

# Training loop
for i, (inputs, targets) in enumerate(dataloader):
    # Forward pass with smaller batch
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps  # Scale loss
    loss.backward()  # Accumulate gradients
    
    # Update weights after accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Memory Monitoring

Add memory monitoring without disrupting training:

```python
# Memory usage monitoring function
import torch
import gc

def log_memory_usage(prefix=""):
    """Log current memory usage without disrupting training"""
    gc.collect()
    torch.mps.empty_cache()
    
    # Get memory info for MPS
    memory_allocated = torch.mps.current_allocated_memory() / (1024 ** 3)  # GB
    memory_reserved = torch.mps.driver_allocated_memory() / (1024 ** 3)    # GB
    
    print(f"{prefix} Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    
    return memory_allocated, memory_reserved

# Call periodically at end of epochs (not during training)
log_memory_usage("End of epoch")
```

## Proper Checkpoint & Interrupt System

```python
# Example training loop with proper interrupt handling
try:
    for epoch in range(start_epoch, num_epochs):
        # Training code...
        
        # Regular checkpointing (not just on improvement)
        if epoch % checkpoint_frequency == 0:
            save_path = checkpoint_dir / f"regular_checkpoint_epoch{epoch}.pt"
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': val_metrics
            }, save_path)
            
except KeyboardInterrupt:
    print("\n🛑 Training interrupted by user. Saving checkpoint...")
    save_path = checkpoint_dir / f"interrupted_checkpoint_epoch{epoch}.pt"
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)
    print(f"💾 Saved interrupted checkpoint to {save_path}")
    
finally:
    # Clean up code
    torch.mps.empty_cache()
    print("Training complete or interrupted. Resources cleaned up.")
```

## Data Pipeline Optimization

```python
# Efficient data loading
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,           # Increase based on CPU cores available
    pin_memory=True,         # Faster data transfer to GPU
    persistent_workers=True, # Keep workers alive between epochs
    prefetch_factor=2        # Pre-fetch batches ahead of time
)
```

These optimization strategies should help improve training efficiency on your M1 Mac Mini, bringing performance closer to dedicated GPUs while maintaining stability.
