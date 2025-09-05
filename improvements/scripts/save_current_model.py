"""
Script to save the current state of a model in training.
This can be executed separately to save the model state at any point.
"""

import torch
from pathlib import Path
import datetime

# Create checkpoints directory if it doesn't exist
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Create a timestamp for the manual checkpoint
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Assuming 'model', 'optimizer', and relevant metrics are in scope
# from the training environment
try:
    # Save checkpoint with timestamp
    save_path = checkpoint_dir / f"manual_checkpoint_{timestamp}.pt"
    
    # You'll need to make sure these variables are accessible
    # when this script is executed
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if 'optimizer' in globals() else None,
        'epoch': epoch if 'epoch' in globals() else None,
        'metrics': {
            'cell_accuracy': val_metrics['cell_accuracy'] if 'val_metrics' in globals() else None,
            'exact_match_rate': val_metrics['exact_match_rate'] if 'val_metrics' in globals() else None,
            'valid_solution_rate': val_metrics['valid_solution_rate'] if 'val_metrics' in globals() else None
        }
    }
    
    torch.save(checkpoint, save_path)
    print(f"💾 Manually saved checkpoint to {save_path}")

except Exception as e:
    print(f"❌ Error saving checkpoint: {e}")
