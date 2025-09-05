# Running HRM on macOS with Apple Silicon

This guide explains how to run the Hierarchical Reasoning Model (HRM) on macOS with Apple Silicon chips (M1/M2/M3).

## What's been added?

I've made the following adjustments to enable HRM to run on your Apple Silicon Mac:

1. **Modified Dataset Generation**: Updated `dataset/build_sudoku_dataset.py` to use proper formatting for empty cells.

2. **Added Test Scripts**: Created test scripts to verify MPS (Metal Performance Shaders) functionality.

3. **Created Simple Training Example**: Added `test_mps_training.py` that demonstrates a simplified training loop using MPS acceleration.

## How to run

### 1. Generate the dataset (if not already done)

To generate the improved dataset with proper formatting (empty cells as 0 displayed as "*"):

```bash
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
```

### 2. Run the test script to validate MPS functionality

```bash
python test_mps_training.py --data_path data/sudoku-extreme-1k-aug-1000
```

### 3. For the full HRM model

Due to the complexity of the original codebase, for Apple Silicon (M1/M2/M3) Macs, the recommended approach is to create a simplified version of the model that's compatible with MPS. The full HRM architecture can be adapted by:

1. Starting with the `test_mps_training.py` example
2. Gradually incorporating components from the original model
3. Testing each addition to ensure MPS compatibility

## Training Performance

On Apple Silicon:
- Training will be slower than on NVIDIA GPUs but still significantly faster than CPU-only training
- You may want to use a smaller batch size for better performance
- Some PyTorch operations might fall back to CPU if not supported by MPS

## Checking results

You can use the validation script to check the dataset:

```bash
python validate_data_new.py --data_path data/sudoku-extreme-1k-aug-1000
```

## Notes for developers

If you want to port the full HRM model to MPS, keep these points in mind:

1. **Device Handling**: Always use a device-agnostic approach:
   ```python
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   model = model.to(device)
   ```

2. **Distributed Training**: MPS doesn't support distributed training with PyTorch's DistributedDataParallel. Use a single GPU approach.

3. **MPS Limitations**: Some operations might not be supported by MPS and will fall back to CPU. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to enable automatic fallback.

4. **Numerical Precision**: MPS may have different numerical precision than CUDA, which could affect training stability.

## CUDA/NVIDIA Compatibility

The original codebase remains fully compatible with CUDA/NVIDIA GPUs. The dataset changes and other improvements are beneficial for all platforms.
