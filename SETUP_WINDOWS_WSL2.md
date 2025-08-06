# Windows WSL2 Setup Guide for HRM

This guide provides step-by-step instructions for setting up the Hierarchical Reasoning Model (HRM) on Windows using WSL2 (Windows Subsystem for Linux 2).

## Why WSL2?

The HRM project requires CUDA extensions (FlashAttention, adam-atan2) that are challenging to compile on native Windows due to:
- Complex Visual Studio/CUDA toolchain compatibility issues
- Missing Linux-specific compilation tools
- Incompatible build environments

WSL2 provides a full Linux environment with proper CUDA support, making it the recommended approach for Windows users.

## Prerequisites

### System Requirements
- Windows 10 version 2004+ or Windows 11
- NVIDIA GPU with CUDA support (tested with RTX 3050 Ti)
- At least 16GB RAM (recommended)
- 10GB+ free disk space

### Enable WSL2
1. Open PowerShell as Administrator
2. Enable WSL2:
```powershell
wsl --install Ubuntu-22.04
```
3. Restart your computer when prompted
4. Set Ubuntu as default WSL distribution:
```powershell
wsl --set-default Ubuntu-22.04
```

### Install NVIDIA CUDA Drivers
1. Install the latest NVIDIA drivers for Windows from [NVIDIA's website](https://www.nvidia.com/drivers/)
2. **Do not install CUDA toolkit in Windows** - we'll install it in WSL2

## WSL2 Environment Setup

### 1. Update Ubuntu and Install Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential curl wget git python3-full python3-dev python3-pip

# Install NVIDIA CUDA Toolkit via apt (Ubuntu package manager)
sudo apt install -y nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version
nvidia-smi  # Should show your GPU
```

### 2. Verify GPU Access
```bash
nvidia-smi
```
You should see output showing your NVIDIA GPU. If not, restart WSL2:
```powershell
# In Windows PowerShell
wsl --shutdown
wsl
```

### 3. Clone and Setup HRM Project
```bash
# Clone the repository
git clone https://github.com/sapientinc/HRM.git
cd HRM

# Initialize git submodules for datasets
git submodule update --init --recursive
```

### 4. Install Python Dependencies
```bash
# Install PyTorch with CUDA support (Linux version)
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --break-system-packages

# Install project requirements
python3 -m pip install -r requirements.txt --break-system-packages

# Install FlashAttention (this will compile from source)
python3 -m pip install flash-attn --no-build-isolation --break-system-packages
```

**Note**: The `--break-system-packages` flag is needed due to Ubuntu 22.04's externally managed Python environment policy.

### 5. Verify Installation
Create a test script to verify everything works:
```bash
cat > test_setup.py << 'EOF'
import torch
import flash_attn
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name()}')
print(f'✅ FlashAttention version: {flash_attn.__version__}')

# Test FlashAttention
if torch.cuda.is_available():
    from flash_attn import flash_attn_func
    q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
    k = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
    v = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
    out = flash_attn_func(q, k, v)
    print('✅ FlashAttention CUDA test passed!')
EOF

python3 test_setup.py
```

Expected output:
```
✅ PyTorch version: 2.5.1+cu121
✅ CUDA available: True
✅ GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
✅ FlashAttention version: 2.8.2
✅ FlashAttention CUDA test passed!
```

## Running HRM

### From Windows PowerShell
You can run HRM commands from Windows PowerShell using the `wsl` prefix:

```powershell
# Check training options
wsl python3 /mnt/c/Development/HRM/pretrain.py --help

# Start training
wsl python3 /mnt/c/Development/HRM/pretrain.py arch=hrm_v1 data_path=data/arc-aug-1000
```

### From WSL2 Terminal
Or work directly in the WSL2 environment:

```bash
# Enter WSL2
wsl

# Navigate to project
cd /mnt/c/Development/HRM

# Run training
python3 pretrain.py arch=hrm_v1 data_path=data/arc-aug-1000
```

## Performance Notes

- **Memory**: WSL2 can use up to 50% of your system RAM by default
- **GPU Performance**: Expect ~95% of native Linux performance
- **File I/O**: Accessing files on Windows filesystem (`/mnt/c/`) is slower than WSL2 filesystem (`~/`)
- **Compilation**: First-time FlashAttention compilation takes ~10-15 minutes

## Troubleshooting

### Common Issues

#### 1. "CUDA not available" Error
```bash
# Check if nvidia-smi works
nvidia-smi

# If not working, restart WSL2
# In Windows PowerShell:
wsl --shutdown
wsl
```

#### 2. FlashAttention Compilation Fails
```bash
# Install additional build dependencies
sudo apt install -y ninja-build

# Retry FlashAttention installation
python3 -m pip install flash-attn --no-build-isolation --force-reinstall --break-system-packages
```

#### 3. Out of Memory During Compilation
```bash
# Increase WSL2 memory limit in Windows
# Create/edit: %UserProfile%\.wslconfig
[wsl2]
memory=16GB
processors=8
```

#### 4. Permission Errors with Virtual Environments
Use the system Python with `--break-system-packages` instead of virtual environments in WSL2.

### Performance Optimization

1. **Move project to WSL2 filesystem for better I/O**:
```bash
# Copy project to WSL2 home directory
cp -r /mnt/c/Development/HRM ~/HRM-wsl2
cd ~/HRM-wsl2
```

2. **Set WSL2 resource limits** in `%UserProfile%\.wslconfig`:
```ini
[wsl2]
memory=16GB
processors=8
swap=4GB
```

## Success Metrics

After successful setup, you should be able to:
- [x] Access NVIDIA GPU from WSL2 (`nvidia-smi` shows GPU)
- [x] Import PyTorch with CUDA support
- [x] Import and use FlashAttention
- [x] Run HRM training scripts without errors
- [x] See CUDA operations executing on GPU

## Additional Resources

- [WSL2 CUDA Documentation](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [FlashAttention Installation Guide](https://github.com/Dao-AILab/flash-attention)
- [HRM Paper](https://arxiv.org/abs/2506.21734)

---

*This setup was tested and verified on Windows 11 with WSL2 Ubuntu 22.04 and NVIDIA RTX 3050 Ti.*
