# AlphaZero Quick Start Guide

This guide will help you get started with training and using the AlphaZero AI for Yolah.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA compute capability 7.5+ (RTX 2000 series or newer)
- 8GB+ GPU memory (16GB+ recommended for larger batch sizes)
- 16GB+ system RAM
- Multi-core CPU (8+ cores recommended for self-play)

### Software Requirements
- Ubuntu 20.04+ or similar Linux distribution
- CUDA Toolkit 11.0+ ([Download](https://developer.nvidia.com/cuda-downloads))
- cuDNN 8.0+ ([Download](https://developer.nvidia.com/cudnn))
- CMake 3.20+
- GCC 11+ with C++23 support
- Python 3.8+ with PyTorch 2.0+

## Installation

### 1. Install CUDA and cuDNN

```bash
# Install CUDA Toolkit (example for CUDA 12.0)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify installation
nvcc --version
nvidia-smi
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv alphazero_env
source alphazero_env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy tqdm
```

### 3. Build the Project

```bash
cd /path/to/Yolah

# Create build directory
mkdir build && cd build

# Configure with CUDA support
cmake .. -DENABLE_CUDA=ON

# Build AlphaZero components
make alphazero_selfplay -j$(nproc)
make Yolah -j$(nproc)

# Verify build
ls ../alphazero/
```

## Quick Training Pipeline

### Step 1: Generate Initial Self-Play Data (30 minutes)

```bash
cd /path/to/Yolah

# Generate 100 games with random policy (no weights needed)
./build/alphazero_selfplay \
    --games 100 \
    --threads 8 \
    --simulations 200

# Output: selfplay_data/training_data_*.bin
```

### Step 2: Train Initial Model (1 hour)

```bash
cd alphazero

# Activate Python environment
source ../alphazero_env/bin/activate

# Train for 100 epochs
python trainer.py \
    --data-file ../selfplay_data/training_data_*.bin \
    --epochs 100 \
    --batch-size 128 \
    --checkpoint-dir checkpoints \
    --save-every 20

# Monitor with: nvidia-smi -l 1
```

### Step 3: Export Model for C++ Inference

```bash
# Still in alphazero directory
python export_model.py \
    --checkpoint checkpoints/model_epoch_100.pt \
    --output weights/alphazero_weights.bin \
    --info

# Verify export
ls -lh weights/alphazero_weights.bin
```

### Step 4: Test the Model

```bash
cd ..

# Play against AlphaZero
./build/Yolah \
    --player1 config/human_player.cfg \
    --player2 config/alphazero_player.cfg \
    --nb-games 1

# Evaluate against existing AI
./build/Yolah \
    --player1 config/alphazero_player.cfg \
    --player2 config/mm_nnue_quantized_player.cfg \
    --nb-games 10
```

## Iterative Training Loop

After initial training, iterate to improve:

### Iteration Script

```bash
#!/bin/bash
# train_iteration.sh

ITER=$1  # Iteration number

echo "=== AlphaZero Training Iteration $ITER ==="

# 1. Self-play with current best model (4 hours for 1000 games)
echo "Step 1/4: Generating self-play data..."
./build/alphazero_selfplay \
    --games 1000 \
    --threads 8 \
    --simulations 400 \
    --weights alphazero/weights/best_model.bin

# 2. Train on new data (2 hours)
echo "Step 2/4: Training network..."
cd alphazero
source ../alphazero_env/bin/activate

python trainer.py \
    --data-file ../selfplay_data/training_data_*.bin \
    --epochs 50 \
    --batch-size 256 \
    --checkpoint-dir checkpoints \
    --checkpoint checkpoints/best_checkpoint.pt \
    --save-every 10

# 3. Export new model
echo "Step 3/4: Exporting model..."
python export_model.py \
    --checkpoint checkpoints/model_epoch_50.pt \
    --output weights/model_iter_${ITER}.bin

cd ..

# 4. Evaluate new vs best (30 minutes)
echo "Step 4/4: Evaluating new model..."
# Update config files to use new/old models
# Play 100 games between them
# If new wins >55%, promote to best

echo "Iteration $ITER complete!"
```

Run iterations:
```bash
chmod +x train_iteration.sh
./train_iteration.sh 1
./train_iteration.sh 2
# ... continue until convergence
```

## Configuration Tuning

### For Fast Training (Weak AI)
```json
{
    "type": "alphazero",
    "network_weights": "weights/alphazero_weights.bin",
    "num_simulations": 200,
    "num_parallel_games": 4,
    "c_puct": 1.5,
    "temperature": 0.0
}
```

### For Strong Play (Slow)
```json
{
    "type": "alphazero",
    "network_weights": "weights/alphazero_weights.bin",
    "num_simulations": 1600,
    "num_parallel_games": 8,
    "c_puct": 2.0,
    "temperature": 0.0
}
```

### For Tournament Play (Balanced)
```json
{
    "type": "alphazero",
    "network_weights": "weights/alphazero_weights.bin",
    "num_simulations": 800,
    "num_parallel_games": 8,
    "c_puct": 1.5,
    "temperature": 0.0
}
```

## Troubleshooting

### CUDA Out of Memory Error

**Symptom**: Training crashes with "CUDA out of memory"

**Solution**:
```bash
# Reduce batch size
python trainer.py --batch-size 64 ...  # instead of 128

# Or reduce network size
python trainer.py --num-blocks 5 --num-filters 128 ...
```

### Slow Self-Play Generation

**Symptom**: Less than 10 games/hour

**Solutions**:
- Reduce simulations: `--simulations 200` (instead of 400)
- Increase threads: `--threads 16` (match CPU cores)
- Use faster GPU
- Reduce network size during export

### Model Not Improving

**Symptom**: Win rate not increasing after multiple iterations

**Solutions**:
1. Generate more self-play data (2000+ games)
2. Increase network capacity (more blocks/filters)
3. Train for more epochs (200+)
4. Adjust learning rate: `--learning-rate 0.0001`
5. Check for bugs in MCTS implementation

### Build Errors

**CMake can't find CUDA**:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
cmake .. -DENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

**Missing cuDNN**:
- Download from NVIDIA website
- Extract to `/usr/local/cuda/`
- Verify: `ls /usr/local/cuda/lib64/libcudnn*`

## Performance Benchmarks

Expected performance on RTX 3090:

| Task | Speed | Time for Milestone |
|------|-------|-------------------|
| Self-play (200 sims) | ~60 games/hour | 100 games: 1.7 hours |
| Self-play (400 sims) | ~30 games/hour | 1000 games: 33 hours |
| Training | ~5000 examples/sec | 100 epochs on 50K examples: 1 hour |
| Inference (single) | ~100 pos/sec | 1 game (50 moves): 0.5 seconds |
| Inference (batch) | ~10000 pos/sec | N/A |

## Next Steps

1. **Train for more iterations**: 10-20 iterations minimum
2. **Increase self-play games**: 5000+ games per iteration
3. **Implement evaluation system**: Track ELO rating over time
4. **Add data augmentation**: Use board symmetries
5. **Tune hyperparameters**: Grid search for optimal settings
6. **Distributed training**: Use multiple GPUs/machines

## Resources

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [Full Documentation](alphazero/README.md)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Getting Help

If you encounter issues:

1. Check the detailed [README](alphazero/README.md)
2. Verify CUDA installation: `nvidia-smi` and `nvcc --version`
3. Check GPU memory usage: `nvidia-smi -l 1`
4. Review error logs in detail
5. Open an issue with full error output

Happy training! 🚀
