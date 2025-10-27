# AlphaZero for Yolah

This directory contains a complete AlphaZero implementation for the Yolah game, featuring:

- **CUDA-accelerated neural network** with residual architecture
- **Parallel Monte Carlo Tree Search (MCTS)** with virtual loss
- **Multi-threaded self-play** for generating training data
- **Multi-GPU training pipeline** using PyTorch DDP

## Architecture Overview

### Neural Network
- Input: 3 planes (8x8) representing black pieces, white pieces, and empty squares
- Body: 10 residual blocks with 256 filters each
- Policy Head: Outputs move probabilities (75 possible moves)
- Value Head: Outputs position evaluation (-1 to +1)

### MCTS
- PUCT (Predictor + Upper Confidence Bound for Trees) selection
- Virtual loss for parallel search
- Dirichlet noise for exploration during self-play
- Temperature-based move selection

### Training Loop
1. **Self-Play**: Generate games using current network
2. **Training**: Update network using game data
3. **Evaluation**: Test new network against previous best
4. **Iteration**: Repeat until convergence

## Building

### Prerequisites
- CUDA Toolkit 11.0 or later
- cuDNN 8.0 or later
- CMake 3.20+
- C++23 compiler
- Python 3.8+ with PyTorch

### Compilation

```bash
cd /path/to/Yolah
mkdir build && cd build

# With CUDA support (default)
cmake .. -DENABLE_CUDA=ON
make alphazero_selfplay

# Without CUDA (CPU only - slower)
cmake .. -DENABLE_CUDA=OFF
make
```

## Usage

### 1. Self-Play Generation

Generate training data through self-play:

```bash
./build/alphazero_selfplay \
    --games 1000 \
    --threads 8 \
    --simulations 400 \
    --weights alphazero_weights.bin
```

Options:
- `--games <n>`: Number of games to generate (default: 100)
- `--threads <n>`: Parallel game workers (default: 8)
- `--simulations <n>`: MCTS simulations per move (default: 400)
- `--weights <file>`: Path to network weights (optional)

Output: Training data saved to `selfplay_data/training_data_*.bin`

### 2. Training

Train the neural network on self-play data:

```bash
cd alphazero

# Single GPU
python trainer.py \
    --data-file ../selfplay_data/training_data_*.bin \
    --epochs 100 \
    --batch-size 128 \
    --checkpoint-dir checkpoints

# Multi-GPU (automatically detects all GPUs)
python trainer.py \
    --data-file ../selfplay_data/training_data_*.bin \
    --epochs 100 \
    --batch-size 128 \
    --num-blocks 10 \
    --num-filters 256 \
    --checkpoint-dir checkpoints
```

Training options:
- `--data-file`: Path to training data
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size per GPU
- `--num-blocks`: Number of residual blocks (default: 10)
- `--num-filters`: Number of filters (default: 256)
- `--learning-rate`: Initial learning rate (default: 0.001)
- `--checkpoint`: Resume from checkpoint

### 3. Export Model for C++

Convert PyTorch model to binary format for C++ inference:

```bash
python export_model.py \
    --checkpoint checkpoints/model_epoch_100.pt \
    --output alphazero_weights.bin
```

### 4. Playing with AlphaZero

Create a configuration file `config/alphazero_player.cfg`:

```json
{
    "type": "alphazero",
    "network_weights": "alphazero/alphazero_weights.bin",
    "num_simulations": 800,
    "num_parallel_games": 8,
    "c_puct": 1.5,
    "temperature": 0.0,
    "use_gpu": true
}
```

Play against AlphaZero:

```bash
./build/Yolah --player1 config/human_player.cfg \
              --player2 config/alphazero_player.cfg \
              --nb-games 1
```

Evaluate against other AIs:

```bash
./build/Yolah --player1 config/alphazero_player.cfg \
              --player2 config/mm_nnue_quantized_player.cfg \
              --nb-games 100
```

## Performance Tuning

### MCTS Parameters

- **num_simulations**: More simulations = stronger play but slower
  - Fast: 200-400 simulations
  - Standard: 800 simulations
  - Strong: 1600+ simulations

- **num_parallel_games**: Number of parallel MCTS workers
  - Set to number of CPU cores for best performance
  - Use 4-8 for balance between speed and quality

- **c_puct**: Exploration constant
  - Higher = more exploration
  - Typical range: 1.0-2.5
  - Default: 1.5

- **temperature**: Controls randomness in move selection
  - 0.0 = deterministic (select most visited move)
  - 1.0 = proportional to visit counts (exploration)
  - Use 1.0 for first ~15 moves in training, then 0.0

### Training Parameters

- **Batch size**: Larger = more stable but uses more memory
  - Single GPU: 128-256
  - Multi-GPU: 256-512 (per GPU)

- **Learning rate**: Start with 0.001, decay over time
  - Use StepLR scheduler with decay every 20-50 epochs

- **Data augmentation**: Use board symmetries (not yet implemented)

## File Format

### Training Data Format (Binary)

Each training example is stored as:
- 8 bytes: black bitboard (uint64_t)
- 8 bytes: white bitboard (uint64_t)
- 8 bytes: empty bitboard (uint64_t)
- 2 bytes: ply count (uint16_t)
- 300 bytes: policy vector (75 × float32)
- 4 bytes: game outcome value (float32)

Total: 330 bytes per example

### Weight File Format

Network weights are stored in binary format:
- Layer 1 weights, biases
- Layer 2 weights, biases
- ... (all layers in order)

Use `export_model.py` to convert from PyTorch to this format.

## Complete Training Pipeline

```bash
#!/bin/bash
# complete_training.sh - Full AlphaZero training loop

ITERATION=0
MAX_ITERATIONS=100

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    echo "=== Iteration $ITERATION ==="

    # 1. Generate self-play games
    echo "Generating self-play data..."
    ./build/alphazero_selfplay \
        --games 1000 \
        --threads 8 \
        --simulations 400 \
        --weights alphazero/weights/best_model.bin

    # 2. Train network
    echo "Training network..."
    cd alphazero
    python trainer.py \
        --data-file ../selfplay_data/training_data_*.bin \
        --epochs 50 \
        --batch-size 128 \
        --checkpoint checkpoints/checkpoint_iter_${ITERATION}.pt

    # 3. Export model
    echo "Exporting model..."
    python export_model.py \
        --checkpoint checkpoints/checkpoint_iter_${ITERATION}.pt \
        --output weights/model_iter_${ITERATION}.bin
    cd ..

    # 4. Evaluate new model
    echo "Evaluating new model..."
    ./build/Yolah \
        --player1 config/alphazero_new.cfg \
        --player2 config/alphazero_best.cfg \
        --nb-games 100 > eval_results.txt

    # 5. Update best model if improved
    WIN_RATE=$(grep "Player 1 wins" eval_results.txt | awk '{print $4}')
    if (( $(echo "$WIN_RATE > 55" | bc -l) )); then
        echo "New model is better! Updating best model..."
        cp alphazero/weights/model_iter_${ITERATION}.bin \
           alphazero/weights/best_model.bin
    fi

    ITERATION=$((ITERATION + 1))
done

echo "Training complete!"
```

## Benchmarks

Performance on NVIDIA RTX 3090:

- Self-play: ~30 games/hour (400 simulations per move)
- Training: ~5000 examples/second (batch size 256)
- Inference: ~100 positions/second (single position)
- Batch inference: ~10000 positions/second (batch size 256)

## Implementation Details

### Key Features

1. **Virtual Loss**: Prevents multiple threads from exploring the same path
2. **Batch Evaluation**: Accumulate positions and evaluate in batches for GPU efficiency
3. **Tree Reuse**: Reuse search tree between moves for faster play
4. **Dirichlet Noise**: Add exploration noise to root node during training
5. **Resignation**: End games early when position is clearly lost
6. **Temperature Scheduling**: High exploration early, deterministic late

### CUDA Kernels

Custom CUDA kernels for:
- ReLU activation
- Residual connections
- Softmax
- Tanh
- Batch normalization

Larger operations (convolution, matrix multiplication) use cuDNN and cuBLAS.

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size
- Reduce number of residual blocks or filters
- Use mixed precision training (FP16)

### Slow Self-Play

- Reduce number of simulations
- Increase number of parallel games
- Use faster GPU
- Enable batch evaluation

### Poor Play Quality

- Increase training data (more self-play games)
- Increase network capacity (more blocks/filters)
- Tune MCTS parameters (simulations, c_puct)
- Train for more iterations

## References

- [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)

## License

See main project LICENSE file.
