# Memory-Mapped Dataset Usage Guide

## Overview

Your training script has been updated to use memory-mapped (memmap) files instead of loading the entire dataset into RAM. This provides several benefits:

- **No RAM duplication**: All worker processes access the same disk-backed data
- **Efficient multi-worker loading**: Each worker reads only what it needs
- **Faster startup**: Dataset is preprocessed once, then reused
- **Scalability**: Works with datasets larger than available RAM

## How It Works

### Step 1: Preprocess Dataset (One-time)

Before training, convert your game files to memory-mapped format:

```bash
python preprocess_memmap.py
```

This creates:
- `data/dataset_inputs.npy` - Input features (one-hot encoded board states)
- `data/dataset_outputs.npy` - Target labels (game outcomes)

**Note**: This step only needs to run once. The files persist and are reused for all future training runs.

### Step 2: Train

Run training as normal. The script will automatically use the memory-mapped files:

```bash
python nnue_multigpu3.py
```

The training script will:
1. Check if memory-mapped files exist
2. If not found, automatically run preprocessing
3. Load the dataset from the pre-computed files
4. Distribute data across workers efficiently

## Key Changes to Your Code

### New Classes

#### `preprocess_dataset_to_memmap(games_dir, output_prefix)`
Converts raw game files to memory-mapped numpy arrays. Run once, output reusable.

#### `GameDatasetMemmap(data_prefix)`
Loads data from pre-computed memory-mapped files. Replaces `GameDataset` for training.

### Updated Main Script

The main script now:
1. Checks for existing memory-mapped files
2. Runs preprocessing if needed
3. Loads dataset efficiently
4. Passes to distributed training workers

## Performance Benefits

### Before (Original Implementation)
```
Main Process:        Load GameDataset (full RAM)
↓ (fork)
Worker 1:            Copy of full dataset in RAM
Worker 2:            Copy of full dataset in RAM
Worker 3:            Copy of full dataset in RAM
Worker 4:            Copy of full dataset in RAM
───────────────────────────────────────────────
Total RAM: dataset_size × (1 + num_workers)
```

### After (Memory-Mapped)
```
Disk File:           dataset_inputs.npy, dataset_outputs.npy
↓ (memory-mapped)
Main Process:        Header info only (< 1 MB)
↓ (fork)
Worker 1:            Memory-maps same disk file (no copy)
Worker 2:            Memory-maps same disk file (no copy)
Worker 3:            Memory-maps same disk file (no copy)
Worker 4:            Memory-maps same disk file (no copy)
───────────────────────────────────────────────
Total RAM: minimal overhead × num_workers (OS page cache)
```

## Troubleshooting

### Error: "Memory-mapped files not found"

Run preprocessing first:
```bash
python preprocess_memmap.py
```

### Want to regenerate memmap files?

Delete the existing files and rerun preprocessing:
```bash
rm data/dataset_inputs.npy data/dataset_outputs.npy
python preprocess_memmap.py
```

### Memory usage still high?

Check that:
1. You're using `GameDatasetMemmap` not `GameDataset`
2. The preprocessing completed successfully
3. DataLoader has appropriate `num_workers` (currently 4)

### Slow preprocessing?

The preprocessing is I/O bound. To speed it up:
- Increase `max_workers` in `GameDataset.__init__()`
- Use an SSD instead of HDD for game files
- Run on a fast disk (not network mounted)

## Advanced: Manual Preprocessing

If needed, you can call preprocessing directly from Python:

```python
from nnue_multigpu3 import preprocess_dataset_to_memmap

# Preprocess from your game files
preprocess_dataset_to_memmap("data", "data/dataset")

# Now you can train
python nnue_multigpu3.py
```

## File Sizes Reference

With `INPUT_SIZE = 193` (64+64+64+1):
- Each input: 193 floats × 4 bytes = 772 bytes
- Each output: 1 int64 × 8 bytes = 8 bytes
- Per sample total: ~780 bytes

Example for different dataset sizes:
- 100k samples: ~78 MB
- 1M samples: ~780 MB
- 10M samples: ~7.8 GB

## Reverting to Original Implementation

If you want to go back to loading the full dataset in memory:

```python
# In main section, replace:
# dataset = GameDatasetMemmap(DATA_PREFIX)
# with:
dataset = GameDataset("data")
```

However, this will cause RAM duplication across workers.
