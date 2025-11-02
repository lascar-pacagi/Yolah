#!/usr/bin/env python3
"""
Preprocessing script to convert game files to memory-mapped format.

Run this ONCE before training:
    python preprocess_memmap.py

This creates:
    - data/dataset_inputs.npy (memory-mapped input features)
    - data/dataset_outputs.npy (memory-mapped target labels)

These files can then be efficiently loaded by all workers without duplication.
"""

from nnue_multigpu3 import preprocess_dataset_to_memmap

if __name__ == "__main__":
    print("=" * 60)
    print("GameDataset Preprocessing for Memory-Mapped Loading")
    print("=" * 60)
    print()

    # Run preprocessing
    preprocess_dataset_to_memmap(
        games_dir="data",
        output_prefix="data/dataset"
    )

    print()
    print("=" * 60)
    print("Preprocessing complete!")
    print("You can now run training with:")
    print("  python nnue_multigpu3.py")
    print("=" * 60)
