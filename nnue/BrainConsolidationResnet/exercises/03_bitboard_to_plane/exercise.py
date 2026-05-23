# I AM NOT DONE
"""
03_bitboard_to_plane — expand a Yolah 64-bit bitboard into an (8, 8) plane.
"""

# ── (filled in from earlier exercises) ─────────────────────────────────────
from tqdm import tqdm
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys
import gc
import json
import random
import threading
import queue as queue_mod
import numpy as np

sys.path.append("../server")
from yolah import Yolah, Move, Square

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

NUM_ACTIONS         = 64 * 64
POLICY_IGNORE_INDEX = -1
IN_CHANNELS         = 4


# ── TODO: implement _bitboard_to_plane ─────────────────────────────────────
#
# Spec:
#   • input  : Python int (interpreted as 64-bit unsigned)
#   • output : np.ndarray shape (8, 8) dtype float32, values in {0.0, 1.0}
#   • orientation: MSB-first. Flat element i holds bit (63 - i) of n,
#     then reshaped to (8, 8). Do NOT change this orientation — it must
#     match what preprocess.py writes to the cache.
#
# See notes.md for diagrams and worked examples.

def _bitboard_to_plane(n: int) -> np.ndarray:
    """64-bit int → (8, 8) float32 plane, MSB-first to match preprocess.py."""
    # TODO: replace this stub
    raise NotImplementedError("Fill me in")

# ── End TODO ────────────────────────────────────────────────────────────────
