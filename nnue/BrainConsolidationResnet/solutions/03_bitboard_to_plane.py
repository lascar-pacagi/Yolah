"""Reference solution — exercise 03."""

# ── (carried from earlier exercises) ───────────────────────────────────────
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


def _bitboard_to_plane(n: int) -> np.ndarray:
    """
    Expand a 64-bit Yolah bitboard into an 8×8 float32 plane.

    MSB-first to match preprocess.py's np.unpackbits on a big-endian view:
        flat[i] receives bit (63 - i) of n.

    A vectorised alternative is:
        np.unpackbits(np.array([n], dtype='>u8').view(np.uint8))
          .astype(np.float32).reshape(8, 8)
    Both produce identical output; the loop is kept for readability since
    this is the inference-time encoder and not on the training hot path.
    """
    b = np.zeros(64, dtype=np.float32)
    for i in range(64):
        if n & (1 << (63 - i)):
            b[i] = 1.0
    return b.reshape(8, 8)
