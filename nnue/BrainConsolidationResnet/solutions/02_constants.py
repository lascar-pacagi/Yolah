"""Reference solution — exercise 02."""

# ── (carried from exercise 01) ─────────────────────────────────────────────
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

# ── Constants ───────────────────────────────────────────────────────────────
NUM_ACTIONS         = 64 * 64       # 4096 — Yolah from*64 + to flattened
POLICY_IGNORE_INDEX = -1            # CrossEntropyLoss skips rows with this target
IN_CHANNELS         = 4             # black, white, empty, turn planes
