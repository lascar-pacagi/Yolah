"""Reference solution — exercise 01."""

# ── Imports ─────────────────────────────────────────────────────────────────
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

# ── Module-level settings ───────────────────────────────────────────────────
torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

_module_loaded = True
