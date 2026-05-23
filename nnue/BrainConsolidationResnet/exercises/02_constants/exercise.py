# I AM NOT DONE
"""
02_constants — set the three module-level constants the rest of the trainer
will read.
"""

# ── (filled in from exercise 01) ───────────────────────────────────────────
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


# ── TODO: set NUM_ACTIONS, POLICY_IGNORE_INDEX, IN_CHANNELS ────────────────
#
# NUM_ACTIONS           : Yolah action space size, from*64 + to flattened.
# POLICY_IGNORE_INDEX   : sentinel for terminal positions (CE loss skips them).
# IN_CHANNELS           : number of board planes per encoded position.
#
# Replace the three None values below.

NUM_ACTIONS         = None
POLICY_IGNORE_INDEX = None
IN_CHANNELS         = None

# ── End TODO ────────────────────────────────────────────────────────────────
