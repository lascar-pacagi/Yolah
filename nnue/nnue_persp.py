"""
nnue_persp.py — the NNUE seen from the side to move, with one head per game phase
(step 1 of doc/nnue_roadmap.qmd).

    192 inputs, two points of view ──► 1024 (shared fc1) ──► K heads 64 → 32 → 1

Inputs. The network no longer knows the colours. Its 192 inputs are
    0–63    MY pieces       (those of the owner of the point of view)
    64–127  HIS pieces
    128–191 the holes
Two points of view exist, one per colour:
    black view : mine = black, his = white, squares as they are
    white view : mine = white, his = black, squares mirrored a <-> h
The mirror maps the black starting squares (a1, h8, d5, e4) onto the white ones
(h1, a8, e5, d4), so both colours see the same start and share the weights.
In the cache layout (MSB first: index i is square 63 − i) the mirror of index i
is i ^ 7, as for squares.

The network evaluates the position for the side to move with "us" = the view
of the side to move. With THEM=True it also reads "them" = the other view:
fc2 sees [clamp(fc1(us)), clamp(fc1(them))], always in that order.

Heads. The phase is the number of holes (one per move played, 0..56):
    bucket = holes · K // 57          (0..K−1; the C++ uses the same formula)
fc2 computes the K heads' 64 outputs at once; fc3 and the value head are one
small matrix per head. All heads are computed and the right one is kept.

Data. The cache of preprocess_nnue.py (193 inputs: black, white, holes, turn)
and its teacher labels are used as they are: `perspective()` converts a batch
on the GPU. The targets (z and the teacher value) are already for the side to
move.
"""
import torch
from torch import nn

CACHE_INPUT_SIZE = 193          # the cache of preprocess_nnue.py
INPUT_SIZE = 192                # mine, his, holes
MAX_HOLES = 56
WEIGHT_MAX = 127 / 64           # int8 at scale 64


def mirror_index():
    """Index permutation of the 192 inputs for the a <-> h mirror (i ^ 7 in each block)."""
    sq = torch.arange(64) ^ 7
    return torch.cat([sq, sq + 64, sq + 128])


def bucket_of(holes, k):
    """Head of each position from its number of holes (tensor of ints)."""
    return torch.clamp(holes * k // (MAX_HOLES + 1), max=k - 1)


class NetPersp(nn.Module):
    def __init__(self, k=8, them=False, l1_size=1024, l2_size=64, l3_size=32):
        super().__init__()
        self.k, self.them = k, them
        self.l1_size, self.l2_size, self.l3_size = l1_size, l2_size, l3_size
        self.fc1 = nn.Linear(INPUT_SIZE, l1_size)
        self.fc2 = nn.Linear(l1_size * (2 if them else 1), l2_size * k)
        # One (l3 × l2) matrix and one (l3) value head per phase, initialised like nn.Linear.
        fc3 = [nn.Linear(l2_size, l3_size) for _ in range(k)]
        fc4 = [nn.Linear(l3_size, 1) for _ in range(k)]
        self.w3 = nn.Parameter(torch.stack([m.weight.data for m in fc3]))           # (K, l3, l2)
        self.b3 = nn.Parameter(torch.stack([m.bias.data for m in fc3]))             # (K, l3)
        self.w4 = nn.Parameter(torch.stack([m.weight.data[0] for m in fc4]))        # (K, l3)
        self.b4 = nn.Parameter(torch.stack([m.bias.data[0] for m in fc4]))          # (K,)
        self.register_buffer("mirror", mirror_index(), persistent=False)

    def perspective(self, x):
        """Cache batch (bs, 193) → (us, them, bucket)."""
        black, white, holes = x[:, 0:64], x[:, 64:128], x[:, 128:192]
        white_to_move = x[:, 192:193] > 0.5
        black_view = torch.cat([black, white, holes], dim=1)
        white_view = torch.cat([white, black, holes], dim=1)[:, self.mirror]
        us = torch.where(white_to_move, white_view, black_view)
        them = torch.where(white_to_move, black_view, white_view)
        bucket = bucket_of(holes.sum(dim=1).round().long(), self.k)
        return us, them, bucket

    def forward(self, x):
        us, them, bucket = self.perspective(x)
        h1 = torch.clamp(self.fc1(us), 0.0, 1.0)
        if self.them:
            h1 = torch.cat([h1, torch.clamp(self.fc1(them), 0.0, 1.0)], dim=1)
        h2 = torch.clamp(self.fc2(h1), 0.0, 1.0).view(-1, self.k, self.l2_size)          # (bs, K, l2)
        h3 = torch.clamp(torch.einsum('bki,koi->bko', h2, self.w3) + self.b3, 0.0, 1.0)  # (bs, K, l3)
        v = (h3 * self.w4).sum(dim=2) + self.b4                                         # (bs, K)
        v = v.gather(1, bucket.unsqueeze(1)).squeeze(1)
        return torch.tanh(v)

    def clip(self):
        # The int8 trunk (fc1, fc2, w3) stays quantizable; the value heads are free (float in C++).
        for p in (self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias, self.w3, self.b3):
            p.data.clamp_(-WEIGHT_MAX, WEIGHT_MAX)

    @torch.no_grad()
    def init_from_nnue193(self, sd):
        """Start from a 193 → 1024 → 64 → 32 → 1 network (nnue193x1024x64x32x1.py).

        fc1 keeps the black / white / holes rows (black = mine) and drops the
        turn row; every head gets a copy of fc2–fc4, and the "them" half of fc2
        starts at 0. The new network then gives exactly the old one's values on
        the positions with black to move."""
        sd = {(k[7:] if k.startswith("module.") else k): v.float() for k, v in sd.items()}
        if sd["fc1.weight"].shape != (self.l1_size, CACHE_INPUT_SIZE):
            raise ValueError(f"fc1 {tuple(sd['fc1.weight'].shape)}: not a 193 → {self.l1_size} network")
        self.fc1.weight.copy_(sd["fc1.weight"][:, :INPUT_SIZE])
        self.fc1.bias.copy_(sd["fc1.bias"])
        self.fc2.weight.zero_()
        for k in range(self.k):
            rows = slice(k * self.l2_size, (k + 1) * self.l2_size)
            self.fc2.weight[rows, :self.l1_size] = sd["fc2.weight"]
            self.fc2.bias[rows] = sd["fc2.bias"]
            self.w3[k] = sd["fc3.weight"]
            self.b3[k] = sd["fc3.bias"]
            self.w4[k] = sd["fc4.weight"][0]
            self.b4[k] = sd["fc4.bias"][0]


def config_of(sd):
    """(k, them, l1, l2, l3) of a NetPersp state dict."""
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    k, l3, l2 = sd["w3"].shape
    l1 = sd["fc1.weight"].shape[0]
    them = sd["fc2.weight"].shape[1] == 2 * l1
    return k, them, l1, l2, l3


def load_persp(path, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = {(k[7:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    k, them, l1, l2, l3 = config_of(sd)
    net = NetPersp(k, them, l1, l2, l3)
    net.load_state_dict(sd)
    return net
