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

Space features (YOLAH_SPACE_MASK, step 2): the second layer also reads up to
16 numbers computed from the position (nnue/space_features.h: corrected flood
fill, queen distance, zones reached late, mobility, for the side to move then
the other side), precomputed for the cache by `space_features compute` into
space.u8 (N × 16, integers 0–64). The batch then has 193 + 16 columns; the
bits of `space_mask` select the features, each given as value / 64 (in C++ the
int8 at scale 64 is the integer itself: no quantization error). fc2 sees
[h1(us), h1(them), space]; the space columns start at 0, so a network
initialised from one without them (init_from_persp) gives the same values.

Data. The cache of preprocess_nnue.py (193 inputs: black, white, holes, turn)
and its teacher labels are used as they are: `perspective()` converts a batch
on the GPU. The targets (z and the teacher value) are already for the side to
move.
"""
import torch
from torch import nn

CACHE_INPUT_SIZE = 193          # the cache of preprocess_nnue.py
SPACE_SIZE = 16                 # space.u8, nnue/space_features.h
SPACE_SCALE = 64
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


def space_columns(mask):
    """The space features selected by a mask (bit i = feature i), in order."""
    return [i for i in range(SPACE_SIZE) if mask >> i & 1]


class NetPersp(nn.Module):
    def __init__(self, k=8, them=False, l1_size=1024, l2_size=64, l3_size=32, space_mask=0):
        super().__init__()
        self.k, self.them = k, them
        self.l1_size, self.l2_size, self.l3_size = l1_size, l2_size, l3_size
        self.space_mask = int(space_mask)
        self.n_space = len(space_columns(self.space_mask))
        self.fc1 = nn.Linear(INPUT_SIZE, l1_size)
        self.fc2 = nn.Linear(l1_size * (2 if them else 1) + self.n_space, l2_size * k)
        # One (l3 × l2) matrix and one (l3) value head per phase, initialised like nn.Linear.
        fc3 = [nn.Linear(l2_size, l3_size) for _ in range(k)]
        fc4 = [nn.Linear(l3_size, 1) for _ in range(k)]
        self.w3 = nn.Parameter(torch.stack([m.weight.data for m in fc3]))           # (K, l3, l2)
        self.b3 = nn.Parameter(torch.stack([m.bias.data for m in fc3]))             # (K, l3)
        self.w4 = nn.Parameter(torch.stack([m.weight.data[0] for m in fc4]))        # (K, l3)
        self.b4 = nn.Parameter(torch.stack([m.bias.data[0] for m in fc4]))          # (K,)
        self.register_buffer("mirror", mirror_index(), persistent=False)
        self.register_buffer("space_cols", torch.tensor(space_columns(self.space_mask), dtype=torch.long),
                             persistent=False)
        # saved with the weights: export and load_persp read the mask from it
        self.register_buffer("space_mask_t", torch.tensor(self.space_mask, dtype=torch.long))

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
        us, them, bucket = self.perspective(x[:, :CACHE_INPUT_SIZE])
        h1 = torch.clamp(self.fc1(us), 0.0, 1.0)
        if self.them:
            h1 = torch.cat([h1, torch.clamp(self.fc1(them), 0.0, 1.0)], dim=1)
        if self.n_space:
            space = x[:, CACHE_INPUT_SIZE:CACHE_INPUT_SIZE + SPACE_SIZE][:, self.space_cols] / SPACE_SCALE
            h1 = torch.cat([h1, space.to(h1.dtype)], dim=1)
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


    @torch.no_grad()
    def init_from_persp(self, sd):
        """Start from another NetPersp with the same K / them / sizes and FEWER (or
        no) space features: everything is copied, the fc2 columns of the space
        features it lacks start at 0 — same values as the source network."""
        sd = {(k[7:] if k.startswith("module.") else k): v.float() for k, v in sd.items()}
        k, them, l1, l2, l3, mask = config_of(sd)
        if (k, them, l1, l2, l3) != (self.k, self.them, self.l1_size, self.l2_size, self.l3_size) \
                or mask & ~self.space_mask:
            raise ValueError(f"init_from_persp: source K={k} them={them} mask={mask:#x} does not fit "
                             f"K={self.k} them={self.them} mask={self.space_mask:#x}")
        for name in ("fc1.weight", "fc1.bias", "fc2.bias", "w3", "b3", "w4", "b4"):
            self.get_parameter(name).copy_(sd[name])
        trunk = l1 * (2 if them else 1)
        self.fc2.weight.zero_()
        self.fc2.weight[:, :trunk] = sd["fc2.weight"][:, :trunk]
        src_cols = space_columns(mask)
        for j, c in enumerate(space_columns(self.space_mask)):
            if c in src_cols:
                self.fc2.weight[:, trunk + j] = sd["fc2.weight"][:, trunk + src_cols.index(c)]


def config_of(sd):
    """(k, them, l1, l2, l3, space_mask) of a NetPersp state dict."""
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    k, l3, l2 = sd["w3"].shape
    l1 = sd["fc1.weight"].shape[0]
    mask = int(sd["space_mask_t"]) if "space_mask_t" in sd else 0
    n_space = len(space_columns(mask))
    them = sd["fc2.weight"].shape[1] - n_space == 2 * l1
    return k, them, l1, l2, l3, mask


def load_persp(path, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = {(k[7:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    k, them, l1, l2, l3, mask = config_of(sd)
    net = NetPersp(k, them, l1, l2, l3, mask)
    sd.setdefault("space_mask_t", torch.tensor(mask, dtype=torch.long))   # networks saved before step 2
    net.load_state_dict(sd)
    return net
