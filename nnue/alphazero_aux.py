"""
alphazero_aux.py — KataGo's auxiliary heads for the Yolah ResNet (optional:
used by alphazero_learn.py only with --aux; without it nothing here runs).

Why (see the doc, "KataGo's auxiliary heads"): the value head learns from one
noisy result per game. Two extra heads, which exist ONLY during training, ask
the shared trunk for richer predictions, and their gradient shapes the trunk:

  • ownership — for each square not yet a hole, who will LEAVE it before the
    game ends: the side to move, the opponent, or nobody. In Yolah every move
    leaves a hole on the square it comes from and scores one point, so the
    map is the rest of the score, square by square.
  • score margin — the distribution of the rest of the score,
    d = (my future moves) − (the opponent's future moves) ∈ [−56, 56],
    which is exactly the sum of the ownership map (so it needs no extra data).

Both heads read the trunk's output (B, C, 8, 8), next to the value and policy
heads. At export they are dropped: base_state_dict() keeps only the Net
parameters, so the .pt/.ts written are the plain two-headed network and the
C++ backends are unchanged.
"""
import torch
from torch import nn
import torch.nn.functional as F

from cnn_resnet_value_policy_chunked import Net

# ── targets: the future ownership map of a row (TrainingSampleAux::own) ──────
OWN_NONE, OWN_MINE, OWN_OPP, OWN_PAST = 0, 1, 2, 3   # as in alphazero_selfplay.h
OWN_BYTES = 16                                       # 64 squares × 2 bits
MAX_MARGIN = 56                                      # 64 squares − 8 pieces
SCORE_BINS = 2 * MAX_MARGIN + 1                      # d ∈ [−56, 56]

# Parameters of the auxiliary heads: their names start with these prefixes.
AUX_PREFIXES = ("own_", "score_")


def decode_ownership(own_bytes):
    """
    (B, 16) uint8 tensor, 2 bits per square (square q in byte q//4, bits
    2·(q%4)) → (B, 64) long tensor of OWN_* codes in PLANE CELL order, i.e.
    aligned with the network input: cell c holds square 63 − c (see
    nn_evaluator.h), hence the flip. The same symmetry gather as the input
    planes then applies to it.
    """
    b = own_bytes.long().unsqueeze(2)                         # (B, 16, 1)
    shifts = torch.arange(0, 8, 2, device=b.device)           # 0, 2, 4, 6
    by_square = ((b >> shifts) & 3).reshape(b.shape[0], 64)   # square q = 4·byte + k
    return by_square.flip(1)                                  # → cell order


def margin_from_ownership(own):
    """The rest of the score for the side to move: #mine − #opponent's."""
    return (own == OWN_MINE).sum(1) - (own == OWN_OPP).sum(1)


# ── the network ─────────────────────────────────────────────────────────────
class AuxNet(Net):
    """
    Net (same trunk, same value and policy heads, same parameter names) plus:

      own_conv   : 1×1 conv C → 3, one softmax per square over
                   (nobody, the side to move, the opponent) — a per-square
                   classification, exactly as KataGo's ownership head;
      score_fc1/2: global pooling of the trunk (mean and max over the 64
                   squares, 2C numbers) → FC 256 → ReLU → FC 113 logits, one
                   per possible margin. Pooling rather than flattening: the
                   margin is a property of the whole board.

    forward() returns (value, policy) like Net, so an AuxNet can be used
    anywhere a Net is; forward_all() adds the two auxiliary outputs.
    """
    def __init__(self, channels=256, nb_blocks=30, value_fc_size=256, num_actions=64 * 64):
        super().__init__(channels=channels, nb_blocks=nb_blocks,
                         value_fc_size=value_fc_size, num_actions=num_actions)
        self.own_conv = nn.Conv2d(channels, 3, kernel_size=1)
        self.score_fc1 = nn.Linear(2 * channels, 256)
        self.score_fc2 = nn.Linear(256, SCORE_BINS)

    def trunk(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        return torch.relu(self.output_bn(x))                  # (B, C, 8, 8)

    def heads(self, h):
        """The two heads of Net, verbatim (see Net.forward)."""
        v = torch.relu(self.value_bn(self.value_conv(h))).flatten(1)
        v = torch.tanh(self.value_fc2(torch.relu(self.value_fc1(v)))).squeeze(-1)
        p = torch.relu(self.policy_bn(self.policy_conv(h))).flatten(1)
        return v, self.policy_fc(p)

    def forward(self, x):
        return self.heads(self.trunk(x))

    def forward_all(self, x):
        h = self.trunk(x)
        v, p = self.heads(h)
        own = self.own_conv(h).flatten(2)                      # (B, 3, 64), cell order
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        score = self.score_fc2(torch.relu(self.score_fc1(pooled)))   # (B, 113)
        return v, p, own, score


def is_aux_key(k):
    return k.startswith(AUX_PREFIXES)


def base_state_dict(sd):
    """The Net part of a (possibly AuxNet) state_dict: what gets exported."""
    return {k: v for k, v in sd.items() if not is_aux_key(k)}


def make_aux_net(sd):
    """
    An AuxNet sized from a state_dict. A plain Net state_dict (the initial
    network, or a run that did not use --aux) leaves the auxiliary heads at
    their random initialisation — the only keys allowed to be missing.
    """
    C = sd["input_conv.0.weight"].shape[0]
    B = sum(1 for k in sd if k.endswith(".conv1.weight"))
    net = AuxNet(channels=C, nb_blocks=B, value_fc_size=sd["value_fc1.weight"].shape[0],
                 num_actions=sd["policy_fc.weight"].shape[0])
    missing, unexpected = net.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not is_aux_key(k)] + list(unexpected)
    if bad:
        raise KeyError(f"state_dict does not fit AuxNet: {bad[:5]}")
    return net, bool(missing)          # True = fresh auxiliary heads


# ── the losses ──────────────────────────────────────────────────────────────
def aux_losses(own_logits, score_logits, own):
    """
    own_logits (B, 3, 64), score_logits (B, 113), own (B, 64) OWN_* codes
    (already transformed by the batch's symmetry).

    Ownership: cross-entropy per square, averaged over the 64 squares (a hole
    — OWN_PAST — contributes 0), i.e. KataGo's (1/b²)·Σ_points. Rows without
    auxiliary data (version-1 rows, all OWN_PAST) contribute nothing.

    Score: cross-entropy of the predicted distribution against the true
    margin d (the "pdf" term) plus the squared distance between the
    predicted and the true cumulative distributions (the "cdf" term, which
    makes predicting +11 for a true +12 cheaper than predicting −30).

    Returns (own_loss, score_pdf_loss, score_cdf_loss, own_accuracy).
    """
    has_aux = (own != OWN_PAST).any(1)                         # (B,)
    n = has_aux.sum()
    if n.item() == 0:
        zero = own_logits.sum() * 0.0
        return zero, zero, zero, float("nan")
    own_logits, score_logits, own = own_logits[has_aux], score_logits[has_aux], own[has_aux]

    # ownership: classes (nobody, mine, opponent's) = codes 0, 1, 2; code 3 ignored
    ce = F.cross_entropy(own_logits.float(), own, ignore_index=OWN_PAST, reduction="none")  # (n, 64)
    own_loss = ce.sum(1).mean() / 64.0
    valid = own != OWN_PAST
    acc = ((own_logits.argmax(1) == own) & valid).sum().item() / max(1, valid.sum().item())

    # score margin
    target = (margin_from_ownership(own).clamp(-MAX_MARGIN, MAX_MARGIN) + MAX_MARGIN)   # (n,)
    logp = F.log_softmax(score_logits.float(), dim=1)
    pdf_loss = F.nll_loss(logp, target)
    cdf_pred = logp.exp().cumsum(1)
    cdf_true = (torch.arange(SCORE_BINS, device=target.device)[None, :] >= target[:, None]).float()
    cdf_loss = ((cdf_pred - cdf_true) ** 2).sum(1).mean()
    return own_loss, pdf_loss, cdf_loss, acc
