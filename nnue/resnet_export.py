"""
resnet_export.py — export the trained two-headed ResNet (cnn_resnet_256x30_
value_policy.pt) for C++ inference by the AlphaZero-style MCTS player.

Three artefacts are produced next to the checkpoint:

  1. <name>.bin  — a flat little-endian float32 dump with every BatchNorm
                   folded into the adjacent conv / affine op. This is what the
                   pure-C++ CPU backend (nnue/resnet.h) loads. No PyTorch needed
                   at run time.
  2. <name>.ts   — a TorchScript module (traced in eval mode) for the optional
                   libtorch backend (nnue/resnet_torch.h, built with
                   -DENABLE_TORCH=ON). This one runs on the GPU.
  3. <name>.test.bin — a handful of random positions with the exact PyTorch
                   outputs (value + 4096 policy logits) so the C++ backends can
                   self-check bit-for-bit-ish (see test/alphazero_mcts_check.cpp).

Usage (from the nnue/ directory, the module imports ../server/yolah.py):

    python3 resnet_export.py [checkpoint.pt] [--no-ts] [--nb-test 32]

Binary layout of <name>.bin (all little-endian; float32 unless stated)
──────────────────────────────────────────────────────────────────────
    char[8]  magic        = "YOLAHRSN"
    uint32   version      = 1
    uint32   in_channels  = 4
    uint32   channels     = C   (256)
    uint32   nb_blocks    = B   (30)
    uint32   value_fc     = F   (256)
    uint32   num_actions  = A   (4096)
    ── trunk ──
    stem_w   [C][9][4]      conv weight, BN folded, layout (out, tap, in) with
                            tap = dr*3 + dc, dr/dc ∈ {0,1,2} (kernel row/col)
    stem_b   [C]            folded bias
    for each block b in 0..B-1:
        bn1_scale [C], bn1_shift [C]      pre-activation affine (y = s*x + t)
        conv1_w   [C][9][C]               (out, tap, in), no bias
        bn2_scale [C], bn2_shift [C]
        conv2_w   [C][9][C]
    out_bn_scale [C], out_bn_shift [C]
    ── value head ──
    value_conv_w [C]        1x1 conv (C→1) with value_bn folded
    value_conv_b [1]
    value_fc1_w  [F][64], value_fc1_b [F]
    value_fc2_w  [1][F],  value_fc2_b [1]
    ── policy head ──
    policy_conv_w [2][C]    1x1 conv (C→2) with policy_bn folded
    policy_conv_b [2]
    policy_fc_w   [A][128], policy_fc_b [A]

Why fold BatchNorm: in eval mode BN is a per-channel affine map
y = γ(x-μ)/√(σ²+ε) + β = s·x + t. Where BN directly follows a conv (stem,
value_conv, policy_conv) it folds into the conv weights (W·s) and a bias (t).
Where BN precedes a ReLU+conv (the pre-activation residual blocks and the
trunk's output_bn) it cannot be merged into the conv, so we keep (s, t) and the
C++ code applies max(0, s·x+t) while it gathers the im2col patches — no extra
pass over the activations.
"""
import os
import sys
import struct
import argparse
import numpy as np
import torch

sys.path.append("../server")
from yolah import Yolah, Move, Square                     # noqa: E402
from cnn_resnet_value_policy_chunked import Net, encode_cnn  # noqa: E402

MAGIC   = b"YOLAHRSN"
VERSION = 1
BN_EPS  = 1e-5   # nn.BatchNorm2d default


def bn_affine(sd, prefix):
    """Return (scale, shift) float32 arrays for the eval-mode BatchNorm `prefix`."""
    gamma = sd[f"{prefix}.weight"].double()
    beta  = sd[f"{prefix}.bias"].double()
    mean  = sd[f"{prefix}.running_mean"].double()
    var   = sd[f"{prefix}.running_var"].double()
    scale = gamma / torch.sqrt(var + BN_EPS)
    shift = beta - mean * scale
    return scale.float().numpy(), shift.float().numpy()


def conv_to_out_tap_in(w):
    """(out, in, 3, 3) torch conv weight → (out, 9, in) float32 numpy, tap = dr*3+dc."""
    w = w.detach().double().numpy()
    out, inn, kh, kw = w.shape
    assert kh == 3 and kw == 3
    return np.ascontiguousarray(w.transpose(0, 2, 3, 1).reshape(out, 9, inn)).astype(np.float32)


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.nb_floats = 0

    def u32(self, v):
        self.f.write(struct.pack("<I", v))

    def floats(self, arr):
        arr = np.ascontiguousarray(arr, dtype="<f4")
        self.f.write(arr.tobytes())
        self.nb_floats += arr.size

    def close(self):
        self.f.close()


def export_bin(net, sd, path):
    C = net.input_conv[0].weight.shape[0]
    B = len(net.res_blocks)
    F = net.value_fc1.weight.shape[0]
    A = net.policy_fc.weight.shape[0]

    w = Writer(path)
    w.f.write(MAGIC)
    for v in (VERSION, 4, C, B, F, A):
        w.u32(v)

    # ── stem: conv(4→C, no bias) followed by BN → fold BN into the conv ──
    s, t = bn_affine(sd, "input_conv.1")
    stem = conv_to_out_tap_in(sd["input_conv.0.weight"]) * s[:, None, None]
    w.floats(stem)
    w.floats(t)

    # ── residual blocks (pre-activation: BN→ReLU→conv, twice) ──
    for b in range(B):
        p = f"res_blocks.{b}"
        s1, t1 = bn_affine(sd, f"{p}.bn1")
        w.floats(s1); w.floats(t1)
        w.floats(conv_to_out_tap_in(sd[f"{p}.conv1.weight"]))
        s2, t2 = bn_affine(sd, f"{p}.bn2")
        w.floats(s2); w.floats(t2)
        w.floats(conv_to_out_tap_in(sd[f"{p}.conv2.weight"]))

    # ── trunk output BN (followed by ReLU in forward) ──
    so, to = bn_affine(sd, "output_bn")
    w.floats(so); w.floats(to)

    # ── value head: 1x1 conv(C→1, no bias) + BN(1) folded ──
    sv, tv = bn_affine(sd, "value_bn")
    vconv = sd["value_conv.weight"].double().numpy().reshape(1, C) * sv[:, None]
    w.floats(vconv.astype(np.float32))
    w.floats(tv)
    w.floats(sd["value_fc1.weight"].numpy()); w.floats(sd["value_fc1.bias"].numpy())
    w.floats(sd["value_fc2.weight"].numpy()); w.floats(sd["value_fc2.bias"].numpy())

    # ── policy head: 1x1 conv(C→2, no bias) + BN(2) folded ──
    sp, tp = bn_affine(sd, "policy_bn")
    pconv = sd["policy_conv.weight"].double().numpy().reshape(2, C) * sp[:, None]
    w.floats(pconv.astype(np.float32))
    w.floats(tp)
    w.floats(sd["policy_fc.weight"].numpy()); w.floats(sd["policy_fc.bias"].numpy())

    w.close()
    print(f"wrote {path}: C={C} blocks={B} value_fc={F} actions={A}, "
          f"{w.nb_floats:,} floats ({os.path.getsize(path) / 1e6:.1f} MB)")


def export_torchscript(net, path):
    """Trace the eval-mode network on a dummy batch and save it for libtorch."""
    example = torch.zeros(1, 4, 8, 8)
    with torch.no_grad():
        traced = torch.jit.trace(net, example)
    traced.save(path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def random_position(rng, max_plies=60):
    """Play a random game for a random number of plies; return the Yolah state."""
    y = Yolah()
    n = int(rng.integers(0, max_plies))
    for _ in range(n):
        if y.game_over():
            break
        moves = y.moves()
        y.play(moves[int(rng.integers(0, len(moves)))])
    return y


def export_test_vectors(net, path, nb, seed=1234):
    """
    Record `nb` random (non-terminal preferred) positions with PyTorch's
    fp32 outputs. Record layout, repeated nb times after a uint32 count:
        uint64 black, uint64 white, uint64 empty, uint32 turn(0/1),
        float32 value, float32[4096] policy logits
    """
    rng = np.random.default_rng(seed)
    positions = [Yolah()]                         # always include the initial position
    while len(positions) < nb:
        y = random_position(rng)
        if not y.game_over():
            positions.append(y)
    X = torch.stack([encode_cnn(y) for y in positions])
    with torch.no_grad():
        v, p = net(X)
    v = v.numpy().astype("<f4"); p = p.numpy().astype("<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", nb))
        for i, y in enumerate(positions):
            f.write(struct.pack("<QQQI", y.black, y.white, y.empty, y.nb_plies() & 1))
            f.write(struct.pack("<f", float(v[i])))
            f.write(p[i].tobytes())
    print(f"wrote {path}: {nb} positions "
          f"(value range [{v.min():+.3f}, {v.max():+.3f}])")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", default="cnn_resnet_256x30_value_policy.pt")
    ap.add_argument("--no-ts", action="store_true", help="skip the TorchScript export")
    ap.add_argument("--nb-test", type=int, default=32, help="number of test vectors")
    args = ap.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu")
    # Tolerate a DDP-wrapped checkpoint ("module." prefix).
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    C = sd["input_conv.0.weight"].shape[0]
    B = sum(1 for k in sd if k.endswith(".conv1.weight"))
    net = Net(channels=C, nb_blocks=B, value_fc_size=sd["value_fc1.weight"].shape[0],
              num_actions=sd["policy_fc.weight"].shape[0])
    net.load_state_dict(sd)
    net.eval()

    base = os.path.splitext(args.checkpoint)[0]
    export_bin(net, sd, base + ".bin")
    if not args.no_ts:
        export_torchscript(net, base + ".ts")
    export_test_vectors(net, base + ".test.bin", args.nb_test)


if __name__ == "__main__":
    main()
