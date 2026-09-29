"""
check_value_nets.py — PyTorch reference values for nets_check (test/nets_check_main.cpp).

    ./nets_check dump positions.bin 2000
    python3 check_value_nets.py positions.bin refs.bin [../models]
    ./nets_check check positions.bin refs.bin ../models

Writes, for each network of NETWORKS (in this order), the value of every
position for the side to move, as float32:
  • the NNUE networks see the 193 inputs of preprocess_nnue.py: black, white
    and holes as 64 bits each, most significant bit first (the C++ 63 − square
    indexing), then the side to move;
  • the features networks see the feature bytes written by nets_check (the
    same YolahFeatures::set_features that encoded their training data) / 255.
The old NNUE has 3 outputs (black wins, draw, white wins): its value for the
side to move is ±(P(black) − P(white)), exactly as the minimax players use it.
"""
import os
import struct
import sys
import numpy as np
import torch

NB_FEATURES = 119
NETWORKS = ["nnue", "nnue_193x1024x64x32x1", "nnue_193x1024x64x32x1_distill",
            "features_119x256x64x1", "features_119x256x64x1_distill"]


def read_positions(path):
    data = open(path, "rb").read()
    i, recs = 0, []
    while i < len(data):
        b, w, e, turn = struct.unpack_from("<QQQB", data, i); i += 25
        feats = np.frombuffer(data, dtype=np.uint8, count=NB_FEATURES, offset=i).copy(); i += NB_FEATURES
        (n,) = struct.unpack_from("<i", data, i); i += 4 + 2 * n
        recs.append((b, w, e, turn, feats))
    return recs


def bits_msb_first(x):
    return np.unpackbits(np.array([x], dtype=">u8").view(np.uint8)).astype(np.float32)


def load_layers(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = {(k[7:] if k.startswith("module.") else k): v.float() for k, v in sd.items()}
    out, k = [], 1
    while f"fc{k}.weight" in sd:
        out.append((sd[f"fc{k}.weight"], sd[f"fc{k}.bias"]))
        k += 1
    return out


def forward(layers, x):
    """clamp(·, 0, 1) after every layer but the last (as in all the training scripts)."""
    for k, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if k < len(layers) - 1:
            x = x.clamp(0.0, 1.0)
    return x


def main():
    pos_path, out_path = sys.argv[1], sys.argv[2]
    models = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "../models")
    recs = read_positions(pos_path)
    turn = torch.tensor([r[3] for r in recs], dtype=torch.float32)
    x_nnue = torch.tensor(np.stack([np.concatenate([bits_msb_first(b), bits_msb_first(w), bits_msb_first(e),
                                                    np.array([t], np.float32)]) for b, w, e, t, _ in recs]))
    x_feat = torch.tensor(np.stack([r[4] for r in recs]).astype(np.float32) / 255.0)
    sign = 1.0 - 2.0 * turn                                   # +1 black to move, −1 white
    with open(out_path, "wb") as f:
        for name in NETWORKS:
            layers = load_layers(os.path.join(models, name + ".pt"))
            with torch.no_grad():
                if name.startswith("features"):
                    v = torch.tanh(forward(layers, x_feat)).squeeze(-1)
                else:
                    out = forward(layers, x_nnue)
                    if out.shape[1] == 3:                     # (black, draw, white) logits
                        p = torch.softmax(out, dim=1)
                        v = sign * (p[:, 0] - p[:, 2])
                    else:
                        v = torch.tanh(out).squeeze(-1)       # already for the side to move
            f.write(v.numpy().astype("<f4").tobytes())
            print(f"{name:32s} value range [{v.min():+.3f}, {v.max():+.3f}], mean |v| {v.abs().mean():.3f}")


if __name__ == "__main__":
    main()
