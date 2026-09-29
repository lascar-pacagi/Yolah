"""
export_value_nets.py — write the C++ weights files of the minimax networks,
in float and in quantized form, next to their .pt in models/.

    python3 export_value_nets.py [../models]

For every known .pt found in the directory:

  nnue.pt                              → nnue.float.txt, nnue.quantized.txt
  nnue_193x1024x64x32x1[_distill].pt   → <name>.float.txt, <name>.quantized.txt
  features_119x256x64x1[_distill].pt   → <name>.float.txt, <name>.quantized.txt

Formats (read by nnue.cpp / nnue_quantized.cpp / ffnn_value.h):
  every layer as "W\\n<rows>\\n<cols>\\n" + the weights row by row (one value
  per line, PyTorch's (out, in) order), then "B\\n<n>\\n" + the biases.

  float     : the values as they are.
  quantized : mirrors NNUE::save_quantized (nnue.cpp) and save_quantized
              (features_network.py): weights × 64 rounded to int8, biases × 64
              for the first layer and × 64² for the others (they add to
              products of two scale-64 numbers).
              Exceptions:
              • the VALUE head of the 1-output networks (the last layer of
                nnue_193x…x1 and features_119x…x1) is written in FLOAT: it is
                not clamped during training (weights up to ~13 and ~80), far
                outside int8 ×64 (|w| ≤ 127/64); the C++ classes compute it in
                float;
              • the features value networks are quantized in INT16 at scale
                1024 (biases ×1024 and ×1024²) instead of int8 ×64: at int8
                their value is off by ~0.1 on average (see ffnn_value.h).
"""
import os
import sys
import torch

SCALE = 64


def load_sd(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return {(k[7:] if k.startswith("module.") else k): v.float() for k, v in sd.items()}


def layers(sd):
    """[(weight (out, in), bias (out,)), ...] in forward order: fc1, fc2, ..."""
    out, i = [], 1
    while f"fc{i}.weight" in sd:
        out.append((sd[f"fc{i}.weight"], sd[f"fc{i}.bias"]))
        i += 1
    return out


def write_matrix(f, w, fmt):
    m, n = w.shape
    f.write(f"W\n{m}\n{n}\n")
    f.write("\n".join(fmt(v) for v in w.reshape(-1).tolist()) + "\n")


def write_bias(f, b, fmt):
    f.write(f"B\n{b.shape[0]}\n")
    f.write("\n".join(fmt(v) for v in b.tolist()) + "\n")


def as_float(v):
    return repr(float(v))


def quantizer(scale, lo, hi, what):
    def q(v):
        x = int(round(scale * v))
        if not lo <= x <= hi:
            raise ValueError(f"{what}: {v} × {scale} = {x} outside [{lo}, {hi}]")
        return str(x)
    return q


def export(path, out_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    ls = layers(load_sd(path))
    value_head = ls[-1][0].shape[0] == 1          # 1 output = value network
    with open(os.path.join(out_dir, name + ".float.txt"), "w") as f:
        for w, b in ls:
            write_matrix(f, w, as_float)
            write_bias(f, b, as_float)
    int16 = value_head and name.startswith("features")
    scale, wmax, bmax = (1024, 32767, 2**31 - 1) if int16 else (SCALE, 127, 32767)
    with open(os.path.join(out_dir, name + ".quantized.txt"), "w") as f:
        for k, (w, b) in enumerate(ls):
            last = k == len(ls) - 1
            if last and value_head:
                write_matrix(f, w, as_float)
                write_bias(f, b, as_float)
                continue
            write_matrix(f, w, quantizer(scale, -wmax, wmax, f"{name} fc{k + 1} weight"))
            bias_scale = scale if k == 0 else scale * scale
            write_bias(f, b, quantizer(bias_scale, -bmax - 1, bmax, f"{name} fc{k + 1} bias"))
    shapes = " → ".join([str(ls[0][0].shape[1])] + [str(w.shape[0]) for w, _ in ls])
    kind = ("value head in float, trunk int16 ×1024" if int16 else "value head in float, trunk int8 ×64") \
        if value_head else "3 outputs, int8 ×64"
    print(f"{name}: {shapes} ({kind})"
          f" → {name}.float.txt, {name}.quantized.txt")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "../models")
    names = ["nnue", "nnue_193x1024x64x32x1", "nnue_193x1024x64x32x1_distill",
             "features_119x256x64x1", "features_119x256x64x1_distill"]
    for n in names:
        p = os.path.join(out_dir, n + ".pt")
        if os.path.exists(p):
            export(p, out_dir)
        else:
            print(f"(skipped: no {p})")


if __name__ == "__main__":
    main()
