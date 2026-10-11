"""
check_persp_net.py — PyTorch reference values of a NetPersp (nnue_persp.py) for
`nets_check check_persp` (test/nets_check_main.cpp).

    ./nets_check dump positions.bin 2000
    python3 check_persp_net.py positions.bin refs.bin ../models/nnue_persp_k8_distill.pt
    ./nets_check check_persp positions.bin refs.bin ../models/nnue_persp_k8_distill.quantized.txt \
                 [../models/nnue_193x1024x64x32x1_distill.quantized.txt]

The network sees the 193 cache inputs (as check_value_nets.py builds them) and
converts them itself; refs.bin = one float32 per position, the value for the
side to move.
"""
import sys
import numpy as np
import torch

from check_value_nets import read_positions, bits_msb_first
from nnue_persp import load_persp


def main():
    pos_path, out_path, model = sys.argv[1], sys.argv[2], sys.argv[3]
    recs = read_positions(pos_path)
    x = torch.tensor(np.stack([np.concatenate([bits_msb_first(b), bits_msb_first(w), bits_msb_first(e),
                                               np.array([t], np.float32)]) for b, w, e, t, _ in recs]))
    net = load_persp(model).eval()
    with torch.no_grad():
        v = net(x)
    open(out_path, "wb").write(v.numpy().astype("<f4").tobytes())
    print(f"{model}: K = {net.k}, them = {net.them}; value range [{v.min():+.3f}, {v.max():+.3f}], "
          f"mean |v| {v.abs().mean():.3f}")


if __name__ == "__main__":
    main()
