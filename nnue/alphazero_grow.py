"""
alphazero_grow.py — make a wider AlphaZero network (more channels, optional
global pooling in the heads) that computes EXACTLY what a trained one does,
so that self-play can go on with more capacity and no loss at the start.

    python3 alphazero_grow.py <az_XXXXXXXX.pt | checkpoint.pt> <out.pt> \
            [--channels 320] [--global-pool] [--blocks 30]

How the outputs are kept: the trunk carries C old channels and D new ones.
  • every weight that READS a new channel (from the residual stream or from a
    new hidden channel of a block) into an old one, into a head or into the
    pooled vector is zero;
  • the weights that WRITE the new channels are random (same spread as the
    old weights of that layer), so the new channels are not all equal and
    the gradient reaches them from the first step.
The new reading weights then get a gradient (their inputs are not zero) and
the network starts using the new channels as training goes on. Extra blocks
(--blocks) are identity blocks (conv2 = 0). The global-pooling layers start
at zero.

From a checkpoint.pt of a run, the weight average (EMA, what the run exports)
is used, auxiliary heads included (their reading weights are widened with
zeros as well). The output is a plain state_dict that alphazero_learn.py
takes as --init-model; with --aux it resumes the auxiliary heads from it.
Writes <out>.ts as well and checks the outputs on random and real positions.
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../server"))
sys.path.append(HERE)
from cnn_resnet_value_policy_chunked import net_from_state_dict  # noqa: E402
from alphazero_aux import AuxNet, is_aux_key  # noqa: E402


def load(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "optimizer" in sd:          # checkpoint.pt of alphazero_learn
        if sd.get("ema") is not None:
            sd = {k[7:]: v for k, v in sd["ema"].items() if k.startswith("module.")}
        else:
            sd = sd["model"]
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def random_like(t, shape):
    """Random values with the spread of the existing weights `t`."""
    std = t.float().std().item() if t.numel() > 1 else 0.05
    return torch.randn(shape) * std


def grow(sd, C2, B2, global_pool):
    C = sd["input_conv.0.weight"].shape[0]
    B = sum(1 for k in sd if k.endswith(".conv1.weight"))
    assert C2 >= C and B2 >= B, "the network can only grow"
    D = C2 - C
    out = {}

    def bn(prefix, new_prefix=None):
        new_prefix = new_prefix or prefix
        for name, fill in (("weight", 1.0), ("bias", 0.0), ("running_mean", 0.0), ("running_var", 1.0)):
            t = sd[f"{prefix}.{name}"]
            out[f"{new_prefix}.{name}"] = torch.cat([t, torch.full((D,), fill, dtype=t.dtype)])
        out[f"{new_prefix}.num_batches_tracked"] = sd[f"{prefix}.num_batches_tracked"].clone()

    def read_only(w, dim=1, pooled=False):
        """A layer that only reads the trunk: zero weights on the new channels
        (for the pooled vector, [mean(C) | max(C)] → [mean(C2) | max(C2)])."""
        if not pooled:
            shape = list(w.shape); shape[dim] = D
            return torch.cat([w, torch.zeros(shape, dtype=w.dtype)], dim=dim)
        a, b = w.split(C, dim=1)
        z = torch.zeros(w.shape[0], D, dtype=w.dtype)
        return torch.cat([a, z, b, z], dim=1)

    # stem: new output channels random
    w = sd["input_conv.0.weight"]
    out["input_conv.0.weight"] = torch.cat([w, random_like(w, (D, *w.shape[1:]))])
    bn("input_conv.1")
    # residual blocks
    for i in range(B):
        p = f"res_blocks.{i}"
        bn(f"{p}.bn1"); bn(f"{p}.bn2")
        for c in ("conv1", "conv2"):
            w = sd[f"{p}.{c}.weight"]
            n = torch.zeros(C2, C2, *w.shape[2:])
            n[:C, :C] = w                                  # old → old
            n[C:, :] = random_like(w, (D, C2, *w.shape[2:]))   # all → new
            out[f"{p}.{c}.weight"] = n                     # new → old stays 0
    for i in range(B, B2):                                 # identity blocks
        p, q = f"res_blocks.{i}", f"res_blocks.{B - 1}"
        for b in ("bn1", "bn2"):
            for name in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                out[f"{p}.{b}.{name}"] = (torch.ones(C2) if name in ("weight", "running_var") else
                                          torch.zeros(C2) if name != "num_batches_tracked" else
                                          sd[f"{q}.{b}.num_batches_tracked"].clone())
        out[f"{p}.conv1.weight"] = random_like(sd[f"{q}.conv1.weight"], (C2, C2, 3, 3))
        out[f"{p}.conv2.weight"] = torch.zeros(C2, C2, 3, 3)
    bn("output_bn")
    # heads: read the trunk with zero weights on the new channels
    for k, v in sd.items():
        if k in out or k.startswith(("input_conv", "res_blocks", "output_bn")):
            continue
        if k in ("value_conv.weight", "policy_conv.weight", "own_conv.weight"):
            out[k] = read_only(v)
        elif k in ("score_fc1.weight", "value_gp_fc.weight", "policy_gp_fc.weight"):
            out[k] = read_only(v, pooled=True)
        else:
            out[k] = v.clone()
    if global_pool and "value_gp_fc.weight" not in out:
        out["value_gp_fc.weight"] = torch.zeros(sd["value_fc1.weight"].shape[0], 2 * C2)
        out["policy_gp_fc.weight"] = torch.zeros(sd["policy_fc.weight"].shape[0], 2 * C2)
    return out


def check(sd_old, sd_new, positions=None):
    """Largest |difference| of value and policy, eval mode, on random boards
    and (if given) on real positions."""
    cls = AuxNet if any(is_aux_key(k) for k in sd_new) else None
    a = net_from_state_dict(sd_old, cls=cls)[0].eval().double()
    b = net_from_state_dict(sd_new, cls=cls)[0].eval().double()
    g = torch.Generator().manual_seed(0)
    cells = torch.randint(0, 3, (512, 64), generator=g)
    x = torch.stack([(cells == 0), (cells == 1), (cells == 2),
                     torch.randint(0, 2, (512, 1), generator=g).expand(512, 64).bool()], 1)
    x = x.double().view(512, 4, 8, 8)
    if positions is not None:
        x = torch.cat([x, positions.double()])
    with torch.no_grad():
        if cls is AuxNet:
            ra, rb = a.forward_all(x), b.forward_all(x)
        else:
            ra, rb = a(x), b(x)
    return [(ta - tb).abs().max().item() for ta, tb in zip(ra, rb)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="az_XXXXXXXX.pt (a plain state_dict) or a run's checkpoint.pt")
    ap.add_argument("out", help="the wider network (.pt); <out>.ts is written next to it")
    ap.add_argument("--channels", type=int, default=320)
    ap.add_argument("--blocks", type=int, default=0, help="0 = as many as the source")
    ap.add_argument("--global-pool", action="store_true", help="add the global-pooling layers to the heads")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    sd = load(args.src)
    C = sd["input_conv.0.weight"].shape[0]
    B = sum(1 for k in sd if k.endswith(".conv1.weight"))
    B2 = args.blocks or B
    new = grow(sd, args.channels, B2, args.global_pool)
    errs = check(sd, new)
    names = ["value", "policy", "ownership", "score"][:len(errs)]
    print(f"{C}x{B} → {args.channels}x{B2}{' + global pooling' if args.global_pool else ''}: "
          f"{sum(v.numel() for k, v in sd.items() if 'running' not in k) / 1e6:.1f}M → "
          f"{sum(v.numel() for k, v in new.items() if 'running' not in k) / 1e6:.1f}M parameters")
    print("largest output difference (float64, 512 boards): "
          + ", ".join(f"{n} {e:.2e}" for n, e in zip(names, errs)))
    if max(errs) > 1e-9:
        sys.exit("ERROR: the wider network does not compute the same outputs")
    torch.save(new, args.out)
    net = net_from_state_dict({k: v for k, v in new.items() if not is_aux_key(k)})[0].float().eval()
    with torch.no_grad():
        torch.jit.trace(net, torch.zeros(1, 4, 8, 8)).save(os.path.splitext(args.out)[0] + ".ts")
    print(f"wrote {args.out} and {os.path.splitext(args.out)[0]}.ts")


if __name__ == "__main__":
    main()
