"""
loop_resnet.py — one round of the distillation loop for the ResNet
(nnue/distill_loop.sh): the value/policy network of the previous round,
fine-tuned on the round's cache (loop_prepare.py).

    python3 loop_resnet.py CACHE --init PREV.pt --out-dir DIR --step ROUND
                           [--epochs 1] [--lr 5e-5] [--batch 1024] [--q-weight 0.7]
                           [--policy-weight 1.0] [--ema 0.999] [--amp auto]

Targets
  value   (1 − q)·z + q·s: z the game's result, s the search target of the cache
          (the exact result near the end, else the dev player's search value;
          the old teacher's value on the old positions)
  policy  cross-entropy with the move the dev player played (one-hot) or, for
          a node of a search tree, its best move (no policy for the passes and
          the old positions)
Each batch takes one random symmetry of the board per row (the move index
through the same symmetry, alphazero_learn.symmetry_tables). The positions are
read by chunks of consecutive rows (loop_prepare.py mixed the games and the
old positions by blocks), each chunk shuffled; the last 5 % are the validation
set, as for the NNUE trainers. The exported network (DIR/models/az_<step>.pt
and .ts, alphazero_learn.export_model) is the exponential average of the
weights; a progress line per epoch goes to DIR/resnet_log.jsonl.
"""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from alphazero_learn import load_state_dict, make_net, export_model, symmetry_tables

NO_POLICY = 65535
ROW = 193


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


class Data:
    def __init__(self, cache):
        with open(os.path.join(cache, "meta.json")) as f:
            self.n = int(json.load(f)["n_positions"])
        mm = lambda name, dt, shape: np.memmap(os.path.join(cache, name), dt, "r", shape=shape)
        self.inputs = mm("inputs.u8", np.uint8, (self.n, ROW))
        self.z = mm("values.i8", np.int8, (self.n,))
        self.s = mm("search.f16", np.float16, (self.n,))
        self.policy = mm("policy.u2", np.uint16, (self.n,))
        self.n_train = int(0.95 * self.n)

    def chunk(self, lo, hi):
        return (np.ascontiguousarray(self.inputs[lo:hi]), self.z[lo:hi].astype(np.float32),
                self.s[lo:hi].astype(np.float32), self.policy[lo:hi].astype(np.int64))


class Batches:
    """Board rows → (x, value target, policy index, policy mask) on the GPU, with a random symmetry."""
    def __init__(self, device, q_weight):
        self.dev = device
        self.q = q_weight
        src, act = symmetry_tables()
        self.src = src.to(device)
        self.act = act.to(device)

    def __call__(self, rows, z, s, pol, augment):
        dev = self.dev
        rows = torch.from_numpy(rows).to(dev, non_blocking=True)
        B = rows.shape[0]
        planes = torch.cat([rows[:, :192].view(B, 3, 64).float(),
                            rows[:, 192].float().view(B, 1, 1).expand(B, 1, 64)], dim=1)
        target = torch.from_numpy((1.0 - self.q) * z + self.q * s).to(dev)
        pol = torch.from_numpy(pol).to(dev)
        has = pol != NO_POLICY
        pol = torch.where(has, pol, torch.zeros_like(pol))
        if augment:
            sym = torch.randint(0, 8, (B,), device=dev)
            planes = planes.gather(2, self.src[sym].unsqueeze(1).expand(B, 4, 64))
            pol = self.act[sym, pol]
        x = planes.view(B, 4, 8, 8).contiguous(memory_format=torch.channels_last)
        return x, target, pol, has


def losses(net, x, target, pol, has, amp_dtype, device):
    with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        v, logits = net(x)
    v = v.float().view(-1)
    v_loss = F.mse_loss(v, target)
    logp = F.log_softmax(logits.float(), dim=1).gather(1, pol.unsqueeze(1)).squeeze(1)
    n_pol = has.sum().clamp_min(1)
    p_loss = -(logp * has).sum() / n_pol
    top1 = ((logits.argmax(1) == pol) & has).sum().float() / n_pol
    return v, v_loss, p_loss, top1


@torch.no_grad()
def validate(net, data, batches, amp_dtype, device, batch, max_rows=2_000_000):
    net.eval()
    lo, hi = data.n_train, min(data.n, data.n_train + max_rows)
    acc = dict(v=0.0, vz=0.0, p=0.0, top1=0.0, n=0, npol=0)
    for a in range(lo, hi, batch):
        rows, z, s, pol = data.chunk(a, min(hi, a + batch))
        x, target, p, has = batches(rows, z, s, pol, augment=False)
        v, v_loss, p_loss, top1 = losses(net, x, target, p, has, amp_dtype, device)
        k = len(z)
        acc["v"] += v_loss.item() * k
        acc["vz"] += F.mse_loss(v, torch.from_numpy(z).to(device)).item() * k
        npol = int(has.sum())
        acc["p"] += p_loss.item() * npol
        acc["top1"] += top1.item() * npol
        acc["n"] += k
        acc["npol"] += npol
    net.train()
    n, m = max(1, acc["n"]), max(1, acc["npol"])
    return dict(mse_target=acc["v"] / n, mse_z=acc["vz"] / n, policy_ce=acc["p"] / m, policy_top1=acc["top1"] / m,
                positions=acc["n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--init", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=1 << 22)
    ap.add_argument("--q-weight", type=float, default=0.7)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if a.amp == "auto":
        amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.bfloat16 if a.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)

    data = Data(a.cache)
    net = make_net(load_state_dict(a.init)).to(device).to(memory_format=torch.channels_last)
    net.train()
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(a.ema)) if a.ema > 0 else None
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    batches = Batches(device, a.q_weight)
    os.makedirs(os.path.join(a.out_dir, "models"), exist_ok=True)
    log_path = os.path.join(a.out_dir, "resnet_log.jsonl")
    total_steps = max(1, int(a.epochs * data.n_train / a.batch))
    log(f"{a.init} → step {a.step}: {data.n:,} positions ({data.n_train:,} training), {total_steps:,} steps "
        f"of {a.batch}, lr {a.lr:g}, q weight {a.q_weight}, policy weight {a.policy_weight}, {amp_dtype}, {device}")

    def record(epoch, m):
        m = dict(step=a.step, epoch=epoch, **m)
        log(f"validation (epoch {epoch}): " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                                         for k, v in m.items()))
        with open(log_path, "a") as f:
            f.write(json.dumps(m) + "\n")

    record(0, validate(net, data, batches, amp_dtype, device, a.batch))
    step = 0
    t0 = time.time()
    acc = dict(v=0.0, p=0.0, top1=0.0, n=0)
    epoch = 0
    while step < total_steps:
        starts = rng.permutation(np.arange(0, data.n_train, a.chunk))
        for c in starts:
            if step >= total_steps:
                break
            rows, z, s, pol = data.chunk(c, min(data.n_train, c + a.chunk))
            perm = rng.permutation(len(z))
            rows, z, s, pol = rows[perm], z[perm], s[perm], pol[perm]
            for b in range(0, len(z) - a.batch + 1, a.batch):
                if step >= total_steps:
                    break
                # Linear warm-up, then a cosine down to a tenth.
                lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.55 + 0.45 * math.cos(math.pi * step / total_steps))
                for g in opt.param_groups:
                    g["lr"] = lr
                x, target, p, has = batches(rows[b:b + a.batch], z[b:b + a.batch], s[b:b + a.batch],
                                            pol[b:b + a.batch], augment=True)
                _, v_loss, p_loss, top1 = losses(net, x, target, p, has, amp_dtype, device)
                loss = v_loss + a.policy_weight * p_loss
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    ema.update_parameters(net)
                step += 1
                acc["v"] += v_loss.item(); acc["p"] += p_loss.item(); acc["top1"] += top1.item(); acc["n"] += 1
                if step % 2000 == 0:
                    n = acc["n"]
                    log(f"step {step:,}/{total_steps:,}  v {acc['v']/n:.4f}  p {acc['p']/n:.4f}  "
                        f"top1 {acc['top1']/n:.3f}  lr {lr:.2e}  {step * a.batch / (time.time() - t0):.0f} samples/s")
                    acc = dict(v=0.0, p=0.0, top1=0.0, n=0)
        epoch += 1
        final = ema.module if ema is not None else net
        record(epoch, validate(final, data, batches, amp_dtype, device, a.batch))
    final = ema.module if ema is not None else net
    pt, ts = export_model(final, a.out_dir, a.step)
    log(f"exported {pt} and {ts}")


if __name__ == "__main__":
    main()
