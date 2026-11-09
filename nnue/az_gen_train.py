"""
az_gen_train.py — the training of one generation of the generations loop
(nnue/az_gen_loop.sh): the network of generation G − 1, trained for EPOCHS
epochs on a window of the newest WINDOW self-play rows, becomes generation G.

    python3 az_gen_train.py ROOT --gen G [--window 50000000] [--epochs 10] [--lr 5e-5]
                            [--old DIR ...] [--q-weight 0] [--fast-policy-weight 0]

ROOT/models/az_<G-1>.pt   the network trained further (generation 0: the start network)
ROOT/gen_<g>/selfplay/    the rows of generation g (self-play by az_<g-1>)
--old-cache DIR           an older training cache (preprocess_nnue.py layout:
                          inputs.u8, values.i8; cache_az256), which fills the
                          window while the generations do not
--old DIR                 older self-play rows (another run's selfplay/), used
                          after the generations, before the old cache

The window: the rows of gen_G (all of them), then gen_{G-1}, gen_{G-2}, ...,
then the --old directories, newest file first, then positions of the old
cache, until WINDOW rows. With generations of WINDOW / 4 rows: generation 1 =
its rows + 3/4 of the window from the old cache, generation 2 = 2/4 + 2/4,
generation 3 = 3/4 + 1/4, then only generations; a generation's rows leave the
window 4 generations later. The old cache's positions are drawn by blocks of
--old-block consecutive positions in one fixed random order (--old-seed): each
generation takes the beginning of it, so the old positions of a generation are
among those of the generation before (they leave the window little by little).

Validation: one file in 50 of gen_G (kept out of the training).

Targets
  value   z (the game's result; the exact result where the solver gave it,
          flags SAMPLE_EXACT), blended with the search value root_q if
          --q-weight > 0 (not where z is exact): for every row
  policy  cross-entropy with the search's π: the rows of the full searches
          only by default (the fast searches, flags SAMPLE_FAST, few visits
          and no noise, weigh --fast-policy-weight = 0; the old cache's
          positions, flags OLD, have no π)
One random symmetry of the board per row (alphazero_learn.BatchMaker).

The learning rate: a linear warm-up, then a cosine down to a tenth over the
generation's training. The exported network (ROOT/models/az_<G>.pt and .ts) is
the exponential average of the weights. A checkpoint (ROOT/gen_<G>/train_ckpt.pt)
every --ckpt-minutes and on SIGTERM / SIGUSR1: the training resumes from it when
the script is run again. Progress: ROOT/train_log.jsonl (one line per epoch).
"""
import argparse
import glob
import json
import math
import os
import signal
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alphazero_learn import (SAMPLE_DTYPE, BatchMaker, export_model, load_state_dict,  # noqa: E402
                             make_net, model_paths, publish_latest, read_header, read_samples)

FAST, EXACT, OLD = 1, 2, 4
ROW = 193
STOP = {"flag": False}


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def window_files(root, gen, old_dirs, window, val_every):
    """(training parts [(file, rows to take from its end)], validation files): newest first."""
    sources = [os.path.join(root, f"gen_{g:03d}", "selfplay") for g in range(gen, 0, -1)] + list(old_dirs)
    parts, val, total = [], [], 0
    for k, d in enumerate(sources):
        files = sorted(glob.glob(os.path.join(d, "*.bin")))
        if k == 0:
            val = files[::val_every]
            files = [f for f in files if f not in set(val)]
        for f in reversed(files):
            if total >= window:
                break
            try:
                n = read_header(f)[0]
            except (ValueError, OSError) as e:
                log(f"skipping {f}: {e}")
                continue
            take = min(n, window - total)
            parts.append((f, take))
            total += take
        log(f"  {d}: {len(files):,} files, window now {total:,} rows")
        if total >= window:
            break
    return parts, val, total


def old_cache_rows(path, n, block, seed):
    """n positions of an old cache as self-play rows (no π, flags OLD), by blocks in a fixed random order."""
    with open(os.path.join(path, "meta.json")) as f:
        on = int(json.load(f)["n_positions"])
    inputs = np.memmap(os.path.join(path, "inputs.u8"), np.uint8, "r", shape=(on, ROW))
    values = np.memmap(os.path.join(path, "values.i8"), np.int8, "r", shape=(on,))
    starts = np.random.default_rng(seed).permutation(on // block) * block
    nb = min(len(starts), -(-n // block))
    take = np.sort(starts[:nb])
    out = np.zeros(nb * block, dtype=SAMPLE_DTYPE)
    for i, st in enumerate(take):
        r = np.asarray(inputs[st:st + block])
        o = out[i * block:(i + 1) * block]
        for k, name in enumerate(("black", "white", "empty")):
            # bit 63 − j of the bitboard is byte j of the row (loop_prepare.board_rows)
            o[name] = np.packbits(r[:, 64 * k:64 * (k + 1)], axis=1).view(">u8").ravel().astype("<u8")
        o["turn"] = r[:, 192]
        o["z"] = values[st:st + block]
        if i % 1000 == 0:
            log(f"  old cache: {i * block:,} / {nb * block:,} positions")
    out["flags"] = OLD
    out = out[:n]
    log(f"  old cache {path}: {len(out):,} of its {on:,} positions")
    return out


def load_rows(parts, total):
    rows = np.empty(total, dtype=SAMPLE_DTYPE)
    pos = 0
    for i, (f, take) in enumerate(parts):
        r = read_samples(f, SAMPLE_DTYPE)
        rows[pos:pos + take] = r[len(r) - take:]
        pos += take
        if i % 2000 == 0:
            log(f"  read {pos:,} / {total:,} rows")
    assert pos == total
    return rows


class Losses:
    def __init__(self, device, amp_dtype, q_weight, fast_policy_weight):
        self.dev, self.amp, self.q, self.fw = device, amp_dtype, q_weight, fast_policy_weight
        self.make = BatchMaker(device, aux=False, q_weight=0.0)

    def __call__(self, net, rows, augment):
        x, z, actions, probs = self.make(rows, augment=augment)
        flags = torch.from_numpy(rows["flags"].astype(np.int64)).to(self.dev)
        exact = (flags & EXACT) != 0
        fast = (flags & FAST) != 0
        no_pi = (flags & OLD) != 0
        target = z
        if self.q > 0:
            q = torch.from_numpy(rows["root_q"].astype(np.float32)).to(self.dev).clamp(-1.0, 1.0)
            target = torch.where(exact, z, (1.0 - self.q) * z + self.q * q)
        with torch.autocast(self.dev.type, dtype=self.amp, enabled=self.dev.type == "cuda"):
            v, logits = net(x)
        v = v.float().view(-1)
        v_loss = F.mse_loss(v, target)
        logp = F.log_softmax(logits.float(), dim=1).gather(1, actions)
        ce = -(probs * logp).sum(1)
        w = torch.where(fast, torch.full_like(ce, self.fw), torch.ones_like(ce))
        w = torch.where(no_pi, torch.zeros_like(ce), w)
        p_loss = (w * ce).sum() / w.sum().clamp_min(1e-6)
        best = actions.gather(1, probs.argmax(1, keepdim=True)).squeeze(1)
        top1 = (logits.argmax(1) == best).float()
        return dict(v=v, z=z, target=target, exact=exact, fast=fast, v_loss=v_loss, p_loss=p_loss, ce=ce, top1=top1)


@torch.no_grad()
def validate(net, rows, losses, batch):
    net.eval()
    acc = dict(mse_z=0.0, n=0, ce_full=0.0, top1_full=0.0, n_full=0, mse_exact=0.0, n_exact=0)
    for a in range(0, len(rows), batch):
        o = losses(net, rows[a:a + batch], augment=False)
        err = (o["v"] - o["z"]) ** 2
        full = ~o["fast"]
        acc["mse_z"] += err.sum().item(); acc["n"] += len(err)
        acc["ce_full"] += o["ce"][full].sum().item(); acc["top1_full"] += o["top1"][full].sum().item()
        acc["n_full"] += int(full.sum())
        acc["mse_exact"] += err[o["exact"]].sum().item(); acc["n_exact"] += int(o["exact"].sum())
    net.train()
    return dict(val_mse_z=acc["mse_z"] / max(1, acc["n"]), val_policy_ce=acc["ce_full"] / max(1, acc["n_full"]),
                val_top1=acc["top1_full"] / max(1, acc["n_full"]),
                val_mse_exact=acc["mse_exact"] / max(1, acc["n_exact"]), val_rows=acc["n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--window", type=int, default=50_000_000)
    ap.add_argument("--old", action="append", default=[], help="older self-play directories (after the generations)")
    ap.add_argument("--old-cache", default="", help="an older training cache, which fills the window")
    ap.add_argument("--old-block", type=int, default=4096)
    ap.add_argument("--old-seed", type=int, default=12345)
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--q-weight", type=float, default=0.0)
    ap.add_argument("--fast-policy-weight", type=float, default=0.0)
    ap.add_argument("--val-every", type=int, default=50, help="one file of the new generation in N for the validation")
    ap.add_argument("--val-rows", type=int, default=1_000_000)
    ap.add_argument("--ckpt-minutes", type=float, default=30.0)
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    for s in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(s, lambda *_: STOP.update(flag=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if a.amp == "auto":
        amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.bfloat16 if a.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)

    gen_dir = os.path.join(a.root, f"gen_{a.gen:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    ckpt_path = os.path.join(gen_dir, "train_ckpt.pt")
    init_pt, _ = model_paths(a.root, a.gen - 1)
    log(f"generation {a.gen}: {init_pt} → az_{a.gen:08d}, window {a.window:,} rows, {a.epochs:g} epochs, "
        f"lr {a.lr:g}, q weight {a.q_weight}, fast policy weight {a.fast_policy_weight}, {amp_dtype}, {device}")
    parts, val_files, total = window_files(a.root, a.gen, a.old, a.window, a.val_every)
    rows = load_rows(parts, total)
    if a.old_cache and total < a.window:
        rows = np.concatenate([rows, old_cache_rows(a.old_cache, a.window - total, a.old_block, a.old_seed)])
        total = len(rows)
    if total == 0:
        sys.exit("no rows")
    val = np.concatenate([read_samples(f, SAMPLE_DTYPE) for f in val_files])[:a.val_rows] if val_files else rows[:0]
    share = lambda bit: ((rows["flags"] & bit) != 0).mean()
    log(f"window: {total:,} rows ({share(OLD):.1%} old cache, {share(FAST):.1%} fast, {share(EXACT):.1%} exact), "
        f"validation {len(val):,} rows")

    net = make_net(load_state_dict(init_pt)).to(device).to(memory_format=torch.channels_last)
    net.train()
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(a.ema)) if a.ema > 0 else None
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    losses = Losses(device, amp_dtype, a.q_weight, a.fast_policy_weight)
    steps_per_epoch = total // a.batch
    total_steps = max(1, int(a.epochs * steps_per_epoch))
    step = 0
    if os.path.exists(ckpt_path):
        c = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); scaler.load_state_dict(c["scaler"])
        if ema is not None and c.get("ema") is not None:
            ema.load_state_dict(c["ema"])
        step = c["step"]
        if c.get("total") != total:
            log(f"WARNING: the checkpoint's window had {c.get('total'):,} rows, now {total:,}")
        log(f"resumed from {ckpt_path} at step {step:,} / {total_steps:,}")

    def save_ckpt():
        tmp = ckpt_path + ".tmp"
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                        ema=ema.state_dict() if ema is not None else None, step=step, total=total), tmp)
        os.replace(tmp, ckpt_path)

    log_path = os.path.join(a.root, "train_log.jsonl")

    def record(epoch):
        final = ema.module if ema is not None else net
        m = dict(gen=a.gen, epoch=epoch, step=step, window=total, **validate(final, val, losses, a.batch))
        log("validation: " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()))
        with open(log_path, "a") as f:
            f.write(json.dumps(m) + "\n")

    if step == 0 and len(val):
        record(0)
    t0, s0, last_ckpt = time.time(), step, time.time()
    acc = dict(v=0.0, p=0.0, n=0)
    while step < total_steps:
        epoch = step // steps_per_epoch
        perm = np.random.default_rng(a.seed * 1000 + a.gen * 100 + epoch).permutation(total)
        for b in range(step % steps_per_epoch, steps_per_epoch):
            if step >= total_steps:
                break
            lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.55 + 0.45 * math.cos(math.pi * step / total_steps))
            for g in opt.param_groups:
                g["lr"] = lr
            idx = np.sort(perm[b * a.batch:(b + 1) * a.batch])
            o = losses(net, rows[idx], augment=True)
            loss = o["v_loss"] + o["p_loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update_parameters(net)
            step += 1
            acc["v"] += o["v_loss"].item(); acc["p"] += o["p_loss"].item(); acc["n"] += 1
            if step % 2000 == 0:
                n = acc["n"]
                log(f"step {step:,}/{total_steps:,} (epoch {step / steps_per_epoch:.2f})  v {acc['v'] / n:.4f}  "
                    f"p {acc['p'] / n:.4f}  lr {lr:.2e}  {(step - s0) * a.batch / (time.time() - t0):.0f} samples/s")
                acc = dict(v=0.0, p=0.0, n=0)
            if STOP["flag"] or time.time() - last_ckpt > 60 * a.ckpt_minutes:
                save_ckpt()
                last_ckpt = time.time()
                if STOP["flag"]:
                    log(f"stopped at step {step:,}: checkpoint {ckpt_path}, run again to resume")
                    sys.exit(3)
        if step % steps_per_epoch == 0 or step >= total_steps:
            if len(val):
                record(step / steps_per_epoch)
    final = ema.module if ema is not None else net
    pt, ts = export_model(final, a.root, a.gen)
    publish_latest(a.root, a.gen)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    log(f"exported {pt} and {ts}")


if __name__ == "__main__":
    main()
