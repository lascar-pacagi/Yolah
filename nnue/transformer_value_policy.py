"""
transformer_value_policy.py
===========================
A two-headed TRANSFORMER of the same size as the supervised ResNet 256x30
(cnn_resnet_256x30_value_policy, ~36 M parameters, ~4.5 GFLOP per position),
trained on the SAME data with the SAME targets (the cache of preprocess.py:
game outcome z for the player to move, move played), so that the two can be
compared on the validation set and then inside the same MCTS (the exported
TorchScript has the ResNet's interface: (B, 4, 8, 8) → value (B,), logits
(B, 4096), action = from*64 + to in Yolah's square numbering).

Network (pre-LayerNorm encoder, one token per square)
─────────────────────────────────────────────────────
    token i  = cell i of the cache planes (cell i is square 63 - i, see
               cnn_resnet_value_policy_chunked._bitboard_to_plane)
    input    : Linear(4 → d) of the 4 planes of the square (black, white,
               empty, turn) + a learned position embedding (64 × d)
    L blocks : x += Attn(LN(x)) ; x += MLP(LN(x))   (GELU, hidden 4d)
               attention has a learned bias per head and per pair of squares
               (h × 64 × 64), so geometry (lines, diagonals, neighbours) can
               be learned directly
    value    : LN → [mean, max] over the squares (2d) → FC 256 → ReLU → FC 1 → tanh
    policy   : LN → q = W_from x, k = W_to x (128 each) → logit(from, to) = q_from · k_to / √128
               (a "from-to attention" policy head, as in Lc0's attention policy)

Default size d = 512, 8 heads, 12 layers: 37.9 M parameters, ≈ 4.8 GFLOP per
position (the ResNet: 36.2 M, ≈ 4.6 GFLOP).

Training (differences with the ResNet script, both necessary for transformers)
───────────────────────────────────────────────────────────────────────────────
    AdamW (weight decay 0.05 on the matrices only), lr warmup 2000 steps then
    cosine to 1 % per STEP; gradient clipping 1.0; bf16 autocast.
    Same loader (ChunkedShuffleLoader), same 95 % / 5 % split, same logged
    metrics (value_mse, value_sign_acc, policy_ce, policy_acc).
    Checkpoint every SAVE_CHUNKS chunks (model + optimizer + step): a stopped
    job resumes where it was, chunk-exact (time limit, scancel).

YOLAH_TF_ARCH = transformer (default) | resnet_ft (the ResNet 256x30 with this
from-to policy head, to separate the head from attention) | resnet (the
ResNet unchanged, trained with this recipe: the fair baseline).

Usage (inside the AlphaZero image; see transformer_value_policy.sh):
    python3 transformer_value_policy.py train  --cache /cache --out /work
    python3 transformer_value_policy.py eval   --cache /cache --model X.pt [--resnet] [--chunks 8]
    python3 transformer_value_policy.py export --model /work/transformer.pt   # → .ts + .test.bin
    python3 transformer_value_policy.py selftest                              # shapes, params, trace, fp16
Environment: YOLAH_TF_DIM (512), YOLAH_TF_HEADS (8), YOLAH_TF_LAYERS (12),
YOLAH_TF_LR (5e-4), YOLAH_NB_EPOCHS (2), YOLAH_TF_BATCH (1024 per GPU).
"""

import argparse
import gc
import json
import math
import os
import random
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "../server"))
import cnn_resnet_value_policy_chunked as R          # noqa: E402  (loader, encoder, ResNet)

torch.set_float32_matmul_precision('high')

DIM        = int(os.environ.get("YOLAH_TF_DIM", "512"))
HEADS      = int(os.environ.get("YOLAH_TF_HEADS", "8"))
LAYERS     = int(os.environ.get("YOLAH_TF_LAYERS", "12"))
LR         = float(os.environ.get("YOLAH_TF_LR", "5e-4"))
NB_EPOCHS  = int(os.environ.get("YOLAH_NB_EPOCHS", "2"))
BATCH      = int(os.environ.get("YOLAH_TF_BATCH", "1024"))
WARMUP     = 2000
WEIGHT_DECAY = 0.05
CHUNK        = int(os.environ.get("YOLAH_TF_CHUNK", str(R.CHUNK_SIZE)))   # positions per chunk (tests: small)
SAVE_CHUNKS  = int(os.environ.get("YOLAH_TF_SAVE_CHUNKS", "4"))   # checkpoint every 4 chunks per rank
POLICY_DIM = 128
VALUE_FC   = 256


# ── Network ─────────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1  = nn.LayerNorm(d)
        self.qkv  = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2  = nn.LayerNorm(d)
        self.fc1  = nn.Linear(d, 4 * d)
        self.fc2  = nn.Linear(4 * d, d)
        self.bias = nn.Parameter(torch.zeros(h, 64, 64))   # learned bias per head, per (square, square)

    def forward(self, x):                                  # x: (B, 64, d)
        d = x.shape[-1]
        q, k, v = self.qkv(self.ln1(x)).reshape(-1, 64, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias.type_as(q))   # type_as: traced dynamically (fp16 in C++)
        x = x + self.proj(a.transpose(1, 2).reshape(-1, 64, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TransformerNet(nn.Module):
    def __init__(self, dim=DIM, heads=HEADS, layers=LAYERS):
        super().__init__()
        self.config = {"arch": "transformer", "dim": dim, "heads": heads, "layers": layers}
        self.embed  = nn.Linear(4, dim)
        self.pos    = nn.Parameter(torch.randn(64, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.ln_f   = nn.LayerNorm(dim)
        self.value_fc1 = nn.Linear(2 * dim, VALUE_FC)
        self.value_fc2 = nn.Linear(VALUE_FC, 1)
        self.pol_from  = nn.Linear(dim, POLICY_DIM)
        self.pol_to    = nn.Linear(dim, POLICY_DIM)
        self.apply(self._init)
        for b in self.blocks:                              # GPT-2 style: residual outputs scaled down
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.normal_(b.fc2.weight,  std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):                                  # x: (B, 4, 8, 8), any memory format
        t = x.reshape(-1, 4, 64).transpose(1, 2)           # (B, 64, 4): token i = cell i = square 63 - i
        h = self.embed(t) + self.pos
        for b in self.blocks:
            h = b(h)
        h = self.ln_f(h)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1)
        v = torch.tanh(self.value_fc2(torch.relu(self.value_fc1(pooled)))).squeeze(-1)
        s = self.pol_from(h) @ self.pol_to(h).transpose(1, 2) * (1.0 / math.sqrt(POLICY_DIM))  # (B, 64 cells, 64 cells)
        p = s.flip(1).flip(2).reshape(-1, 64 * 64)         # cell order → square order: index from*64 + to
        return v, p

    def clip(self):
        pass


class ResNetFromTo(R.Net):
    """
    The ResNet 256x30 (same trunk, same value head) with the transformer's
    from-to policy head instead of its 2-channel one: two 1x1 convolutions
    give every square a 128-number "from" vector q and "to" vector k, and
    logit(from, to) = q_from · k_to / √128. It separates the effect of the
    policy head from the effect of attention. GPU (.ts) only: the C++ CPU
    backend knows only the original head.
    """
    def __init__(self, channels=256, nb_blocks=30):
        super().__init__(channels=channels, nb_blocks=nb_blocks)
        self.config = {"arch": "resnet_ft", "channels": channels, "nb_blocks": nb_blocks}
        del self.policy_conv, self.policy_bn, self.policy_fc
        self.pol_from = nn.Conv2d(channels, POLICY_DIM, kernel_size=1)
        self.pol_to   = nn.Conv2d(channels, POLICY_DIM, kernel_size=1)

    def heads(self, x):                                    # x: (B, C, 8, 8), relu'd trunk
        v = torch.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = torch.tanh(self.value_fc2(torch.relu(self.value_fc1(v)))).squeeze(-1)
        q = self.pol_from(x).reshape(-1, POLICY_DIM, 64)   # cell i = square 63 - i (as the planes)
        k = self.pol_to(x).reshape(-1, POLICY_DIM, 64)
        s = q.transpose(1, 2) @ k * (1.0 / math.sqrt(POLICY_DIM))
        return v, s.flip(1).flip(2).reshape(-1, 64 * 64)


class ResNetPlain(R.Net):
    """The ResNet 256x30 unchanged, trained with THIS script's recipe (fair baseline)."""
    def __init__(self, channels=256, nb_blocks=30):
        super().__init__(channels=channels, nb_blocks=nb_blocks)
        self.config = {"arch": "resnet", "channels": channels, "nb_blocks": nb_blocks}


ARCH = os.environ.get("YOLAH_TF_ARCH", "transformer")   # transformer | resnet_ft | resnet


def make_net(cfg=None):
    cfg = dict(cfg or {"arch": ARCH})
    arch = cfg.pop("arch", "transformer")
    if arch == "transformer":
        return TransformerNet(**{"dim": DIM, "heads": HEADS, "layers": LAYERS, **cfg})
    if arch == "resnet_ft":
        return ResNetFromTo(**cfg)
    if arch == "resnet":
        return ResNetPlain(**cfg)
    raise ValueError(f"unknown arch {arch}")


def load_net(path):
    ck = torch.load(path, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    net = make_net(ck.get("config") if "model" in ck else None)
    net.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()})
    return net


# ── Loaders: either cache, and an exact resume ──────────────────────────────

class _PlanesOfNNUE:
    """Rows of an NNUE cache (N, 193: black, white, holes, turn; cell c = square
    63 - c) seen as the ResNet's (N, 4, 8, 8) planes: the same cells, so
    positions[a:b] gives what preprocess.py would have written."""
    def __init__(self, inputs):
        self.inputs = inputs
        self.shape = (inputs.shape[0], 4, 8, 8)

    def __getitem__(self, sl):
        a = np.asarray(self.inputs[sl])
        out = np.empty((len(a), 4, 8, 8), dtype=np.uint8)
        out[:, :3] = a[:, :192].reshape(-1, 3, 8, 8)
        out[:, 3] = a[:, 192, None, None]
        return out


class CacheLoader(R.ChunkedShuffleLoader):
    """The ResNet loader, which also reads an NNUE cache (preprocess_nnue.py)
    once policies.i16 has been added to it by policies_from_nnue_cache.py."""
    def _open(self):
        with open(os.path.join(self.cache_dir, "meta.json")) as f:
            meta = json.load(f)
        if "positions" in meta:
            return super()._open()
        n = int(meta["n_positions"])
        pol = os.path.join(self.cache_dir, "policies.i16")
        if not os.path.isfile(pol):
            raise FileNotFoundError(f"{pol}: run policies_from_nnue_cache.py {self.cache_dir} first")
        inputs = np.memmap(os.path.join(self.cache_dir, meta["inputs"]["path"]), dtype=np.uint8, mode='r',
                           shape=tuple(meta["inputs"]["shape"]))
        self.positions = _PlanesOfNNUE(inputs)
        self.values = np.memmap(os.path.join(self.cache_dir, meta["values"]["path"]), dtype=np.int8, mode='r', shape=(n,))
        self.policies = np.memmap(pol, dtype=np.int16, mode='r', shape=(n,))


class ResumableLoader(CacheLoader):
    skip_chunks = 0

    def __iter__(self):
        import queue as queue_mod, threading
        order = list(range(len(self.my_chunks)))
        if self.shuffle:
            random.Random(self.epoch).shuffle(order)       # same order as the ResNet loader
        order = order[self.skip_chunks:]
        q = queue_mod.Queue(maxsize=self.queue_depth)
        t = threading.Thread(target=self._producer, args=(order, q), daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        t.join()


# ── Metrics (the ResNet script's) ───────────────────────────────────────────

class Meter:
    def __init__(self):
        self.n = self.v = self.p = self.sign = self.pc = self.pt = 0.0

    def add(self, v_pred, p_logits, v_t, p_t, v_loss, p_loss):
        bs = len(v_t)
        self.n += bs
        self.v += v_loss.item() * bs
        self.p += p_loss.item() * bs
        self.sign += (torch.sign(v_pred) == torch.sign(v_t)).sum().item()
        mask = p_t != R.POLICY_IGNORE_INDEX
        self.pc += ((p_logits.argmax(1) == p_t) & mask).sum().item()
        self.pt += mask.sum().item()

    def text(self):
        n = max(self.n, 1)
        return (f"value_mse: {self.v / n:.4f} value_sign_acc: {self.sign / n:.4f} "
                f"policy_ce: {self.p / n:.4f} policy_acc: {self.pc / max(self.pt, 1):.4f}")


def losses(v_pred, p_logits, v_t, p_t):
    v_loss = F.mse_loss(v_pred.float(), v_t)
    p_loss = F.cross_entropy(p_logits.float(), p_t, ignore_index=R.POLICY_IGNORE_INDEX)
    return v_loss, p_loss


def evaluate(model, loader, device, amp_dtype, max_batches=None):
    model.eval()
    m = Meter()
    with torch.no_grad():
        for i, (X, v_t, p_t) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            X, v_t, p_t = X.to(device, non_blocking=True), v_t.to(device), p_t.to(device)
            with torch.autocast('cuda', dtype=amp_dtype):
                v_pred, p_logits = model(X)
            m.add(v_pred.float(), p_logits, v_t, p_t, *losses(v_pred, p_logits, v_t, p_t))
    model.train()
    return m


# ── Training (one process per GPU, DDP) ─────────────────────────────────────

def train_worker(rank, world, args):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    os.environ.setdefault("MASTER_ADDR", "localhost")
    # one port per job: two jobs on one node (crn25 has several GPUs) must not share the rendezvous
    os.environ.setdefault("MASTER_PORT", str(20000 + int(os.environ.get("SLURM_JOB_ID", "45439")) % 40000))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(rank)[0] >= 8 else torch.float16

    with open(os.path.join(args.cache, "meta.json")) as f:
        n_total = int(json.load(f)["n_positions"])
    n_train = int(0.95 * n_total)
    train_loader = ResumableLoader(args.cache, 0, n_train, BATCH, rank, world, chunk_size=CHUNK, shuffle=True)
    val_loader   = CacheLoader(args.cache, n_train, n_total, BATCH, rank, world, chunk_size=CHUNK, shuffle=False)
    per_epoch = len(train_loader)
    total = NB_EPOCHS * per_epoch

    net = make_net().to(device).to(memory_format=torch.channels_last)
    decay    = [p for n, p in net.named_parameters() if p.dim() >= 2 and n != "pos" and not n.endswith(".bias")]
    no_decay = [p for n, p in net.named_parameters() if not (p.dim() >= 2 and n != "pos" and not n.endswith(".bias"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.95), fused=True)
    step = 0
    ck_path = os.path.join(args.out, "transformer_last.pt")
    if os.path.isfile(ck_path):
        ck = torch.load(ck_path, map_location="cpu")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    if rank == 0:
        print(f"dataset {n_total:,} positions, train {n_train:,}; {world} GPU(s), batch {BATCH}/GPU; "
              f"{per_epoch:,} steps/epoch, {NB_EPOCHS} epochs; params {sum(p.numel() for p in net.parameters()):,}; "
              f"start at step {step:,}", flush=True)

    model = torch.compile(DDP(net, device_ids=[rank], gradient_as_bucket_view=True, static_graph=True))

    def lr_at(s):
        if s < WARMUP:
            return LR * (s + 1) / WARMUP
        x = (s - WARMUP) / max(1, total - WARMUP)
        return LR * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, x))))

    def save(tag=None):
        if rank != 0:
            return
        ck = {"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "config": net.config}
        tmp = ck_path + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, ck_path)
        if tag is not None:
            torch.save({"model": net.state_dict(), "config": net.config, "step": step},
                       os.path.join(args.out, f"transformer.{tag}.pt"))

    bpc = train_loader.batches_per_chunk
    gc.disable()
    while step < total:
        epoch = step // per_epoch
        in_epoch = step % per_epoch
        assert in_epoch % bpc == 0, "checkpoints are taken at chunk boundaries"
        train_loader.set_epoch(epoch)
        train_loader.skip_chunks = in_epoch // bpc
        meter, t0 = Meter(), time.time()
        for X, v_t, p_t in train_loader:
            X, v_t, p_t = X.to(device, non_blocking=True), v_t.to(device, non_blocking=True), p_t.to(device, non_blocking=True)
            X = X.contiguous(memory_format=torch.channels_last)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast('cuda', dtype=amp_dtype):
                v_pred, p_logits = model(X)
            v_loss, p_loss = losses(v_pred, p_logits, v_t, p_t)
            opt.zero_grad(set_to_none=True)
            (v_loss + p_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                meter.add(v_pred.detach().float(), p_logits.detach(), v_t, p_t, v_loss.detach(), p_loss.detach())
            if step % 5000 == 0 and rank == 0:
                rate = 5000 * BATCH * world / (time.time() - t0)
                t0 = time.time()
                print(f"step {step:,}/{total:,} epoch {step / per_epoch:.3f} lr {lr_at(step):.2e} "
                      f"train(sampled) {meter.text()} | {rate:,.0f} pos/s", flush=True)
                meter = Meter()
                gc.collect()
            if step % (bpc * SAVE_CHUNKS) == 0:
                save()
            if step >= total:
                break
        # end of epoch: validation on the whole 5 % (every rank its share), checkpoint
        m = evaluate(model, val_loader, device, amp_dtype)
        t = torch.tensor([m.n, m.v, m.p, m.sign, m.pc, m.pt], device=device, dtype=torch.float64)
        dist.all_reduce(t)
        m.n, m.v, m.p, m.sign, m.pc, m.pt = t.tolist()
        if rank == 0 and m.n > 0:
            print(f"epoch {step / per_epoch:.2f} val {m.text()}", flush=True)
            with open(os.path.join(args.out, "val.jsonl"), "a") as f:
                f.write(json.dumps({"epoch": step / per_epoch, "step": step, "value_mse": m.v / m.n,
                                    "policy_ce": m.p / m.n, "policy_acc": m.pc / m.pt,
                                    "value_sign_acc": m.sign / m.n}) + "\n")
        save(tag=f"e{round(step / per_epoch)}")
    if rank == 0:
        torch.save({"model": net.state_dict(), "config": net.config, "step": step},
                   os.path.join(args.out, "transformer.pt"))
        print("done", flush=True)
    dist.destroy_process_group()


# ── Evaluation of a checkpoint (transformer or ResNet) on the validation set ─

def cmd_eval(args):
    device = torch.device("cuda")
    if args.resnet:
        sd = torch.load(args.model, map_location="cpu")
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        net, _ = R.net_from_state_dict(sd)
    else:
        net = load_net(args.model)
    net = net.to(device).eval()
    with open(os.path.join(args.cache, "meta.json")) as f:
        n_total = int(json.load(f)["n_positions"])
    n_train = int(0.95 * n_total)
    loader = CacheLoader(args.cache, n_train, n_total, BATCH, 0, 1, chunk_size=CHUNK, shuffle=False)
    max_b = None if args.chunks <= 0 else args.chunks * loader.batches_per_chunk
    m = evaluate(net, loader, device, torch.bfloat16, max_b)
    print(f"{os.path.basename(args.model)}: {int(m.n):,} validation positions: {m.text()}")


# ── Export for the C++ MCTS (libtorch backend) ──────────────────────────────

def cmd_export(args):
    net = load_net(args.model).eval()
    base = os.path.splitext(args.model)[0]
    with torch.no_grad():
        traced = torch.jit.trace(net, torch.zeros(1, 4, 8, 8))
    traced.save(base + ".ts")
    print(f"wrote {base}.ts")
    import resnet_export as E
    E.export_test_vectors(net, base + ".test.bin", 32)


def cmd_selftest(args):
    torch.manual_seed(0)
    net = TransformerNet().eval()
    res = R.Net()
    print(f"transformer {sum(p.numel() for p in net.parameters()):,} params, "
          f"ResNet 256x30 {sum(p.numel() for p in res.parameters()):,}")
    # the action index must follow Yolah's squares: compare with a direct square-order computation
    x = torch.randint(0, 2, (3, 4, 8, 8)).float()
    v, p = net(x)
    assert v.shape == (3,) and p.shape == (3, 4096)
    with torch.no_grad():
        t = x.reshape(-1, 4, 64).transpose(1, 2)
        h = net.embed(t) + net.pos
        for b in net.blocks:
            h = b(h)
        h = net.ln_f(h)
        q, k = net.pol_from(h), net.pol_to(h)
        for (fr, to) in [(0, 63), (12, 5), (63, 0), (40, 41)]:
            direct = (q[:, 63 - fr] * k[:, 63 - to]).sum(-1) / math.sqrt(POLICY_DIM)
            assert torch.allclose(direct, p[:, fr * 64 + to], atol=1e-5), (fr, to)
    # channels_last input gives the same result
    v2, p2 = net(x.contiguous(memory_format=torch.channels_last))
    assert torch.allclose(p, p2, atol=1e-6)
    rf = ResNetFromTo().eval()
    print(f"ResNet 256x30 + from-to head {sum(p.numel() for p in rf.parameters()):,} params")
    with torch.no_grad():
        vr, pr = rf(x)
        tr = torch.relu(rf.output_bn(rf.res_blocks(rf.input_conv(x))))
        qf, kt = rf.pol_from(tr).flatten(2), rf.pol_to(tr).flatten(2)
        for (fr, to) in [(0, 63), (12, 5), (63, 0)]:
            direct = (qf[:, :, 63 - fr] * kt[:, :, 63 - to]).sum(-1) / math.sqrt(POLICY_DIM)
            assert torch.allclose(direct, pr[:, fr * 64 + to], atol=1e-4), ("resnet_ft", fr, to)
        trf = torch.jit.trace(rf, torch.zeros(1, 4, 8, 8))
        assert torch.allclose(trf(x)[1], pr, atol=1e-4)
    traced = torch.jit.trace(net, torch.zeros(1, 4, 8, 8))
    vt, pt = traced(x)
    assert torch.allclose(vt, v, atol=1e-5) and torch.allclose(pt, p, atol=1e-4), "trace"
    if torch.cuda.is_available():
        import copy
        th = torch.jit.trace(copy.deepcopy(net), torch.zeros(1, 4, 8, 8)).cuda().half()   # (a trace shares the net's parameters)
        for bs in (1, 64):
            xb = torch.randint(0, 2, (bs, 4, 8, 8)).float()
            vf, pf = net(xb)
            vh, ph = th(xb.cuda().half())
            print(f"fp16 batch {bs}: max |dv| {(vh.float().cpu() - vf).abs().max():.4f}, "
                  f"max |dlogit| {(ph.float().cpu() - pf).abs().max():.4f}")
        xb = torch.zeros(64, 4, 8, 8).cuda().half()
        for _ in range(5):
            th(xb)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(50):
            th(xb)
        torch.cuda.synchronize()
        print(f"fp16 batch 64 forward: {(time.time() - t0) / 50 * 1e3:.2f} ms")
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "eval", "export", "selftest"])
    ap.add_argument("--cache", default=os.environ.get("YOLAH_CACHE_DIR", "/cache"))
    ap.add_argument("--out", default="/work")
    ap.add_argument("--model")
    ap.add_argument("--resnet", action="store_true", help="eval: the checkpoint is the ResNet")
    ap.add_argument("--chunks", type=int, default=0, help="eval: only the first N chunks of the validation set (0: all)")
    args = ap.parse_args()
    if args.cmd == "train":
        os.makedirs(args.out, exist_ok=True)
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        world = len([c for c in cvd.split(",") if c.strip()]) if cvd else torch.cuda.device_count()
        torch.multiprocessing.spawn(train_worker, args=(world, args), nprocs=world)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "export":
        cmd_export(args)
    else:
        cmd_selftest(args)


if __name__ == "__main__":
    main()
