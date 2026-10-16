"""
nnue_persp_distill.py — distil the value/policy ResNet into the NNUE seen from
the side to move, with one head per game phase (NetPersp, nnue_persp.py; step 1
of doc/nnue_roadmap.qmd).

Same data, loss, metrics and loop as nnue193x1024x64x32x1_distill.py: the
cache of preprocess_nnue.py (193 inputs) and its teacher labels are used as
they are, NetPersp converts each batch to its 192 inputs on the GPU.

    L = α · MSE(v_s, v_t) + (1 - α) · MSE(v_s, z)        α = YOLAH_DISTILL_ALPHA

The student starts from the distilled 193 network (YOLAH_INIT_MODEL): fc1
without its turn row, each head a copy of fc2–fc4 (see
NetPersp.init_from_nnue193). It then gives the old values on every position
with black to move, and learns the rest.

Env (in addition to those of nnue193x1024x64x32x1_distill.py):
    YOLAH_PERSP_K         number of phase heads            (default 8)
    YOLAH_PERSP_THEM      1: fc2 also reads the other view (default 0)
    YOLAH_INIT_MODEL      starting weights, a 193 network OR a NetPersp
                          (default /mnt/nnue_193x1024x64x32x1_distill.pt)
    YOLAH_SPACE_MASK      step 2: the space features fc2 reads (bit i = feature i of
                          nnue/space_features.h; 65535 = all 16, 65278 = without the
                          corrected flood fill); needs CACHE/space.u8 (space_features
                          compute). Default 0: none.
    YOLAH_SPACE_GROUPS    the ablation (doc/nnue_roadmap.qmd): the GROUPS of candidate
                          features fc2 reads (bit g = group g of space_features.h:
                          0 corr, 1 qdist, 2 qfirst, 3 qw, 4 late9, 5 mob, 6 infl,
                          7 connect, 8 encircle; 0x1ff = all); needs CACHE/space_all.u8
                          (space_features compute-all). Not with YOLAH_SPACE_MASK.
    YOLAH_SEED            another shuffling of the training data (default 0), to
                          measure the noise between two otherwise identical runs
    YOLAH_NAME_SUFFIX     appended to the model name (default "_space" with a mask,
                          "_g<groups in hex>" with groups)
After each epoch the validation numbers (overall and per phase) are appended to
/mnt/<model name>.val.jsonl (one JSON object per line), read by ablation.py.
A NetPersp start with fewer space features keeps its values (init_from_persp:
the new fc2 columns start at 0).
The model is saved as /mnt/nnue_persp_k<K>[_them]<suffix>_distill.<epoch>.pt; a
/mnt/nnue_persp_k<K>[_them]<suffix>_distill.pt takes precedence (resume).
"""
from tqdm import tqdm
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import gc
import json
import numpy as np

from distill_common import (DistillLoader, Metrics, labelled_ranges, open_teacher_labels,
                            read_meta, TEACHER_META)
from nnue_persp import NetPersp, CACHE_INPUT_SIZE as INPUT_SIZE, SPACE_SIZE

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

# ── configuration ──────────────────────────────────────────────────────────
NB_EPOCHS  = int(os.environ.get("YOLAH_NB_EPOCHS", "20"))
ALPHA      = float(os.environ.get("YOLAH_DISTILL_ALPHA", "0.8"))
LR         = float(os.environ.get("YOLAH_LR", "3e-4"))
BATCH_SIZE = int(os.environ.get("YOLAH_BATCH_SIZE", "1024"))
CHUNK_SIZE = int(os.environ.get("YOLAH_CHUNK_SIZE", str(2048 * 2048)))
MODEL_PATH = os.environ.get("YOLAH_MODEL_DIR", "/mnt/")
PERSP_K    = int(os.environ.get("YOLAH_PERSP_K", "8"))
PERSP_THEM = os.environ.get("YOLAH_PERSP_THEM", "0") == "1"
SPACE_MASK = int(os.environ.get("YOLAH_SPACE_MASK", "0"), 0)
SPACE_GROUPS = int(os.environ.get("YOLAH_SPACE_GROUPS", "0"), 0)
SEED = int(os.environ.get("YOLAH_SEED", "0"))
if SPACE_MASK and SPACE_GROUPS:
    raise SystemExit("YOLAH_SPACE_MASK and YOLAH_SPACE_GROUPS are exclusive")
NAME_SUFFIX = os.environ.get("YOLAH_NAME_SUFFIX", "_space" if SPACE_MASK else
                             f"_g{SPACE_GROUPS:03x}" if SPACE_GROUPS else "")
MODEL_NAME = f"nnue_persp_k{PERSP_K}{'_them' if PERSP_THEM else ''}{NAME_SUFFIX}_distill"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt"
INIT_MODEL = os.environ.get("YOLAH_INIT_MODEL", f"{MODEL_PATH}nnue_193x1024x64x32x1_distill.pt")
CACHE_DIR  = os.environ.get("YOLAH_CACHE_DIR", "/cache")
PHASES     = ["opening 40-56", "middle 20-39", "endgame 10-19", "endgame 0-9"]   # free squares


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.environ.get("YOLAH_DDP_PORT", "65434")
    # One GPU: gloo. NCCL brings nothing without a second GPU, and it needs NVML,
    # which fails on nodes whose driver was updated without a reboot
    # ("nvmlInit_v2() failed: Driver/library version mismatch").
    backend = os.environ.get("YOLAH_DDP_BACKEND", "nccl" if world_size > 1 else "gloo")
    init_process_group(backend=backend, rank=rank, world_size=world_size)


def open_cache(cache_dir):
    """memmaps (inputs, values, teacher) + the labelled ranges + n_positions."""
    meta = read_meta(cache_dir)
    n = int(meta["n_positions"])
    if meta["inputs"]["shape"][1] != INPUT_SIZE:
        raise ValueError(f"cache input width {meta['inputs']['shape'][1]} != {INPUT_SIZE}")
    inputs = np.memmap(os.path.join(cache_dir, meta["inputs"]["path"]), dtype=np.uint8, mode='r',
                       shape=tuple(meta["inputs"]["shape"]))
    values = np.memmap(os.path.join(cache_dir, meta["values"]["path"]), dtype=np.int8, mode='r', shape=(n,))
    teacher, _ = open_teacher_labels(cache_dir, n, mode='r')
    space, mask = None, SPACE_MASK
    if SPACE_MASK or SPACE_GROUPS:
        meta_name = "space_meta.json" if SPACE_MASK else "space_all_meta.json"
        with open(os.path.join(cache_dir, meta_name)) as f:
            sm = json.load(f)
        width = SPACE_SIZE if SPACE_MASK else len(sm["groups"])
        if sm["n_positions"] != n or sm["shape"][1] != width:
            raise ValueError(f"{sm['path']}: {sm['shape']} does not match the cache ({n} positions × {width})")
        space = np.memmap(os.path.join(cache_dir, sm["path"]), dtype=np.uint8, mode='r', shape=(n, width))
        if SPACE_GROUPS:
            mask = sum(1 << c for c, g in enumerate(sm["groups"]) if SPACE_GROUPS >> g & 1)
    return inputs, values, teacher, labelled_ranges(cache_dir, n), n, space, mask


class TrainerDDP:
    """Owns one GPU's model replica, optimizer, and the train/validate loops."""

    def __init__(self, gpu_id, model, train_loader, val_loader, save_every=1):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.save_every = save_every
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=NB_EPOCHS,
                                                                    eta_min=LR / 30)
        major, _ = torch.cuda.get_device_capability(gpu_id)
        self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        self.model = DDP(self.model, device_ids=[gpu_id], gradient_as_bucket_view=True, static_graph=True)
        self.model = torch.compile(self.model)
        self.stream = torch.cuda.Stream(device=gpu_id)

    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(), f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _log_val(self, epoch, m):
        """Validation numbers of the epoch → /mnt/<model>.val.jsonl (and the per-phase mse on stdout)."""
        sse, cnt = (t.cpu().tolist() for t in m.phases)
        phases = {name: (a / b if b else None) for name, a, b in zip(PHASES, sse, cnt)}
        print(f'epoch {epoch + 1} val   [vs teacher] mse per phase: '
              + '  '.join(f'{k} {v:.4f}' for k, v in phases.items() if v is not None), flush=True)
        n = max(m.n, 1)
        rec = dict(epoch=epoch + 1, model=MODEL_NAME, groups=SPACE_GROUPS, seed=SEED, mask=hex(self.model.module.space_mask),
                   n=m.n, loss=m.loss / n, mse_teacher=m.mse_t / n, mse_outcome=m.mse_z / n,
                   sign_teacher=m.sign_t / n, phases=phases, phase_n=dict(zip(PHASES, cnt)))
        with open(f"{MODEL_PATH}{MODEL_NAME}.val.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _h2d_async(self, batch_cpu):
        with torch.cuda.stream(self.stream):
            return tuple(t.to(self.gpu_id, non_blocking=True) for t in batch_cpu)

    @staticmethod
    def _loss(v_s, z, v_t):
        # Distillation: pull towards the teacher value, keep the outcome as a
        # (down-weighted) anchor so the student cannot drift with the teacher's
        # systematic errors.
        return ALPHA * nn.functional.mse_loss(v_s, v_t) + (1.0 - ALPHA) * nn.functional.mse_loss(v_s, z)

    def _epoch(self, loader, train, epoch):
        """One pass over `loader`; returns the accumulated Metrics."""
        self.model.train(train)
        metrics = Metrics()
        phase_sse = torch.zeros(len(PHASES), device=self.gpu_id)
        phase_n = torch.zeros(len(PHASES), device=self.gpu_id)
        metrics.phases = (phase_sse, phase_n)
        it = 0
        loader_iter = iter(loader)
        cpu_cur = next(loader_iter, None)
        if cpu_cur is None:
            return metrics
        gpu_cur = self._h2d_async(cpu_cur)
        pbar = tqdm(total=len(loader), disable=(self.gpu_id != 0))
        with torch.set_grad_enabled(train):
            while True:
                torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)
                X, z, v_t = gpu_cur
                cpu_next = next(loader_iter, None)
                if cpu_next is not None:
                    gpu_next = self._h2d_async(cpu_next)

                if train:
                    self.optimizer.zero_grad()
                with torch.autocast('cuda', dtype=self.amp_dtype):
                    v_s = self.model(X)
                    loss = self._loss(v_s, z, v_t)
                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.model.module.clip()            # int8 trunk weight clamp
                metrics.update(v_s.float(), z, v_t, loss.item())
                if not train:                           # mse vs the teacher per phase (free squares)
                    free = 56 - X[:, 128:192].sum(dim=1)
                    ph = (free < 40).long() + (free < 20).long() + (free < 10).long()
                    phase_sse.index_add_(0, ph, (v_s.float() - v_t) ** 2)
                    phase_n.index_add_(0, ph, torch.ones_like(v_t))

                it += 1
                if it % 1000 == 0:
                    gc.collect()
                pbar.update(1)
                if cpu_next is None:
                    break
                cpu_cur, gpu_cur = cpu_next, gpu_next
        pbar.close()
        return metrics

    def train(self, nb_epochs):
        for epoch in range(nb_epochs):
            self.train_loader.set_epoch(epoch)
            train_metrics = self._epoch(self.train_loader, True, epoch)
            self.scheduler.step()
            val_metrics = self._epoch(self.val_loader, False, epoch)
            if self.gpu_id == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(train_metrics.report(f'epoch {epoch + 1} train'), flush=True)
                print(val_metrics.report(f'epoch {epoch + 1} val  '), flush=True)
                print(f'epoch {epoch + 1} lr: {lr:.6f} alpha: {ALPHA}', flush=True)
                self._log_val(epoch, val_metrics)
                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)
        if self.gpu_id == 0:
            self._save_checkpoint(nb_epochs - 1)


def main(rank, world_size, batch_size, cache_dir):
    ddp_setup(rank, world_size)
    inputs, values, teacher, ranges, n_total, space, space_mask = open_cache(cache_dir)
    n_train = int(0.95 * n_total)
    train_loader = DistillLoader(cache_dir, inputs, values, teacher, ranges, 0, n_train, batch_size,
                                 rank, world_size, chunk_size=CHUNK_SIZE, shuffle=True, extra=space,
                                 seed=SEED)
    val_loader = DistillLoader(cache_dir, inputs, values, teacher, ranges, n_train, n_total, batch_size,
                               rank, world_size, chunk_size=CHUNK_SIZE, shuffle=False, extra=space)
    if rank == 0:
        labelled = sum(hi - lo for lo, hi in ranges)
        print(f'Dataset: {n_total:,} positions, {labelled:,} labelled by the teacher '
              f'({read_meta(cache_dir).get("n_games", "?")} games)', flush=True)
        print(f'Teacher: {open(os.path.join(cache_dir, TEACHER_META)).read().strip()}', flush=True)
        print(f'Batches/rank/epoch: train {len(train_loader):,}  val {len(val_loader):,}  '
              f'(chunk_size={CHUNK_SIZE:,}, batch={batch_size}, alpha={ALPHA}, lr={LR})', flush=True)
        if len(train_loader) == 0:
            print('WARNING: no complete labelled training chunk — label more positions or lower '
                  'YOLAH_CHUNK_SIZE', flush=True)

    net = NetPersp(PERSP_K, PERSP_THEM, space_mask=space_mask)
    if os.path.isfile(LAST_MODEL):
        net.init_from_persp(torch.load(LAST_MODEL, map_location='cpu'))
        src = LAST_MODEL
    elif os.path.isfile(INIT_MODEL):
        sd = torch.load(INIT_MODEL, map_location='cpu')
        if "w3" in sd:
            net.init_from_persp(sd)
        else:
            net.init_from_nnue193(sd)
        src = INIT_MODEL
    else:
        src = 'random init'
    if rank == 0:
        print(f'Student initialised from: {src} (K={PERSP_K}, them={PERSP_THEM}, '
              f'space mask={space_mask:#x}, groups={SPACE_GROUPS:#x}, {net.n_space} space features)', flush=True)
        print(net, flush=True)
        print(f'Parameters: {sum(p.numel() for p in net.parameters()):,}', flush=True)

    gc.disable()
    trainer = TrainerDDP(rank, net, train_loader, val_loader)
    trainer.train(NB_EPOCHS)
    destroy_process_group()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    world_size = (len([x for x in cvd.split(",") if x.strip()]) if cvd else torch.cuda.device_count())
    print(world_size, flush=True)
    mp.spawn(main, args=(world_size, BATCH_SIZE, CACHE_DIR), nprocs=world_size)
