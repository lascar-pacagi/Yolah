"""
nnue193x1024x64x32x1_distill.py — distil the value/policy ResNet into the
193 → 1024 → 64 → 32 → 1 NNUE (same architecture and cache as
nnue193x1024x64x32x1.py; see distill_common.py for the method).

Pipeline (nnue193x1024x64x32x1_distill.sh does it on the cluster):
    1. preprocess_nnue.py            → /cache (inputs.u8, values.i8, meta.json)
    2. distill_teacher_labels.py     → /cache/teacher_values.f16 (+ done/meta)
    3. this script                   → /mnt/nnue_193x1024x64x32x1_distill.<epoch>.pt

The student starts from the weights of the last non-distilled run
(YOLAH_INIT_MODEL, default /mnt/nnue_193x1024x64x32x1.pt) and is trained on

    L = α · MSE(v_s, v_t) + (1 - α) · MSE(v_s, z)        α = YOLAH_DISTILL_ALPHA

Every epoch prints, for the training shard and the validation shard, the
student against the outcome z ("training set"), the student against the
teacher targets ("training set augmented") and the teacher itself against z.

Env:
    YOLAH_CACHE_DIR       cache directory                  (default /cache)
    YOLAH_INIT_MODEL      starting weights                  (default /mnt/nnue_193x1024x64x32x1.pt)
    YOLAH_NB_EPOCHS       epochs                            (default 20)
    YOLAH_DISTILL_ALPHA   weight of the teacher term        (default 0.8)
    YOLAH_LR              initial learning rate             (default 3e-4; the student is fine-tuned)
    YOLAH_BATCH_SIZE      per-GPU batch size                (default 1024)
    YOLAH_CHUNK_SIZE      loader chunk (positions)          (default 4194304; smaller for tiny caches)
    YOLAH_DDP_PORT        rendezvous port                   (default 65434)

If a checkpoint of THIS run exists (/mnt/nnue_193x1024x64x32x1_distill.pt) it
takes precedence over YOLAH_INIT_MODEL (resume).
"""
from tqdm import tqdm
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import gc
import numpy as np

from distill_common import (DistillLoader, Metrics, labelled_ranges, open_teacher_labels,
                            read_meta, TEACHER_META)
from nnue193x1024x64x32x1 import Net, INPUT_SIZE          # the student, unchanged

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

# ── configuration ──────────────────────────────────────────────────────────
NB_EPOCHS  = int(os.environ.get("YOLAH_NB_EPOCHS", "20"))
ALPHA      = float(os.environ.get("YOLAH_DISTILL_ALPHA", "0.8"))
LR         = float(os.environ.get("YOLAH_LR", "3e-4"))
BATCH_SIZE = int(os.environ.get("YOLAH_BATCH_SIZE", "1024"))
CHUNK_SIZE = int(os.environ.get("YOLAH_CHUNK_SIZE", str(2048 * 2048)))
MODEL_PATH = os.environ.get("YOLAH_MODEL_DIR", "/mnt/")
MODEL_NAME = "nnue_193x1024x64x32x1_distill"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt"
INIT_MODEL = os.environ.get("YOLAH_INIT_MODEL", f"{MODEL_PATH}nnue_193x1024x64x32x1.pt")
CACHE_DIR  = os.environ.get("YOLAH_CACHE_DIR", "/cache")


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.environ.get("YOLAH_DDP_PORT", "65434")
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


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
    return inputs, values, teacher, labelled_ranges(cache_dir, n), n


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
                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)
        if self.gpu_id == 0:
            self._save_checkpoint(nb_epochs - 1)


def main(rank, world_size, batch_size, cache_dir):
    ddp_setup(rank, world_size)
    inputs, values, teacher, ranges, n_total = open_cache(cache_dir)
    n_train = int(0.95 * n_total)
    train_loader = DistillLoader(cache_dir, inputs, values, teacher, ranges, 0, n_train, batch_size,
                                 rank, world_size, chunk_size=CHUNK_SIZE, shuffle=True)
    val_loader = DistillLoader(cache_dir, inputs, values, teacher, ranges, n_train, n_total, batch_size,
                               rank, world_size, chunk_size=CHUNK_SIZE, shuffle=False)
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

    net = Net()
    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL, map_location='cpu'))
        src = LAST_MODEL
    elif os.path.isfile(INIT_MODEL):
        net.load_state_dict(torch.load(INIT_MODEL, map_location='cpu'))
        src = INIT_MODEL
    else:
        src = 'random init'
    if rank == 0:
        print(f'Student initialised from: {src}', flush=True)
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
