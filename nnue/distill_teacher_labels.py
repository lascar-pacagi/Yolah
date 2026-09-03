"""
distill_teacher_labels.py — label a position cache with the ResNet teacher.

Reads   <cache_dir>/<boards>   (N, 193) uint8 board rows — inputs.u8 of a
                               preprocess_nnue.py cache, or boards.u8 written
                               by distill_boards_for_features.py next to a
                               preprocess_features.py cache
Writes  <cache_dir>/teacher_values.f16   (N,) float16  teacher value v_t for
                                          the player to move
        <cache_dir>/teacher_done.u8      one byte per LABEL_CHUNK positions,
                                          1 once that chunk is written
        <cache_dir>/teacher_meta.json    what produced the labels

Usage (from nnue/, the teacher module imports ../server/yolah.py):

    python3 distill_teacher_labels.py <cache_dir> [--boards inputs.u8]
        [--model /mnt/cnn_resnet_256x30_value_policy.pt] [--batch 2048]
        [--tta] [--start 0 --end N] [--gpus 0,1]

Env:  YOLAH_TEACHER_MODEL   default model path
      CUDA_VISIBLE_DEVICES  honoured like the trainers (one process per GPU)

Resumable: chunks already flagged in teacher_done.u8 are skipped, so an
interrupted job (SLURM time limit) is simply resubmitted. Several GPUs work
on disjoint chunks (rank r takes chunks r, r+world, ...). --start/--end
restrict the labelled range (e.g. label a prefix to start training early —
the distillation trainers use only the labelled chunks).

Cost: one forward of the 256×30 ResNet per position (×8 with --tta). On an
RTX 3080 in fp16 that is ~6-8k positions/s, i.e. ~40 h per 10⁹ positions.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp

from distill_common import (BOARD_ROW, LABEL_CHUNK, TEACHER_META, load_teacher,
                            teacher_values, open_teacher_labels, read_meta)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir")
    ap.add_argument("--boards", default=None,
                    help="board file inside cache_dir (default: inputs.u8 if present, else boards.u8)")
    ap.add_argument("--model", default=os.environ.get("YOLAH_TEACHER_MODEL",
                                                      "/mnt/cnn_resnet_256x30_value_policy.pt"))
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--tta", action="store_true", help="average over the 8 board symmetries")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--gpus", default=None, help="comma separated GPU indices (default: all visible)")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (slow; for tests)")
    return ap.parse_args()


def worker(rank, world_size, args, n_positions, boards_path, devices):
    device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{devices[rank]}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    net = load_teacher(args.model, device)

    boards = np.memmap(boards_path, dtype=np.uint8, mode="r", shape=(n_positions, BOARD_ROW))
    values, done = open_teacher_labels(args.cache_dir, n_positions, mode="r+")

    end = n_positions if args.end is None else min(args.end, n_positions)
    first_chunk, last_chunk = args.start // LABEL_CHUNK, (end - 1) // LABEL_CHUNK
    my_chunks = [c for c in range(first_chunk, last_chunk + 1) if c % world_size == rank and not done[c]]
    if rank == 0:
        total = sum(1 for c in range(first_chunk, last_chunk + 1) if not done[c])
        print(f"{total} chunks of {LABEL_CHUNK:,} positions to label on {world_size} device(s), "
              f"model {args.model}, tta={args.tta}", flush=True)

    t0 = time.time()
    n_done = 0
    for k, c in enumerate(my_chunks):
        lo, hi = c * LABEL_CHUNK, min((c + 1) * LABEL_CHUNK, n_positions)
        rows = np.array(boards[lo:hi])                      # one sequential read
        out = np.empty(hi - lo, dtype=np.float16)
        for b in range(0, hi - lo, args.batch):
            v = teacher_values(net, rows[b:b + args.batch], device, tta=args.tta)
            out[b:b + args.batch] = v.numpy().astype(np.float16)
        values[lo:hi] = out
        values.flush()
        done[c] = 1
        done.flush()
        n_done += hi - lo
        if rank == 0 and (k % 4 == 0 or k == len(my_chunks) - 1):
            rate = n_done / max(time.time() - t0, 1e-6)
            remaining = sum(min((cc + 1) * LABEL_CHUNK, n_positions) - cc * LABEL_CHUNK for cc in my_chunks[k + 1:])
            print(f"  rank 0: chunk {c} done, {rate:,.0f} pos/s, ETA {remaining / max(rate, 1e-6) / 3600:.2f} h",
                  flush=True)
    if rank == 0:
        print(f"rank 0 finished {n_done:,} positions in {time.time() - t0:.0f}s", flush=True)


def main():
    args = parse_args()
    meta = read_meta(args.cache_dir)
    n_positions = int(meta["n_positions"])
    if args.boards is None:
        args.boards = "inputs.u8" if os.path.isfile(os.path.join(args.cache_dir, "inputs.u8")) else "boards.u8"
    boards_path = os.path.join(args.cache_dir, args.boards)
    if not os.path.isfile(boards_path):
        sys.exit(f"ERROR: {boards_path} not found (for a features cache run distill_boards_for_features.py first)")
    if os.path.getsize(boards_path) != n_positions * BOARD_ROW:
        sys.exit(f"ERROR: {boards_path} has {os.path.getsize(boards_path)} bytes, expected {n_positions * BOARD_ROW}")
    if not os.path.isfile(args.model):
        sys.exit(f"ERROR: teacher checkpoint {args.model} not found")

    # Create the label files once (sparse), never truncate existing ones.
    n_chunks = (n_positions + LABEL_CHUNK - 1) // LABEL_CHUNK
    for name, size in ((os.path.join(args.cache_dir, "teacher_values.f16"), n_positions * 2),
                       (os.path.join(args.cache_dir, "teacher_done.u8"), n_chunks)):
        if not os.path.isfile(name) or os.path.getsize(name) != size:
            with open(name, "wb") as f:
                f.truncate(size)
    meta_path = os.path.join(args.cache_dir, TEACHER_META)
    if os.path.isfile(meta_path):
        old = json.load(open(meta_path))
        if old.get("model") != os.path.basename(args.model) or old.get("tta") != args.tta:
            sys.exit(f"ERROR: {meta_path} was produced with model={old.get('model')} tta={old.get('tta')}; "
                     f"delete teacher_* to relabel with different settings")
    with open(meta_path, "w") as f:
        json.dump({"model": os.path.basename(args.model), "tta": args.tta, "chunk": LABEL_CHUNK,
                   "n_positions": n_positions, "boards": args.boards}, f, indent=2)

    if args.cpu:
        worker(0, 1, args, n_positions, boards_path, [None])
        return
    if args.gpus:
        devices = [int(x) for x in args.gpus.split(",")]
    else:
        # Honour CUDA_VISIBLE_DEVICES (SLURM) like the trainers; torch then
        # numbers the visible devices 0..k-1.
        devices = list(range(torch.cuda.device_count()))
    if not devices:
        sys.exit("ERROR: no GPU visible (use --cpu for a test run)")
    world_size = len(devices)
    if world_size == 1:
        worker(0, 1, args, n_positions, boards_path, devices)
    else:
        mp.spawn(worker, args=(world_size, args, n_positions, boards_path, devices), nprocs=world_size)

    _, done = open_teacher_labels(args.cache_dir, n_positions, mode="r")
    print(f"labelled chunks: {int(np.asarray(done).sum())}/{n_chunks}", flush=True)


if __name__ == "__main__":
    main()
