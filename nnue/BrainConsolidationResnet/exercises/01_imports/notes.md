# 01 — Imports & module setup

## Concepts

The training script is one Python file, but the very first lines decide what
the rest of the program *can do*. Pick the wrong sharing strategy or the wrong
matmul precision and you'll spend hours later debugging crashes or speed.

Three categories of import live here:

1. **Big libraries.** `torch`, `numpy`, `tqdm`. Standard.
2. **`torch.multiprocessing` (`mp`) + DDP.** We will spawn one process per
   GPU with `mp.spawn`; each spawned process initialises a NCCL collective
   via `init_process_group` and is wrapped in `DistributedDataParallel`. All
   three names come from `torch.{multiprocessing, distributed,
   nn.parallel}` — they are separate things.
3. **Concurrency primitives for the loader.** `threading.Thread` for the
   background producer, `queue.Queue` for the bounded ping-pong buffer
   between producer and consumer, `gc` because we disable Python's cyclic
   GC inside `main()` so a Gen-2 collection cannot stall the GPU mid-step.

Two module-level *settings* matter:

- **TF32.** `torch.set_float32_matmul_precision('high')` lets Ampere+ GPUs
  use TF32 (19-bit mantissa) for fp32 matmuls. Small accuracy loss, ~2×
  speed-up. Safe for value+policy training; the loss landscape is smooth.
- **`file_system` IPC sharing.** PyTorch dataloader workers exchange shared
  tensors through `/dev/shm` by default. On SLURM + Singularity, `/dev/shm`
  is capped tiny by the cgroup → SIGBUS on the first batch. The
  `file_system` strategy puts those shared objects in regular tmp files
  instead.

The chunked loader doesn't actually use DataLoader workers — but we set the
sharing strategy anyway, because *any* torch IPC will go through it.

## API

| Symbol                                              | From                          | Why we need it                            |
|-----------------------------------------------------|-------------------------------|-------------------------------------------|
| `tqdm`                                              | `tqdm`                        | progress bar inside `_run_epoch`          |
| `torch`, `nn`                                       | `torch`, `torch.nn`           | tensors + model definition                |
| `mp`                                                | `torch.multiprocessing`       | `mp.spawn` for one-process-per-GPU launch |
| `DDP`                                               | `torch.nn.parallel`           | wraps the model for gradient all-reduce   |
| `init_process_group`, `destroy_process_group`       | `torch.distributed`           | rendezvous + cleanup for the NCCL group   |
| `os`, `sys`                                         | stdlib                        | env vars + sys.path for ../server         |
| `gc`                                                | stdlib                        | `gc.disable()` inside `main()`            |
| `json`                                              | stdlib                        | read `meta.json` from the cache           |
| `random`                                            | stdlib                        | epoch-level chunk-order shuffle           |
| `threading`                                         | stdlib                        | background producer thread                |
| `queue as queue_mod`                                | stdlib                        | bounded hand-off queue (consumer ↔ producer) |
| `np`                                                | `numpy`                       | memmap + bulk dtype casts                 |
| `Yolah, Move, Square`                               | `yolah` (sys.path ../server)  | only used by `encode_cnn` for inference   |

Two non-imports that must follow:

```python
torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')
```

(Note: there is no need to set the strategy a second time on `mp` — the
function lives on the public `torch.multiprocessing` namespace.)

## Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  module load                                                      │
│                                                                   │
│  ┌────────────┐   ┌─────────┐   ┌──────────┐   ┌──────────────┐   │
│  │ tqdm       │   │ torch / │   │ stdlib   │   │ ../server    │   │
│  │ numpy      │   │ mp / DDP│   │ os sys gc│   │ Yolah Move   │   │
│  │            │   │         │   │ threading│   │ Square       │   │
│  └────────────┘   └─────────┘   │ queue    │   └──────────────┘   │
│                                 │ random   │                      │
│                                 │ json     │                      │
│                                 └──────────┘                      │
│                       │                                           │
│                       ▼                                           │
│       set_float32_matmul_precision('high')      ← TF32 on Ampere+ │
│       set_sharing_strategy('file_system')       ← /dev/shm escape │
└──────────────────────────────────────────────────────────────────┘
```

## Hints

Peek only when stuck.

<details>
<summary>Hint 1 — naming</summary>

Six imports are aliased; the rest aren't. The aliases are
`tqdm.tqdm → tqdm` (already inside the package), `torch.nn → nn`,
`torch.multiprocessing → mp`, `queue → queue_mod` (the `queue` name shadows
a local variable elsewhere, hence the alias), and `numpy → np`. DDP is
`from torch.nn.parallel import DistributedDataParallel as DDP`.

</details>

<details>
<summary>Hint 2 — sys.path for the yolah module</summary>

The yolah module lives in `../server/yolah.py`. Before importing it, append
`"../server"` to `sys.path`. This is fine in the live trainer (file system
path resolves), and the test injects a stub so the import succeeds even
outside the real repo.

</details>

<details>
<summary>Hint 3 — order matters slightly</summary>

It is conventional (PEP 8) to group: (1) stdlib, (2) third-party,
(3) local. The original file groups by "kind of thing it does" instead
(`tqdm` first for visibility, then torch, then mp, then DDP, then dist,
then stdlib, then numpy, then yolah). Either ordering passes the test —
the test only checks that every required name is bound.

</details>
