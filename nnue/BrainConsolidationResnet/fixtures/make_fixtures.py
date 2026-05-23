"""Build a tiny synthetic cache that mimics what preprocess.py produces.

Real cache: ~1B positions × 256 bytes ≈ 253 GB.
This fixture:    N=16384 positions × 256 bytes ≈ 4 MB.

Why 16384?
  • a multiple of 2048 → chunk math (CHUNK_SIZE=2048 for tests) divides cleanly;
  • big enough that within-chunk shuffling is meaningful;
  • small enough to regenerate in <1 s and commit-friendly disk.

We DO NOT commit the binaries — they regenerate deterministically (seed=0)
the first time runner.py is invoked.
"""
import json
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SEED = 0

# Test-scale constants. The real script uses CHUNK_SIZE=2048*2048; we shrink
# it for fixtures so loaders fit several chunks within 16K positions.
N_POSITIONS  = 16384
TEST_CHUNK   = 2048           # used by the loader's tests, NOT a real constant
IN_CHANNELS  = 4
NUM_ACTIONS  = 64 * 64        # 4096

# What the trainer-side meta.json schema looks like:
#   {
#     "n_positions": N,
#     "positions": {"path": "positions.u8", "dtype": "uint8", "shape": [N, 4, 8, 8]},
#     "values":    {"path": "values.i8",    "dtype": "int8",  "shape": [N]},
#     "policies":  {"path": "policies.i16", "dtype": "int16", "shape": [N]},
#     "encoding_planes": 4
#   }


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # ── positions: random binary planes ───────────────────────────────────
    # Each "position" is (4, 8, 8) uint8 in {0, 1}. We don't worry that they
    # form legal Yolah positions — the loader only cares about byte layout.
    positions = rng.integers(0, 2, size=(N_POSITIONS, IN_CHANNELS, 8, 8),
                             dtype=np.uint8)
    pos_path = CACHE / "positions.u8"
    positions.tofile(pos_path)

    # ── values: z ∈ {-1, 0, +1} ───────────────────────────────────────────
    values = rng.choice([-1, 0, 1], size=N_POSITIONS).astype(np.int8)
    val_path = CACHE / "values.i8"
    values.tofile(val_path)

    # ── policies: action index in [0, 4095], with ~10% terminal (-1) ──────
    policies = rng.integers(0, NUM_ACTIONS, size=N_POSITIONS).astype(np.int16)
    terminal = rng.random(N_POSITIONS) < 0.10
    policies[terminal] = -1
    pol_path = CACHE / "policies.i16"
    policies.tofile(pol_path)

    # ── meta.json sidecar (same schema as preprocess.py) ──────────────────
    meta = {
        "n_positions": N_POSITIONS,
        "n_games":     1234,                              # not used by loader
        "positions":   {"path": "positions.u8", "dtype": "uint8",
                        "shape": [N_POSITIONS, IN_CHANNELS, 8, 8]},
        "values":      {"path": "values.i8",    "dtype": "int8",
                        "shape": [N_POSITIONS]},
        "policies":    {"path": "policies.i16", "dtype": "int16",
                        "shape": [N_POSITIONS]},
        "encoding_planes": IN_CHANNELS,
        "_fixture": True,                                 # marker so tests can detect
    }
    with open(CACHE / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Fixture cache ready at {CACHE}:")
    print(f"  positions.u8  {positions.nbytes:>10,d} B")
    print(f"  values.i8     {values.nbytes:>10,d} B")
    print(f"  policies.i16  {policies.nbytes:>10,d} B")


if __name__ == "__main__":
    main()
