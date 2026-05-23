# 02 — Constants & encoding context

## Concepts

Three module-level constants set the shapes that every downstream piece —
preprocessor, dataset, network, loss — must agree on. Get one wrong and
either the data won't load or the loss will silently train on garbage.

### `NUM_ACTIONS = 64 * 64`

Yolah's move space is **(from_square, to_square)**, each square in
`0..63`. We flatten that 2-D action into a single index:

```
action_idx = from_sq * 64 + to_sq
```

So the policy head outputs a 4096-vector of logits and the policy target is
a single int in `[0, 4096)`.

This works because Yolah's board is 8×8 = 64 squares and the from/to pair
is unordered enough that we can represent every legal move as one of the
4096 indices. Illegal indices simply never appear as targets — but the
network may still output non-zero logits for them.

### `POLICY_IGNORE_INDEX = -1`

Some positions in the cache are **terminal** (game over, no next move).
They still have a value (z = the final outcome from current player's POV)
but no policy target. `preprocess.py` writes `-1` in `policies.i16` for
those.

`torch.nn.CrossEntropyLoss(ignore_index=-1)` skips rows whose target equals
`-1` when computing the loss — no contribution to gradient, but they still
flow through the forward pass.

Conceptual check: if you set this to any value in `[0, 4096)`, the policy
loss would silently include nonsense targets and the network would learn
to predict whatever index you picked for every terminal position. Bad.

### `IN_CHANNELS = 4`

Four planes per position:

| Plane | What                            |
|-------|---------------------------------|
| 0     | Black pieces (binary mask)      |
| 1     | White pieces (binary mask)      |
| 2     | Empty/scored squares            |
| 3     | Turn (all-0 = black, all-1 = white) |

`preprocess.py` writes exactly this layout. The loader checks
`meta["positions"]["shape"][1] == IN_CHANNELS` and refuses to start
otherwise — preventing the silent disaster of training a 4-plane net on
a 6-plane cache (or vice-versa).

## API

| Symbol                | Type | Value     | Used by                              |
|-----------------------|------|-----------|--------------------------------------|
| `NUM_ACTIONS`         | int  | 4096      | `Net.policy_fc`, CE loss target range|
| `POLICY_IGNORE_INDEX` | int  | -1        | `CrossEntropyLoss(ignore_index=…)`   |
| `IN_CHANNELS`         | int  | 4         | `Net.input_conv`, meta.json guard    |

## Diagram

```
Policy target encoding
──────────────────────

  legal move (a3 → b4):
       from = 16  (a3)
       to   = 25  (b4)
       action_idx = 16 * 64 + 25 = 1049

  terminal position:
       action_idx = -1   ←  CE loss skips this row

Cache layout match
──────────────────

  meta.json["positions"]["shape"]    →   [N, 4, 8, 8]   ✓ must match IN_CHANNELS
                                            ↑
                                            │
                                  IN_CHANNELS = 4
                                  Net.input_conv: Conv2d(IN_CHANNELS, …)
```

## Hints

<details>
<summary>Hint 1 — just three lines</summary>

Three plain integer assignments. No imports needed for this exercise.

</details>

<details>
<summary>Hint 2 — why `-1` and not e.g. `4097`?</summary>

`CrossEntropyLoss` expects targets in `[0, num_classes)` or `ignore_index`.
Choosing `-1` is conventional because it can't collide with any legal class
index (which are non-negative). Any negative integer would work, but `-1`
is the documented default of `ignore_index` in many APIs.

</details>
