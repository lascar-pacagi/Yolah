# 03 — `_bitboard_to_plane`

## Concepts

A Yolah position lives as **three 64-bit integers** (black bitboard, white
bitboard, empty bitboard). To feed it to a CNN we need three `(8, 8)`
float32 planes — one cell per board square. This function expands one
bitboard into one plane.

### What is a bitboard?

A 64-bit unsigned integer where **bit `s` corresponds to square `s`**.
Yolah numbers squares 0..63 starting from `a1` = bit 0:

```
bit 56 57 58 59 60 61 62 63           a8 b8 c8 d8 e8 f8 g8 h8
bit 48 49 50 51 52 53 54 55           a7 b7 c7 d7 e7 f7 g7 h7
   ...                                ...
bit  8  9 10 11 12 13 14 15           a2 b2 c2 d2 e2 f2 g2 h2
bit  0  1  2  3  4  5  6  7           a1 b1 c1 d1 e1 f1 g1 h1
```

To "have a black stone on square `s`" means bit `s` of the black bitboard
is `1`.

### Why MSB-first into the plane?

The training cache (`positions.u8`) was produced by `preprocess.py` using
`np.unpackbits` on a **big-endian** view of the bitboard:

```python
np.unpackbits(np.array([n], dtype='>u8').view(np.uint8))
```

This returns 64 bits **MSB first** — i.e. array element 0 holds the value
of bit 63, element 1 holds bit 62, …, element 63 holds bit 0. So the
flattened plane is in *reverse* square order compared to the Yolah enum.

That sounds wrong, but it's harmless:

1. A CNN is translation-equivariant. The absolute orientation of the 8×8
   plane doesn't matter — what matters is the *relative* positions of
   stones. Black-next-to-white looks the same whether the board is
   right-side up or upside down.
2. What **does** matter is that the inference encoder and the cached
   training tensors agree, *down to the bit*. If you change the orientation
   here, the trained network sees rotated boards at inference and produces
   garbage.

So this function MUST stay byte-identical to what `preprocess.py` writes.
Do not "fix" the orientation.

### The function

```python
def _bitboard_to_plane(n: int) -> np.ndarray:
    """64-bit int → (8, 8) float32 plane, MSB-first."""
```

Pure Python loop is fine here — this is *inference-time only* (called by
`encode_cnn`), not in the hot training path. Performance is not a concern.

## API

| Param  | Type           | Notes                                            |
|--------|----------------|--------------------------------------------------|
| `n`    | `int`          | 64-bit bitboard, conceptually a `uint64`         |
| return | `np.ndarray`   | shape `(8, 8)`, dtype `float32`, values in {0,1} |

## Diagram

```
Bitboard (uint64)
   63 62 61 60 59 58 57 56  ←  MSB
    1  0  1  0  0  1  0  0
   55 54 53 52 51 50 49 48
    ...
    7  6  5  4  3  2  1  0  ←  LSB

         │
         ▼  unpack MSB-first into flat (64,)

flat[i] = bit (63 - i) of n
   ┌──────────────────────────┐
i  │ 0  1  2  3  4  5  6  7   │  ← bit 63 .. bit 56
   │ 8  9 10 11 12 13 14 15   │  ← bit 55 .. bit 48
   │  ...                     │
   │56 57 58 59 60 61 62 63   │  ← bit  7 .. bit  0
   └──────────────────────────┘

         │
         ▼  .reshape(8, 8)

         (8, 8) float32 plane

Example
───────
n = 1                       (only bit 0 set, i.e. square a1)
flat[63] = 1, all others 0
plane[7, 7] = 1, rest 0    ← bottom-right of the plane

n = 1 << 63                 (only bit 63 set, top-left in Yolah enum)
flat[0] = 1, all others 0
plane[0, 0] = 1, rest 0    ← top-left of the plane
```

## Hints

<details>
<summary>Hint 1 — outline</summary>

Allocate a flat `(64,)` float32 array, loop `i` from 0 to 63, and set
`flat[i] = 1.0` iff bit `(63 - i)` of `n` is set. Then `.reshape(8, 8)`.

</details>

<details>
<summary>Hint 2 — testing bit (63 - i)</summary>

`(n >> (63 - i)) & 1` is one way. `n & (1 << (63 - i)) != 0` is another.

</details>

<details>
<summary>Hint 3 — vectorised alternative (off-topic but instructive)</summary>

```python
return np.unpackbits(np.array([n], dtype='>u8').view(np.uint8)) \
         .astype(np.float32).reshape(8, 8)
```

This is exactly what `preprocess.py` does. It is much faster than the
Python loop, but the loop version is easier to read — and this code path
is not on the hot loop, so either works for the test.

</details>
