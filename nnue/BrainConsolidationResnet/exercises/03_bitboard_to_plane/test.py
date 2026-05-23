"""Test for exercise 03 — _bitboard_to_plane."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fixtures"))
import _stub_yolah  # noqa: F401

from _test_utils import load_exercise


def test_shape_and_dtype():
    ex = load_exercise(__file__)
    p = ex._bitboard_to_plane(0)
    assert p.shape == (8, 8), f"shape should be (8, 8), got {p.shape}"
    assert p.dtype == np.float32, f"dtype should be float32, got {p.dtype}"


def test_zero_bitboard_is_all_zeros():
    ex = load_exercise(__file__)
    p = ex._bitboard_to_plane(0)
    assert (p == 0).all(), "bitboard=0 must produce all-zero plane"


def test_all_ones_bitboard():
    ex = load_exercise(__file__)
    p = ex._bitboard_to_plane((1 << 64) - 1)
    assert (p == 1).all(), "bitboard=0xFFFF...F must produce all-one plane"


def test_lsb_lands_at_bottom_right():
    """bit 0 set → flat element 63 set → plane[7, 7] = 1."""
    ex = load_exercise(__file__)
    p = ex._bitboard_to_plane(1)
    assert p[7, 7] == 1.0
    # everything else zero
    p2 = p.copy()
    p2[7, 7] = 0
    assert (p2 == 0).all(), "only plane[7, 7] should be 1 for bitboard=1"


def test_msb_lands_at_top_left():
    """bit 63 set → flat element 0 set → plane[0, 0] = 1."""
    ex = load_exercise(__file__)
    p = ex._bitboard_to_plane(1 << 63)
    assert p[0, 0] == 1.0
    p2 = p.copy()
    p2[0, 0] = 0
    assert (p2 == 0).all(), "only plane[0, 0] should be 1 for bitboard=1<<63"


def test_matches_unpackbits_reference():
    """
    The function MUST match what preprocess.py writes. The reference is:
        np.unpackbits(np.array([n], dtype='>u8').view(np.uint8)).reshape(8,8)
    on a random sample of bitboards.
    """
    ex = load_exercise(__file__)
    rng = np.random.default_rng(42)
    for _ in range(64):
        n = int(rng.integers(0, 1 << 63))   # any 63-bit number
        n |= int(rng.integers(0, 1 << 63))   # mix some more bits
        expected = (np.unpackbits(np.array([n], dtype='>u8').view(np.uint8))
                      .astype(np.float32)
                      .reshape(8, 8))
        got = ex._bitboard_to_plane(n)
        assert np.array_equal(got, expected), (
            f"mismatch for n=0x{n:016x}\nexpected:\n{expected}\ngot:\n{got}"
        )


def test_specific_pattern():
    """A concrete pattern check — diagonal."""
    ex = load_exercise(__file__)
    # Bits 0, 9, 18, 27, 36, 45, 54, 63 set
    n = sum(1 << (i * 9) for i in range(8))
    p = ex._bitboard_to_plane(n)
    # In MSB-first order, bit (63 - flat_idx). Flat indices that are set:
    # 63-(0), 63-9, 63-18, 63-27, 63-36, 63-45, 63-54, 63-63 = 63,54,45,36,27,18,9,0
    expected_flat = np.zeros(64, dtype=np.float32)
    for k in (0, 9, 18, 27, 36, 45, 54, 63):
        expected_flat[63 - k] = 1.0
    assert np.array_equal(p.reshape(64), expected_flat)
