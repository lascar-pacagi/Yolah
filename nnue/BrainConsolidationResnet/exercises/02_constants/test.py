"""Test for exercise 02 — constants."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fixtures"))
import _stub_yolah  # noqa: F401

from _test_utils import load_exercise


def test_num_actions():
    ex = load_exercise(__file__)
    assert ex.NUM_ACTIONS == 64 * 64, (
        f"NUM_ACTIONS should be 64*64 = 4096, got {ex.NUM_ACTIONS!r}")


def test_policy_ignore_index():
    ex = load_exercise(__file__)
    assert ex.POLICY_IGNORE_INDEX == -1, (
        f"POLICY_IGNORE_INDEX should be -1, got {ex.POLICY_IGNORE_INDEX!r}")


def test_in_channels():
    ex = load_exercise(__file__)
    assert ex.IN_CHANNELS == 4, (
        f"IN_CHANNELS should be 4, got {ex.IN_CHANNELS!r}")


def test_constants_are_ints():
    """Plain ints — not np.int64 wrappers or strings."""
    ex = load_exercise(__file__)
    for name in ("NUM_ACTIONS", "POLICY_IGNORE_INDEX", "IN_CHANNELS"):
        val = getattr(ex, name)
        assert isinstance(val, int), f"{name} should be a plain int, got {type(val)}"
        assert not isinstance(val, bool), f"{name} should be int, not bool"
